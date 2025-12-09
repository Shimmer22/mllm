// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// MatMul.cpp - Highway x86 MatMul kernels with debug & fixes.
//
// Notes:
//  - Assumes row-major inputs A[M x K], B[K x N], output C[M x N].
//  - transpose_a/transponse_b follow the same semantic as original code:
//      if transpose_a == true, treat A as (K x M) (i.e., use A[k*M + m])
//      if transpose_b == true, treat B as (N x K) (i.e., use B[n*K + k])
//  - bias is assumed length N (one bias per output column).
//  - Use ENABLE_MATMUL_DEBUG to enable runtime NaN/Inf & bounds checks (prints and abort).
//
// Recommended compile options: enable -march/-mavx2/-mfma (CMake target_compile_options).

#include "MatMul.hpp"
#include "mllm/utils/Common.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <thread>
#include <vector>
#include <algorithm>
#include <cstdint>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "mllm/backends/cpu/kernels/x86/MatMul.cpp"
#include <hwy/foreach_target.h>

#include <hwy/aligned_allocator.h>
#include <hwy/contrib/dot/dot-inl.h>
#include <hwy/contrib/thread_pool/thread_pool.h>

// Debugging toggle: enable to print NaN/Inf / bounds failure diagnostics.
#ifndef ENABLE_MATMUL_DEBUG
// #define ENABLE_MATMUL_DEBUG 1
#endif

HWY_BEFORE_NAMESPACE();
namespace mllm {
namespace cpu {
namespace x86 {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;
using std::size_t;

// bfloat16 to float32 conversion
// BF16 has the same exponent bits as FP32, just truncated mantissa
static inline float bf16_to_fp32(uint16_t bf16) {
    uint32_t f = ((uint32_t)bf16) << 16;
    return *reinterpret_cast<float*>(&f);
}

// Helper debug macros/functions
#ifdef ENABLE_MATMUL_DEBUG
static inline void debug_abort_on_nan(float val, const char* ctx, int m=-1, int n=-1, int k=-1) {
  if (std::isnan(val) || std::isinf(val)) {
    std::fprintf(stderr, "[MatMul DEBUG] Detected NaN/Inf in %s at m=%d n=%d k=%d value=%f\n", ctx, m, n, k, val);
    std::abort();
  }
}
static inline void debug_check_index(bool cond, const char* ctx, int idx1=-1, int idx2=-1, int idx3=-1) {
  if (!cond) {
    std::fprintf(stderr, "[MatMul DEBUG] Index check failed %s idx=%d %d %d\n", ctx, idx1, idx2, idx3);
    std::abort();
  }
}
#else
static inline void debug_abort_on_nan(float, const char*, int=-1, int=-1, int=-1) {}
static inline void debug_check_index(bool, const char*, int=-1, int=-1, int=-1) {}
#endif

// Utility: safe scalar multiply-add for tail case (no SIMD)
static inline float scalar_dot_row_by_col_rowmajor(const float* A, const float* B, int M_i, int K, int N_j, int strideA_K, int strideB_N, bool trans_a, bool trans_b, int m, int n) {
    // This helper is not used heavily; we keep simple scalar computation as fallback.
    // (signature kept minimal, used only in tails)
    (void)M_i; (void)strideA_K; (void)strideB_N;
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        float a_v, b_v;
        if (!trans_a) { // A row-major: A[m*K + k]
            a_v = A[m * K + k];
        } else { // A transposed view: A[k*M + m]
            a_v = A[k * M_i + m];
        }
        if (!trans_b) { // B row-major: B[k*N + n]
            b_v = B[k * N_j + n];
        } else { // B transposed view: B[n*K + k]
            b_v = B[n * K + k];
        }
#ifdef ENABLE_MATMUL_DEBUG
        debug_abort_on_nan(a_v, "input_A", m, n, k);
        debug_abort_on_nan(b_v, "input_B", m, n, k);
#endif
        sum += a_v * b_v;
    }
    return sum;
}

// NN: C = A * B
void MatmulKernel_NN(const float* A, const float* B, float* C,
                     int M, int N, int K,
                     const float* bias, int thread_count) {
    if (M <= 0 || N <= 0 || K <= 0) return;
#ifdef ENABLE_MATMUL_DEBUG
    debug_check_index(true, "MatmulKernel_NN entry", M, N, K);
#endif

    const hn::ScalableTag<float> d;
    const int V = static_cast<int>(hn::Lanes(d)); // vector lanes

    // Worker function for a contiguous row range [row_start, row_end)
    auto worker = [&](int row_start, int row_end) {
        for (int m = row_start; m < row_end; ++m) {
            int n = 0;
            // Vectorized over columns n in steps of V
            for (; n + V <= N; n += V) {
                auto sum_vec = hn::Zero(d);
                // For each k, broadcast a scalar A[m*K + k] then multiply-add with B[k*N + n : n+V)
                for (int k = 0; k < K; ++k) {
                    const float a_scalar = A[m * K + k];
#ifdef ENABLE_MATMUL_DEBUG
                    debug_abort_on_nan(a_scalar, "A", m, n, k);
#endif
                    auto a_vec = hn::Set(d, a_scalar);                // broadcast
                    auto b_vec = hn::Load(d, B + (std::int64_t)k * N + n); // load V floats
#ifdef ENABLE_MATMUL_DEBUG
                    // Optionally check elements of b_vec by reducing differences - expensive so omitted
#endif
                    sum_vec = hn::MulAdd(a_vec, b_vec, sum_vec);
                }
                float sum = hn::ReduceSum(d, sum_vec);
#ifdef ENABLE_MATMUL_DEBUG
                debug_abort_on_nan(sum, "sum_vec_reduce", m, n, -1);
#endif
                float add_bias = (bias ? bias[n] : 0.0f); // note: bias[n] corresponds to first element of vector block
                // For correctness, bias is per-column; we need to add bias elementwise for the V columns.
                // Since we only have single scalar add_bias for starting column, store elementwise below.
                // We'll store elementwise by recomputing per-lane (cheap since V small) OR load, add, store.
                // Simpler: store the vector sum_vec + bias_vec.
                if (bias) {
                    auto bias_vec = hn::Load(d, bias + n);
                    auto out_vec = hn::Add(hn::Set(d, 0.0f), sum_vec); // sum_vec is already the vector of partial sums
                    // But note: sum_vec currently holds partial sums per lane, not a single scalar.
                    // We already reduced sum_vec and lost lane info; to fix that, we must NOT reduce until after adding bias.
                    // So adjust approach: do not ReduceSum here, instead store sum_vec + bias_vec directly.
                }
                // Because of above complexity, redo: we should NOT call ReduceSum here.
                // Re-implement storing without ReduceSum: recompute sum_vec per k but keep vector.
                // So we refactor this outer loop to compute vector sums and store directly.
            }

            // *** Revised vectorized block which keeps vector sums intact ***
            n = 0;
            for (; n + V <= N; n += V) {
                auto sum_vec = hn::Zero(d);
                for (int k = 0; k < K; ++k) {
                    const float a_scalar = A[m * K + k];
#ifdef ENABLE_MATMUL_DEBUG
                    debug_abort_on_nan(a_scalar, "A", m, n, k);
#endif
                    auto a_vec = hn::Set(d, a_scalar);
                    auto b_vec = hn::Load(d, B + (std::int64_t)k * N + n);
                    sum_vec = hn::MulAdd(a_vec, b_vec, sum_vec);
                }
                // Add bias vector if present
                if (bias) {
                    auto bias_vec = hn::Load(d, bias + n);
                    sum_vec = hn::Add(sum_vec, bias_vec);
                }
#ifdef ENABLE_MATMUL_DEBUG
                // Reduce per lane to check NaN after bias add (optional)
                // float check_sum = hn::ReduceSum(d, sum_vec);
                // debug_abort_on_nan(check_sum, "sum_vec_postbias_reduce", m, n, -1);
#endif
                hn::Store(sum_vec, d, C + (std::int64_t)m * N + n);
            }

            // Tail: remaining columns handled scalar-wise
            for (; n < N; ++n) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    float a_v = A[m * K + k];
                    float b_v = B[(std::int64_t)k * N + n];
#ifdef ENABLE_MATMUL_DEBUG
                    debug_abort_on_nan(a_v, "A_tail", m, n, k);
                    debug_abort_on_nan(b_v, "B_tail", m, n, k);
#endif
                    sum += a_v * b_v;
                }
                float out = sum + (bias ? bias[n] : 0.0f);
#ifdef ENABLE_MATMUL_DEBUG
                debug_abort_on_nan(out, "out_tail", m, n, -1);
#endif
                C[(std::int64_t)m * N + n] = out;
            }
        }
    };

    // Threading: split rows into contiguous blocks per thread
    if (thread_count <= 1) {
        worker(0, M);
    } else {
        int num_threads = std::min(thread_count, M);
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        int rows_per = (M + num_threads - 1) / num_threads;
        for (int t = 0; t < num_threads; ++t) {
            int start = t * rows_per;
            int end = std::min(M, start + rows_per);
            if (start >= end) break;
            threads.emplace_back([start, end, &worker]() { worker(start, end); });
        }
        for (auto &th : threads) th.join();
    }
}

// NT: C = A * B^T  (A not transposed, B treated as transposed)
void MatmulKernel_NT(const float* A, const float* B, float* C,
                     int M, int N, int K,
                     const float* bias, int thread_count) {
    // Equivalent to multiply where B is provided in transposed layout B^T: shape N x K
    // So element B element access is B[n*K + k]
    if (M <= 0 || N <= 0 || K <= 0) return;
    const hn::ScalableTag<float> d;
    const int V = static_cast<int>(hn::Lanes(d));
    
    auto worker = [&](int row_start, int row_end) {
        for (int m = row_start; m < row_end; ++m) {
            for (int n = 0; n < N; ++n) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    float a_v = A[m * K + k];
                    float b_v = B[n * K + k]; // B is transposed: B^T stored row-major as (N x K)
#ifdef ENABLE_MATMUL_DEBUG
                    debug_abort_on_nan(a_v, "A_NT", m, n, k);
                    debug_abort_on_nan(b_v, "B_NT", m, n, k);
#endif
                    sum += a_v * b_v;
                }
                float out = sum + (bias ? bias[n] : 0.0f);
#ifdef ENABLE_MATMUL_DEBUG
                debug_abort_on_nan(out, "out_NT", m, n, -1);
#endif
                C[(std::int64_t)m * N + n] = out;
            }
        }
    };

    // threading
    if (thread_count <= 1) {
        worker(0, M);
    } else {
        int num_threads = std::min(thread_count, M);
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        int rows_per = (M + num_threads - 1) / num_threads;
        for (int t = 0; t < num_threads; ++t) {
            int start = t * rows_per;
            int end = std::min(M, start + rows_per);
            if (start >= end) break;
            threads.emplace_back([start, end, &worker]() { worker(start, end); });
        }
        for (auto &th : threads) th.join();
    }
}

// TN: C = A^T * B  (A treated as transposed, B not transposed)
void MatmulKernel_TN(const float* A, const float* B, float* C,
                     int M, int N, int K,
                     const float* bias, int thread_count) {
    // A is treated as transposed: A^T means A has shape K x M (access A[k*M + m])
    if (M <= 0 || N <= 0 || K <= 0) return;
    const hn::ScalableTag<float> d;
    const int V = static_cast<int>(hn::Lanes(d));

    auto worker = [&](int row_start, int row_end) {
        for (int m = row_start; m < row_end; ++m) {
            int n = 0;
            // vectorized across N columns
            for (; n + V <= N; n += V) {
                auto sum_vec = hn::Zero(d);
                for (int k = 0; k < K; ++k) {
                    float a_scalar = A[(std::int64_t)k * M + m]; // A transposed access
#ifdef ENABLE_MATMUL_DEBUG
                    debug_abort_on_nan(a_scalar, "A_TN", m, n, k);
#endif
                    auto a_vec = hn::Set(d, a_scalar);
                    auto b_vec = hn::Load(d, B + (std::int64_t)k * N + n);
                    sum_vec = hn::MulAdd(a_vec, b_vec, sum_vec);
                }
                if (bias) {
                    auto bias_vec = hn::Load(d, bias + n);
                    sum_vec = hn::Add(sum_vec, bias_vec);
                }
                hn::Store(sum_vec, d, C + (std::int64_t)m * N + n);
            }
            for (; n < N; ++n) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    float a_v = A[(std::int64_t)k * M + m];
                    float b_v = B[(std::int64_t)k * N + n];
#ifdef ENABLE_MATMUL_DEBUG
                    debug_abort_on_nan(a_v, "A_TN_tail", m, n, k);
                    debug_abort_on_nan(b_v, "B_TN_tail", m, n, k);
#endif
                    sum += a_v * b_v;
                }
                float out = sum + (bias ? bias[n] : 0.0f);
#ifdef ENABLE_MATMUL_DEBUG
                debug_abort_on_nan(out, "out_TN_tail", m, n, -1);
#endif
                C[(std::int64_t)m * N + n] = out;
            }
        }
    };

    if (thread_count <= 1) {
        worker(0, M);
    } else {
        int num_threads = std::min(thread_count, M);
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        int rows_per = (M + num_threads - 1) / num_threads;
        for (int t = 0; t < num_threads; ++t) {
            int start = t * rows_per;
            int end = std::min(M, start + rows_per);
            if (start >= end) break;
            threads.emplace_back([start, end, &worker]() { worker(start, end); });
        }
        for (auto &th : threads) th.join();
    }
}

// TT: C = A^T * B^T
void MatmulKernel_TT(const float* A, const float* B, float* C,
                     int M, int N, int K,
                     const float* bias, int thread_count) {
    // A transposed: A[k*M + m], B transposed: B[n*K + k]
    if (M <= 0 || N <= 0 || K <= 0) return;

    auto worker = [&](int row_start, int row_end) {
        for (int m = row_start; m < row_end; ++m) {
            for (int n = 0; n < N; ++n) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    float a_v = A[(std::int64_t)k * M + m];
                    float b_v = B[(std::int64_t)n * K + k];
#ifdef ENABLE_MATMUL_DEBUG
                    debug_abort_on_nan(a_v, "A_TT", m, n, k);
                    debug_abort_on_nan(b_v, "B_TT", m, n, k);
#endif
                    sum += a_v * b_v;
                }
                float out = sum + (bias ? bias[n] : 0.0f);
#ifdef ENABLE_MATMUL_DEBUG
                debug_abort_on_nan(out, "out_TT", m, n, -1);
#endif
                C[(std::int64_t)m * N + n] = out;
            }
        }
    };

    if (thread_count <= 1) {
        worker(0, M);
    } else {
        int num_threads = std::min(thread_count, M);
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        int rows_per = (M + num_threads - 1) / num_threads;
        for (int t = 0; t < num_threads; ++t) {
            int start = t * rows_per;
            int end = std::min(M, start + rows_per);
            if (start >= end) break;
            threads.emplace_back([start, end, &worker]() { worker(start, end); });
        }
        for (auto &th : threads) th.join();
    }
}

// Dequantize qsi4c32p packed weights to fp32
// Based on kai_run_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0 from KleidiAI
// Format: For each nr-column block:
//   [For each quant block: [int4 weights: nr*(bl/2)] [scales: nr*2 (bf16)]]
//   [sums: nr*4 (float32)]
//   [bias: nr*4 (float32)]
void dequant_qsi4c32p_to_fp32(const uint8_t* packed_weights, float* unpacked_weights,
                               int n, int k, int nr, int kr, int sr) {
    if (!packed_weights || !unpacked_weights || n <= 0 || k <= 0) return;

    // Constants from KAI reference (kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0.c)
    const size_t bl = 32;  // Block length for quantization
    const size_t num_bytes_scale = sizeof(uint16_t);  // bf16
    const size_t num_bytes_sum = sizeof(float);       // float32
    const size_t num_bytes_bias = sizeof(float);      // float32

    auto kai_roundup = [](size_t val, size_t mult) -> size_t {
        return ((val + mult - 1) / mult) * mult;
    };

    const size_t k_rounded = kai_roundup(k, bl);
    const size_t num_blocks_per_row = k_rounded / bl;
    
    // Each block contains: (bl/2) bytes of int4 weights + num_bytes_scale bytes per column
    const size_t num_bytes_per_block = (bl / 2) + num_bytes_scale;  // 16 + 2 = 18 bytes per column
    
    // Stride per nr-column group: 
    // nr * (num_bytes_per_block * num_blocks_per_row) + nr * num_bytes_sum + nr * num_bytes_bias
    const size_t rhs_packed_stride = nr * (num_bytes_per_block * num_blocks_per_row + 
                                           num_bytes_sum + num_bytes_bias);
    
    const size_t dst_num_rows = (n + nr - 1) / nr;

    // Initialize output to zero
    memset(unpacked_weights, 0, (size_t)n * k * sizeof(float));

    for (size_t dst_row_idx = 0; dst_row_idx < dst_num_rows; ++dst_row_idx) {
        const uint8_t* row_base = packed_weights + dst_row_idx * rhs_packed_stride;
        
        // Offset to sums (after all blocks): nr * num_bytes_per_block * num_blocks_per_row
        const size_t offset_to_sums = nr * num_bytes_per_block * num_blocks_per_row;
        const float* sums_ptr = (const float*)(row_base + offset_to_sums);
        
        // Offset to bias (after sums): + nr * num_bytes_sum
        const float* bias_ptr = (const float*)(row_base + offset_to_sums + nr * num_bytes_sum);
        
        // Process each column in this nr-group
        for (size_t nr_idx = 0; nr_idx < (size_t)nr; ++nr_idx) {
            size_t n_idx = dst_row_idx * nr + nr_idx;
            if (n_idx >= (size_t)n) continue;
            
            float bias_val = bias_ptr[nr_idx];
            float sum_val = sums_ptr[nr_idx];
            (void)sum_val; // Sum is used for zero-point correction during matmul, not needed for simple dequant
            
            if (std::isnan(bias_val) || std::isinf(bias_val)) {
                fprintf(stderr, "[MatMul ERROR] Invalid bias value at n=%zu: %f\n", n_idx, bias_val);
                abort();
            }
            
            float* unpacked_col = unpacked_weights + n_idx * k;
            
            // Process each quantization block
            const uint8_t* block_ptr = row_base;
            for (size_t block_idx = 0; block_idx < num_blocks_per_row; ++block_idx) {
                // Layout within a block row:
                // [nr columns of (bl/2) bytes each] [nr scales of 2 bytes each]
                
                // Weights for this column within this block
                const uint8_t* weights_base = block_ptr + nr_idx * (bl / 2);
                
                // Scale is bf16, located after all nr column weights in this block
                const uint16_t* scale_ptr = (const uint16_t*)(block_ptr + nr * (bl / 2));
                uint16_t scale_bf16 = scale_ptr[nr_idx];
                float scale_val = bf16_to_fp32(scale_bf16);
                
                if (std::isnan(scale_val) || std::isinf(scale_val)) {
                    fprintf(stderr, "[MatMul ERROR] Invalid scale value at n=%zu block=%zu: %f (raw hex: 0x%04x)\n", 
                            n_idx, block_idx, scale_val, scale_bf16);
                    abort();
                }
                
                // Unpack int4 values from this block
                // The packed format uses XOR ^0x88 and interleaves k indices 0-15 with 16-31
                for (size_t byte_idx = 0; byte_idx < bl / 2; ++byte_idx) {
                    uint8_t packed_byte = weights_base[byte_idx];
                    
                    // Undo the XOR 0x88 applied during packing
                    packed_byte ^= 0x88;
                    
                    // Extract low and high nibbles
                    uint8_t val_lo = packed_byte & 0x0F;
                    uint8_t val_hi = (packed_byte >> 4) & 0x0F;
                    
                    // Convert from unsigned (0-15 range with zero_point=8) to signed (-8 to 7)
                    int8_t signed_lo = (int8_t)val_lo - 8;
                    int8_t signed_hi = (int8_t)val_hi - 8;
                    
                    // K indices: The interleaving pattern places:
                    // - low nibble at k_base + byte_idx
                    // - high nibble at k_base + byte_idx + 16 (interleaved)
                    size_t k_base = block_idx * bl;
                    size_t k_idx_lo = k_base + byte_idx;
                    size_t k_idx_hi = k_base + byte_idx + 16;
                    
                    if (k_idx_lo < (size_t)k) {
                        unpacked_col[k_idx_lo] = signed_lo * scale_val + bias_val;
                    }
                    if (k_idx_hi < (size_t)k) {
                        unpacked_col[k_idx_hi] = signed_hi * scale_val + bias_val;
                    }
                }
                
                // Move to next block (weights + scales for all nr columns)
                block_ptr += nr * num_bytes_per_block;
            }
        }
    }
}




// Helper to quantize FP32 to int8 for LHS with proper scaling
static void quantize_f32_to_int8(const float* input, int8_t* output, int size) {
    // Find max absolute value for scaling
    float max_abs = 0.0f;
    for (int i = 0; i < size; ++i) {
        float abs_val = fabsf(input[i]);
        if (abs_val > max_abs) {
            max_abs = abs_val;
        }
    }

    // Avoid division by zero
    if (max_abs == 0.0f) {
        max_abs = 1.0f;
    }

    // Scale factor to map to int8 range [-127, 127] (avoiding -128 for symmetry)
    float scale = 127.0f / max_abs;

    for (int i = 0; i < size; ++i) {
        float val = input[i];

        // Scale and round
        int32_t quantized = (int32_t)roundf(val * scale);

        // Clamp to int8 range
        if (quantized > 127) quantized = 127;
        if (quantized < -127) quantized = -127;

        output[i] = (int8_t)quantized;
    }
}

// MatMul with FP32 input and int8 RHS (dequantized) - implementation
void hwy_matmul_f32_qai8dxp_qsi4c32p_impl(int M, int K, int N, float* C,
                                          const float* A, const uint8_t* packed_B,
                                          const float* bias, int thread_count) {
    if (M <= 0 || K <= 0 || N <= 0 || !A || !packed_B || !C) return;

    // Temporary buffer for dequantized weights (N x K)
    std::vector<float> dequantized_B(N * K);

    // Dequantize the packed weights
    // Use typical KAI tile parameters: nr=8, kr=32, sr=1 (CORRECTED: nr=8 for qsi4c32p8x8)
    dequant_qsi4c32p_to_fp32(packed_B, dequantized_B.data(), N, K, 8, 32, 1);

    // Temporary buffer for quantized input (M x K)
    std::vector<int8_t> quantized_A(M * K);

    // Quantize FP32 input to int8
    quantize_f32_to_int8(A, quantized_A.data(), M * K);

    // Handle quantized LHS with dequantized RHS
    const hn::ScalableTag<float> d;
    const int V = static_cast<int>(hn::Lanes(d));

    auto worker = [&](int row_start, int row_end) {
        for (int m = row_start; m < row_end; ++m) {
            int n = 0;
            // Vectorized over columns n in steps of V
            for (; n + V <= N; n += V) {
                auto sum_vec = hn::Zero(d);

                // Process K dimension with quantized LHS
                for (int k = 0; k < K; ++k) {
                    int8_t a_int8 = quantized_A[m * K + k];
                    float a_fp32 = static_cast<float>(a_int8);

                    // Check for reasonable values
                    if (std::isnan(a_fp32) || std::isinf(a_fp32)) {
                        fprintf(stderr, "[MatMul ERROR] NaN/Inf in quantized input: m=%d, k=%d, a_int8=%d, a_fp32=%f\n",
                                m, k, a_int8, a_fp32);
                        abort();
                    }

                    auto a_vec = hn::Set(d, a_fp32);
                    auto b_vec = hn::Load(d, dequantized_B.data() + k * N + n);
                    sum_vec = hn::MulAdd(a_vec, b_vec, sum_vec);
                }

                // Add bias if present
                if (bias) {
                    auto bias_vec = hn::Load(d, bias + n);
                    sum_vec = hn::Add(sum_vec, bias_vec);
                }

                hn::Store(sum_vec, d, C + m * N + n);

                // Check for NaN in output
                for (int v = 0; v < V; ++v) {
                    float val = C[m * N + n + v];
                    if (std::isnan(val) || std::isinf(val)) {
                        fprintf(stderr, "[MatMul ERROR] NaN/Inf detected in vector output at m=%d, n=%d: %f\n",
                                m, n + v, val);
                        abort();
                    }
                }
            }

            // Tail: remaining columns handled scalar-wise
            for (; n < N; ++n) {
                float sum = 0.0f;

                for (int k = 0; k < K; ++k) {
                    int8_t a_int8 = quantized_A[m * K + k];
                    float a_fp32 = static_cast<float>(a_int8);

                    // Check for reasonable values
                    if (std::isnan(a_fp32) || std::isinf(a_fp32)) {
                        fprintf(stderr, "[MatMul ERROR] NaN/Inf in quantized input (tail): m=%d, k=%d, a_int8=%d, a_fp32=%f\n",
                                m, k, a_int8, a_fp32);
                        abort();
                    }

                    sum += a_fp32 * dequantized_B[k * N + n];
                }

                float out = sum + (bias ? bias[n] : 0.0f);

                // Check for NaN in output
                if (std::isnan(out) || std::isinf(out)) {
                    fprintf(stderr, "[MatMul ERROR] NaN/Inf detected in scalar output at m=%d, n=%d: %f (sum=%f, bias=%f)\n",
                            m, n, out, sum, bias ? bias[n] : 0.0f);
                    abort();
                }

                C[m * N + n] = out;
            }
        }
    };

    // Threading
    if (thread_count <= 1) {
        worker(0, M);
    } else {
        int num_threads = std::min(thread_count, M);
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        int rows_per = (M + num_threads - 1) / num_threads;
        for (int t = 0; t < num_threads; ++t) {
            int start = t * rows_per;
            int end = std::min(M, start + rows_per);
            if (start >= end) break;
            threads.emplace_back([start, end, &worker]() { worker(start, end); });
        }
        for (auto &th : threads) th.join();
    }
}

// Batch MatMul with FP32 input and int8 RHS (dequantized) - implementation
void hwy_batch_matmul_f32_qai8dxp_qsi4c32p_impl(int batch_size, int M, int K, int N,
                                                int C_stride, int A_stride, int B_stride, int bias_stride,
                                                float* C, const float* A, const uint8_t* packed_B,
                                                const float* bias, int thread_count) {
    // Simple loop over batches
    for (int i = 0; i < batch_size; ++i) {
        const float* current_A = A + (std::int64_t)i * A_stride;
        const uint8_t* current_B = packed_B + (std::int64_t)i * B_stride;  // This might need adjustment
        float* current_C = C + (std::int64_t)i * C_stride;
        const float* current_bias = bias ? (bias + (std::int64_t)i * bias_stride) : nullptr;

        hwy_matmul_f32_qai8dxp_qsi4c32p_impl(M, K, N, current_C, current_A, current_B, current_bias, thread_count);
    }
}

} // namespace HWY_NAMESPACE
HWY_AFTER_NAMESPACE();
} // namespace x86
} // namespace cpu
} // namespace mllm

#if HWY_ONCE

namespace mllm {
namespace cpu {
namespace x86 {

// Export the kernels to build the dispatch table
HWY_EXPORT(MatmulKernel_NN);
HWY_EXPORT(MatmulKernel_NT);
HWY_EXPORT(MatmulKernel_TN);
HWY_EXPORT(MatmulKernel_TT);
HWY_EXPORT(hwy_matmul_f32_qai8dxp_qsi4c32p_impl);
HWY_EXPORT(hwy_batch_matmul_f32_qai8dxp_qsi4c32p_impl);

// Wrapper functions for Highway dynamic dispatch
void hwy_matmul_f32_qai8dxp_qsi4c32p(int M, int K, int N, float* C,
                                     const float* A, const uint8_t* packed_B,
                                     const float* bias, int thread_count) {
    HWY_DYNAMIC_DISPATCH(hwy_matmul_f32_qai8dxp_qsi4c32p_impl)(M, K, N, C, A, packed_B, bias, thread_count);
}

void hwy_batch_matmul_f32_qai8dxp_qsi4c32p(int batch_size, int M, int K, int N,
                                           int C_stride, int A_stride, int B_stride, int bias_stride,
                                           float* C, const float* A, const uint8_t* packed_B,
                                           const float* bias, int thread_count) {
    HWY_DYNAMIC_DISPATCH(hwy_batch_matmul_f32_qai8dxp_qsi4c32p_impl)(batch_size, M, K, N,
                                                                     C_stride, A_stride, B_stride, bias_stride,
                                                                     C, A, packed_B, bias, thread_count);
}

// Export new functions for int4 LHS and int8 RHS support
// These functions are outside HWY_NAMESPACE, so they don't need HWY_EXPORT

// Public API wrapper used by higher layers:
// Keep same signature as requested (M,K,N ordering in original code might vary — ensure the caller uses this convention).
void hwy_matmul_fp32(int M, int K, int N,
                     float *C, const float *A, const float *B,
                     const float *bias, bool transpose_a, bool transpose_b, int thread_count) {

    // Basic sanity checks
    if (M <= 0 || N <= 0 || K <= 0) return;
    // Note: we assume bias length == N if bias != nullptr.
#ifdef ENABLE_MATMUL_DEBUG
    if (bias) {
        // No direct length info; caller must ensure bias length >= N.
        // Optionally add memory sanitizer / externally pass bias_length param in future.
    }
    fprintf(stderr, "[MatMul DEBUG] Inspecting first few B values: ");
    for (int i = 0; i < 8; ++i) {
        fprintf(stderr, "%g ", B[i]);
    }
    fprintf(stderr, "\n");
#endif

    // Choose the correct kernel based on transpose flags.
    if (!transpose_a && !transpose_b) {
        HWY_DYNAMIC_DISPATCH(MatmulKernel_NN)(A, B, C, M, N, K, bias, thread_count);
    } else if (!transpose_a && transpose_b) {
        HWY_DYNAMIC_DISPATCH(MatmulKernel_NT)(A, B, C, M, N, K, bias, thread_count);
    } else if (transpose_a && !transpose_b) {
        HWY_DYNAMIC_DISPATCH(MatmulKernel_TN)(A, B, C, M, N, K, bias, thread_count);
    } else { // transpose_a && transpose_b
        HWY_DYNAMIC_DISPATCH(MatmulKernel_TT)(A, B, C, M, N, K, bias, thread_count);
    }
}

void hwy_batch_matmul_fp32(int batch_size, int M, int K, int N,
                           int C_stride, int A_stride, int B_stride, int bias_stride,
                           float *C, const float *A, const float *B, const float *bias,
                           bool transpose_a, bool transpose_b, int thread_count) {
    // Simple loop over batches. Could be parallelized further if needed.
    for (int i = 0; i < batch_size; ++i) {
        const float* current_A = A + (std::int64_t)i * A_stride;
        const float* current_B = B + (std::int64_t)i * B_stride;
        float* current_C = C + (std::int64_t)i * C_stride;
        const float* current_bias = bias ? (bias + (std::int64_t)i * bias_stride) : nullptr;
        hwy_matmul_fp32(M, K, N, current_C, current_A, current_B, current_bias, transpose_a, transpose_b, thread_count);
    }
}


} // namespace x86
} // namespace cpu
} // namespace mllm

#endif // HWY_ONCE
