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
// EXACT reverse of kai_run_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0
// 
// Parameters from kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm:
//   nr = 8, kr = 16, sr = 2, bl = 32
//   block_length_in_bytes = kr / sr = 8
//
// Pack layout per nr-column group:
//   For each block (bl=32): [interleaved int4 data] [nr*2 bytes of bf16 scales]
//   After all blocks: [nr*4 bytes float sums] [nr*4 bytes float bias]
//
// Critical insight from packing code:
//   Each uint16 (2 bytes) is packed as:
//     bits 0-3:   k0 low nibble
//     bits 4-7:   k0+16 low nibble
//     bits 8-11:  k0+1 high nibble
//     bits 12-15: k0+16+1 high nibble
//   XOR 0x8888 applied, zero_point = 8
void dequant_qsi4c32p_to_fp32(const uint8_t* packed_weights, float* unpacked_weights,
                               int n, int k, int nr, int kr, int sr) {
    if (!packed_weights || !unpacked_weights || n <= 0 || k <= 0) return;

    const size_t bl = 32;  // Block length (quantization block size)
    const size_t num_bytes_scale = sizeof(uint16_t);  // bf16 = 2 bytes
    const size_t num_bytes_sum = sizeof(float);       // 4 bytes
    const size_t num_bytes_bias = sizeof(float);      // 4 bytes
    
    // Use actual parameters: kr=16, sr=2 for qsi4c32p8x8_4x8x32
    const size_t block_length_in_bytes = kr / sr;  // = 16/2 = 8

    auto kai_roundup = [](size_t val, size_t mult) -> size_t {
        return ((val + mult - 1) / mult) * mult;
    };

    const size_t k_rounded = kai_roundup(k, bl);
    const size_t num_blocks_per_row = k_rounded / bl;
    const size_t num_bytes_per_block_k = bl / 2;  // 16 bytes = 32 nibbles
    
    // Per-block: [nr columns × 16 bytes int4] + [nr × 2 bytes bf16 scale]
    const size_t num_bytes_int4_per_block = nr * num_bytes_per_block_k;
    const size_t num_bytes_scale_per_block = nr * num_bytes_scale;
    const size_t num_bytes_per_block_total = num_bytes_int4_per_block + num_bytes_scale_per_block;
    
    // Offset to sums (after all blocks)
    const size_t offset_to_sums = num_bytes_per_block_total * num_blocks_per_row;
    
    // Stride per nr-column group
    const size_t rhs_packed_stride = offset_to_sums + nr * num_bytes_sum + nr * num_bytes_bias;
    
    const size_t dst_num_rows = kai_roundup(n, nr);

    // Initialize output to zero
    memset(unpacked_weights, 0, (size_t)n * k * sizeof(float));

    // Debug print once
    static bool debug_printed = false;

    for (size_t dst_row_idx = 0; dst_row_idx < dst_num_rows; dst_row_idx += nr) {
        const uint8_t* row_base = packed_weights + (dst_row_idx / nr) * rhs_packed_stride;
        
        // Process each quantization block
        const uint8_t* block_ptr = row_base;
        for (size_t block_idx = 0; block_idx < num_blocks_per_row; ++block_idx) {
            // Scale for this block (bf16) - located after all int4 data in block
            const uint16_t* scale_base = (const uint16_t*)(block_ptr + num_bytes_int4_per_block);
            
            // Create local scale array for this block
            float scales[8];
            for (size_t i = 0; i < (size_t)nr && (dst_row_idx + i) < (size_t)n; ++i) {
                scales[i] = bf16_to_fp32(scale_base[i]);
            }
            
            if (!debug_printed && block_idx == 0 && dst_row_idx == 0) {
                fprintf(stderr, "[DEBUG v5] scale_bf16[0]=0x%04x scale_val=%f\n",
                        scale_base[0], scales[0]);
                fprintf(stderr, "[DEBUG v5] kr=%d sr=%d block_length_in_bytes=%zu\n",
                        kr, sr, block_length_in_bytes);
                fprintf(stderr, "[DEBUG v5] rhs_packed_stride=%zu offset_to_sums=%zu\n",
                        rhs_packed_stride, offset_to_sums);
                fprintf(stderr, "[DEBUG v5] num_blocks_per_row=%zu\n", num_blocks_per_row);
                fprintf(stderr, "[DEBUG v5] First 16 bytes of block: ");
                for (int i = 0; i < 16; i++) {
                    fprintf(stderr, "0x%02x ", block_ptr[i]);
                }
                fprintf(stderr, "\n");
                
                // Print first 4 uint16 values after XOR
                fprintf(stderr, "[DEBUG v5] First 4 uint16 values (col 0, seg 0):\n");
                for (int i = 0; i < 4; i++) {
                    uint16_t u16 = *((const uint16_t*)(block_ptr + i*2)) ^ 0x8888;
                    uint8_t x0_lo = (u16 >> 0) & 0xF;
                    uint8_t x0_hi = (u16 >> 4) & 0xF;
                    uint8_t x1_lo = (u16 >> 8) & 0xF;
                    uint8_t x1_hi = (u16 >> 12) & 0xF;
                    fprintf(stderr, "  u16[%d] = 0x%04x -> k%d=%d k%d=%d k%d=%d k%d=%d\n",
                            i, u16, i*2, (int)x0_lo-8, i*2+16, (int)x0_hi-8,
                            i*2+1, (int)x1_lo-8, i*2+17, (int)x1_hi-8);
                }
            }
            
            // Mirror the exact packing loop structure:
            // for (dst_byte_idx = 0; dst_byte_idx < 16; dst_byte_idx += 16)  -- only once
            //   for (segment_idx = 0; segment_idx < 16/8; segment_idx++)  -- 2 times
            //     for (nr_idx = 0; nr_idx < 8; nr_idx++)  -- 8 columns
            //       for (block_byte_idx = 0; block_byte_idx < 8; block_byte_idx += 2) -- 4 uint16s
            
            const uint8_t* src_ptr = block_ptr;
            size_t k0_idx_i = block_idx * bl;
            
            for (size_t dst_byte_idx = 0; dst_byte_idx < num_bytes_per_block_k; dst_byte_idx += 16) {
                for (size_t segment_idx = 0; segment_idx < 16 / block_length_in_bytes; ++segment_idx) {
                    for (size_t nr_idx = 0; nr_idx < (size_t)nr; ++nr_idx) {
                        size_t n_idx = dst_row_idx + nr_idx;
                        if (n_idx >= (size_t)n) {
                            // Skip padding columns but still advance pointer
                            src_ptr += block_length_in_bytes;
                            continue;
                        }
                        
                        float* unpacked_col = unpacked_weights + n_idx * k;
                        float scale_val = scales[nr_idx];
                        
                        size_t k0_idx = k0_idx_i;
                        size_t k1_idx = k0_idx_i + 16;
                        
                        for (size_t block_byte_idx = 0; block_byte_idx < block_length_in_bytes; block_byte_idx += 2) {
                            // Read uint16 and undo XOR
                            uint16_t packed_u16 = *((const uint16_t*)src_ptr) ^ 0x8888;
                            
                            // Extract 4 nibbles
                            uint8_t src_x0_lo = (packed_u16 >> 0) & 0x0F;   // k0 low nibble
                            uint8_t src_x0_hi = (packed_u16 >> 4) & 0x0F;   // k0+16 low nibble
                            uint8_t src_x1_lo = (packed_u16 >> 8) & 0x0F;   // k0+1 high nibble
                            uint8_t src_x1_hi = (packed_u16 >> 12) & 0x0F;  // k0+16+1 high nibble
                            
                            // Convert to signed: subtract zero_point (8)
                            int8_t val_k0 = (int8_t)src_x0_lo - 8;      // k = k0
                            int8_t val_k16 = (int8_t)src_x0_hi - 8;     // k = k0 + 16
                            int8_t val_k1 = (int8_t)src_x1_lo - 8;      // k = k0 + 1
                            int8_t val_k17 = (int8_t)src_x1_hi - 8;     // k = k0 + 16 + 1
                            
                            // Store dequantized values
                            if (k0_idx < (size_t)k) {
                                unpacked_col[k0_idx] = val_k0 * scale_val;
                            }
                            if (k1_idx < (size_t)k) {
                                unpacked_col[k1_idx] = val_k16 * scale_val;
                            }
                            if (k0_idx + 1 < (size_t)k) {
                                unpacked_col[k0_idx + 1] = val_k1 * scale_val;
                            }
                            if (k1_idx + 1 < (size_t)k) {
                                unpacked_col[k1_idx + 1] = val_k17 * scale_val;
                            }
                            
                            k0_idx += 2;
                            k1_idx += 2;
                            src_ptr += 2;
                        }
                    }
                    k0_idx_i += block_length_in_bytes;
                }
                k0_idx_i += 16;
            }
            
            // Skip past scales
            block_ptr += num_bytes_per_block_total;
        }
        
        // Print first weights after first row group is processed
        if (!debug_printed && dst_row_idx == 0) {
            debug_printed = true;
            fprintf(stderr, "[DEBUG v5] After unpacking, col 0 weights [0..7]: ");
            for (int i = 0; i < 8 && i < k; i++) {
                fprintf(stderr, "%.4f ", unpacked_weights[0 * k + i]);
            }
            fprintf(stderr, "\n");
            fprintf(stderr, "[DEBUG v5] After unpacking, col 0 weights [16..23]: ");
            for (int i = 16; i < 24 && i < k; i++) {
                fprintf(stderr, "%.4f ", unpacked_weights[0 * k + i]);
            }
            fprintf(stderr, "\n");
            
            // Calculate checksum for col 0
            float checksum = 0;
            for (int i = 0; i < k; i++) {
                checksum += unpacked_weights[0 * k + i];
            }
            fprintf(stderr, "[DEBUG v5] Col 0 checksum (sum of all weights): %.6f\n", checksum);
            
            // Count zeros
            int zero_count = 0;
            for (int i = 0; i < k; i++) {
                if (unpacked_weights[0 * k + i] == 0.0f) zero_count++;
            }
            fprintf(stderr, "[DEBUG v5] Col 0 zero count: %d out of %d\n", zero_count, k);
        }
    }
}


// MatMul with FP32 input and packed int4 RHS (dequantized to fp32) - implementation
// A: [M x K], B (packed): [N x K] after unpacking, C: [M x N]
// Note: The packed format contains embedded bias, so the external bias parameter is optional
void hwy_matmul_f32_qai8dxp_qsi4c32p_impl(int M, int K, int N, float* C,
                                          const float* A, const uint8_t* packed_B,
                                          const float* bias, int thread_count) {
    if (M <= 0 || K <= 0 || N <= 0 || !A || !packed_B || !C) return;

    // Temporary buffer for dequantized weights
    // Layout: N x K (each row is a column of the original weight matrix)
    std::vector<float> dequantized_B(N * K);

    // Extract embedded bias from packed weights
    std::vector<float> embedded_bias(N, 0.0f);
    
    // Calculate layout parameters to find embedded bias location
    const int nr = 8;  // For qsi4c32p8x8
    const size_t bl = 32;
    const size_t num_bytes_scale = 2;  // bf16
    const size_t num_bytes_per_block_k = bl / 2;
    const size_t k_rounded = ((K + bl - 1) / bl) * bl;
    const size_t num_blocks_per_row = k_rounded / bl;
    const size_t num_bytes_int4_per_block = nr * num_bytes_per_block_k;
    const size_t num_bytes_scale_per_block = nr * num_bytes_scale;
    const size_t num_bytes_per_block_total = num_bytes_int4_per_block + num_bytes_scale_per_block;
    const size_t offset_to_sums = num_bytes_per_block_total * num_blocks_per_row;
    const size_t rhs_packed_stride = offset_to_sums + nr * sizeof(float) * 2;  // sums + bias
    const size_t dst_num_rows = (N + nr - 1) / nr;
    
    // Extract bias from each NR column group
    for (size_t row_group = 0; row_group < dst_num_rows; ++row_group) {
        const uint8_t* group_base = packed_B + row_group * rhs_packed_stride;
        const float* bias_ptr = (const float*)(group_base + offset_to_sums + nr * sizeof(float));
        for (int i = 0; i < nr && (row_group * nr + i) < (size_t)N; ++i) {
            embedded_bias[row_group * nr + i] = bias_ptr[i];
        }
    }
    
    // Debug: print first 8 embedded bias values
    static bool bias_debug_printed = false;
    if (!bias_debug_printed) {
        bias_debug_printed = true;
        fprintf(stderr, "[DEBUG BIAS] First 8 embedded bias values: ");
        for (int i = 0; i < 8 && i < N; i++) {
            fprintf(stderr, "%.6f ", embedded_bias[i]);
        }
        fprintf(stderr, "\n");
        
        // Also print the sums values
        const uint8_t* group_base = packed_B;
        const float* sums_ptr = (const float*)(group_base + offset_to_sums);
        fprintf(stderr, "[DEBUG SUMS] First 8 sums values: ");
        for (int i = 0; i < 8; i++) {
            fprintf(stderr, "%.6f ", sums_ptr[i]);
        }
        fprintf(stderr, "\n");
    }

    // Dequantize the packed weights
    // For qsi4c32p8x8_4x8x32: nr=8, kr=16, sr=2 (from kai_matmul header)
    dequant_qsi4c32p_to_fp32(packed_B, dequantized_B.data(), N, K, 8, 16, 2);

    // Handle fp32 LHS with dequantized RHS
    // C[m,n] = sum_k(A[m,k] * B[n,k])  -- note: B is stored as N x K (transposed)
    const hn::ScalableTag<float> d;
    const int V = static_cast<int>(hn::Lanes(d));

    auto worker = [&](int row_start, int row_end) {
        for (int m = row_start; m < row_end; ++m) {
            for (int n = 0; n < N; ++n) {
                float sum = 0.0f;
                
                // Dot product: A[m,:] * B[n,:]
                // A is M x K, B is N x K (weight matrix transposed)
                for (int k = 0; k < K; ++k) {
                    float a_val = A[m * K + k];
                    float b_val = dequantized_B[n * K + k];  // B is N x K layout
                    sum += a_val * b_val;
                }
                
                // Add embedded bias (from packed weights)
                float out = sum + embedded_bias[n];
                
                // Also add external bias if provided (for compatibility)
                if (bias) {
                    out += bias[n];
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
