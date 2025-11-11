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

} // namespace HWY_NAMESPACE
} // namespace x86
} // namespace cpu
} // namespace mllm
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace mllm {
namespace cpu {
namespace x86 {

// Export the kernels to build the dispatch table
HWY_EXPORT(MatmulKernel_NN);
HWY_EXPORT(MatmulKernel_NT);
HWY_EXPORT(MatmulKernel_TN);
HWY_EXPORT(MatmulKernel_TT);

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
