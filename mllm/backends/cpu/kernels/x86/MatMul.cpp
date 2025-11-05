// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "MatMul.hpp"
#include "mllm/utils/Common.hpp"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "mllm/backends/cpu/kernels/x86/MatMul.cpp"
#include <hwy/foreach_target.h>

#include <hwy/aligned_allocator.h>
#include <hwy/contrib/dot/dot-inl.h>
#include <hwy/contrib/thread_pool/thread_pool.h>

// Highway dispatch wrapper
HWY_BEFORE_NAMESPACE();
namespace mllm {
namespace cpu {
namespace x86 {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// Kernel for C = A * B (transpose_a=false, transpose_b=false)
void MatmulKernel_NN(const float* A, const float* B, float* C, int M, int N, int K, const float* bias, int thread_count) {
    const hn::ScalableTag<float> d;
    const int V = hn::Lanes(d);
    hwy::ThreadPool pool(thread_count);
    pool.Run(0, M, [&](const int m, int thread) {
        for (int n = 0; n < N; ++n) {
            auto sum_vec = hn::Zero(d);
            for (int k = 0; k < K; ++k) {
                const auto a_val = hn::Set(d, A[m * K + k]);
                const auto b_val = hn::Set(d, B[k * N + n]);
                sum_vec = hn::MulAdd(a_val, b_val, sum_vec);
            }
            float sum = hn::ReduceSum(d, sum_vec);
            C[m * N + n] = sum + (bias ? bias[n] : 0.0f);
        }
    });
}

// Kernel for C = A * B^T (transpose_a=false, transpose_b=true)
void MatmulKernel_NT(const float* A, const float* B, float* C, int M, int N, int K, const float* bias, int thread_count) {
    const hn::ScalableTag<float> d;
    const int V = hn::Lanes(d);
    hwy::ThreadPool pool(thread_count);
    pool.Run(0, M, [&](const int m, int thread) {
        for (int n = 0; n < N; ++n) {
            auto sum_vec = hn::Zero(d);
            for (int k = 0; k < K; k += V) {
                const auto a_vec = hn::LoadU(d, &A[m * K + k]);
                const auto b_vec = hn::LoadU(d, &B[n * K + k]);
                sum_vec = hn::MulAdd(a_vec, b_vec, sum_vec);
            }
            float sum = hn::ReduceSum(d, sum_vec);
            C[m * N + n] = sum + (bias ? bias[n] : 0.0f);
        }
    });
}

// Kernel for C = A^T * B (transpose_a=true, transpose_b=false)
void MatmulKernel_TN(const float* A, const float* B, float* C, int M, int N, int K, const float* bias, int thread_count) {
    const hn::ScalableTag<float> d;
    const int V = hn::Lanes(d);
    hwy::ThreadPool pool(thread_count);
    pool.Run(0, M, [&](const int m, int thread) {
        for (int n = 0; n < N; ++n) {
            auto sum_vec = hn::Zero(d);
            for (int k = 0; k < K; ++k) {
                const auto a_val = hn::Set(d, A[k * M + m]);
                const auto b_val = hn::Set(d, B[k * N + n]);
                sum_vec = hn::MulAdd(a_val, b_val, sum_vec);
            }
            float sum = hn::ReduceSum(d, sum_vec);
            C[m * N + n] = sum + (bias ? bias[n] : 0.0f);
        }
    });
}

// Kernel for C = A^T * B^T (transpose_a=true, transpose_b=true)
void MatmulKernel_TT(const float* A, const float* B, float* C, int M, int N, int K, const float* bias, int thread_count) {
    const hn::ScalableTag<float> d;
    const int V = hn::Lanes(d);
    hwy::ThreadPool pool(thread_count);
    pool.Run(0, M, [&](const int m, int thread) {
        for (int n = 0; n < N; ++n) {
            auto sum_vec = hn::Zero(d);
            for (int k = 0; k < K; k+=V) {
                const auto a_vec = hn::LoadU(d, &A[k * M + m]);
                const auto b_vec = hn::LoadU(d, &B[n * K + k]);
                sum_vec = hn::MulAdd(a_vec, b_vec, sum_vec);
            }
            float sum = hn::ReduceSum(d, sum_vec);
            C[m * N + n] = sum + (bias ? bias[n] : 0.0f);
        }
    });
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

// Get the function pointer for the best target for each kernel.
HWY_EXPORT(MatmulKernel_NN);
HWY_EXPORT(MatmulKernel_NT);
HWY_EXPORT(MatmulKernel_TN);
HWY_EXPORT(MatmulKernel_TT);

// This is the public API that will be called from other parts of the code.
void hwy_matmul_fp32(int M, int K, int N, float *C, const float *A, const float *B, const float *bias, bool transpose_a, bool transpose_b, int thread_count) {
    
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

void hwy_batch_matmul_fp32(int batch_size, int M, int K, int N, int C_stride, int A_stride, int B_stride, int bias_stride, float *C, const float *A, const float *B, const float *bias, bool transpose_a, bool transpose_b, int thread_count) {
    // Simple loop over batches. Could be parallelized further.
    for (int i = 0; i < batch_size; ++i) {
        const float* current_A = A + i * A_stride;
        const float* current_B = B + i * B_stride;
        float* current_C = C + i * C_stride;
        const float* current_bias = bias ? bias + i * bias_stride : nullptr;
        hwy_matmul_fp32(M, K, N, current_C, current_A, current_B, current_bias, transpose_a, transpose_b, thread_count);
    }
}

} // namespace x86
} // namespace cpu
} // namespace mllm

#endif // HWY_ONCE