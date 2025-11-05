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

// This is the actual kernel implementation that will be compiled for each target.
void MatmulKernel(const float* A, const float* B, float* C, int M, int N, int K, const float* bias, int thread_count) {
    const hn::ScalableTag<float> d;
    const int V = hn::Lanes(d);

    // Simple tiled implementation for better cache usage.
    const int TILE_M = 4;
    const int TILE_N = 4 * V;
    const int TILE_K = 128;

    hwy::ThreadPool pool(thread_count);
    pool.Run(0, M, [&](const int m_start, int thread) {
        for (int m0 = m_start; m0 < m_start + 1 && m0 < M; m0 += TILE_M) {
            for (int n0 = 0; n0 < N; n0 += TILE_N) {
                // Zero the C tile
                for (int m = m0; m < std::min(m0 + TILE_M, M); ++m) {
                    for (int n = n0; n < std::min(n0 + TILE_N, N); ++n) {
                        C[m * N + n] = 0;
                    }
                }

                for (int k0 = 0; k0 < K; k0 += TILE_K) {
                    int m_end = std::min(m0 + TILE_M, M);
                    int n_end = std::min(n0 + TILE_N, N);
                    int k_end = std::min(k0 + TILE_K, K);

                    for (int m = m0; m < m_end; ++m) {
                        for (int k = k0; k < k_end; ++k) {
                            const auto a_val = hn::Set(d, A[m * K + k]);
                            for (int n = n0; n < n_end; n += V) {
                                const auto b_vec = hn::LoadU(d, &B[k * N + n]);
                                auto c_vec = hn::LoadU(d, &C[m * N + n]);
                                c_vec = hn::MulAdd(a_val, b_vec, c_vec);
                                hn::StoreU(c_vec, d, &C[m * N + n]);
                            }
                        }
                    }
                }
                // Add bias after all K tiles are processed for a C tile
                if (bias) {
                    for (int m = m0; m < std::min(m0 + TILE_M, M); ++m) {
                        for (int n = n0; n < std::min(n0 + TILE_N, N); n += V) {
                             auto c_vec = hn::LoadU(d, &C[m * N + n]);
                             const auto bias_vec = hn::LoadU(d, &bias[n]);
                             c_vec = hn::Add(c_vec, bias_vec);
                             hn::StoreU(c_vec, d, &C[m * N + n]);
                        }
                    }
                }
            }
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

// Get the function pointer for the best target.
HWY_EXPORT(MatmulKernel);

// This is the public API that will be called from other parts of the code.
void hwy_matmul_fp32(int M, int K, int N, float *C, const float *A, const float *B, const float *bias, bool transpose_a, bool transpose_b, int thread_count) {
    // For now, only support transpose_b = true, transpose_a = false
    if (transpose_a || !transpose_b) {
        NYI("hwy_matmul_fp32 only supports transpose_a=false and transpose_b=true");
        return;
    }
    // The `MatmulKernel` function pointer is resolved by HWY_DYNAMIC_DISPATCH.
    // This will call the best version of the kernel for the current CPU.
    HWY_DYNAMIC_DISPATCH(MatmulKernel)(A, B, C, M, N, K, bias, thread_count);
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
