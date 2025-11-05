#ifndef MLLM_MATMUL_HPP
#define MLLM_MATMUL_HPP

#include "mllm/core/DataTypes.hpp"

namespace mllm {
namespace cpu {
namespace x86 {

void hwy_matmul_fp32(int M, int K, int N, float *C, const float *A, const float *B, const float *bias, bool transpose_a, bool transpose_b, int thread_count);
void hwy_batch_matmul_fp32(int batch_size, int M, int K, int N, int C_stride, int A_stride, int B_stride, int bias_stride, float *C, const float *A, const float *B, const float *bias, bool transpose_a, bool transpose_b, int thread_count);

} // namespace x86
} // namespace cpu
} // namespace mllm

#endif // MLLM_MATMUL_HPP
