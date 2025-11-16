#ifndef MLLM_MATMUL_HPP
#define MLLM_MATMUL_HPP

#include "mllm/core/DataTypes.hpp"

namespace mllm {
namespace cpu {
namespace x86 {

void hwy_matmul_fp32(int M, int K, int N, float *C, const float *A, const float *B, const float *bias, bool transpose_a, bool transpose_b, int thread_count);
void hwy_batch_matmul_fp32(int batch_size, int M, int K, int N, int C_stride, int A_stride, int B_stride, int bias_stride, float *C, const float *A, const float *B, const float *bias, bool transpose_a, bool transpose_b, int thread_count);
// Dequantize qsi4c32p packed weights to fp32
void dequant_qsi4c32p_to_fp32(const uint8_t* packed_weights, float* unpacked_weights,
                               int n, int k, int nr, int kr, int sr);

// MatMul with FP32 LHS and int8 RHS (dequantized) - quantizes input to int8 internally
void hwy_matmul_f32_qai8dxp_qsi4c32p(int M, int K, int N, float* C,
                                     const float* A, const uint8_t* packed_B,
                                     const float* bias, int thread_count);

// Batch MatMul with FP32 LHS and int8 RHS (dequantized)
void hwy_batch_matmul_f32_qai8dxp_qsi4c32p(int batch_size, int M, int K, int N,
                                           int C_stride, int A_stride, int B_stride, int bias_stride,
                                           float* C, const float* A, const uint8_t* packed_B,
                                           const float* bias, int thread_count);
} // namespace x86
} // namespace cpu
} // namespace mllm

#endif // MLLM_MATMUL_HPP
