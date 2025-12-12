// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Kernel launcher header for TileLang-generated kernels.
// Only contains function declarations, actual launch is in .cu file.

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace mllm::cuda::generated {

// Forward declaration of the launcher (defined in tl_elementwise_add.cu)
extern "C" void launch_elem_add_kernel(half* A, half* B, half* C, int numel, cudaStream_t stream);

/**
 * @brief Launch the elementwise add kernel.
 * 
 * This kernel performs C = A + B for FP16 tensors.
 * 
 * @param A Input tensor A (device pointer, FP16)
 * @param B Input tensor B (device pointer, FP16)
 * @param C Output tensor C (device pointer, FP16)
 * @param numel Total number of elements
 * @param stream CUDA stream
 */
inline void launch_tl_elementwise_add(half* A, half* B, half* C, int numel, cudaStream_t stream = 0) {
    launch_elem_add_kernel(A, B, C, numel, stream);
}

}  // namespace mllm::cuda::generated
