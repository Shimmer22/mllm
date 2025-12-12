// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cuda/ops/TLElementwiseAddOp.hpp"
#include "mllm/backends/cuda/kernels/generated/tl_elementwise_add.cuh"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"

#include <cuda_fp16.h>

namespace mllm::cuda {

TLElementwiseAddOp::TLElementwiseAddOp(const TLElementwiseAddOpOptions& options)
    : BaseOp(OpTypes::kTLElementwiseAdd), options_(options) {}

void TLElementwiseAddOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_RT_ASSERT_EQ(inputs.size(), 2);
  MLLM_RT_ASSERT_EQ(outputs.size(), 1);

  auto& A = inputs[0];
  auto& B = inputs[1];
  auto& C = outputs[0];

  // Validate dtypes
  MLLM_RT_ASSERT_EQ(A.dtype(), kFloat16);
  MLLM_RT_ASSERT_EQ(B.dtype(), kFloat16);
  MLLM_RT_ASSERT_EQ(C.dtype(), kFloat16);

  // Get total number of elements
  int numel = A.numel();

  // Launch the simplified kernel
  generated::launch_tl_elementwise_add(
      A.ptr<half>(),
      B.ptr<half>(),
      C.ptr<half>(),
      numel,
      /* stream */ 0
  );

  // Synchronize for correctness (remove in production for async)
  cudaDeviceSynchronize();
}

void TLElementwiseAddOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_RT_ASSERT_EQ(inputs.size(), 2);
  
  auto& A = inputs[0];
  auto& B = inputs[1];
  
  // Validate shapes match
  MLLM_RT_ASSERT(A.shape() == B.shape());
  
  // Output has same shape as inputs
  outputs.emplace_back(Tensor::empty(A.shape(), A.dtype(), kCUDA));
}

}  // namespace mllm::cuda
