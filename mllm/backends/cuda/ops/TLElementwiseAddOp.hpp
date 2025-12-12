// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"

namespace mllm::cuda {

/**
 * @brief Options for the TileLang-generated ElementwiseAdd Op.
 */
struct TLElementwiseAddOpOptions : public BaseOpOptions<TLElementwiseAddOpOptions> {};

/**
 * @brief CUDA Op wrapper for TileLang-generated elementwise add kernel.
 * 
 * This Op performs C = A + B for FP16 tensors on CUDA.
 */
class TLElementwiseAddOp final : public BaseOp {
 public:
  explicit TLElementwiseAddOp(const TLElementwiseAddOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override {}

  void load(const ParameterFile::ptr_t& ploader) override {}

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override {}

 protected:
  TLElementwiseAddOpOptions options_;
};

/**
 * @brief Factory for creating TLElementwiseAddOp instances.
 */
class TLElementwiseAddOpFactory : public TypedOpFactory<OpTypes::kTLElementwiseAdd, TLElementwiseAddOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const TLElementwiseAddOpOptions& options) override {
    return std::make_shared<TLElementwiseAddOp>(options);
  }
};

}  // namespace mllm::cuda
