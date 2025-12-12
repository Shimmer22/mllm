// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Simple test to verify TLElementwiseAddOp integration.

#include <iostream>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "mllm/mllm.hpp"
#include "mllm/engine/Context.hpp"
#include "mllm/backends/cuda/CudaBackend.hpp"
#include "mllm/backends/cuda/ops/TLElementwiseAddOp.hpp"

int main() {
    std::cout << "=== TileLang Integration Test ===" << std::endl;

    // Initialize mllm context
    mllm::initializeContext();

    // Initialize CUDA backend
    mllm::initCudaBackend();

    std::cout << "[OK] Context and CUDA backend initialized." << std::endl;

    // Check if the Op type is registered
    std::cout << "[OK] OpType kTLElementwiseAdd = " 
              << static_cast<int>(mllm::OpTypes::kTLElementwiseAdd) << std::endl;

    // Create the Op via factory
    auto& ctx = mllm::Context::instance();
    auto cuda_backend = ctx.getBackend(mllm::kCUDA);
    
    if (cuda_backend) {
        std::cout << "[OK] CUDA backend retrieved successfully." << std::endl;
        
        // Try to create the Op
        mllm::cuda::TLElementwiseAddOpOptions options;
        auto op = cuda_backend->createOp(mllm::OpTypes::kTLElementwiseAdd, options);
        
        if (op) {
            std::cout << "[OK] TLElementwiseAddOp created successfully!" << std::endl;
        } else {
            std::cout << "[FAIL] Failed to create TLElementwiseAddOp." << std::endl;
            return 1;
        }
    } else {
        std::cout << "[FAIL] Failed to get CUDA backend." << std::endl;
        return 1;
    }

    // Shutdown
    mllm::shutdownContext();
    std::cout << "[OK] Context shutdown." << std::endl;
    
    std::cout << "=== All tests passed! ===" << std::endl;
    return 0;
}
