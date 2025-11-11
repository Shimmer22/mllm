// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/kernels/x86/softmax.hpp"

#if defined(MLLM_HOST_ARCH_X86) || defined(MLLM_HOST_ARCH_X86_64)

#include <hwy/highway.h>
#include <hwy/contrib/math/math-inl.h>
#include <cmath>
#include <algorithm>
#include <limits>
#include <iostream>

#include "mllm/backends/cpu/kernels/x86/math.hpp"

namespace mllm::cpu::x86 {
namespace hn = hwy::HWY_NAMESPACE;

// NOTE [CUSTOM_OPTIMIZE_OR_ALIGN]:  
// We use hn::LoadU / hn::StoreU here (instead of aligned Load/Store)  
// because some input pointers may not meet required alignment.  
// However, this is a fallback / workaround — performance may degrade compared  
// to aligned access.  
// Future improvement:  
//   – ensure X and Y buffers are allocated with proper alignment (e.g. 32-byte)  
//   – switch back to hn::Load / hn::Store when alignment is guaranteed  
//   – benchmark aligned vs unaligned for this kernel on target CPU(s)  
void softmax_v1_fp32(const mllm_fp32_t* __restrict X, mllm_fp32_t* __restrict Y, int len, int stride, int thread_count) {
  if (stride != 1 || len <= 16) {
    std::cout << "[DEBUG] Softmax entering scalar path. len=" << len << ", stride=" << stride << std::endl;
    float max_value = std::numeric_limits<float>::lowest();
    for (int i = 0; i < len; ++i) { max_value = std::max(max_value, X[i * stride]); }
    std::cout << "[DEBUG] max_value: " << max_value << std::endl;

    float sum = 0.f;
    for (int i = 0; i < len; ++i) {
      auto val = X[i * stride] - max_value;
      auto tmp = expf(val);
      if (i < 4) { // Log first few values
          std::cout << "[DEBUG] X[" << i << "]=" << X[i*stride] << ", val=" << val << ", expf(val)=" << tmp << std::endl;
      }
      if (std::isnan(tmp) || std::isinf(tmp)) {
          std::cout << "[ERROR] NaN/Inf detected after expf. Input to expf was: " << val << std::endl;
          abort();
      }
      Y[i * stride] = tmp;
      sum += tmp;
    }
    std::cout << "[DEBUG] sum: " << sum << std::endl;
    if (sum == 0.f) {
        std::cout << "[ERROR] Sum is zero, division will result in Inf." << std::endl;
    }
    sum = 1.f / sum;
    std::cout << "[DEBUG] 1/sum: " << sum << std::endl;

    for (int i = 0; i < len; ++i) { 
        Y[i * stride] *= sum; 
        if (i < 4 && (std::isnan(Y[i*stride]) || std::isinf(Y[i*stride]))) {
            std::cout << "[ERROR] NaN/Inf detected in final output Y[" << i << "]" << std::endl;
            abort();
        }
    }
    return;
  }

  const hn::ScalableTag<float> d;
  using V = hn::Vec<decltype(d)>;
  int i = 0;
  V max_vec = hn::Set(d, std::numeric_limits<float>::lowest());
  for (; i + hn::Lanes(d) <= len; i += hn::Lanes(d)) {
    const V x_vec = hn::LoadU(d, X + i);
    max_vec = hn::Max(max_vec, x_vec);
  }
  float max_value = hn::ReduceMax(d, max_vec);
  for (; i < len; ++i) { max_value = std::max(max_value, X[i]); }
  V sum_vec = hn::Zero(d);
  const V max_vec_broadcast = hn::Set(d, max_value);
  i = 0;
  for (; i + hn::Lanes(d) <= len; i += hn::Lanes(d)) {
    const V x_vec = hn::LoadU(d, X + i);
    const V normalized = hn::Sub(x_vec, max_vec_broadcast);
    const V exp_vec = hn::Exp(d, normalized);
    hn::StoreU(exp_vec, d, Y + i);
    sum_vec = hn::Add(sum_vec, exp_vec);
  }
  float sum_value = hn::ReduceSum(d, sum_vec);
  for (; i < len; ++i) {
    float tmp = expf(X[i] - max_value);
    Y[i] = tmp;
    sum_value += tmp;
  }
  sum_value = 1.f / sum_value;
  const V inv_sum_vec = hn::Set(d, sum_value);
  i = 0;
  for (; i + hn::Lanes(d) <= len; i += hn::Lanes(d)) {
    const V y_vec = hn::LoadU(d, Y + i);
    const V result = hn::Mul(y_vec, inv_sum_vec);
    hn::StoreU(result, d, Y + i);
  }
  for (; i < len; ++i) { Y[i] *= sum_value; }
}

}  // namespace mllm::cpu::x86

#endif