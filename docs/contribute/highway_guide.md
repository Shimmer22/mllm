# Google Highway 库在 MLLM 项目中的使用指南

**版本**: 1.0
**日期**: 2025-11-05

## 1. 目的

本文档旨在为 `mllm` 项目的开发者提供一个关于如何正确使用 [Google Highway](https://github.com/google/highway) 库的清晰、规范的指南。

`mllm` 项目的 x86 后端严重依赖 `highway` 来实现核心算子（如 `MatMul`, `Linear`）的 SIMD 加速。根据历史开发日志，不规范的使用曾导致了大量的编译失败和调试困难。本指南旨在总结成功经验、提炼标准实践（以 Google `gemma.cpp` 为范本），帮助开发者高效、正确地为 `mllm` 贡献新的 x86 内核，避免重蹈覆辙。

## 2. 核心概念：Highway 为何如此特殊？

`highway` 的强大之处在于其**性能可移植性**，它允许我们编写一份 SIMD 代码，然后通过动态分发在支持不同指令集（SSE, AVX2, AVX-512 等）的 CPU 上以最优方式运行。

为了实现这一点，`highway` 使用了一套复杂的宏系统，其工作原理如下：
编译器在 `HWY_TARGET_INCLUDE` 宏的驱动下，**多次包含同一个源文件**。每一次包含，都会定义不同的目标指令集宏（如 `HWY_TARGET_AVX2`），从而为该指令集生成一个特定版本的内核函数。最后，`HWY_ONCE` 块中的代码会生成一个**分发表（Dispatch Table）**，允许在运行时根据 CPU 能力选择最高效的函数版本。

这种机制决定了我们**必须严格遵守 `highway` 的代码结构规范**，任何微小的偏差都可能导致难以理解的编译或链接错误。

## 3. 项目构建与配置

在编写代码之前，必须确保构建环境已正确配置。

### 3.1. CMake 编译选项

要使 `highway` 正常工作，必须在 `CMake` 中设置正确的编译选项。对于 `mllm` 项目，这通常在 `tasks/` 目录下的 `.yaml` 配置文件中完成。

1.  **启用 `contrib` 模块**: `highway` 的许多实用功能（如 `dot`）位于 `contrib` 目录中，默认可能不启用。这是导致 `hwy/contrib/dot/dot.h: No such file or directory` 错误的原因。
    ```yaml
    # 在 tasks/build_x86.yaml 或类似文件中确保以下设置为 ON
    - -DHWY_ENABLE_CONTRIB=ON
    ```

2.  **启用目标指令集**: 必须告知编译器生成 x86 SIMD 指令。使用 `-march=native` 是最简单的方式，它会为你当前的 CPU 启用所有支持的指令集。
    ```yaml
    # 确保编译选项正确无误，没有多余的引号
    - -DMLLM_CPU_BACKEND_COMPILE_OPTIONS=-march=native
    ```

### 3.2. 手动构建命令

为了绕过 `task.py` 的封装并获得更清晰的构建日志，推荐使用以下手动 `cmake` 命令（在干净的 `build` 目录中执行）：

```bash
# 1. 配置项目 (在项目根目录运行)
cmake -S . -G Ninja -B build -DCMAKE_BUILD_TYPE=Release \
    -DHWY_ENABLE_CONTRIB=ON \
    -DHWY_ENABLE_TESTS=OFF \
    -DHWY_ENABLE_EXAMPLES=OFF \
    -DMLLM_CPU_BACKEND_COMPILE_OPTIONS=-march=native \
    -DMLLM_KERNEL_USE_THREADS=ON \
    -DMLLM_KERNEL_THREADS_VENDOR_OPENMP=ON

# 2. 编译项目
cmake --build build
```

## 4. 标准实现模式 (Canonical Pattern)

**永远不要“黑盒试错”！** 请严格遵循 `highway` 官方的 `skeleton.cc` 示例和 Google `gemma.cpp` 项目中的实践。`gemma.cpp` 通过清晰分层将调度与实现解耦，这是 `mllm` 应该效仿的最佳实践。

以下是为 `mllm` 添加新内核（以 `MatMul` 为例）的规范步骤。

### 4.1. 推荐文件结构

```
mllm/backends/cpu/kernels/x86/
├── MatMul.hpp         # 公共接口头文件，供 MatMulOp.cpp 调用
├── MatMul.cpp         # 桥接文件，负责 highway 宏定义和分发
└── MatMul_kernel.hpp  # 【推荐】真正的 SIMD 内核实现
```
这种结构将 `highway` 的复杂宏机制（在 `MatMul.cpp` 中）与纯粹的 SIMD 算法逻辑（在 `MatMul_kernel.hpp` 中）分离开，极大地提高了代码的可读性和可维护性。

### 4.2. 内核实现 (`MatMul_kernel.hpp`)

这个文件被 `MatMul.cpp` 多次包含，它只包含位于 `HWY_NAMESPACE` 中的 SIMD 计算逻辑。

```cpp
// mllm/backends/cpu/kernels/x86/MatMul_kernel.hpp

// 此文件没有 include guard，因为它需要被多次包含

// 所有的实现代码都必须在这个命名空间内
namespace mllm {
namespace cpu {
namespace x86 {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// SIMD 内核的具体实现。
// 它可以是模板，也可以是普通函数。
// 为了清晰，我们将其命名为 Impl
void MatMulKernelImpl(const float* A, const float* B, float* C, size_t M, size_t N, size_t K) {
    const hn::ScalableTag<float> d;
    // ... 真正的 SIMD 算法逻辑 ...
    // 例如：使用 hn::Load, hn::MulAdd, hn::Store 等
}

// 【关键】创建一个非模板的、符合 HWY_EXPORT 签名的包装器。
// 这是为了解决模板参数中的 '>' 符号与宏粘贴冲突的问题。
// `HWY_EXPORT` 将作用于这个函数。
void MatMulKernel(const float* A, const float* B, float* C, size_t M, size_t N, size_t K) {
    return MatMulKernelImpl(A, B, C, M, N, K);
}

} // namespace HWY_NAMESPACE
} // namespace x86
} // namespace cpu
} // namespace mllm
```

### 4.3. 桥接与分发 (`MatMul.cpp`)

这个文件是 `highway` 宏魔法发生的地方。

```cpp
// mllm/backends/cpu/kernels/x86/MatMul.cpp

// 1. 定义 HWY_TARGET_INCLUDE, 它指向包含具体实现的内核文件
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "mllm/backends/cpu/kernels/x86/MatMul_kernel.hpp"

// 2. 包含 highway 的核心头文件，顺序不能错
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

// 3. 包含内核实现文件
#include "mllm/backends/cpu/kernels/x86/MatMul_kernel.hpp"

// 4. 将所有 highway 相关代码包裹在 BEFORE/AFTER 宏中
HWY_BEFORE_NAMESPACE();

// 通过 using 指令，将内核实现引入当前命名空间
namespace mllm {
namespace cpu {
namespace x86 {
HWY_USING_NAMESPACE();
} // namespace x86
} // namespace cpu
} // namespace mllm

HWY_AFTER_NAMESPACE();


// 5. 在 HWY_ONCE 块中导出并定义分发函数
// 此块代码在所有目标编译后，只会执行一次
#if HWY_ONCE
namespace mllm {
namespace cpu {
namespace x86 {

// 声明一个外部函数指针表，名为 MatMulKernelHighwayDispatchTable
// HWY_EXPORT 的参数必须与内核文件中的包装函数名完全一致
HWY_EXPORT(MatMulKernel);

// 定义我们最终暴露给外部的公共接口函数
void hwy_matmul_fp32(const float* A, const float* B, float* C, size_t M, size_t N, size_t K) {
    // 使用 HWY_DYNAMIC_DISPATCH 调用运行时最高效的内核版本
    // 它会从上面导出的分发表中查找并调用函数
    return HWY_DYNAMIC_DISPATCH(MatMulKernel)(A, B, C, M, N, K);
}

} // namespace x86
} // namespace cpu
} // namespace mllm

// 最后，包含公共头文件以确保函数签名一致
#include "mllm/backends/cpu/kernels/x86/MatMul.hpp"

#endif // HWY_ONCE
```

### 4.4. 公共头文件 (`MatMul.hpp`)

这个文件应该非常简洁，只声明对外的接口，供 `MatMulOp.cpp` 等上层模块调用。

```cpp
// mllm/backends/cpu/kernels/x86/MatMul.hpp
#pragma once

#include <cstddef>

namespace mllm {
namespace cpu {
namespace x86 {

// 只声明最终的调用接口
void hwy_matmul_fp32(const float* A, const float* B, float* C, size_t M, size_t N, size_t K);

} // namespace x86
} // namespace cpu
} // namespace mllm
```

## 5. 常见陷阱与排错指南 (Troubleshooting)

| 错误信息 | 根本原因 (mllm 历史问题) | 解决方案 | 
| :--- | :--- | :--- |
| `pasting " > " and "HighwayDispatchTable" does not produce a valid preprocessing token` | `HWY_EXPORT` 的参数是一个模板特化（如 `MyKernel<float>`）。C++的 `>` 符号破坏了宏的 `##` 粘贴操作。 | **永远不要**将模板化函数名直接传给 `HWY_EXPORT`。按照第 4 节的模式，创建一个非模板的包装函数，并导出那个包装函数。 |
| `‘MatmulKernelHighwayDispatchTable’ is not a member of ‘...’` 或 `redefinition of ‘...’` | 宏结构错误。`HWY_EXPORT` 和 `HWY_DYNAMIC_DISPATCH` 不在同一个 C++ 命名空间内，或者 `HWY_ONCE` 块使用不当。 | **严格对照本文档第 4 节的模式**检查你的代码结构。确保 `HWY_EXPORT` 和 `HWY_DYNAMIC_DISPATCH` 位于 `#if HWY_ONCE` 块内，且处于相同的命名空间。 |
| `fatal error: hwy/contrib/dot/dot.h: No such file or directory` | `highway` 的 `contrib` 模块未启用或未下载。 | 在 CMake 配置中添加 `-DHWY_ENABLE_CONTRIB=ON`，并清理构建目录后重试。同时注意，正确的文件名可能是 `dot-inl.h`。 |
| `fatal error: mllm/core/Types.hpp: No such file or directory` | 项目头文件名或路径错误。 | **不要猜测！** 立即检查文件系统确认正确的文件名（例如，是 `DataTypes.hpp` 而非 `Types.hpp`）。 |
| `‘mllm::cpu::arm’ has not been declared` | 在 x86 环境下编译了未被宏保护的 ARM 专属代码。 | 为平台专属代码块添加正确的预处理器宏保护，如 `#if defined(MLLM_HOST_ARCH_ARM64)`。 |

## 6. 进阶参考

-   **黄金标准**: Google 的 [gemma.cpp](https://github.com/google/gemma.cpp) 项目是学习 `highway` 高级用法的最佳参考。重点阅读其 `ops/` 目录，学习**内存对齐 (Alignment)**, **数据预取 (Prefetching)** 和 **循环分块 (Tiling/Packing)** 等高级性能优化技巧。
-   **官方文档**:
    -   [Highway 官方示例 `skeleton.cc`](https://github.com/google/highway/blob/master/hwy/examples/skeleton.cc): 必须遵守的结构模板。
    -   [Highway FAQ](https://google.github.io/highway/en/master/faq.html): 解答了许多关于构建和宏使用的常见问题。

遵循本指南，可以显著减少在 `highway` 集成上花费的时间，让开发者专注于 SIMD 内核本身的逻辑与优化。
