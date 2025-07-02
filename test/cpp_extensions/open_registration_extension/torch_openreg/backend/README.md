# OpenReg - CPU-based CUDA Runtime Simulator

OpenReg 是一个用于教育和调试目的的C++库。它在CPU上模拟了部分CUDA Runtime API，特别是围绕内存管理和设备上下文的部分。

其核心特性是**内存隔离**：通过 `openregMalloc` 分配的“设备内存”在默认情况下是受CPU保护的，任何直接访问都会导致段错误（Segmentation Fault），这与真实GPU设备内存的行为类似。只有在模拟的内存拷贝操作期间，库内部才会临时开放访问权限。

## 特性

- **设备内存 (`openregMalloc`)**: 分配受保护的、CPU不可直接访问的内存。
- **Pinned主机内存 (`openregMallocHost`)**: 分配CPU可直接访问的、由本库管理的内存。
- **内存拷贝 (`openregMemcpy`)**: 安全地在不同内存类型之间拷贝数据。
- **设备管理**:
  - `openregGetDeviceCount`: 获取模拟的设备数量。
  - `openregSetDevice`/`openregGetDevice`: 管理每个主机线程的当前设备上下文（使用`thread_local`实现）。

## 目录结构

```
.
├── CMakeLists.txt      # 主构建脚本
├── README.md           # 本文档
├── include/
│   └── openreg.h       # 公共API头文件
└── csrc/
    ├── device.cpp      # 设备管理API实现
    └── memory.cpp      # 内存管理API实现
```

## 如何构建

项目使用CMake进行构建。

```bash
# 1. 创建并进入构建目录
mkdir build
cd build

# 2. 运行CMake配置项目
cmake ..

# 3. 编译库
make

# 编译完成后，将在 build/csrc/ 目录下生成 libopenreg.so
```

## 如何使用

将 `include` 目录添加到你的项目包含路径，并链接到编译生成的 `libopenreg.so` 库。

```cpp
#include "openreg.h"

// ...

OPENREG_CHECK(openregSetDevice(1));
int* d_ptr = nullptr;
OPENREG_CHECK(openregMalloc((void**)&d_ptr, 1024));

// ...
```
