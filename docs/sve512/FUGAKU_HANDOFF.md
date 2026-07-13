# Fugaku 接力开发指南

从 x86/QEMU 环境迁移到 Fugaku（A64FX，SVE VL=512）真机调试。本文档描述 GitHub 仓库状态、迁移前准备、Fugaku 原生构建与验证步骤。

## 仓库与分支

| 位置 | 远程 | 分支 | 用途 |
|------|------|------|------|
| mainline 开发 | `git@github.com:nahso/pytorch.git` | `sve512-main` | 上游 PR 目标，**Fugaku 主调试线** |
| 2.10 自用 | `git@github.com:nahso/pytorch-sve512.git` | `sve512-2.10` | 稳定 2.10 + kblas 服务器脚本（路径需按 Fugaku 改） |
| 参考（只读） | `pytorch-fugaku-1.13.1/` | `origin/r1.13_for_a64fx` | 富士通官方 SVE512 实现参考 |

本地 worktree：

- `pytorch-main-sve512/` — mainline，分支 `sve512-main`
- `pytorch-2.10-sve512/` — 2.10 backport，分支 `sve512-2.10`

## 离开 x86 前：推送到 GitHub

GitHub 上的 `nahso/pytorch:sve512-main` 可能落后于本地。Fugaku 接力前务必 push 最新代码（含 A64FX BF16 修复）。

```bash
cd pytorch-main-sve512
git status -sb
git log --oneline -3

# 提交 Fugaku 相关改动（示例）
git add aten/src/ATen/native/DispatchStub.cpp \
        cmake/Codegen.cmake cmake/Modules/FindARM.cmake \
        torch/_inductor/cpu_vec_isa.py
git commit -m "Enable SVE512 without FEAT_BF16 and extend Inductor SVE512"

git push fork sve512-main
```

`fork` 远程 = `git@github.com:nahso/pytorch.git`。

根目录 `scripts/`（交叉编译脚本）目前不在 GitHub；Fugaku 原生构建不依赖它们。若需同步，可后续放入 `nahso/pytorch` 的 `scripts/fugaku/` 或单独 meta 仓库。

## Fugaku 上获取代码

```bash
git clone --branch sve512-main --recursive \
  git@github.com:nahso/pytorch.git pytorch-main-sve512
cd pytorch-main-sve512
git submodule update --init --depth 1 --recursive
```

## Fugaku 原生构建（不用 QEMU / 交叉工具链）

Fugaku 登录/计算节点为 native aarch64。使用**标准 gcc/clang**，**不要用 FCC**（`fcc`、`tcsds`）。

### 1. 工具链

```bash
module load gcc/15    # 按节点可用模块调整；推荐 GCC 13–15
export CC=gcc CXX=g++
gcc --version
```

### 2. Host 工具（sleef codegen + protoc）

在登录节点执行（与 x86 交叉构建相同逻辑，但在 aarch64 上 native 跑）：

```bash
cd pytorch-main-sve512

cmake -S third_party/sleef -B build-sleef-native -GNinja -DSLEEF_BUILD_TESTS=OFF
ninja -C build-sleef-native mkdisp mkrename mkalias

# protoc：优先用仓库脚本；或系统 protoc（需与 bundled protobuf 版本兼容）
bash scripts/build_host_protoc.sh
# 若无脚本：which protoc 后传给 -DCAFFE2_CUSTOM_PROTOC_EXECUTABLE=
```

### 3. CMake 配置

```bash
cmake -S . -B build-sve512-native \
  -DNATIVE_BUILD_DIR=$(pwd)/build-sleef-native \
  -DCAFFE2_CUSTOM_PROTOC_EXECUTABLE=$(pwd)/build_host_protoc/bin/protoc \
  -Dprotobuf_BUILD_PROTOC_BINARIES=OFF \
  -DUSE_PRIORITIZED_TEXT_FOR_LD=OFF \
  -DUSE_SYSTEM_LIBS=OFF \
  -DUSE_FBGEMM=OFF -DUSE_KINETO=OFF -DUSE_NNPACK=OFF \
  -DUSE_XNNPACK=OFF -DUSE_PYTORCH_QNNPACK=OFF -DUSE_KLEIDIAI=OFF \
  -DBUILD_PYTHON=ON -DBUILD_CAFFE2=OFF \
  -DUSE_CUDA=OFF -DUSE_ROCM=OFF -DUSE_NUMPY=OFF \
  -DUSE_DISTRIBUTED=OFF -DUSE_MPI=OFF -DUSE_NCCL=OFF \
  -DCMAKE_BUILD_TYPE=Release -Wno-dev -GNinja
```

配置成功后应看到 `CXX_SVE512_FOUND=1`。

### 4. 编译

```bash
ninja -C build-sve512-native libtorch_cpu.so vec_test_all_types_SVE512

# Python 绑定（可选）
ninja -C build-sve512-native torch_python _C
```

### 5. 验证

**Vec 全量测试：**

```bash
./build-sve512-native/bin/vec_test_all_types_SVE512
# 期望：360/360 PASSED
```

**ATen / Python 冒烟：**

```bash
export PYTHONPATH=$(pwd):$(pwd)/build-sve512-native
export LD_LIBRARY_PATH=$(pwd)/build-sve512-native/lib

python3 -c "
import torch
print('cpu_capability=', torch._C._get_cpu_capability())
print('sve_max_length=', torch.cpu.get_capabilities().get('sve_max_length'))
a = torch.ones(512)
b = torch.ones(512)
print('add_sum=', float((a + b).sum()))
"
# 期望：cpu_capability= SVE512, sve_max_length= 512, add_sum= 1024.0
```

**强制 SVE512 路径（检测异常时）：**

```bash
export ATEN_CPU_CAPABILITY=sve512
```

## A64FX 特别注意

| 项 | 说明 |
|----|------|
| **无 FEAT_BF16** | A64FX 是 ARMv8.2-A+SVE，无 v8.6 BF16。全局 `-march=...+bf16` 可能 SIGILL |
| **SVE512 编译 flag** | 仅 SVE512 TU 使用 `-march=armv8-a+sve -msve-vector-bits=512`（不带 `+bf16`） |
| **运行时分发** | 已去掉 `cpuinfo_has_arm_bf16()` 对 SVE 路径的硬性要求 |
| **bfloat16 算子** | 可能走 scalar/DEFAULT 回退；先验证 float32 |
| **不用 FCC** | 跳过 `vec/sve/fcc/`、`fjlapackexsve.so` 等富士通专用路径 |

## 建议调试顺序

1. `git clone` + `submodule update`
2. cmake native 配置（确认 `CXX_SVE512_FOUND`）
3. `vec_test_all_types_SVE512`（360/360）
4. `import torch` + `cpu_capability` / `sve_max_length`
5. 简单 ATen：`ones`、`add`、`mm`
6. `torch.compile` 简单 kernel（Inductor SVE512）
7. 性能与回归（对比 DEFAULT/NEON）

## x86 本地调试（交叉 + QEMU，可选）

在推送到 Fugaku 前，x86 上仍可用：

```bash
# libtorch + vec 测试
bash scripts/cross_build_sve512.sh

# Python 交叉 import（实验性）
bash scripts/cross_build_sve512_python.sh
```

QEMU 用 `-cpu max` 有 bf16，**测不出** A64FX 无 BF16 问题；真机验证不可替代。

## 2.10 线（可选）

若用 2.10 而非 mainline：

```bash
git clone --branch sve512-2.10 \
  git@github.com:nahso/pytorch-sve512.git pytorch-2.10-sve512
```

`scripts/third-party/build_torch_2.10.0_gcc_kblas_sve512.sh` 内路径为 Kunpeng/HPCKit 环境，迁移到 Fugaku 需改 `HPCKIT_ROOT`、`CONDA_ROOT`、`ACL_ROOT_DIR` 等。

## 相关文件

| 文件 | 作用 |
|------|------|
| `CLAUDE.md` | 项目全貌、决策记录 |
| `scripts/cross_build_sve512.sh` | x86 交叉 libtorch + QEMU vec 测试 |
| `scripts/cross_build_sve512_python.sh` | x86 交叉 Python 构建 |
| `scripts/aarch64-toolchain.cmake` | 交叉工具链（Fugaku 不需要） |
| `scripts/sve512_smoke_test.cpp` | 最小 ATen C++ 冒烟 |
