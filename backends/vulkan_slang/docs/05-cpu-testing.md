# CPU-First Development & Testing Strategy

**You do NOT need a GPU to develop or test this project.** The entire implementation can be built, tested, and validated on CPU.

---

## Three CPU Mechanisms

### 1. SwiftShader — Software Vulkan (Primary CI Target)

Google's software Vulkan implementation. Runs entirely on CPU. Supports Vulkan 1.1+ compute shaders. Slow (~100x slower than a real GPU) but functionally correct.

```bash
# Ubuntu/Debian
sudo apt-get install swiftshader

# Or build from source for latest features
git clone https://github.com/nickvdp/swiftshader.git
cd swiftshader && mkdir build && cd build
cmake .. -DSWIFTSHADER_BUILD_TESTS=OFF
cmake --build . -j$(nproc)

# Point Vulkan loader to SwiftShader
export VK_ICD_FILENAMES=/usr/share/swiftshader/vk_swiftshader_icd.json

# Verify
vulkaninfo --summary  # Should show "SwiftShader Device"
```

**Known limitations:**
- No float64 support
- Limited subgroup size (may be 1)
- No `KHR_cooperative_matrix` extension
- No float16 storage in some builds
- ~100x slower than real GPU — use small tensor sizes in tests

### 2. Slang CPU Target — Fastest Iteration (No Vulkan At All)

`slangc shader.slang -target cpp` emits plain C++ from the same shader source. Test shader math as a native CPU function without any Vulkan infrastructure.

```bash
slangc shaders/activation/gelu.slang -target cpp -o cpu_tests/generated/gelu.cpp
cd cpu_tests && cmake --build build && ./test_activation_cpu
```

### 3. lavapipe — Mesa CPU Vulkan

Mesa's CPU-based Vulkan driver for Linux. Another software Vulkan option, sometimes better supported than SwiftShader for newer Vulkan features.

---

## Development Flow

```
Phase 1 (Shader math correctness):
  Edit .slang → slangc -target cpp → compile → run CPU unit test
  Fastest iteration. Tests pure math: does gelu compute the right values?
  Does the backward pass produce correct gradients?

Phase 2 (Vulkan pipeline correctness):
  Edit .slang → slangc -target spirv → load into SwiftShader Vulkan
  Tests the full Vulkan path: buffer allocation, descriptor sets, dispatch,
  host↔device copies, synchronization. All on CPU.

Phase 3 (Performance on real GPU):
  Same SPIR-V → run on actual GPU hardware
  Only needed for benchmarking and performance optimization.
  Correctness should already be verified from Phase 1+2.
```

---

## Three Test Tiers

### Tier 1: CPU Shader Unit Tests (`cpu_tests/`)

No Vulkan at all. Slang compiles to C++. Tests shader math in isolation.

```cpp
// cpu_tests/test_activation_cpu.cpp
#include "generated/gelu_cpu.h"
#include <cassert>
#include <cmath>

void test_gelu_forward() {
    float input[] = {-2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f};
    for (float x : input) {
        float slang_result = gelu_exact(x);
        float reference = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x*x*x)));
        assert(fabsf(slang_result - reference) < 1e-6f);
    }
}

void test_gelu_backward() {
    float x = 0.5f;
    float h = 1e-4f;
    float numerical_grad = (gelu_exact(x + h) - gelu_exact(x - h)) / (2.0f * h);

    DifferentialPair<float> dp = diffPair(x, 0.0f);
    bwd_diff_gelu_exact(dp, 1.0f);
    float autodiff_grad = dp.getDifferential();

    assert(fabsf(autodiff_grad - numerical_grad) < 1e-3f);
}

int main() {
    test_gelu_forward();
    test_gelu_backward();
    printf("All CPU shader tests passed!\n");
    return 0;
}
```

These tests verify:
- Forward math correctness (does `gelu(0.5)` return the right value?)
- Autodiff correctness (does `bwd_diff(gelu)` match the analytical derivative?)
- Edge cases (NaN, inf, denormals, zero)
- Dtype variants (f32 vs f16 precision)

Fast iteration: edit `.slang` → compile to C++ (~1 sec) → run test (~instant).

**Implementation tasks:**
- [ ] `cpu_tests/CMakeLists.txt` — builds CPU tests independently of Vulkan
- [ ] `cpu_tests/test_trivial.cpp` — compile trivial Slang to C++, verify output
- [ ] `cpu_tests/test_unary_cpu.cpp` — all unary op forward + backward
- [ ] `cpu_tests/test_binary_cpu.cpp` — all binary op forward + backward
- [ ] `cpu_tests/test_activation_cpu.cpp` — all activation forward + backward
- [ ] `cpu_tests/test_matmul_cpu.cpp` — naive matmul correctness
- [ ] `cpu_tests/test_autodiff_cpu.cpp` — systematic bwd_diff vs numerical gradients

### Tier 2: Vulkan Integration Tests (`tests/`)

Requires Vulkan (SwiftShader OK). Tests the full pipeline end-to-end.

```python
def test_add_vulkan():
    a = torch.randn(64, 64)
    b = torch.randn(64, 64)
    result = (a.to("vulkan") + b.to("vulkan")).cpu()
    torch.testing.assert_close(a + b, result, rtol=1e-5, atol=1e-5)
```

These verify:
- Buffer allocation and host↔device copies
- Descriptor set binding
- Command buffer recording and submission
- Pipeline barriers and synchronization
- PyTorch dispatch integration
- Autograd backward pass end-to-end

**Implementation tasks:**
- [ ] `tests/conftest.py` — auto-skip if no Vulkan device, `vulkan_device` fixture
- [ ] `tests/test_basic_ops.py` — element-wise forward correctness
- [ ] `tests/test_matmul.py` — GEMM correctness (naive vs tiled)
- [ ] `tests/test_conv.py` — convolution tests
- [ ] `tests/test_autograd.py` — backward pass via `gradcheck()`
- [ ] `tests/test_training.py` — end-to-end training loops (MNIST, MLP)
- [ ] `tests/test_amp.py` — mixed precision training
- [ ] `tests/test_serialization.py` — save/load round-trip
- [ ] `tests/test_slang_autodiff.py` — Slang bwd_diff matches numerical gradients

### Tier 3: GPU Benchmarks (`tests/benchmarks/`)

Only for performance optimization. Not needed for correctness. Requires real GPU hardware.

- [ ] `tests/benchmarks/bench_matmul.py` — GEMM throughput (GFLOPS)
- [ ] `tests/benchmarks/bench_conv.py` — convolution throughput
- [ ] `tests/benchmarks/bench_e2e.py` — end-to-end training throughput

---

## conftest.py Auto-Detection

```python
# tests/conftest.py
import pytest
import torch

def pytest_configure(config):
    try:
        import torch_vulkan
        if not torch.vulkan.is_available():
            pytest.skip("No Vulkan device (install SwiftShader for CPU testing)")
    except ImportError:
        pytest.skip("torch_vulkan not installed")

@pytest.fixture
def vulkan_device():
    return torch.device("vulkan:0")  # SwiftShader counts as a device
```

---

## CI Configuration

```yaml
# .github/workflows/test.yml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install Vulkan SDK + SwiftShader
        run: |
          wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo tee /etc/apt/trusted.gpg.d/lunarg.asc
          sudo wget -qO /etc/apt/sources.list.d/lunarg.list https://packages.lunarg.com/vulkan/lunarg-vulkan-jammy.list
          sudo apt-get update && sudo apt-get install -y vulkan-sdk swiftshader
      - name: Install Slang
        run: |
          SLANG_VER=$(cat tools/slang_version.txt)
          wget https://github.com/shader-slang/slang/releases/download/v${SLANG_VER}/slang-linux-x64.tar.gz
          tar xf slang-linux-x64.tar.gz && sudo mv bin/slangc /usr/local/bin/
      - name: Build & Test
        env:
          VK_ICD_FILENAMES: /usr/share/swiftshader/vk_swiftshader_icd.json
        run: |
          pip install -e .
          pytest tests/ -x --timeout=300
```

---

## Testing Best Practices

1. **Small tensors in SwiftShader tests.** Use 64×64 matmul, not 4096×4096. SwiftShader is 100x slower.
2. **CPU tests first.** Edit shader → compile to C++ → run test → instant feedback. Only move to SwiftShader for pipeline correctness.
3. **Feature-gate tests.** SwiftShader lacks float64, cooperative matrix, some subgroup ops. Skip those tests on SwiftShader.
4. **Generous timeouts.** `--timeout=300` in pytest for SwiftShader runs.
5. **Validation layers always on.** Catch API misuse early. SwiftShader supports validation layers.
6. **Deterministic seeds.** Always set `torch.manual_seed()` for reproducible tests.

---

## SwiftShader Integration Tasks

- [ ] Verify SwiftShader installs and provides a `VkPhysicalDevice` on CI
- [ ] Set `VK_ICD_FILENAMES` environment variable to SwiftShader ICD JSON
- [ ] Verify `vulkaninfo --summary` shows SwiftShader Device
- [ ] Document SwiftShader limitations (no float64, limited subgroup support)
- [ ] Set generous test timeouts for SwiftShader (100x slower than GPU)
