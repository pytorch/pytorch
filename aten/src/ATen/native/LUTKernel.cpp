#include <ATen/native/TensorIterator.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/cpu/Loops.h>
#include <c10/util/BFloat16.h>
#include <ATen/native/cpu/Gelu.h>
#include <cmath>
#include <array>
#include <iostream>
#include <cstring>
#include <ATen/native/LUTKernel.h>
#include <ATen/Parallel.h>

namespace at::native {

// Exact GELU formula
static float gelu_exact(float x) {
    return x * 0.5f * (1.0f + std::erf(x / std::sqrt(2.0f)));
}

// Precomputed GELU LUT for BF16 bit patterns
static std::array<c10::BFloat16, 1 << 16> build_lut() {
    std::array<c10::BFloat16, 1 << 16> lut{};
    for (uint32_t i = 0; i < (1 << 16); ++i) {
        uint16_t raw = static_cast<uint16_t>(i);
        uint32_t expanded = static_cast<uint32_t>(raw) << 16;
        float val;
        std::memcpy(&val, &expanded, sizeof(float));
        lut[i] = c10::BFloat16(gelu_exact(val));
    }
    return lut;
}

static const auto gelu_lut = build_lut();

void gelu_bf16_lut_kernel(at::TensorIteratorBase& iter) {
  // std::cout << "GELU LUT kernel for BF16 triggered!\n";
    if (iter.dtype() == at::kBFloat16) {
        using scalar_t = c10::BFloat16;

        int64_t batch_size = iter.numel();
        auto input = static_cast<const scalar_t*>(iter.data_ptr(1));
        auto output = static_cast<scalar_t*>(iter.data_ptr(0));

        try {
            //  Main parallel implementation using at::parallel_for and manually retrieved indicies. 
            at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
                for (int64_t i = start; i < end; ++i) {
                    uint16_t raw;
                    std::memcpy(&raw, &input[i], sizeof(uint16_t));
                    output[i] = gelu_lut[raw];
                }
            });
        } catch (...) {
            // Fallback: Use PyTorch's built-in cpu_kernel if something fails
            std::cout << "[Fallback] Using cpu_kernel instead of manual parallel_for.\n";
            cpu_kernel(iter, [](scalar_t x) -> scalar_t {
                uint16_t raw;
                std::memcpy(&raw, &x, sizeof(uint16_t));
                return gelu_lut[raw];
            });
        }

    } else {
        TORCH_CHECK(false, "gelu_bf16_lut_kernel: only supports bfloat16");
    }
}

} // namespace at::native
