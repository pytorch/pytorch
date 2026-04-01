#include "dispatch.h"
#include "../generated/shaders.h"

#include <torch/library.h>

namespace torch_vulkan { namespace ops {

// ── arange ──────────────────────────────────────────────────────
// Create on CPU, then transfer to Vulkan
at::Tensor vulkan_arange(const at::Scalar& start, const at::Scalar& end,
                          const at::Scalar& step,
                          std::optional<at::ScalarType> dtype_opt,
                          std::optional<at::Layout> layout_opt,
                          std::optional<at::Device> device_opt,
                          std::optional<bool> pin_memory_opt) {
    // Default to float32 since Vulkan backend only supports float32 compute
    auto dtype = dtype_opt.value_or(at::kFloat);
    auto device = device_opt.value_or(c10::Device(c10::DeviceType::PrivateUse1, 0));

    auto cpu_result = at::arange(start, end, step,
                                  at::TensorOptions().dtype(dtype).device(at::kCPU));
    return cpu_result.to(device);
}

// ── linspace ────────────────────────────────────────────────────
at::Tensor vulkan_linspace(const at::Scalar& start, const at::Scalar& end,
                            int64_t steps,
                            std::optional<at::ScalarType> dtype_opt,
                            std::optional<at::Layout> layout_opt,
                            std::optional<at::Device> device_opt,
                            std::optional<bool> pin_memory_opt) {
    auto dtype = dtype_opt.value_or(at::kFloat);
    auto device = device_opt.value_or(c10::Device(c10::DeviceType::PrivateUse1, 0));
    auto cpu_result = at::linspace(start, end, steps,
                                    at::TensorOptions().dtype(dtype).device(at::kCPU));
    return cpu_result.to(device);
}

// ── eye ─────────────────────────────────────────────────────────
at::Tensor vulkan_eye(int64_t n,
                       std::optional<at::ScalarType> dtype_opt,
                       std::optional<at::Layout> layout_opt,
                       std::optional<at::Device> device_opt,
                       std::optional<bool> pin_memory_opt) {
    auto dtype = dtype_opt.value_or(at::kFloat);
    auto device = device_opt.value_or(c10::Device(c10::DeviceType::PrivateUse1, 0));
    auto cpu_result = at::eye(n, at::TensorOptions().dtype(dtype).device(at::kCPU));
    return cpu_result.to(device);
}

at::Tensor vulkan_eye_m(int64_t n, int64_t m,
                         std::optional<at::ScalarType> dtype_opt,
                         std::optional<at::Layout> layout_opt,
                         std::optional<at::Device> device_opt,
                         std::optional<bool> pin_memory_opt) {
    auto dtype = dtype_opt.value_or(at::kFloat);
    auto device = device_opt.value_or(c10::Device(c10::DeviceType::PrivateUse1, 0));
    auto cpu_result = at::eye(n, m, at::TensorOptions().dtype(dtype).device(at::kCPU));
    return cpu_result.to(device);
}

// ── full ────────────────────────────────────────────────────────
at::Tensor vulkan_full(at::IntArrayRef size, const at::Scalar& fill_value,
                        std::optional<at::ScalarType> dtype_opt,
                        std::optional<at::Layout> layout_opt,
                        std::optional<at::Device> device_opt,
                        std::optional<bool> pin_memory_opt) {
    auto dtype = dtype_opt.value_or(at::kFloat);
    auto device = device_opt.value_or(c10::Device(c10::DeviceType::PrivateUse1, 0));
    auto cpu_result = at::full(size, fill_value,
                                at::TensorOptions().dtype(dtype).device(at::kCPU));
    return cpu_result.to(device);
}

// ── scalar_tensor ───────────────────────────────────────────────
at::Tensor vulkan_scalar_tensor(const at::Scalar& s,
                                 std::optional<at::ScalarType> dtype_opt,
                                 std::optional<at::Layout> layout_opt,
                                 std::optional<at::Device> device_opt,
                                 std::optional<bool> pin_memory_opt) {
    auto dtype = dtype_opt.value_or(at::kFloat);
    auto device = device_opt.value_or(c10::Device(c10::DeviceType::PrivateUse1, 0));
    auto cpu_result = at::scalar_tensor(s, at::TensorOptions().dtype(dtype).device(at::kCPU));
    return cpu_result.to(device);
}

}} // namespace torch_vulkan::ops
