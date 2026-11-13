#include <torch/csrc/inductor/aoti_torch/c/shim_mps.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/Exception.h>

#include <functional>

using torch::stable::Tensor;

namespace {

const char* SCALE_NEGATE_CLAMP_SHADER = R"MPS_SCALE_CLAMP(
#include <metal_stdlib>
using namespace metal;
kernel void scale_negate_clamp(device const float* input [[buffer(0)]],
                               device float* output      [[buffer(1)]],
                               constant float& scale     [[buffer(2)]],
                               constant bool& negate     [[buffer(3)]],
                               constant float2& bounds   [[buffer(4)]],
                               uint index [[thread_position_in_grid]]) {
  float v = input[index] * scale;
  v = negate ? -v : v;
  output[index] = clamp(v, bounds.x, bounds.y);
}
)MPS_SCALE_CLAMP";

AOTIMetalKernelFunctionHandle get_scale_negate_clamp_kernel() {
  static AOTIMetalShaderLibraryHandle lib_handle = []() {
    AOTIMetalShaderLibraryHandle handle = nullptr;
    STABLE_TORCH_ERROR_CODE_CHECK(
        aoti_torch_mps_create_shader_library(SCALE_NEGATE_CLAMP_SHADER, &handle));
    return handle;
  }();
  AOTIMetalKernelFunctionHandle func = nullptr;
  STABLE_TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_get_kernel_function(lib_handle, "scale_negate_clamp", &func));
  return func;
}

Tensor my_mps_scale_negate_clamp(
    Tensor input,
    double scale,
    bool negate,
    double low,
    double high) {
  STD_TORCH_CHECK(
      input.scalar_type() == torch::headeronly::ScalarType::Float,
      "input must be float32");
  STD_TORCH_CHECK(input.numel() > 0, "input must be non-empty");

  Tensor input_ = torch::stable::contiguous(input);
  Tensor output = torch::stable::empty_like(input_);
  AOTIMetalKernelFunctionHandle func = get_scale_negate_clamp_kernel();

  float scale_f = static_cast<float>(scale);
  float bounds[2] = {static_cast<float>(low), static_cast<float>(high)};
  auto numel = static_cast<uint64_t>(input_.numel());

  std::function<void(AOTIMetalKernelFunctionHandle)> encode =
      [&](AOTIMetalKernelFunctionHandle f) {
        STABLE_TORCH_ERROR_CODE_CHECK(aoti_torch_mps_start_encoding(f));
        STABLE_TORCH_ERROR_CODE_CHECK(
            aoti_torch_mps_set_arg_tensor(f, 0, input_.get()));
        STABLE_TORCH_ERROR_CODE_CHECK(
            aoti_torch_mps_set_arg_tensor(f, 1, output.get()));
        STABLE_TORCH_ERROR_CODE_CHECK(
            torch_mps_set_arg_bytes(f, 2, &scale_f, sizeof(float)));
        STABLE_TORCH_ERROR_CODE_CHECK(
            torch_mps_set_arg_bytes(f, 3, &negate, sizeof(bool)));
        STABLE_TORCH_ERROR_CODE_CHECK(
            torch_mps_set_arg_bytes(f, 4, bounds, sizeof(bounds)));
        STABLE_TORCH_ERROR_CODE_CHECK(aoti_torch_mps_dispatch_single(f, numel));
      };
  STABLE_TORCH_ERROR_CODE_CHECK(aoti_torch_mps_run_command_block(
      func, aoti_torch_mps_shared_callback, &encode));
  return output;
}

Tensor my_mps_set_arg_bytes_raw(Tensor input, int64_t size, bool null_ptr) {
  Tensor input_ = torch::stable::contiguous(input);
  AOTIMetalKernelFunctionHandle func = get_scale_negate_clamp_kernel();

  static const char blob[4096] = {};
  const void* ptr = null_ptr ? nullptr : blob;
  auto size_u = static_cast<uint64_t>(size);

  std::function<void(AOTIMetalKernelFunctionHandle)> encode =
      [&](AOTIMetalKernelFunctionHandle f) {
        STABLE_TORCH_ERROR_CODE_CHECK(aoti_torch_mps_start_encoding(f));
        STABLE_TORCH_ERROR_CODE_CHECK(
            aoti_torch_mps_set_arg_tensor(f, 0, input_.get()));
        STABLE_TORCH_ERROR_CODE_CHECK(
            torch_mps_set_arg_bytes(f, 2, ptr, size_u));
      };
  STABLE_TORCH_ERROR_CODE_CHECK(aoti_torch_mps_run_command_block(
      func, aoti_torch_mps_shared_callback, &encode));
  return torch::stable::empty_like(input_);
}

Tensor my_mps_set_arg_bytes_lifetime(Tensor input) {
  Tensor input_ = torch::stable::contiguous(input);
  Tensor output = torch::stable::empty_like(input_);
  AOTIMetalKernelFunctionHandle func = get_scale_negate_clamp_kernel();
  auto numel = static_cast<uint64_t>(input_.numel());

  std::function<void(AOTIMetalKernelFunctionHandle)> encode =
      [&](AOTIMetalKernelFunctionHandle f) {
        STABLE_TORCH_ERROR_CODE_CHECK(aoti_torch_mps_start_encoding(f));
        STABLE_TORCH_ERROR_CODE_CHECK(
            aoti_torch_mps_set_arg_tensor(f, 0, input_.get()));
        STABLE_TORCH_ERROR_CODE_CHECK(
            aoti_torch_mps_set_arg_tensor(f, 1, output.get()));
        float scale = 3.0f;
        bool negate = false;
        float bounds[2] = {-1e30f, 1e30f};
        STABLE_TORCH_ERROR_CODE_CHECK(
            torch_mps_set_arg_bytes(f, 2, &scale, sizeof(float)));
        STABLE_TORCH_ERROR_CODE_CHECK(
            torch_mps_set_arg_bytes(f, 3, &negate, sizeof(bool)));
        STABLE_TORCH_ERROR_CODE_CHECK(
            torch_mps_set_arg_bytes(f, 4, bounds, sizeof(bounds)));
        scale = -7.0f;
        negate = true;
        bounds[0] = 0.0f;
        bounds[1] = 0.0f;
        STABLE_TORCH_ERROR_CODE_CHECK(aoti_torch_mps_dispatch_single(f, numel));
      };
  STABLE_TORCH_ERROR_CODE_CHECK(aoti_torch_mps_run_command_block(
      func, aoti_torch_mps_shared_callback, &encode));
  return output;
}

} // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def(
      "my_mps_scale_negate_clamp(Tensor input, float scale, bool negate, float low, float high) -> Tensor");
  m.def("my_mps_set_arg_bytes_raw(Tensor input, int size, bool null_ptr) -> Tensor");
  m.def("my_mps_set_arg_bytes_lifetime(Tensor input) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, MPS, m) {
  m.impl("my_mps_scale_negate_clamp", TORCH_BOX(&my_mps_scale_negate_clamp));
  m.impl("my_mps_set_arg_bytes_raw", TORCH_BOX(&my_mps_set_arg_bytes_raw));
  m.impl("my_mps_set_arg_bytes_lifetime", TORCH_BOX(&my_mps_set_arg_bytes_lifetime));
}
