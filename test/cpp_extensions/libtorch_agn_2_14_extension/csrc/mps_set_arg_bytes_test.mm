#include <torch/csrc/inductor/aoti_torch/c/shim_mps.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/Exception.h>

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

struct ScaleNegateClampArgs {
  AtenTensorHandle input;
  AtenTensorHandle output;
  float scale;
  bool negate;
  float bounds[2];
  uint64_t numel;
};

void scale_negate_clamp_encode(
    AOTIMetalKernelFunctionHandle func,
    void* user_data) {
  auto* args = static_cast<ScaleNegateClampArgs*>(user_data);
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_start_encoding(func));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_tensor(func, 0, args->input));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_tensor(func, 1, args->output));
  TORCH_ERROR_CODE_CHECK(
      torch_mps_set_arg_bytes(func, 2, &args->scale, sizeof(float)));
  TORCH_ERROR_CODE_CHECK(
      torch_mps_set_arg_bytes(func, 3, &args->negate, sizeof(bool)));
  TORCH_ERROR_CODE_CHECK(
      torch_mps_set_arg_bytes(func, 4, args->bounds, sizeof(args->bounds)));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_dispatch_single(func, args->numel));
}

AOTIMetalKernelFunctionHandle get_scale_negate_clamp_kernel() {
  static AOTIMetalShaderLibraryHandle lib_handle = []() {
    AOTIMetalShaderLibraryHandle handle = nullptr;
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_mps_create_shader_library(SCALE_NEGATE_CLAMP_SHADER, &handle));
    return handle;
  }();
  AOTIMetalKernelFunctionHandle func = nullptr;
  TORCH_ERROR_CODE_CHECK(
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

  ScaleNegateClampArgs args{
      input_.get(),
      output.get(),
      static_cast<float>(scale),
      negate,
      {static_cast<float>(low), static_cast<float>(high)},
      static_cast<uint64_t>(input_.numel())};
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_run_command_block(func, &scale_negate_clamp_encode, &args));
  return output;
}

struct SetArgBytesInvalidArgs {
  AtenTensorHandle input;
  const void* ptr;
  uint64_t size;
};

void set_arg_bytes_invalid_encode(
    AOTIMetalKernelFunctionHandle func,
    void* user_data) {
  auto* args = static_cast<SetArgBytesInvalidArgs*>(user_data);
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_start_encoding(func));
  TORCH_ERROR_CODE_CHECK(aoti_torch_mps_set_arg_tensor(func, 0, args->input));
  STABLE_TORCH_ERROR_CODE_CHECK(
      torch_mps_set_arg_bytes(func, 2, args->ptr, args->size));
}

// STABLE_TORCH_ERROR_CODE_CHECK (not TORCH_ERROR_CODE_CHECK) here and in the
// encode callback, so the shim's TORCH_CHECK message survives to Python for
// assertRaisesRegex.
Tensor my_mps_set_arg_bytes_invalid(Tensor input, int64_t size, bool null_ptr) {
  Tensor input_ = torch::stable::contiguous(input);
  AOTIMetalKernelFunctionHandle func = get_scale_negate_clamp_kernel();

  static const float dummy = 0.0f;
  SetArgBytesInvalidArgs args{
      input_.get(),
      null_ptr ? nullptr : &dummy,
      static_cast<uint64_t>(size)};
  STABLE_TORCH_ERROR_CODE_CHECK(
      aoti_torch_mps_run_command_block(func, &set_arg_bytes_invalid_encode, &args));
  return torch::stable::empty_like(input_);
}

} // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def(
      "my_mps_scale_negate_clamp(Tensor input, float scale, bool negate, float low, float high) -> Tensor");
  m.def("my_mps_set_arg_bytes_invalid(Tensor input, int size, bool null_ptr) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, MPS, m) {
  m.impl("my_mps_scale_negate_clamp", TORCH_BOX(&my_mps_scale_negate_clamp));
  m.impl("my_mps_set_arg_bytes_invalid", TORCH_BOX(&my_mps_set_arg_bytes_invalid));
}
