#include <torch/extension.h>
#include <ATen/native/mps/OperationUtils.h>
#include <torch/csrc/inductor/aoti_torch/c/shim_mps.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>

// this sample custom kernel is taken from:
// https://developer.apple.com/documentation/metal/performing_calculations_on_a_gpu
static at::native::mps::MetalShaderLibrary lib(R"MPS_ADD_ARRAYS(
#include <metal_stdlib>
using namespace metal;
kernel void add_arrays(device const float* inA,
                       device const float* inB,
                       device float* result,
                       uint index [[thread_position_in_grid]])
{
    result[index] = inA[index] + inB[index];
}

kernel void add_one(device float* data,
                    uint index [[thread_position_in_grid]]) {
  data[index] += 1.0;
}
)MPS_ADD_ARRAYS");

at::Tensor get_cpu_add_output(at::Tensor & cpu_input1, at::Tensor & cpu_input2) {
  return cpu_input1 + cpu_input2;
}

at::Tensor get_mps_add_output(at::Tensor & mps_input1, at::Tensor & mps_input2) {

  // smoke tests
  TORCH_CHECK(mps_input1.is_mps());
  TORCH_CHECK(mps_input2.is_mps());
  TORCH_CHECK(mps_input1.sizes() == mps_input2.sizes());

  using namespace at::native::mps;
  at::Tensor mps_output = at::empty_like(mps_input1);

  @autoreleasepool {
    size_t numThreads = mps_output.numel();
    auto kernelPSO = lib.getPipelineStateForFunc("add_arrays");
    MPSStream* mpsStream = getCurrentMPSStream();

    dispatch_sync(mpsStream->queue(), ^() {
      // Start a compute pass.
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      TORCH_CHECK(computeEncoder, "Failed to create compute command encoder");

      // Encode the pipeline state object and its parameters.
      [computeEncoder setComputePipelineState: kernelPSO];
      mtl_setBuffer(computeEncoder, mps_input1, 0);
      mtl_setBuffer(computeEncoder, mps_input2, 1);
      mtl_setBuffer(computeEncoder, mps_output, 2);
      mtl_dispatch1DJob(computeEncoder, kernelPSO, numThreads);
    });
  }
  return mps_output;
}

void mps_add_one_new_encoder(const at::Tensor& input) {
  using namespace at::native::mps;
  TORCH_CHECK(input.is_mps());
  TORCH_CHECK(input.numel() > 0);

  @autoreleasepool {
  auto kernelPSO = lib.getPipelineStateForFunc("add_one");
  auto serialQueue = torch::mps::get_dispatch_queue();

  dispatch_sync(serialQueue, ^(){
    auto commandBuffer = torch::mps::get_command_buffer();
    // Start a compute pass.
    auto computeEncoder = [commandBuffer computeCommandEncoder];
    TORCH_CHECK(computeEncoder, "Failed to create compute command encoder");
    [computeEncoder setComputePipelineState: kernelPSO];
    mtl_setArgs(computeEncoder, input);
    mtl_dispatch1DJob(computeEncoder, kernelPSO, input.numel());
    [computeEncoder endEncoding];
     torch::mps::commit();
  });
  }
}

// Exercises the stable shim's torch_mps_set_arg_bytes: binds a float
// and a bool scalar inline (Metal setBytes) instead of smuggling them through
// 1-element tensors.
static const char* SCALE_NEGATE_SHADER = R"MPS_SCALE_NEGATE(
#include <metal_stdlib>
using namespace metal;
kernel void scale_negate(device const float* input [[buffer(0)]],
                         device float* output      [[buffer(1)]],
                         constant float& scale     [[buffer(2)]],
                         constant bool& negate     [[buffer(3)]],
                         uint index [[thread_position_in_grid]]) {
  float v = input[index] * scale;
  output[index] = negate ? -v : v;
}
)MPS_SCALE_NEGATE";

struct ScaleNegateArgs {
  AtenTensorHandle input;
  AtenTensorHandle output;
  float scale;
  bool negate;
  uint64_t numel;
};

static void scale_negate_encode(AOTIMetalKernelFunctionHandle func, void* user_data) {
  auto* args = static_cast<ScaleNegateArgs*>(user_data);
  TORCH_CHECK(aoti_torch_mps_start_encoding(func) == AOTI_TORCH_SUCCESS);
  TORCH_CHECK(aoti_torch_mps_set_arg_tensor(func, 0, args->input) == AOTI_TORCH_SUCCESS);
  TORCH_CHECK(aoti_torch_mps_set_arg_tensor(func, 1, args->output) == AOTI_TORCH_SUCCESS);
  TORCH_CHECK(torch_mps_set_arg_bytes(func, 2, &args->scale, sizeof(float)) == AOTI_TORCH_SUCCESS);
  TORCH_CHECK(torch_mps_set_arg_bytes(func, 3, &args->negate, sizeof(bool)) == AOTI_TORCH_SUCCESS);
  TORCH_CHECK(aoti_torch_mps_dispatch_single(func, args->numel) == AOTI_TORCH_SUCCESS);
}

at::Tensor get_mps_scale_negate_output(const at::Tensor& input, double scale, bool negate) {
  TORCH_CHECK(input.is_mps());
  TORCH_CHECK(input.scalar_type() == at::kFloat);
  TORCH_CHECK(input.numel() > 0);

  auto input_ = input.contiguous();
  at::Tensor output = at::empty_like(input_);

  static AOTIMetalShaderLibraryHandle lib_handle = []() {
    AOTIMetalShaderLibraryHandle handle = nullptr;
    TORCH_CHECK(
        aoti_torch_mps_create_shader_library(SCALE_NEGATE_SHADER, &handle) ==
        AOTI_TORCH_SUCCESS);
    return handle;
  }();
  AOTIMetalKernelFunctionHandle func = nullptr;
  TORCH_CHECK(
      aoti_torch_mps_get_kernel_function(lib_handle, "scale_negate", &func) ==
      AOTI_TORCH_SUCCESS);

  ScaleNegateArgs args{
      torch::aot_inductor::tensor_pointer_to_tensor_handle(&input_),
      torch::aot_inductor::tensor_pointer_to_tensor_handle(&output),
      static_cast<float>(scale),
      negate,
      static_cast<uint64_t>(input_.numel())};
  TORCH_CHECK(
      aoti_torch_mps_run_command_block(func, &scale_negate_encode, &args) ==
      AOTI_TORCH_SUCCESS);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_cpu_add_output", &get_cpu_add_output);
  m.def("get_mps_add_output", &get_mps_add_output);
  m.def("mps_add_one_new_context", &mps_add_one_new_encoder);
  m.def("get_mps_scale_negate_output", &get_mps_scale_negate_output);
}
