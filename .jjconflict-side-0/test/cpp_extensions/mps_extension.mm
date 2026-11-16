#include <torch/extension.h>
#include <ATen/native/mps/OperationUtils.h>

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_cpu_add_output", &get_cpu_add_output);
  m.def("get_mps_add_output", &get_mps_add_output);
  m.def("mps_add_one_new_context", &mps_add_one_new_encoder);
}
