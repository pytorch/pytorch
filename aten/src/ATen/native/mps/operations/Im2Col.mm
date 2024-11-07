#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/col2im_native.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/im2col_native.h>
#endif

namespace at::native {
using namespace mps;
#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Im2Col_metallib.h>
#endif

namespace {
static void im2col_out_mps_template(Tensor& output,
                                    const Tensor& input_,
                                    IntArrayRef kernel_size,
                                    IntArrayRef dilation,
                                    IntArrayRef padding,
                                    IntArrayRef stride) {
  TORCH_CHECK(kernel_size.size() == 2, "It is expected kernel_size equals to 2, but got size ", kernel_size.size());

  TORCH_CHECK(dilation.size() == 2, "It is expected dilation equals to 2, but got size ", dilation.size());

  TORCH_CHECK(padding.size() == 2, "It is expected padding equals to 2, but got size ", padding.size());

  TORCH_CHECK(stride.size() == 2, "It is expected stride equals to 2, but got size ", stride.size());

  const auto kernel_height = kernel_size[0];
  int64_t kernel_width = kernel_size[1];
  int64_t dilation_height = dilation[0];
  int64_t dilation_width = dilation[1];
  int64_t pad_height = padding[0];
  int64_t pad_width = padding[1];
  int64_t stride_height = stride[0];
  int64_t stride_width = stride[1];

  Tensor input = input_.contiguous();

  bool batched_input = true;

  if (input.dim() == 3) {
    batched_input = false;
    input = input.unsqueeze(0);
  }

  int64_t batch_size = input.size(0);
  int64_t n_input_plane = input.size(1);
  int64_t input_height = input.size(2);
  int64_t input_width = input.size(3);

  int64_t output_height =
      (input_height + 2 * pad_height - (dilation_height * (kernel_height - 1) + 1)) / stride_height + 1;
  int64_t output_width = (input_width + 2 * pad_width - (dilation_width * (kernel_width - 1) + 1)) / stride_width + 1;
  int64_t n_output_plane = n_input_plane * kernel_width * kernel_height;
  int64_t output_length = output_height * output_width;

  output.resize_({batch_size, n_output_plane, output_length});
  auto stream = getCurrentMPSStream();
  auto device = MPSDevice::getInstance()->device();
  auto im2colPSO = lib.getPipelineStateForFunc("im2col_" + mps::scalarToMetalTypeString(input));
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      std::array<int32_t, 4> kernel_dilation = {static_cast<int32_t>(kernel_width),
                                                static_cast<int32_t>(kernel_height),
                                                static_cast<int32_t>(dilation_width),
                                                static_cast<int32_t>(dilation_height)};
      std::array<int32_t, 4> padding_stride = {static_cast<int32_t>(pad_width),
                                               static_cast<int32_t>(pad_height),
                                               static_cast<int32_t>(stride_width),
                                               static_cast<int32_t>(stride_height)};
      std::array<int64_t, 4> input_sizes = {input_width, input_height, n_input_plane, batch_size};
      std::array<int64_t, 4> input_strides = {input.stride(3), input.stride(2), input.stride(1), input.stride(0)};
      std::array<int64_t, 4> output_strides = {output.stride(2), output.stride(1), output.stride(0), output_width};
      getMPSProfiler().beginProfileKernel(im2colPSO, "im2col", {input, output});
      auto computeEncoder = stream->commandEncoder();
      [computeEncoder setComputePipelineState:im2colPSO];
      mtl_setBuffer(computeEncoder, input, 0);
      mtl_setBuffer(computeEncoder, output, 1);
      mtl_setBytes(computeEncoder, kernel_dilation, 2);
      mtl_setBytes(computeEncoder, padding_stride, 3);
      mtl_setBytes(computeEncoder, input_strides, 4);
      mtl_setBytes(computeEncoder, output_strides, 5);
      mtl_setBytes(computeEncoder, input_sizes, 6);
      [computeEncoder dispatchThreads:MTLSizeMake(output_length, n_input_plane, batch_size)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      getMPSProfiler().endProfileKernel(im2colPSO);
    }
  });
  if (!batched_input) {
    output = output.squeeze(0);
  }
}

} // anonymous namespace
Tensor& im2col_out_mps(const Tensor& input,
                       IntArrayRef kernel_size,
                       IntArrayRef dilation,
                       IntArrayRef padding,
                       IntArrayRef stride,
                       Tensor& output) {
  im2col_out_mps_template(output, input, kernel_size, dilation, padding, stride);
  return output;
}

Tensor im2col_mps(const Tensor& input,
                  IntArrayRef kernel_size,
                  IntArrayRef dilation,
                  IntArrayRef padding,
                  IntArrayRef stride) {
  Tensor output = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  im2col_out_mps_template(output, input, kernel_size, dilation, padding, stride);
  return output;
}
} // namespace at::native
