#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/col2im_native.h>
#endif

namespace at::native {
using namespace mps;

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Col2Im_metallib.h>
#endif

namespace {

static void col2im_out_mps_template(const Tensor& input,
                                    Tensor& output,
                                    IntArrayRef output_size,
                                    IntArrayRef kernel_size,
                                    IntArrayRef dilation,
                                    IntArrayRef padding,
                                    IntArrayRef stride) {
  auto output_height = output_size[0];
  auto output_width = output_size[1];
  auto kernel_height = kernel_size[0];
  auto kernel_width = kernel_size[1];
  auto dilation_height = dilation[0];
  auto dilation_width = dilation[1];
  auto pad_height = padding[0];
  auto pad_width = padding[1];
  auto stride_height = stride[0];
  auto stride_width = stride[1];

  Tensor col_tensor = input.contiguous();
  bool batched_input = true;
  if (col_tensor.dim() == 2) {
    batched_input = false;
    col_tensor = col_tensor.unsqueeze(0);
  }

  auto batch_size = col_tensor.size(0);
  auto n_input_plane = col_tensor.size(1);
  TORCH_CHECK(n_input_plane % (kernel_height * kernel_width) == 0, "col2im_mps: invalid number of input channels");
  auto n_output_plane = n_input_plane / (kernel_height * kernel_width);
  auto input_batch_stride = col_tensor.stride(0);

  output.resize_({batch_size, n_output_plane, output_height, output_width});
  auto output_batch_stride = output.stride(0);

  auto height_col = (output_height + 2 * pad_height - (dilation_height * (kernel_height - 1) + 1)) / stride_height + 1;
  auto width_col = (output_width + 2 * pad_width - (dilation_width * (kernel_width - 1) + 1)) / stride_width + 1;

  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      auto col2imPSO = lib.getPipelineStateForFunc("col2im_kernel_" + mps::scalarToMetalTypeString(input));
      auto computeEncoder = stream->commandEncoder();
      [computeEncoder setComputePipelineState:col2imPSO];
      const uint32_t gridWidth = static_cast<uint32_t>(output_width);
      const uint32_t gridHeight = static_cast<uint32_t>(output_height);
      const uint32_t gridDepth = static_cast<uint32_t>(batch_size * n_output_plane);
      MTLSize gridSize = MTLSizeMake(gridWidth, gridHeight, gridDepth);
      const uint32_t maxThreadsPerGroup = col2imPSO.maxTotalThreadsPerThreadgroup;
      const uint32_t threadExecutionWidth = col2imPSO.threadExecutionWidth;
      uint32_t tgWidth = std::min(gridWidth, threadExecutionWidth);
      uint32_t tgHeight = std::min(gridHeight, maxThreadsPerGroup / tgWidth);
      MTLSize threadgroupSize = MTLSizeMake(tgWidth, tgHeight, 1);
      mtl_setArgs(
          computeEncoder,
          col_tensor,
          output,
          input_batch_stride,
          n_output_plane,
          std::array<uint32_t, 2>{static_cast<uint32_t>(output_height), static_cast<uint32_t>(output_width)}, // im_hw
          std::array<uint32_t, 2>{static_cast<uint32_t>(kernel_height),
                                  static_cast<uint32_t>(kernel_width)}, // kernel_hw
          std::array<uint32_t, 2>{static_cast<uint32_t>(pad_height), static_cast<uint32_t>(pad_width)}, // pad_hw
          std::array<uint32_t, 2>{static_cast<uint32_t>(stride_height),
                                  static_cast<uint32_t>(stride_width)}, // stride_hw
          std::array<uint32_t, 2>{static_cast<uint32_t>(dilation_height),
                                  static_cast<uint32_t>(dilation_width)}, // dilation_hw
          std::array<uint32_t, 2>{static_cast<uint32_t>(height_col), static_cast<uint32_t>(width_col)}, // col_hw
          output_batch_stride);
      [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    }
  });
  if (!batched_input) {
    output = output.squeeze(0);
  }
}

} // anonymous namespace

Tensor& col2im_out_mps(const Tensor& self,
                       IntArrayRef output_size,
                       IntArrayRef kernel_size,
                       IntArrayRef dilation,
                       IntArrayRef padding,
                       IntArrayRef stride,
                       Tensor& out) {
  col2im_out_mps_template(self, out, output_size, kernel_size, dilation, padding, stride);
  return out;
}

Tensor col2im_mps(const Tensor& self,
                  IntArrayRef output_size,
                  IntArrayRef kernel_size,
                  IntArrayRef dilation,
                  IntArrayRef padding,
                  IntArrayRef stride) {
  Tensor out = at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  col2im_out_mps_template(self, out, output_size, kernel_size, dilation, padding, stride);
  return out;
}

} // namespace at::native