#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/PoolingChecks.h>
#include <ATen/native/cpu/MaxUnpoolKernel.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/max_unpool2d_native.h>
#include <ATen/ops/max_unpool3d_native.h>
#endif

namespace at::native {

Tensor& max_unpooling2d_forward_out_cpu(
    const Tensor& self_,
    const Tensor& indices_,
    IntArrayRef output_size,
    Tensor& output) {
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic with duplicate indices
  at::globalContext().alertNotDeterministic("max_unpooling2d_forward_out");

  max_unpooling2d_shape_check(self_, indices_, output_size, "max_unpooling2d_forward_out_cpu()");

  auto oheight = output_size[0];
  auto owidth = output_size[1];

  auto memory_format = self_.suggest_memory_format();
  auto self = self_.contiguous(memory_format);
  auto indices = indices_.contiguous(memory_format);

  if (self.ndimension() == 3) {
    int64_t numChannels = self.size(0);
    output.resize_({numChannels, oheight, owidth});
  } else {
    int64_t numBatch = self.size(0);
    int64_t numChannels = self.size(1);
    output.resize_({numBatch, numChannels, oheight, owidth}, memory_format);
  }
  output.zero_();

  if (output.numel() != 0) {
    max_unpool2d_kernel(kCPU, output, self, indices);
  }

  return output;
}

Tensor max_unpooling2d_forward_cpu(
    const Tensor& self,
    const Tensor& indices,
    IntArrayRef output_size) {
  auto output = at::empty({0}, self.options());
  at::native::max_unpooling2d_forward_out_cpu(self, indices, output_size, output);
  return output;
}

Tensor& max_unpooling3d_forward_out_cpu(const Tensor& self_,
    const Tensor& indices_,
    IntArrayRef output_size,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output) {
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic with duplicate indices
  at::globalContext().alertNotDeterministic("max_unpooling3d_forward_out");

  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");

  auto self = self_.contiguous();
  auto indices = indices_.contiguous();

  max_unpooling3d_shape_check(
      self_, indices_, output_size, stride, padding, "max_unpooling3d_forward_out_cpu()");

  int64_t oT = output_size[0];
  int64_t oH = output_size[1];
  int64_t oW = output_size[2];

  if (self_.ndimension() == 5) {
    output.resize_({self.size(0), self.size(1), oT, oH, oW});
  } else {
    output.resize_({self.size(0), oT, oH, oW});
  }
  output.zero_();
  if (output.numel() != 0) {
    max_unpool3d_kernel(kCPU, output, self, indices);
  }

  return output;
}

Tensor max_unpooling3d_forward_cpu(
    const Tensor& self,
    const Tensor& indices,
    IntArrayRef output_size,
    IntArrayRef stride,
    IntArrayRef padding) {
  auto output = at::empty({0}, self.options());
  at::native::max_unpooling3d_forward_out_cpu(
      self, indices, output_size, stride, padding, output);
  return output;
}

DEFINE_DISPATCH(max_unpool2d_kernel);
DEFINE_DISPATCH(max_unpool3d_kernel);

} // namespace at::native
