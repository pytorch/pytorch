#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/cpu/MaxUnpoolKernel.h>
#include <c10/util/irange.h>

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

  TORCH_CHECK(
      indices_.scalar_type() == at::ScalarType::Long,
      "elements in indices should be type int64 but got: ", indices_.scalar_type());
  TORCH_CHECK(
      output_size.size() == 2,
      "There should be exactly two elements (height, width) in output_size, but got ", output_size.size(), " elements.");
  TORCH_CHECK(
      (self_.ndimension() == 3 || self_.ndimension() == 4),
      "Input to max_unpooling2d should be a 3d or 4d Tensor, but got a tensor with ", self_.ndimension(), " dimensions.");
  TORCH_CHECK(
      self_.sizes() == indices_.sizes(),
      "Expected shape of indices to be same as that of the input tensor (", self_.sizes(),
      ") but got indices tensor with shape: ", indices_.sizes());

  for (const auto i : c10::irange(1, self_.ndimension())) {
    TORCH_CHECK(self_.size(i) > 0, "max_unpooling2d_forward_out_cpu(): ",
                "Expected input to have non-zero size for non-batch dimensions, but got ",
                self_.sizes(), " with dimension ", i , " being empty.");
  }

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

static void max_unpooling3d_shape_check(
    const Tensor& input,
    const Tensor& gradOutput,
    const Tensor& indices,
    IntArrayRef output_size,
    IntArrayRef stride,
    IntArrayRef padding,
    const char *fn_name) {

  TORCH_CHECK(
      indices.scalar_type() == at::ScalarType::Long,
      "elements in indices should be type int64");
  TORCH_CHECK(
      (input.ndimension() == 4 || input.ndimension() == 5),
      "Input to max_unpooling3d should be a 4d or 5d Tensor, but got a tensor with ", input.ndimension(), " dimensions.");
  TORCH_CHECK(
      output_size.size() == 3,
      "There should be exactly three elements (depth, height, width) in output_size, but got ", output_size.size(), " elements.");
  TORCH_CHECK(
      stride.size() == 3,
      "There should be exactly three elements (depth, height, width) in stride, but got: ", stride.size(), " elements.");
  TORCH_CHECK(
      padding.size() == 3,
      "There should be exactly three elements (depth, height, width) in padding, but got: ", padding.size(), " elements.");
  TORCH_CHECK(
      input.sizes() == indices.sizes(),
      "Expected shape of indices to be same as that of the input tensor (", input.sizes(),
      ") but got indices tensor with shape: ", indices.sizes());

  for (const auto i : c10::irange(1, input.ndimension())) {
    TORCH_CHECK(input.size(i) > 0, fn_name,
                ": Expected input to have non-zero size for non-batch dimensions, but got ",
                input.sizes(), " with dimension ", i , " being empty.");
  }

  TORCH_CHECK(
      stride[0] > 0 && stride[1] > 0 && stride[2] > 0,
      "strides should be greater than zero, but got stride: ",
      stride);

  int64_t oT = output_size[0];
  int64_t oH = output_size[1];
  int64_t oW = output_size[2];

  int dimw = 3;
  int dimh = 2;
  int dimt = 1;
  int dimn = 0;

  if (input.ndimension() == 5) {
    dimw++;
    dimh++;
    dimt++;
    dimn++;
  }

  int nslices = input.size(dimn);

  if (gradOutput.defined()) {
    if (oT != gradOutput.size(dimt) || oH != gradOutput.size(dimh) ||
        oW != gradOutput.size(dimw)) {
      TORCH_CHECK(false,
          "Inconsistent gradOutput size. oT= ",
          oT,
          ", oH= ",
          oH,
          ", oW= ",
          oW,
          ". gradOutput: ",
          gradOutput.size(dimt),
          "x",
          gradOutput.size(dimh),
          "x",
          gradOutput.size(dimw));
    }
    TORCH_CHECK(
        gradOutput.ndimension() == input.ndimension() &&
            gradOutput.size(dimn) == nslices,
        "gradOutput and input Tensors should have same number of dimensions and also the same number of channels/slices");
  }
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
    self_, Tensor(), indices_, output_size, stride, padding, "max_unpooling3d_forward_out_cpu()");

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
