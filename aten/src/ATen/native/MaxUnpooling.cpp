#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/cpu/MaxUnpoolKernel.h>

namespace at {
namespace native {

Tensor& max_unpooling2d_forward_out_cpu(
    Tensor& output,
    const Tensor& self_,
    const Tensor& indices_,
    IntArrayRef output_size) {
  auto oheight = output_size[0];
  auto owidth = output_size[1];
  TORCH_CHECK(
      indices_.scalar_type() == at::ScalarType::Long,
      "elements in indices should be type int64");
  TORCH_CHECK(
      output_size.size() == 2,
      "There should be exactly two elements (height, width) in output_size");
  TORCH_CHECK(
      (self_.ndimension() == 3 || self_.ndimension() == 4),
      "Input to max_unpooling2d should be a 3d or 4d Tensor");
  TORCH_CHECK(
      self_.sizes() == indices_.sizes(),
      "Shape of indices should match shape of input");

  TORCH_CHECK(self_.numel() > 0, "Input must be non-empty");

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

  max_unpool2d_kernel(kCPU, output, self, indices);
  return output;
};

Tensor max_unpooling2d_forward_cpu(
    const Tensor& self,
    const Tensor& indices,
    IntArrayRef output_size) {
  auto output = at::empty({0}, self.options());
  max_unpooling2d_forward_out_cpu(output, self, indices, output_size);
  return output;
}

static void max_unpooling3d_shape_check(
    const Tensor& input,
    const Tensor& gradOutput,
    const Tensor& indices,
    IntArrayRef output_size,
    IntArrayRef stride,
    IntArrayRef padding) {
  int64_t oT = output_size[0];
  int64_t oH = output_size[1];
  int64_t oW = output_size[2];
  TORCH_CHECK(
      indices.scalar_type() == at::ScalarType::Long,
      "elements in indices should be type int64");
  TORCH_CHECK(
      (input.ndimension() == 4 || input.ndimension() == 5),
      "Input to max_unpooling3d should be a 4d or 5d Tensor",
      input.sizes());
  TORCH_CHECK(
      output_size.size() == 3,
      "There should be exactly three elements (depth, height, width) in output_size");
  TORCH_CHECK(
      stride.size() == 3,
      "There should be exactly three elements (depth, height, width) in stride");
  TORCH_CHECK(
      padding.size() == 3,
      "There should be exactly three elements (depth, height, width) in padding");
  TORCH_CHECK(
      input.sizes() == indices.sizes(),
      "Shape of indices should match shape of input");

  TORCH_CHECK(input.numel() > 0, "Input must be non-empty");

  TORCH_CHECK(
      stride[0] > 0 && stride[1] > 0 && stride[2] > 0,
      "strides should be greater than zero, but got stride: ",
      stride);

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
      AT_ERROR(
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

Tensor& max_unpooling3d_forward_out_cpu(
    Tensor& output,
    const Tensor& self_,
    const Tensor& indices_,
    IntArrayRef output_size,
    IntArrayRef stride,
    IntArrayRef padding) {
  int64_t oT = output_size[0];
  int64_t oH = output_size[1];
  int64_t oW = output_size[2];

  auto self = self_.contiguous();
  auto indices = indices_.contiguous();

  max_unpooling3d_shape_check(
      self_, Tensor(), indices_, output_size, stride, padding);

  if (self_.ndimension() == 5) {
    output.resize_({self.size(0), self.size(1), oT, oH, oW});
  } else {
    output.resize_({self.size(0), oT, oH, oW});
  }
  output.zero_();

  max_unpool3d_kernel(kCPU, output, self, indices);
  return output;
}

Tensor max_unpooling3d_forward_cpu(
    const Tensor& self,
    const Tensor& indices,
    IntArrayRef output_size,
    IntArrayRef stride,
    IntArrayRef padding) {
  auto output = at::empty({0}, self.options());
  max_unpooling3d_forward_out_cpu(
      output, self, indices, output_size, stride, padding);
  return output;
}

Tensor& max_unpooling2d_backward_out_cpu(
    Tensor& grad_input,
    const Tensor& grad_output_,
    const Tensor& self,
    const Tensor& indices_,
    IntArrayRef output_size) {
  int64_t oheight = output_size[0];
  int64_t owidth = output_size[1];
  int64_t ndim = self.ndimension();
  int64_t dimh = ndim == 3 ? 1 : 2;
  int64_t dimw = ndim == 3 ? 2 : 3;

  TORCH_CHECK(
      indices_.scalar_type() == at::ScalarType::Long,
      "elements in indices should be type int64");
  TORCH_CHECK(
      self.sizes() == indices_.sizes(), "Input shape must match indices shape");
  TORCH_CHECK(output_size.size() == 2, "Output size must be 2");

  auto memory_format = self.suggest_memory_format();
  auto grad_output = grad_output_.contiguous(memory_format);
  auto indices = indices_.contiguous(memory_format);

  grad_input.resize_(self.sizes(), memory_format);
  grad_input.zero_();

  if (owidth != grad_output.size(dimw) || oheight != grad_output.size(dimh)) {
    AT_ERROR(
        "Inconsistent gradOutput size. output height = ",
        oheight,
        ", output width = ",
        owidth,
        ", gradOutput: ",
        grad_output.size(dimh),
        "x",
        grad_output.size(dimw));
  }

  max_unpool2d_backward_kernel(kCPU, grad_input, grad_output, indices);
  return grad_input;
}

Tensor max_unpooling2d_backward_cpu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& indices,
    IntArrayRef output_size) {
  auto grad_input = at::empty({0}, self.options());
  max_unpooling2d_backward_out_cpu(
      grad_input, grad_output, self, indices, output_size);
  return grad_input;
}

Tensor& max_unpooling3d_backward_out_cpu(
    Tensor& grad_input,
    const Tensor& grad_output_,
    const Tensor& self,
    const Tensor& indices_,
    IntArrayRef output_size,
    IntArrayRef stride,
    IntArrayRef padding) {
  int64_t oT = output_size[0];
  int64_t oH = output_size[1];
  int64_t oW = output_size[2];
  int64_t ndim = self.ndimension();
  int64_t dimt = ndim == 4 ? 1 : 2;
  int64_t dimh = ndim == 4 ? 2 : 3;
  int64_t dimw = ndim == 4 ? 3 : 4;

  max_unpooling3d_shape_check(
      self, grad_output_, indices_, output_size, stride, padding);

  /* get contiguous gradOutput */
  auto grad_output = grad_output_.contiguous();
  auto indices = indices_.contiguous();

  /* resize */
  grad_input.resize_as_(self);
  grad_input.zero_();

  if (oW != grad_output.size(dimw) || oH != grad_output.size(dimh) || oT != grad_output.size(dimt)) {
    AT_ERROR(
        "Inconsistent gradOutput size. output depth = ",
        oT,
        ", output height = ",
        oH,
        ", output width = ",
        oW,
        ", gradOutput: ",
        grad_output.size(dimt),
        "x",
        grad_output.size(dimh),
        "x",
        grad_output.size(dimw));
  }

  max_unpool3d_backward_kernel(kCPU, grad_input, grad_output, indices);
  return grad_input;
}

Tensor max_unpooling3d_backward_cpu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& indices,
    IntArrayRef output_size,
    IntArrayRef stride,
    IntArrayRef padding) {
  auto grad_input = at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  max_unpooling3d_backward_out_cpu(
      grad_input, grad_output, self, indices, output_size, stride, padding);
  return grad_input;
}

DEFINE_DISPATCH(max_unpool2d_kernel);
DEFINE_DISPATCH(max_unpool2d_backward_kernel);
DEFINE_DISPATCH(max_unpool3d_kernel);
DEFINE_DISPATCH(max_unpool3d_backward_kernel);

} // namespace native
} // namespace at
