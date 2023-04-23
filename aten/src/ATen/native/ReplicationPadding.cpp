#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorMeta.h>
#include <ATen/native/cpu/PaddingKernel.h>
#include <c10/util/irange.h>
#include <algorithm>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/replication_pad1d_backward_native.h>
#include <ATen/ops/replication_pad1d_native.h>
#include <ATen/ops/replication_pad2d_backward_native.h>
#include <ATen/ops/replication_pad2d_native.h>
#include <ATen/ops/replication_pad3d_backward_native.h>
#include <ATen/ops/replication_pad3d_native.h>
#include <ATen/ops/empty.h>
#endif

namespace at {

namespace meta {

TORCH_META_FUNC(replication_pad1d) (
  const Tensor& input, IntArrayRef padding  // no out argument!
) {
  int64_t dimw = 1;
  int64_t dimslices = 0;
  int64_t nbatch = 1;

  TORCH_CHECK(padding.size() == 2, "padding size is expected to be 2");
  int64_t pad_l = padding[0];
  int64_t pad_r = padding[1];

  // allow empty batch size but not other dimensions.
  at::native::padding::check_valid_input<1>(input);

  int ndim = input.dim();
  if (ndim == 3) {
    nbatch = input.size(0);
    dimw++;
    dimslices++;
  }

  /* sizes */
  int64_t nslices = input.size(dimslices);
  int64_t iwidth = input.size(dimw);
  int64_t owidth = iwidth + pad_l + pad_r;

  TORCH_CHECK(owidth >= 1,
      "input (W: ", iwidth, ") is too small."
      " Calculated output W: ", owidth);

  if (ndim == 2) {
    set_output_raw_strided(0, {nslices, owidth}, {}, input.options());
  } else {
    set_output_raw_strided(0, {nbatch, nslices, owidth}, {}, input.options());
  }
}

TORCH_META_FUNC(replication_pad1d_backward) (
  const Tensor& gradOutput,
  const Tensor& input,
  IntArrayRef padding
) {
  TORCH_CHECK(padding.size() == 2, "padding size is expected to be 2");
  at::native::padding::check_valid_input_backward<1>(
      gradOutput, input, padding);

  set_output_raw_strided(0, input.sizes(), {}, input.options());
}

TORCH_META_FUNC(replication_pad2d) (
  const Tensor& input, IntArrayRef padding
) {
  int64_t dimw = 2;
  int64_t dimh = 1;
  int64_t dimslices = 0;
  int64_t nbatch = 1;

  TORCH_CHECK(padding.size() == 4, "padding size is expected to be 4");
  int64_t pad_l = padding[0];
  int64_t pad_r = padding[1];
  int64_t pad_t = padding[2];
  int64_t pad_b = padding[3];

  // allow empty batch size but not other dimensions.
  at::native::padding::check_valid_input<2>(input);

  int ndim = input.dim();
  if (ndim == 4) {
    nbatch = input.size(0);
    dimw++;
    dimh++;
    dimslices++;
  }

  /* sizes */
  int64_t nslices = input.size(dimslices);
  int64_t iheight = input.size(dimh);
  int64_t iwidth = input.size(dimw);
  int64_t oheight = iheight + pad_t + pad_b;
  int64_t owidth  = iwidth + pad_l + pad_r;

  TORCH_CHECK(owidth >= 1 || oheight >= 1,
      "input (H: ", iheight, ", W: ", iwidth, " ) is too small."
      " Calculated output H: ", oheight, " W: ", owidth);

  if (ndim == 3) {
    set_output_raw_strided(0, {nslices, oheight, owidth}, {}, input.options());
  } else {
    const auto memory_format = input.suggest_memory_format();
    set_output_raw_strided(0, {nbatch, nslices, oheight, owidth}, {},
        input.options().memory_format(memory_format));
  }
}

TORCH_META_FUNC(replication_pad3d) (
  const Tensor& input, IntArrayRef padding
) {
  int64_t dimw = 3;
  int64_t dimh = 2;
  int64_t dimd = 1;
  int64_t dimslices = 0;
  int64_t nbatch = 1;

  TORCH_CHECK(padding.size() == 6, "padding size is expected to be 6");
  int64_t pleft = padding[0];
  int64_t pright = padding[1];
  int64_t ptop = padding[2];
  int64_t pbottom = padding[3];
  int64_t pfront = padding[4];
  int64_t pback = padding[5];

  // allow empty batch size but not other dimensions.
  at::native::padding::check_valid_input<3>(input);

  int ndim = input.dim();
  if (ndim == 5) {
    nbatch = input.size(0);
    dimw++;
    dimh++;
    dimd++;
    dimslices++;
  }

  /* sizes */
  int64_t nslices = input.size(dimslices);
  int64_t idepth = input.size(dimd);
  int64_t iheight = input.size(dimh);
  int64_t iwidth = input.size(dimw);
  int64_t odepth = idepth + pfront + pback;
  int64_t oheight = iheight + ptop + pbottom;
  int64_t owidth  = iwidth + pleft + pright;

  TORCH_CHECK(owidth >= 1 || oheight >= 1 || odepth >= 1,
      "input (D: ", idepth, " H: ", iheight, ", W: ", iwidth,
      ") is too small."
      " Calculated output D: ", odepth, " H: ", oheight, " W: ", owidth);

  /* resize output */
  const auto memory_format = input.suggest_memory_format();
  const auto options = input.options().memory_format(memory_format);
  if (ndim == 4) {
    set_output_raw_strided(0, {nslices, odepth, oheight, owidth}, {}, options);
  } else {
    set_output_raw_strided(0, {nbatch, nslices, odepth, oheight, owidth}, {}, options);
  }
}

} // namespace meta

namespace native {

namespace {

void replication_pad2d_backward_out_cpu_template(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef padding)
{
  TORCH_CHECK(padding.size() == 4, "padding size is expected to be 4");
  at::native::padding::check_valid_input_backward<2>(
      gradOutput, input, padding);

  /* resize */
  gradInput.resize_(input.sizes(), input.suggest_memory_format());
  if (gradInput.numel() == 0) {
    return;
  }

  gradInput.zero_();
  replication_pad2d_backward_kernel(kCPU, gradInput, gradOutput, padding);
}

void replication_pad3d_backward_out_cpu_template(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef padding)
{
  TORCH_CHECK(padding.size() == 6, "padding size is expected to be 6");
  at::native::padding::check_valid_input_backward<3>(
      gradOutput, input, padding);

  /* resize */
  gradInput.resize_(input.sizes(), input.suggest_memory_format());
  if (gradInput.numel() == 0) {
    return;
  }

  gradInput.zero_();
  replication_pad3d_backward_kernel(kCPU, gradInput, gradOutput, padding);
}

} // anonymous namespace

TORCH_IMPL_FUNC(replication_pad1d_out_cpu) (
  const Tensor& input, IntArrayRef padding, const Tensor& output
) {
  replication_pad1d_kernel(kCPU, output, input, padding);
}

TORCH_IMPL_FUNC(replication_pad1d_backward_out_cpu) (
  const Tensor& gradOutput, const Tensor& input, IntArrayRef padding, const Tensor& gradInput
) {
  if (gradInput.numel() == 0) {
    return;
  }
  gradInput.zero_();

  replication_pad1d_backward_kernel(kCPU, gradInput, gradOutput, padding);
}

TORCH_IMPL_FUNC(replication_pad2d_out_cpu) (
  const Tensor& input, IntArrayRef padding, const Tensor& output
) {
  replication_pad2d_kernel(kCPU, output, input, padding);
}

Tensor& replication_pad2d_backward_out_cpu(
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef padding,
    Tensor& gradInput)
{
  replication_pad2d_backward_out_cpu_template(
      gradInput, gradOutput, input, padding);
  return gradInput;
}

Tensor replication_pad2d_backward_cpu(
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef padding)
{
  auto gradInput = at::empty({0}, input.options());
  replication_pad2d_backward_out_cpu_template(
      gradInput, gradOutput, input, padding);
  return gradInput;
}

TORCH_IMPL_FUNC(replication_pad3d_out_cpu) (
  const Tensor& input, IntArrayRef padding, const Tensor& output
) {
  replication_pad3d_kernel(kCPU, output, input, padding);
}

Tensor& replication_pad3d_backward_out_cpu(
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef padding,
    Tensor& gradInput)
{
  replication_pad3d_backward_out_cpu_template(
      gradInput, gradOutput, input, padding);
  return gradInput;
}

Tensor replication_pad3d_backward_cpu(
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef padding)
{
  auto gradInput = at::empty({0}, input.options());
  replication_pad3d_backward_out_cpu_template(
      gradInput, gradOutput, input, padding);
  return gradInput;
}

DEFINE_DISPATCH(replication_pad1d_kernel);
DEFINE_DISPATCH(replication_pad1d_backward_kernel);
DEFINE_DISPATCH(replication_pad2d_kernel);
DEFINE_DISPATCH(replication_pad2d_backward_kernel);
DEFINE_DISPATCH(replication_pad3d_kernel);
DEFINE_DISPATCH(replication_pad3d_backward_kernel);

} // at::native
} // at
