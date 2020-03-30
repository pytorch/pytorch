#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/NativeFunctions.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/Pool.h>
#include <tuple>


namespace at {
namespace native {

namespace {

template <typename scalar_t>
static void max_pool2d_with_indices_single_out_frame(
          scalar_t *input_p,
          scalar_t *output_p,
          int64_t *ind_p,
          int64_t nslices,
          int64_t iwidth,
          int64_t iheight,
          int64_t owidth,
          int64_t oheight,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH,
          int dilationW,
          int dilationH
          )
{
  at::parallel_for(0, nslices, 0, [&](int64_t start, int64_t end) {
    for (auto k = start; k < end; k++)
    {
      /* loop over output */
      int64_t i, j;
      scalar_t *ip = input_p   + k*iwidth*iheight;
      for(i = 0; i < oheight; i++)
      {
        for(j = 0; j < owidth; j++)
        {
          int64_t hstart = i * dH - padH;
          int64_t wstart = j * dW - padW;
          int64_t hend = std::min(hstart + (kH - 1) * dilationH + 1, iheight);
          int64_t wend = std::min(wstart + (kW - 1) * dilationW + 1, iwidth);
          while(hstart < 0)
            hstart += dilationH;
          while(wstart < 0)
            wstart += dilationW;

          /* local pointers */
          scalar_t *op = output_p  + k*owidth*oheight + i*owidth + j;
          int64_t *indp = ind_p   + k*owidth*oheight + i*owidth + j;

          /* compute local max: */
          int64_t maxindex = hstart*iwidth + wstart;
          scalar_t maxval = -std::numeric_limits<scalar_t>::infinity();
          for(int64_t y = hstart; y < hend; y += dilationH)
          {
            for(int64_t x = wstart; x < wend; x += dilationW)
            {
              int64_t tcntr = y*iwidth + x;
              scalar_t val = *(ip + tcntr);
              if ((val > maxval) || std::isnan(val))
              {
                maxval = val;
                maxindex = tcntr;
              }
            }
          }

          /* set output to local max */
          *op = maxval;

          /* store location of max */
          *indp = maxindex;
        }
      }
    }
  });
}

template <typename scalar_t>
static void max_pool2d_with_indices_out_frame(
          scalar_t *input_data,
          scalar_t *output_data,
          int64_t *indices_data,
          int64_t nbatch,
          int64_t nInputPlane,
          int64_t inputWidth,
          int64_t inputHeight,
          int64_t outputWidth,
          int64_t outputHeight,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH,
          int dilationW,
          int dilationH)
{
  at::parallel_for(0, nbatch, 0, [&](int64_t start, int64_t end) {
    for (auto p = start; p < end; p++) {
      max_pool2d_with_indices_single_out_frame(
        input_data+p*nInputPlane*inputWidth*inputHeight,
        output_data+p*nInputPlane*outputWidth*outputHeight,
        indices_data+p*nInputPlane*outputWidth*outputHeight,
        nInputPlane,
        inputWidth, inputHeight,
        outputWidth, outputHeight,
        kW, kH, dW, dH,
        padW, padH,
        dilationW, dilationH);
    }
  });
}

void max_pool2d_with_indices_out_cpu_template(
          Tensor& output,
          Tensor& indices,
          const Tensor& input_,
          IntArrayRef kernel_size,
          IntArrayRef stride,
          IntArrayRef padding,
          IntArrayRef dilation,
          bool ceil_mode)
{
  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
    "max_pool2d: kernel_size must either be a single int, or a tuple of two ints")
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);

  // NB: stride default is not expressible as an integer constant, so we accept
  // empty stride for this case
  TORCH_CHECK(stride.size() == 0 || stride.size() == 1 || stride.size() == 2,
    "max_pool2d: stride must either be omitted, a single int, or a tuple of two ints")
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

  TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
    "max_pool2d: padding must be either be a single int, or a tuple of two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  TORCH_CHECK(dilation.size() == 1 || dilation.size() == 2,
    "max_pool2d: dilation must be either a single int, or a tuple of two ints");
  const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationW = dilation.size() == 1 ? dilationH : safe_downcast<int, int64_t>(dilation[1]);

  TORCH_CHECK((input_.ndimension() == 3 || input_.ndimension() == 4),
    "non-empty 3D or 4D (batch mode) tensor expected for input");

  /* sizes */
  const int64_t nbatch = input_.ndimension() == 4 ? input_.size(-4) : 1;
  const int64_t nInputPlane = input_.size(-3);
  const int64_t inputHeight = input_.size(-2);
  const int64_t inputWidth = input_.size(-1);

  const int64_t outputHeight = pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, dilationH, ceil_mode);
  const int64_t outputWidth = pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, dilationW, ceil_mode);

  pool2d_shape_check(
    input_,
    kH, kW, dH, dW, padH, padW, dilationH, dilationW,
    nInputPlane,
    inputHeight, inputWidth,
    outputHeight, outputWidth);

  /* get contiguous input */
  Tensor input = input_.contiguous();

  /* resize output */
  if (input.ndimension() == 3)
  {
    output.resize_({nInputPlane, outputHeight, outputWidth});
    /* indices will contain the locations for each output point */
    indices.resize_({nInputPlane, outputHeight, outputWidth});

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(),
      "max_pool2d_with_indices_cpu",
      [&] {
        /* get raw pointers */
        scalar_t *input_data = input.data_ptr<scalar_t>();
        scalar_t *output_data = output.data_ptr<scalar_t>();
        int64_t *indices_data = indices.data_ptr<int64_t>();

        max_pool2d_with_indices_single_out_frame(
          input_data, output_data,
          indices_data,
          nInputPlane,
          inputWidth, inputHeight,
          outputWidth, outputHeight,
          kW, kH, dW, dH,
          padW, padH,
          dilationW, dilationH);
      }
    );
  }
  else
  {
    output.resize_({nbatch, nInputPlane, outputHeight, outputWidth});
    /* indices will contain the locations for each output point */
    indices.resize_({nbatch, nInputPlane, outputHeight, outputWidth});

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(),
      "max_pool2d_with_indices_cpu",
      [&] {
        scalar_t *input_data = input.data_ptr<scalar_t>();
        scalar_t *output_data = output.data_ptr<scalar_t>();
        int64_t *indices_data = indices.data_ptr<int64_t>();

        max_pool2d_with_indices_out_frame(
          input_data,
          output_data,
          indices_data,
          nbatch,
          nInputPlane,
          inputWidth, inputHeight,
          outputWidth, outputHeight,
          kW, kH, dW, dH,
          padW, padH,
          dilationW, dilationH); }
    );
  }
}

template <typename scalar_t>
static void max_pool2d_with_indices_backward_single_out_frame(
          scalar_t *gradInput_p,
          scalar_t *gradOutput_p,
          int64_t *ind_p,
          int64_t nInputPlane,
          int64_t inputWidth,
          int64_t inputHeight,
          int64_t outputWidth,
          int64_t outputHeight,
          int dW,
          int dH)
{
  at::parallel_for(0, nInputPlane, 0, [&](int64_t start, int64_t end) {
    for (auto k = start; k < end; k++)
    {
      scalar_t *gradInput_p_k = gradInput_p + k*inputWidth*inputHeight;
      scalar_t *gradOutput_p_k = gradOutput_p + k*outputWidth*outputHeight;
      int64_t *ind_p_k = ind_p + k*outputWidth*outputHeight;

      /* calculate max points */
      int64_t i, j;
      for(i = 0; i < outputHeight; i++)
      {
        for(j = 0; j < outputWidth; j++)
        {
          /* retrieve position of max */
          int64_t maxp = ind_p_k[i*outputWidth + j];
          if (maxp != -1) {
            /* update gradient */
            gradInput_p_k[maxp] += gradOutput_p_k[i*outputWidth + j];
          }
        }
      }
    }
  });
}

template <typename scalar_t>
static void max_pool2d_with_indices_backward_out_frame(
          scalar_t *gradInput_data,
          scalar_t *gradOutput_data,
          int64_t *indices_data,
          int64_t nbatch,
          int64_t nInputPlane,
          int64_t inputWidth,
          int64_t inputHeight,
          int64_t outputWidth,
          int64_t outputHeight,
          int dW,
          int dH)
{
  at::parallel_for(0, nbatch, 0, [&](int64_t start, int64_t end) {
    for (auto p = start; p < end; p++) {
      max_pool2d_with_indices_backward_single_out_frame<scalar_t>(
        gradInput_data+p*nInputPlane*inputWidth*inputHeight,
        gradOutput_data+p*nInputPlane*outputWidth*outputHeight,
        indices_data+p*nInputPlane*outputWidth*outputHeight,
        nInputPlane,
        inputWidth, inputHeight,
        outputWidth, outputHeight,
        dW, dH);
    }
  });
}

Tensor& max_pool2d_with_indices_backward_out_cpu_template(
          Tensor& gradInput,
          const Tensor& gradOutput_,
          const Tensor& input,
          const Tensor& indices,
          IntArrayRef kernel_size,
          IntArrayRef stride,
          IntArrayRef padding,
          IntArrayRef dilation,
          bool ceil_mode)
{
  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
    "max_pool2d: kernel_size must either be a single int, or a tuple of two ints")
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);

  // NB: stride default is not expressible as an integer constant, so we accept
  // empty stride for this case
  TORCH_CHECK(stride.size() == 0 || stride.size() == 1 || stride.size() == 2,
    "max_pool2d: stride must either be omitted, a single int, or a tuple of two ints")
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

  TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
    "max_pool2d: padding must be either be a single int, or a tuple of two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  TORCH_CHECK(dilation.size() == 1 || dilation.size() == 2,
    "max_pool2d: dilation must be either a single int, or a tuple of two ints");
  const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationW = dilation.size() == 1 ? dilationH : safe_downcast<int, int64_t>(dilation[1]);

  TORCH_CHECK((input.ndimension() == 3 || input.ndimension() == 4),
    "non-empty 3D or 4D (batch mode) tensor expected for input");

  /* get contiguous gradOutput */
  const Tensor gradOutput = gradOutput_.contiguous();

  /* resize */
  gradInput.resize_as_(input);
  gradInput.zero_();

  /* sizes */
  const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
  const int64_t nInputPlane = input.size(-3);
  const int64_t inputHeight = input.size(-2);
  const int64_t inputWidth = input.size(-1);
  const int64_t outputHeight = gradOutput.size(-2);
  const int64_t outputWidth = gradOutput.size(-1);

  /* XXX preserve the existing shape check behavior */
  const int64_t outputHeight_for_shape_check = pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, dilationH, ceil_mode);
  const int64_t outputWidth_for_shape_check = pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, dilationW, ceil_mode);

  max_pool2d_backward_shape_check(
    input,
    gradOutput_,
    indices,
    nbatch,
    kH, kW, dH, dW, padH, padW, dilationH, dilationW,
    nInputPlane,
    inputHeight, inputWidth,
    outputHeight_for_shape_check, outputWidth_for_shape_check);

  /* backprop */
  if (input.ndimension() == 3)
  {
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(),
      "max_pool2d_with_indices_backward",
      [&] {
        /* get raw pointers */
        scalar_t *gradInput_data = gradInput.data_ptr<scalar_t>();
        scalar_t *gradOutput_data = gradOutput.data_ptr<scalar_t>();
        int64_t *indices_data = indices.data_ptr<int64_t>();

        max_pool2d_with_indices_backward_single_out_frame(
          gradInput_data, gradOutput_data,
          indices_data,
          nInputPlane,
          inputWidth, inputHeight,
          outputWidth, outputHeight,
          dW, dH);
      }
    );
  }
  else
  {
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(),
      "max_pool2d_with_indices_backward",
      [&] {
        /* get raw pointers */
        scalar_t *gradInput_data = gradInput.data_ptr<scalar_t>();
        scalar_t *gradOutput_data = gradOutput.data_ptr<scalar_t>();
        int64_t *indices_data = indices.data_ptr<int64_t>();

        max_pool2d_with_indices_backward_out_frame<scalar_t>(
          gradInput_data, gradOutput_data,
          indices_data,
          nbatch,
          nInputPlane,
          inputWidth, inputHeight,
          outputWidth, outputHeight,
          dW, dH);
      }
    );
  }

  return gradInput;
}

} // namespace

std::tuple<Tensor&, Tensor&> max_pool2d_with_indices_out_cpu(
  Tensor& output,
  Tensor& indices,
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  IntArrayRef dilation,
  bool ceil_mode)
{
  max_pool2d_with_indices_out_cpu_template(
    output,
    indices,
    input,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode);
  return std::tuple<Tensor&, Tensor&>(output, indices);
}

std::tuple<Tensor, Tensor> max_pool2d_with_indices_cpu(
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  IntArrayRef dilation,
  bool ceil_mode)
{
  NoNamesGuard guard;

  Tensor output = at::empty({0}, input.options());
  Tensor indices = at::empty({0}, input.options().dtype(kLong));
  max_pool2d_with_indices_out_cpu_template(
    output,
    indices,
    input,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode);

  guard.reset();
  namedinference::propagate_names(output, input);
  namedinference::propagate_names(indices, input);

  return std::tuple<Tensor, Tensor>(output, indices);
}

Tensor& max_pool2d_with_indices_backward_out_cpu(
  Tensor& gradInput,
  const Tensor& gradOutput_,
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  IntArrayRef dilation,
  bool ceil_mode,
  const Tensor& indices)
{
  max_pool2d_with_indices_backward_out_cpu_template(
    gradInput,
    gradOutput_,
    input,
    indices,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode);
  return gradInput;
}

Tensor max_pool2d_with_indices_backward_cpu(
  const Tensor& gradOutput_,
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  IntArrayRef dilation,
  bool ceil_mode,
  const Tensor& indices)
{
  auto gradInput = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  max_pool2d_with_indices_backward_out_cpu_template(
    gradInput,
    gradOutput_,
    input,
    indices,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode);
  return gradInput;
}

} // at::native
} // at
