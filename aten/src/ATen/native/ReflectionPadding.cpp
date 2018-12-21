#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <tuple>

namespace at {
namespace native {

namespace {

template <typename scalar_t>
static void reflection_pad1d_out_frame(
    scalar_t *input_p, scalar_t *output_p,
    int64_t nslices,
    int64_t iwidth, int64_t owidth,
    int64_t pad_l, int64_t pad_r) {
  int64_t i_start_x = std::max(0L, (long)-pad_l);
  int64_t o_start_x = std::max(0L, (long)pad_l);

  int64_t k, ip_x;
#pragma omp parallel for private(k, ip_x)

  for (k = 0; k < nslices; k++) {
    for (int64_t j = 0; j < owidth; j++) {
      if (j < pad_l) {
        ip_x = pad_l * 2 - j;
      } else if (j >= pad_l && j < iwidth + pad_l) {
        ip_x = j;
      } else {
        ip_x = (iwidth + pad_l - 1) * 2 - j;
      }
      ip_x = ip_x - o_start_x + i_start_x;

      scalar_t *dest_p = output_p + k*owidth + j;
      scalar_t *src_p = input_p + k*iwidth + ip_x;
      *dest_p = *src_p;
    }
  }
}

void reflection_pad1d_out_template(
    Tensor& output, Tensor const& input_, IntList padding) {
  int64_t dimw = 1;
  int64_t dimslices = 0;
  int64_t nbatch = 1;
  int64_t nslices;
  int64_t iwidth;
  int64_t owidth;

  for (int64_t i = 0; i < input_.ndimension(); ++i) {
    AT_CHECK(input_.size(i) > 0,
      "reflection_pad1d(): expected input to have non-empty temporal "
      "dimensions, but input has sizes ", input_.sizes(), "with dimension ", i,
      " being empty");
  }

  AT_CHECK(input_.ndimension() == 2 || input_.ndimension() == 3, "non-empty 2D "
    "or 3D (batch mode) tensor expected for input, but got: ", input_);

  if (input_.ndimension() == 3) {
    nbatch = input_.size(0);
    dimw++;
    dimslices++;
  }

  /* input size */
  nslices = input_.size(dimslices);
  iwidth = input_.size(dimw);

  auto pad_l = padding[0];
  auto pad_r = padding[1];

  AT_CHECK(pad_l < iwidth && pad_r < iwidth, "Argument #4: Padding size should"
    " be less than the corresponding input dimension, but got: padding (",
    pad_l, ", ", pad_r, ") at dimension ", dimw, " of input ", input_.sizes());

  /* output size */
  owidth  = iwidth + pad_l + pad_r;

  AT_CHECK(owidth >= 1 , 2,
    "input (W: ", iwidth, ")is too small. Calculated output W: ", owidth);

  /* get contiguous input */
  Tensor input = input_.contiguous();

  /* resize output */
  if (input.ndimension() == 2) {
    output.resize_({nslices, owidth});

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.type(), "reflection_pad1d", [&] {
        auto input_data = input.data<scalar_t>();
        auto output_data = output.data<scalar_t>();
        reflection_pad1d_out_frame<scalar_t>(input_data, output_data,
                                             nslices,
                                             iwidth, owidth,
                                             pad_l, pad_r);
        }
    );
  } else {
    output.resize_({nbatch, nslices, owidth});

    int64_t p;
#pragma omp parallel for private(p)
    for (p = 0; p < nbatch; p++) {
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.type(), "reflection_pad1d", [&] {
          auto input_data = input.data<scalar_t>();
          auto output_data = output.data<scalar_t>();

          reflection_pad1d_out_frame<scalar_t>(
            input_data+p*nslices*iwidth,
            output_data+p*nslices*owidth,
            nslices,
            iwidth,
            owidth,
            pad_l, pad_r);
          }
      );
    }
  }
}

template <typename scalar_t>
static void reflection_pad1d_backward_out_frame(
    scalar_t *ginput_p, scalar_t *goutput_p,
    int64_t nslices,
    int64_t iwidth, int64_t owidth,
    int64_t pad_l, int64_t pad_r) {
  int64_t i_start_x = std::max(0L, (long)-pad_l);
  int64_t o_start_x = std::max(0L, (long)pad_l);

  int64_t k, ip_x;
#pragma omp parallel for private(k, ip_x)

  for (k = 0; k < nslices; k++) {
    for (int64_t j = 0; j < owidth; j++) {
      if (j < pad_l) {
        ip_x = pad_l * 2 - j;
      } else if (j >= pad_l && j < iwidth + pad_l) {
        ip_x = j;
      } else {
        ip_x = (iwidth + pad_l - 1) * 2 - j;
      }
      ip_x = ip_x - o_start_x + i_start_x;

      scalar_t *src_p = goutput_p + k*owidth + j;
      scalar_t *dest_p = ginput_p + k*iwidth + ip_x;
      *dest_p += *src_p;
    }
  }
}

void reflection_pad1d_backward_out_template(
    Tensor const& input, Tensor const& grad_output_, Tensor& grad_input,
    IntList padding) {
  int64_t dimw = 1;
  int64_t dimslices = 0;
  int64_t nbatch = 1;
  int64_t nslices;
  int64_t iwidth;
  int64_t owidth;

  if (input.ndimension() == 3) {
    nbatch = input.size(0);
    dimw++;
    dimslices++;
  }

  /* sizes */
  auto pad_l = padding[0];
  auto pad_r = padding[1];
  nslices = input.size(dimslices);
  iwidth = input.size(dimw);
  owidth  = iwidth + pad_l + pad_r;

  AT_CHECK(owidth == grad_output_.size(dimw), "gradOutput width unexpected. "
    "Expected: ", owidth, ", Got: ", grad_output_.size(dimw));

  /* get contiguous gradOutput */
  Tensor grad_output = grad_output_.contiguous();

  /* backprop */
  if (input.ndimension() == 2) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.type(), "reflection_pad1d_backward", [&] {
        reflection_pad1d_backward_out_frame(
          grad_input.data<scalar_t>(),
          grad_output.data<scalar_t>(),
          nslices,
          iwidth,
          owidth,
          pad_l, pad_r);
        }
    );
  } else {
    int64_t p;
#pragma omp parallel for private(p)
    for (p = 0; p < nbatch; p++) {
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.type(), "reflection_pad1d_backward", [&] {
          reflection_pad1d_backward_out_frame(
            grad_input.data<scalar_t>() + p * nslices * iwidth,
            grad_output.data<scalar_t>() + p * nslices * owidth,
            nslices,
            iwidth,
            owidth,
            pad_l, pad_r);
          }
      );
    }
  }
}
} // namespace

Tensor& reflection_pad1d_out_cpu(
    Tensor& output, const Tensor& input, IntList padding) {
  reflection_pad1d_out_template(output, input, padding);
  return output;
}

Tensor reflection_pad1d_cpu(Tensor const& input, IntList padding) {
  auto output = at::empty({0}, input.options());
  reflection_pad1d_out_template(output, input, padding);
  return output;
}

Tensor& reflection_pad1d_backward_out_cpu(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    IntList padding) {
  grad_input = at::zeros_like(input);
  reflection_pad1d_backward_out_template(input, grad_output, grad_input, padding);
  return grad_input;
}

Tensor reflection_pad1d_backward_cpu(
    const Tensor& grad_output,
    const Tensor& input,
    IntList padding) {
  auto grad_input = at::zeros_like(input);
  reflection_pad1d_backward_out_template(
    input, grad_output, grad_input, padding);
  return grad_input;
}

} // namespace native
} // namespace at
