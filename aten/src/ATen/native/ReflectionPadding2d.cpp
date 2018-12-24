#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <tuple>

namespace at {
namespace native {

namespace {

template <typename scalar_t>
static void reflection_pad2d_out_frame(
    scalar_t * input_p, scalar_t * output_p,
    int64_t nslices,
    int64_t iwidth, int64_t iheight,
    int64_t owidth, int64_t oheight,
    int64_t pad_l, int64_t pad_r,
    int64_t pad_t, int64_t pad_b) {
  auto i_start_x = std::max(0L, (long)-pad_l);
  auto i_start_y = std::max(0L, (long)-pad_t);
  auto o_start_x = std::max(0L, (long)pad_l);
  auto o_start_y = std::max(0L, (long)pad_t);

  int64_t k, ip_x, ip_y;
#pragma omp parallel for private(k, ip_x, ip_y)

  for (k = 0; k < nslices; k++) {
    for (int64_t i = 0; i < oheight; i++) {
      for (int64_t j = 0; j < owidth; j++) {
        if (j < pad_l) {
          ip_x = pad_l * 2 - j;
        } else if (j >= pad_l && j < iwidth + pad_l) {
          ip_x = j;
        } else {
          ip_x = (iwidth + pad_l - 1) * 2 - j;
        }
        ip_x = ip_x - o_start_x + i_start_x;

        if (i < pad_t) {
          ip_y = pad_t * 2 - i;
        } else if (i >= pad_t && i < iheight + pad_t) {
          ip_y = i;
        } else {
          ip_y = (iheight + pad_t - 1) * 2 - i;
        }
        ip_y = ip_y - o_start_y + i_start_y;

        scalar_t *dest_p = output_p + k*owidth*oheight + i * owidth + j;
        scalar_t *src_p = input_p + k*iwidth*iheight + ip_y * iwidth + ip_x;
        *dest_p = *src_p;
      }
    }
  }
}

void reflection_pad2d_out_template(
    Tensor &output, const Tensor &input_, IntList padding) {
  int dim_w = 2;
  int dim_h = 1;
  int dim_slices = 0;
  int64_t nbatch = 1;
  int64_t nslices;
  int64_t iheight;
  int64_t iwidth;
  int64_t oheight;
  int64_t owidth;

  AT_CHECK(input_.numel() > 0 &&
    (input_.ndimension() == 3 || input_.ndimension() == 4), "non-empty 3D or 4D "
    "(batch mode) tensor expected for input, but got: ", input_);

  if (input_.ndimension() == 4) {
    nbatch = input_.size(0);
    dim_w++;
    dim_h++;
    dim_slices++;
  }

  int64_t pad_l = padding[0];
  int64_t pad_r = padding[1];
  int64_t pad_t = padding[2];
  int64_t pad_b = padding[3];
  /* input sizes */
  nslices = input_.size(dim_slices);
  iheight = input_.size(dim_h);
  iwidth = input_.size(dim_w);

  AT_CHECK(pad_l < iwidth && pad_r < iwidth,
           "Argument #4: Padding size should be less than the corresponding "
           "input dimension, but got: padding (", pad_l, ", ", pad_r,
           ") at dimension ", dim_w, " of input ", input_.ndimension());

  AT_CHECK(pad_t < iheight && pad_b < iheight,
           "Argument #6: Padding size should be less than the corresponding "
           "input dimension, but got: padding (", pad_t, ", ", pad_b,
           ") at dimension ", dim_h, " of input ", input_.ndimension());

  /* output sizes */
  oheight = iheight + pad_t + pad_b;
  owidth  = iwidth + pad_l + pad_r;

  AT_CHECK(owidth >= 1 || oheight >= 1,
	     "input (H: ", iheight, ", W: ", iwidth, ")is too small. Calculated "
       "output H: ", oheight, " W: ", owidth);

  /* get contiguous input */
  Tensor input = input_.contiguous();

  if (input.ndimension() == 3) {
    /* resize output */
    output.resize_({nslices, oheight, owidth});

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "reflection_pad2d", [&] {
      reflection_pad2d_out_frame(
        input.data<scalar_t>(), output.data<scalar_t>(),
        nslices,
        iwidth, iheight, owidth, oheight,
        pad_l, pad_r, pad_t, pad_b);
    });
  } else {
    int64_t p;
    /* resize output */
    output.resize_({nbatch, nslices, oheight, owidth});

#pragma omp parallel for private(p)
    for (p = 0; p < nbatch; p++) {
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.type(), "reflection_pad2d", [&] {
          reflection_pad2d_out_frame(
            input.data<scalar_t>() + p * nslices * iwidth * iheight,
            output.data<scalar_t>() + p * nslices * owidth * oheight,
            nslices,
            iwidth, iheight, owidth, oheight,
            pad_l, pad_r, pad_t, pad_b);
        }
      );
    }
  }
}

template <typename scalar_t>
static void reflection_pad2d_backward_out_frame(
    scalar_t *ginput_p, scalar_t *goutput_p,
    int64_t nslices,
    int64_t iwidth, int64_t iheight,
    int64_t owidth, int64_t oheight,
    int64_t pad_l, int64_t pad_r,
    int64_t pad_t, int64_t pad_b) {
  auto i_start_x = std::max(0L, (long)-pad_l);
  auto i_start_y = std::max(0L, (long)-pad_t);
  auto o_start_x = std::max(0L, (long)pad_l);
  auto o_start_y = std::max(0L, (long)pad_t);

  int64_t k, ip_x, ip_y;
#pragma omp parallel for private(k, ip_x, ip_y)

  for (k = 0; k < nslices; k++) {
    for (int64_t i = 0; i < oheight; i++) {
      for (int64_t j = 0; j < owidth; j++) {
        if (j < pad_l) {
          ip_x = pad_l * 2 - j;
        } else if (j >= pad_l && j < iwidth + pad_l) {
          ip_x = j;
        } else {
          ip_x = (iwidth + pad_l - 1) * 2 - j;
        }
        ip_x = ip_x - o_start_x + i_start_x;

        if (i < pad_t) {
          ip_y = pad_t * 2 - i;
        } else if (i >= pad_t && i < iheight + pad_t) {
          ip_y = i;
        } else {
          ip_y = (iheight + pad_t - 1) * 2 - i;
        }
        ip_y = ip_y - o_start_y + i_start_y;

        scalar_t *src_p = goutput_p + k*owidth*oheight + i * owidth + j;
        scalar_t *dest_p = ginput_p + k*iwidth*iheight + ip_y * iwidth + ip_x;
        *dest_p += *src_p;
      }
    }
  }
}

void reflection_pad2d_backward_out_template(
    Tensor &grad_input, const Tensor &grad_output_,
    const Tensor &input, IntList padding) {
  int dim_w = 2;
  int dim_h = 1;
  int dim_slices = 0;
  int64_t nbatch = 1;
  int64_t nslices;
  int64_t iheight;
  int64_t iwidth;
  int64_t oheight;
  int64_t owidth;

  if (input.ndimension() == 4) {
    nbatch = input.size(0);
    dim_w++;
    dim_h++;
    dim_slices++;
  }

  int64_t pad_l = padding[0];
  int64_t pad_r = padding[1];
  int64_t pad_t = padding[2];
  int64_t pad_b = padding[3];

  /* sizes */
  nslices = input.size(dim_slices);
  iheight = input.size(dim_h);
  iwidth = input.size(dim_w);
  oheight = iheight + pad_t + pad_b;
  owidth  = iwidth + pad_l + pad_r;

  AT_CHECK(owidth == grad_output_.size(dim_w),
    "gradOutput width unexpected. Expected: ", owidth, ", Got: ",
    grad_output_.size(dim_w));

  AT_CHECK(oheight == grad_output_.size(dim_h),
    "gradOutput height unexpected. Expected: ", oheight, ", Got: ",
    grad_output_.size(dim_h));

  /* get contiguous gradOutput */
  Tensor grad_output = grad_output_.contiguous();

  /* backprop */
  if (input.ndimension() == 3) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.type(), "reflection_pad2d_backward", [&] {
        reflection_pad2d_backward_out_frame(
          grad_input.data<scalar_t>(),
          grad_output.data<scalar_t>(),
          nslices,
          iwidth, iheight, owidth, oheight,
          pad_l, pad_r, pad_t, pad_b);
      }
    );
  } else {
    int64_t p;
#pragma omp parallel for private(p)
    for (p = 0; p < nbatch; p++) {
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        grad_output.type(), "reflection_pad2d_backward", [&] {
          reflection_pad2d_backward_out_frame(
            grad_input.data<scalar_t>() + p * nslices * iheight * iwidth,
            grad_output.data<scalar_t>() + p * nslices * oheight * owidth,
            nslices,
            iwidth, iheight, owidth, oheight,
            pad_l, pad_r, pad_t, pad_b);
        }
      );
    }
  }
}

} // namespace

Tensor& reflection_pad2d_out_cpu(
    Tensor& output, const Tensor& input, IntList padding) {
  reflection_pad2d_out_template(output, input, padding);
  return output;
}

Tensor reflection_pad2d_cpu(Tensor const& input, IntList padding) {
  auto output = at::empty({0}, input.options());
  reflection_pad2d_out_template(output, input, padding);
  return output;
}

Tensor& reflection_pad2d_backward_out_cpu(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    IntList padding) {
  grad_input = at::zeros_like(input);
  reflection_pad2d_backward_out_template(
    grad_input, grad_output, input, padding);
  return grad_input;
}

Tensor reflection_pad2d_backward_cpu(
    const Tensor& grad_output,
    const Tensor& input,
    IntList padding) {
  auto grad_input = at::zeros_like(input);
  reflection_pad2d_backward_out_template(
    grad_input, grad_output, input, padding);
  return grad_input;
}

} // namespace native
} // namespace at
