#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

namespace at {
namespace native {

namespace {

template <typename scalar_t>
static void reflection_pad1d_out_frame(
    scalar_t *input_p, scalar_t *output_p,
    int64_t nplane,
    int64_t input_w, int64_t output_w,
    int64_t pad_l) {
  int64_t i_start_x = std::max(int64_t(0), -pad_l);
  int64_t o_start_x = std::max(int64_t(0), pad_l);

  int64_t k, ip_x;
#pragma omp parallel for private(k, ip_x)

  for (k = 0; k < nplane; k++) {
    for (int64_t j = 0; j < output_w; j++) {
      if (j < pad_l) {
        ip_x = pad_l * 2 - j;
      } else if (j >= pad_l && j < input_w + pad_l) {
        ip_x = j;
      } else {
        ip_x = (input_w + pad_l - 1) * 2 - j;
      }
      ip_x = ip_x - o_start_x + i_start_x;

      scalar_t *dest_p = output_p + k*output_w + j;
      scalar_t *src_p = input_p + k*input_w + ip_x;
      *dest_p = *src_p;
    }
  }
}

template <typename scalar_t>
inline void reflection_pad1d_out_loop(
    scalar_t *input_p, scalar_t *output_p,
    int64_t nbatch, int64_t nplane,
    int64_t input_w, int64_t output_w,
    int64_t pad_l) {
  int64_t p;
#pragma omp parallel for private(p)
  for (p = 0; p < nbatch; p++) {
    reflection_pad1d_out_frame<scalar_t>(
      input_p + p * nplane * input_w,
      output_p + p * nplane * output_w,
      nplane,
      input_w, output_w,
      pad_l);
  }
}

void reflection_pad1d_out_template(
    Tensor& output, const Tensor& input_, IntList padding) {
  int64_t dim_plane = 0;
  int64_t dim_w = 1;
  int64_t nbatch = 1;

  AT_CHECK(input_.numel() > 0 &&
    (input_.ndimension() == 2 || input_.ndimension() == 3), "non-empty 2D "
    "or 3D (batch mode) tensor expected for input, but got: ", input_);

  if (input_.ndimension() == 3) {
    nbatch = input_.size(0);
    dim_w++;
    dim_plane++;
  }

  /* sizes */
  auto pad_l = padding[0];
  auto pad_r = padding[1];

  int64_t nplane = input_.size(dim_plane);
  int64_t input_w = input_.size(dim_w);
  int64_t output_w  = input_w + pad_l + pad_r;

  AT_CHECK(pad_l < input_w && pad_r < input_w, "Argument #4: Padding size "
    "should be less than the corresponding input dimension, but got: padding (",
    pad_l, ", ", pad_r, ") at dimension ", dim_w, " of input ", input_.sizes());

  AT_CHECK(output_w >= 1 , 2,
    "input (W: ", input_w, ")is too small. Calculated output W: ", output_w);

  /* get contiguous input */
  Tensor input = input_.contiguous();

  /* resize output */
  if (input.ndimension() == 2) {
    output.resize_({nplane, output_w});
    AT_DISPATCH_FLOATING_TYPES(input.type(), "reflection_pad1d", [&] {
      reflection_pad1d_out_frame<scalar_t>(
        input.data<scalar_t>(), output.data<scalar_t>(),
        nplane,
        input_w, output_w,
        pad_l);
    });
  } else {
    output.resize_({nbatch, nplane, output_w});
    AT_DISPATCH_FLOATING_TYPES(input.type(), "reflection_pad1d", [&] {
      reflection_pad1d_out_loop<scalar_t>(
        input.data<scalar_t>(), output.data<scalar_t>(),
        nbatch, nplane,
        input_w, output_w,
        pad_l);
    });
  }
}

template <typename scalar_t>
static void reflection_pad1d_backward_out_frame(
    scalar_t * grad_input, scalar_t * grad_output,
    int64_t nplane,
    int64_t input_w, int64_t output_w,
    int64_t pad_l) {
  int64_t i_start_x = std::max(int64_t(0), -pad_l);
  int64_t o_start_x = std::max(int64_t(0), pad_l);

  int64_t k, ip_x;
#pragma omp parallel for private(k, ip_x)

  for (k = 0; k < nplane; k++) {
    for (int64_t j = 0; j < output_w; j++) {
      if (j < pad_l) {
        ip_x = pad_l * 2 - j;
      } else if (j >= pad_l && j < input_w + pad_l) {
        ip_x = j;
      } else {
        ip_x = (input_w + pad_l - 1) * 2 - j;
      }
      ip_x = ip_x - o_start_x + i_start_x;

      scalar_t *src_p = grad_output + k*output_w + j;
      scalar_t *dest_p = grad_input + k*input_w + ip_x;
      *dest_p += *src_p;
    }
  }
}

template <typename scalar_t>
inline void reflection_pad1d_backward_out_loop(
    scalar_t *grad_input, scalar_t *grad_output,
    int64_t nbatch, int64_t nplane,
    int64_t input_w, int64_t output_w,
    int64_t pad_l) {
  int64_t p;
#pragma omp parallel for private(p)
  for (p = 0; p < nbatch; p++) {
    reflection_pad1d_backward_out_frame<scalar_t>(
      grad_input + p * nplane * input_w,
      grad_output + p * nplane * output_w,
      nplane,
      input_w, output_w,
      pad_l);
  }
}

void reflection_pad1d_backward_out_template(
    Tensor& grad_input, const Tensor& grad_output_, const Tensor& input,
    IntList padding) {
  int64_t dim_plane = 0;
  int64_t dim_w = 1;
  int64_t nbatch = 1;

  if (input.ndimension() == 3) {
    nbatch = input.size(0);
    dim_w++;
    dim_plane++;
  }

  /* sizes */
  auto pad_l = padding[0];
  auto pad_r = padding[1];
  int64_t nplane = input.size(dim_plane);
  int64_t input_w = input.size(dim_w);
  int64_t output_w  = input_w + pad_l + pad_r;

  AT_CHECK(output_w == grad_output_.size(dim_w), "grad_output width unexpected."
    " Expected: ", output_w, ", Got: ", grad_output_.size(dim_w));

  /* get contiguous grad_output */
  Tensor grad_output = grad_output_.contiguous();

  /* backprop */
  if (input.ndimension() == 2) {
    AT_DISPATCH_FLOATING_TYPES(
      grad_input.type(), "reflection_pad1d_backward", [&] {
        reflection_pad1d_backward_out_frame(
          grad_input.data<scalar_t>(), grad_output.data<scalar_t>(),
          nplane,
          input_w, output_w,
          pad_l);
        }
    );
  } else {
    AT_DISPATCH_FLOATING_TYPES(
      grad_input.type(), "reflection_pad1d_backward", [&] {
        reflection_pad1d_backward_out_loop(
          grad_input.data<scalar_t>(),
          grad_output.data<scalar_t>(),
          nbatch, nplane,
          input_w, output_w,
          pad_l);
      }
    );
  }
}
} // namespace

Tensor& reflection_pad1d_out_cpu(
    Tensor& output, const Tensor& input, IntList padding) {
  reflection_pad1d_out_template(output, input, padding);
  return output;
}

Tensor reflection_pad1d_cpu(const Tensor& input, IntList padding) {
  auto output = at::empty({0}, input.options());
  reflection_pad1d_out_template(output, input, padding);
  return output;
}

Tensor& reflection_pad1d_backward_out_cpu(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    IntList padding) {
  grad_input.resize_as_(input);
  grad_input.zero_();
  reflection_pad1d_backward_out_template(
    grad_input, grad_output, input, padding);
  return grad_input;
}

Tensor reflection_pad1d_backward_cpu(
    const Tensor& grad_output,
    const Tensor& input,
    IntList padding) {
  auto grad_input = at::zeros_like(input);
  reflection_pad1d_backward_out_template(
    grad_input, grad_output, input, padding);
  return grad_input;
}

} // namespace native
} // namespace at
