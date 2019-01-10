#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/SquareForm.h>

namespace at
{
namespace native
{
template <typename scalar_t>
static void squareform_frame(
    scalar_t *input_p, scalar_t *output_p, int64_t square_size,
    int64_t input_dim, bool checks)
{
  int64_t idx = 0;
  if (input_dim == 1)
  {
    // from 1D to 2D
    for (int64_t i = 0; i < square_size; i++)
    {
      scalar_t *dest_ii = output_p + i * square_size + i;
      *dest_ii = 0;
      for (int64_t j = i + 1; j < square_size; j++)
      {
        scalar_t *dest_ij = output_p + i * square_size + j;
        scalar_t *dest_ji = output_p + j * square_size + i;
        *dest_ij = input_p[idx];
        *dest_ji = input_p[idx];
        idx++;
      }
    }
  }
  else
  {
    // from 2D to 1D
    for (int64_t i = 0; i < square_size; i++)
    {
      if (checks)
      {
        AT_CHECK(*(input_p + i * square_size + i) == 0,
            "input has non-zero in diagnoal");
      }
      for (int64_t j = i + 1; j < square_size; j++)
      {
        output_p[idx] = *(input_p + i * square_size + j);
        idx++;
        if (checks)
        {
          AT_CHECK(*(input_p + i * square_size + j) == *(input_p + j * square_size + i),
              "input is not symmetirc");
        }
      }
    }
  }
}

template <typename scalar_t>
static void squareform_backward_frame(
    scalar_t *grad_input, scalar_t *grad_output, int64_t square_size,
    int64_t input_dim)
{
  int64_t idx = 0;
  if (input_dim == 1)
  {
    /// backward for 2D to 1D
    for (int64_t i = 0; i < square_size; i++)
    {
      for (int64_t j = i + 1; j < square_size; j++)
      {
        scalar_t *dest_ij = grad_output + i * square_size + j;
        scalar_t *dest_ji = grad_output + j * square_size + i;
        *dest_ij = grad_input[idx] / 2.0;
        *dest_ji = grad_input[idx] / 2.0;
        idx++;
      }
    }
  }
  else
  {
    /// backward for 1D to 2D
    for (int64_t i = 0; i < square_size; i++)
    {
      for (int64_t j = i + 1; j < square_size; j++)
      {
        grad_output[idx] = *(grad_input + i * square_size + j);
        grad_output[idx] += *(grad_input + j * square_size + i);
        idx++;
      }
    }
  }
}

Tensor squareform_cpu(const Tensor& self, bool checks)
{
  AT_CHECK((self.dim() == 1 || self.dim() == 2),
      "Expected vector-form distance or square-form distance as input",
      " but the dimension of input is ", self.dim());
  auto input = self.contiguous();
  Tensor out_tensor;
  int64_t vector_size = 0;
  int64_t d = 0;
  if (input.dim() == 1)
  {
    vector_size = input.size(0);
    d = getSquareSize(vector_size);
    out_tensor = at::empty({d, d}, input.options());
  }
  else
  {
    AT_CHECK(input.size(0) == input.size(1),
        "Expected square matrix as input, size(0) ",
        input.size(0), " != size(1) ", input.size(1));
    d = input.size(0);
    vector_size = d * (d - 1) / 2;
    out_tensor = at::empty({vector_size}, input.options());
  }
  AT_DISPATCH_ALL_TYPES(input.type(), "squareform", [&] {
    squareform_frame<scalar_t> (input.data<scalar_t>(),
      out_tensor.data<scalar_t>(), d, input.dim(), checks);
  });
  return out_tensor;
}

Tensor squareform_backward_cpu(const Tensor& grad, const Tensor& self)
{
  AT_CHECK((grad.dim() == 1 || grad.dim() == 2),
      "Expected the input dim is 1 or 2, but got ", grad.dim());
  auto input = grad.contiguous();
  Tensor out_tensor;
  int64_t vector_size = 0;
  int64_t d = 0;
  if (input.dim() == 1)
  {
    vector_size = input.size(0);
    d = getSquareSize(vector_size);
    out_tensor = at::zeros({d, d}, input.options());
  }
  else
  {
    AT_CHECK(input.size(0) == input.size(1),
        "expected square-form distance as input, size(0) ",
        input.size(0), " != size(1) ", input.size(1));
    d = input.size(0);
    vector_size = d * (d - 1) / 2;
    out_tensor = at::empty({vector_size}, input.options());
  }
  AT_DISPATCH_ALL_TYPES(input.type(), "squareform", [&] {
    squareform_backward_frame<scalar_t> (input.data<scalar_t>(),
      out_tensor.data<scalar_t>(), d, input.dim());
  });
  return out_tensor;
}

}  // at::native
}  // at
