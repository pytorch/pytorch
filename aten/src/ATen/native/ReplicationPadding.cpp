#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"

namespace at {
namespace native {

namespace {
template <typename scalar_t>
static void replication_pad1d_out_frame(
    scalar_t *input_p, scalar_t *output_p,
    long nslices,
    long iwidth,
    long owidth,
    int pad_l, int pad_r)
{
  int iStartX = fmax(0, -pad_l);
  int oStartX = fmax(0, pad_l);

  long k, ip_x;
#pragma omp parallel for private(k, ip_x)
  for (k = 0; k < nslices; k++)
  {
    long j;
    for (j = 0; j < owidth; j++) {
      if (j < pad_l) {
        ip_x = pad_l;
      } else if (j >= pad_l && j < iwidth + pad_l) {
        ip_x = j;
      } else {
        ip_x = iwidth + pad_l - 1;
      }
      ip_x = ip_x - oStartX + iStartX;

      scalar_t *dest_p = output_p + k*owidth + j;
      scalar_t *src_p = input_p + k*iwidth + ip_x;
      *dest_p = *src_p;
    }
  }
}

void replication_pad1d_out_cpu_template(
    at::Tensor& output,
    at::Tensor const& input,
    IntList paddingSize)
{
  int dimw = 1;
  int dimslices = 0;
  long nbatch = 1;
  long nslices;
  long iwidth;
  long owidth;
  int pad_l = paddingSize[0];
  int pad_r = paddingSize[1];

  AT_CHECK(input.numel() > 0
      && (input.ndimension() == 2 || input.ndimension() == 3),
      "non-empty 2D or 3D (batch mode) tensor expected for input");

  if (input.ndimension() == 3)
  {
    nbatch = input.size(0);
    dimw++;
    dimslices++;
  }

  /* sizes */
  nslices = input.size(dimslices);
  iwidth = input.size(dimw);
  owidth  = iwidth + pad_l + pad_r;

  AT_CHECK(owidth >= 1,
      "input (W: ", iwidth, ") is too small."
      " Calculated output W: ", owidth);


  /* get contiguous input */
  auto input_ = input.contiguous();

  /* resize output */
  if (input_.ndimension() == 2)
  {
    output.resize_({nslices, owidth});
    AT_DISPATCH_FLOATING_TYPES(input_.type(), "replication_pad1d", [&] {
        auto input_data = input_.data<scalar_t>();
        auto output_data = output.data<scalar_t>();
        replication_pad1d_out_frame<scalar_t> (input_data, output_data,
          nslices,
          iwidth,
          owidth,
          pad_l, pad_r);
        }
        );
  }
  else
  {
    long p;

    output.resize_({nbatch, nslices, owidth});

#pragma omp parallel for private(p)
    for (p = 0; p < nbatch; p++)
    {
      AT_DISPATCH_FLOATING_TYPES(input_.type(), "replication_pad1d", [&] {
          auto input_data = input_.data<scalar_t>();
          auto output_data = output.data<scalar_t>();
          replication_pad1d_out_frame<scalar_t>(
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
static void replication_pad1d_backward_out_frame(
    scalar_t *ginput_p, scalar_t *goutput_p,
    long nslices,
    long iwidth,
    long owidth,
    int pad_l, int pad_r)
{
  int iStartX = fmax(0, -pad_l);
  int oStartX = fmax(0, pad_l);

  long k, ip_x;
#pragma omp parallel for private(k, ip_x)
  for (k = 0; k < nslices; k++)
  {
    long j;
    for (j = 0; j < owidth; j++) {
      if (j < pad_l) {
        ip_x = pad_l;
      } else if (j >= pad_l && j < iwidth + pad_l) {
        ip_x = j;
      } else {
        ip_x = iwidth + pad_l - 1;
      }
      ip_x = ip_x - oStartX + iStartX;

      scalar_t *src_p = goutput_p + k*owidth + j;
      scalar_t *dest_p = ginput_p + k*iwidth + ip_x;
      *dest_p += *src_p;
    }
  }
}

Tensor& replication_pad1d_backward_out_cpu_template(
    Tensor& gradInput,
    const Tensor& gradOutput_,
    const Tensor& input,
    IntList paddingSize)
{
  int dimw = 1;
  int dimslices = 0;
  long nbatch = 1;
  long nslices;
  long iwidth;
  long owidth;
  int pad_l = paddingSize[0];
  int pad_r = paddingSize[1];

  if (input.ndimension() == 3)
  {
    nbatch = input.size(0);
    dimw++;
    dimslices++;
  }

  /* sizes */
  nslices = input.size(dimslices);
  iwidth = input.size(dimw);
  owidth  = iwidth + pad_l + pad_r;

  AT_CHECK(owidth == gradOutput_.size(dimw),
      "gradOutput width unexpected. Expected: ", owidth,
      " Got: ", gradOutput_.size(dimw));

  /* get contiguous gradOutput */
  auto gradOutput = gradOutput_.contiguous();
  gradInput.resize_as_(input);
  gradInput.zero_();

  /* backprop */
  if (input.ndimension() == 2) {
    AT_DISPATCH_FLOATING_TYPES(
        input.type(), "replication_pad1d_backward", [&] {
        scalar_t *gradInput_data = gradInput.data<scalar_t>();
        scalar_t *gradOutput_data = gradOutput.data<scalar_t>();

        replication_pad1d_backward_out_frame<scalar_t> (
          gradInput_data,
          gradOutput_data,
          nslices,
          iwidth,
          owidth,
          pad_l, pad_r);
        }
        );
  } else {
    long p;
#pragma omp parallel for private(p)
    for (p = 0; p < nbatch; p++) {
      AT_DISPATCH_FLOATING_TYPES(
          input.type(), "replication_pad1d_backward", [&] {
          scalar_t *gradInput_data = gradInput.data<scalar_t>();
          scalar_t *gradOutput_data = gradOutput.data<scalar_t>();

          replication_pad1d_backward_out_frame<scalar_t>(
            gradInput_data + p * nslices * iwidth,
            gradOutput_data + p * nslices * owidth,
            nslices,
            iwidth,
            owidth,
            pad_l, pad_r);
          }
          );
    }
  }
  return gradInput;
}
} // namespace

Tensor& replication_pad1d_out_cpu(
    Tensor& output,
    const Tensor& input,
    IntList paddingSize)
{
  replication_pad1d_out_cpu_template(
      output, input, paddingSize);
  return output;
}

Tensor replication_pad1d_cpu(
    at::Tensor const& input,
    IntList paddingSize)
{
  auto output = at::empty({0}, input.options());
  replication_pad1d_out_cpu_template(
      output, input, paddingSize);
  return output;
}

Tensor& replication_pad1d_backward_out_cpu(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input,
    IntList paddingSize)
{
  gradInput.resize_as_(input);
  replication_pad1d_backward_out_cpu_template(
      gradInput, gradOutput, input, paddingSize);
  return gradInput;
}

Tensor replication_pad1d_backward_cpu(
    const Tensor& gradOutput,
    const Tensor& input,
    IntList paddingSize)
{
  auto gradInput = at::zeros_like(input);
  replication_pad1d_backward_out_cpu_template(
      gradInput, gradOutput, input, paddingSize);
  return gradInput;
}

} // at::native
} // at
