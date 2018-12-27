#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"
#include <algorithm>

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
  int iStartX = std::max(0, -pad_l);
  int oStartX = std::max(0, pad_l);

  long k, ip_x;
#pragma omp parallel for private(k, ip_x)
  for (k = 0; k < nslices; k++)
  {
    for (long j = 0; j < owidth; j++) {
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
    Tensor& output,
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
  int iStartX = std::max(0, -pad_l);
  int oStartX = std::max(0, pad_l);

  long k, ip_x;
#pragma omp parallel for private(k, ip_x)
  for (k = 0; k < nslices; k++)
  {
    for (long j = 0; j < owidth; j++) {
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

template <typename scalar_t>
static void replication_pad2d_out_frame(
    scalar_t *input_p, scalar_t *output_p,
    int64_t nslices,
    int64_t iwidth, int64_t iheight,
    int64_t owidth, int64_t oheight,
    int pad_l, int pad_r,
    int pad_t, int pad_b)
{
  int iStartX = std::max(0, -pad_l);
  int iStartY = std::max(0, -pad_t);
  int oStartX = std::max(0, pad_l);
  int oStartY = std::max(0, pad_t);

  int64_t k, ip_x, ip_y;
#pragma omp parallel for private(k, ip_x, ip_y)
  for (k = 0; k < nslices; k++)
  {
    for (int64_t i = 0; i < oheight; i++) {
      for (int64_t j = 0; j < owidth; j++) {
        if (j < pad_l) {
          ip_x = pad_l;
        } else if (j >= pad_l && j < iwidth + pad_l) {
          ip_x = j;
        } else {
          ip_x = iwidth + pad_l - 1;
        }
        ip_x = ip_x - oStartX + iStartX;

        if (i < pad_t) {
          ip_y = pad_t;
        } else if (i >= pad_t && i < iheight + pad_t) {
          ip_y = i;
        } else {
          ip_y = iheight + pad_t - 1;
        }
        ip_y = ip_y - oStartY + iStartY;

        scalar_t *dest_p = output_p + k*owidth*oheight + i * owidth + j;
        scalar_t *src_p = input_p + k*iwidth*iheight + ip_y * iwidth + ip_x;
        *dest_p = *src_p;
      }
    }
  }
}

void replication_pad2d_out_cpu_template(Tensor& output,
    const Tensor& input,
    IntList paddingSize)
{
  int pad_l = paddingSize[0];
  int pad_r = paddingSize[1];
  int pad_t = paddingSize[2];
  int pad_b = paddingSize[3];
  int dimw = 2;
  int dimh = 1;
  int dimslices = 0;
  int64_t nbatch = 1;
  int64_t nslices;
  int64_t iheight;
  int64_t iwidth;
  int64_t oheight;
  int64_t owidth;

  AT_CHECK(input.numel() > 0 && (input.dim() == 3 || input.dim() == 4),
      "3D or 4D (batch mode) tensor expected for input, but got: ", input);

  if (input.dim() == 4)
  {
    nbatch = input.size(0);
    dimw++;
    dimh++;
    dimslices++;
  }

  /* sizes */
  nslices = input.size(dimslices);
  iheight = input.size(dimh);
  iwidth = input.size(dimw);
  oheight = iheight + pad_t + pad_b;
  owidth  = iwidth + pad_l + pad_r;

  AT_CHECK(owidth >= 1 || oheight >= 1,
      "input (H: ", iheight, ", W: ", iwidth, " )is too small."
      " Calculated output H: ", oheight, " W: ", owidth);


  /* get contiguous input */
  auto input_ = input.contiguous();

  /* resize output */
  if (input_.dim() == 3)
  {
    output.resize_({nslices, oheight, owidth});
    AT_DISPATCH_FLOATING_TYPES(input_.type(), "replication_pad2d", [&] {
        auto input_data = input_.data<scalar_t>();
        auto output_data = output.data<scalar_t>();
        replication_pad2d_out_frame<scalar_t> (input_data, output_data,
            nslices,
            iwidth, iheight,
            owidth, oheight,
            pad_l, pad_r,
            pad_t, pad_b);
        }
        );
  }
  else
  {
    int64_t p;

    output.resize_({nbatch, nslices, oheight, owidth});

#pragma omp parallel for private(p)
    for (p = 0; p < nbatch; p++)
    {
      AT_DISPATCH_FLOATING_TYPES(input_.type(), "replication_pad2d", [&] {
          auto input_data = input_.data<scalar_t>();
          auto output_data = output.data<scalar_t>();
          replication_pad2d_out_frame<scalar_t>(
              input_data+p*nslices*iwidth*iheight,
              output_data+p*nslices*owidth*oheight,
              nslices,
              iwidth, iheight,
              owidth, oheight,
              pad_l, pad_r,
              pad_t, pad_b);
          }
          );
    }
  }
}

template <typename scalar_t>
static void replication_pad2d_backward_out_frame(
    scalar_t *ginput_p, scalar_t *goutput_p,
    int64_t nslices,
    int64_t iwidth, int64_t iheight,
    int64_t owidth, int64_t oheight,
    int pad_l, int pad_r,
    int pad_t, int pad_b)
{
  int iStartX = std::max(0, -pad_l);
  int iStartY = std::max(0, -pad_t);
  int oStartX = std::max(0, pad_l);
  int oStartY = std::max(0, pad_t);

  int64_t k, ip_x, ip_y;
#pragma omp parallel for private(k, ip_x, ip_y)
  for (k = 0; k < nslices; k++)
  {
    for (int64_t i = 0; i < oheight; i++) {
      for (int64_t j = 0; j < owidth; j++) {
        if (j < pad_l) {
          ip_x = pad_l;
        } else if (j >= pad_l && j < iwidth + pad_l) {
          ip_x = j;
        } else {
          ip_x = iwidth + pad_l - 1;
        }
        ip_x = ip_x - oStartX + iStartX;

        if (i < pad_t) {
          ip_y = pad_t;
        } else if (i >= pad_t && i < iheight + pad_t) {
          ip_y = i;
        } else {
          ip_y = iheight + pad_t - 1;
        }
        ip_y = ip_y - oStartY + iStartY;

        scalar_t *src_p = goutput_p + k*owidth*oheight + i * owidth + j;
        scalar_t *dest_p = ginput_p + k*iwidth*iheight + ip_y * iwidth + ip_x;
        *dest_p += *src_p;
      }
    }
  }
}

Tensor& replication_pad2d_backward_out_cpu_template(
    Tensor& gradInput,
    const Tensor& gradOutput_,
    const Tensor& input,
    IntList paddingSize)
{
  int pad_l = paddingSize[0];
  int pad_r = paddingSize[1];
  int pad_t = paddingSize[2];
  int pad_b = paddingSize[3];
  int dimw = 2;
  int dimh = 1;
  int dimslices = 0;
  int64_t nbatch = 1;
  int64_t nslices;
  int64_t iheight;
  int64_t iwidth;
  int64_t oheight;
  int64_t owidth;

  if (input.dim() == 4)
  {
    nbatch = input.size(0);
    dimw++;
    dimh++;
    dimslices++;
  }

  /* sizes */
  nslices = input.size(dimslices);
  iheight = input.size(dimh);
  iwidth = input.size(dimw);
  oheight = iheight + pad_t + pad_b;
  owidth  = iwidth + pad_l + pad_r;

  AT_CHECK(owidth == gradOutput_.size(dimw),
      "gradOutput width unexpected. Expected: ", owidth, ", Got: ",
      gradOutput_.size(dimw));
  AT_CHECK(oheight == gradOutput_.size(dimh),
      "gradOutput height unexpected. Expected: ", oheight, ", Got: ",
      gradOutput_.size(dimh));

  /* get contiguous gradOutput */
  auto gradOutput = gradOutput_.contiguous();

  /* resize */
  gradInput.resize_as_(input);
  gradInput.zero_();

  /* backprop */
  if (input.dim() == 3) {
    AT_DISPATCH_FLOATING_TYPES(
        input.type(), "replication_pad2d_backward", [&] {
        replication_pad2d_backward_out_frame<scalar_t>(
            gradInput.data<scalar_t>(),
            gradOutput.data<scalar_t>(),
            nslices,
            iwidth, iheight,
            owidth, oheight,
            pad_l, pad_r,
            pad_t, pad_b);
        }
        );
  } else {
    int64_t p;
#pragma omp parallel for private(p)
    for (p = 0; p < nbatch; p++) {
      AT_DISPATCH_FLOATING_TYPES(
          input.type(), "replication_pad2d_backward", [&] {
          replication_pad2d_backward_out_frame<scalar_t>(
              gradInput.data<scalar_t>() + p * nslices * iheight * iwidth,
              gradOutput.data<scalar_t>() + p * nslices * oheight * owidth,
              nslices,
              iwidth, iheight,
              owidth, oheight,
              pad_l, pad_r,
              pad_t, pad_b);
          }
          );
    }
  }
  return gradInput;
}

static inline void shapeCheck(
    const Tensor& input,
    int pleft, int pright,
    int ptop, int pbottom,
    int pfront, int pback) {
  int dimw = 3;
  int dimh = 2;
  int dimd = 1;
  int dimslices = 0;
  int64_t nslices;
  int64_t idepth;
  int64_t iheight;
  int64_t iwidth;
  int64_t odepth;
  int64_t oheight;
  int64_t owidth;

  AT_CHECK(input.numel() > 0 && (input.dim() == 4 || input.dim() == 5),
      "non-empty 4D or 5D (batch mode) tensor expected for input, but got: ", input);

  if (input.dim() == 5)
  {
    dimw++;
    dimh++;
    dimd++;
    dimslices++;
  }

  /* sizes */
  nslices = input.size(dimslices);
  idepth = input.size(dimd);
  iheight = input.size(dimh);
  iwidth = input.size(dimw);
  odepth = idepth + pfront + pback;
  oheight = iheight + ptop + pbottom;
  owidth  = iwidth + pleft + pright;

  AT_CHECK(owidth >= 1 || oheight >= 1 || odepth >= 1,
      "input (D: ", idepth, " H: ", iheight, ", W: ", iwidth,
      ") is too small."
      " Calculated output D: ", odepth, " H: ", oheight, " W: ", owidth);

}

template <typename scalar_t>
static void replication_pad3d_out_frame(
    scalar_t *input_p, scalar_t *output_p,
    int64_t nslices,
    int64_t iwidth, int64_t iheight, int64_t idepth,
    int64_t owidth, int64_t oheight, int64_t odepth,
    int pleft, int pright,
    int ptop, int pbottom,
    int pfront, int pback)
{
  int iStartX = std::max(0, -pleft);
  int iStartY = std::max(0, -ptop);
  int iStartZ = std::max(0, -pfront);
  int oStartX = std::max(0, pleft);
  int oStartY = std::max(0, ptop);
  int oStartZ = std::max(0, pfront);

  int64_t k, ip_x, ip_y, ip_z;
#pragma omp parallel for private(k, ip_x, ip_y, ip_z)
  for (k = 0; k < nslices; k++) {
    for (int64_t z = 0; z < odepth; z++) {
      for (int64_t i = 0; i < oheight; i++) {
        for (int64_t j = 0; j < owidth; j++) {
          if (j < pleft) {
            ip_x = pleft;
          } else if (j >= pleft && j < iwidth + pleft) {
            ip_x = j;
          } else {
            ip_x = iwidth + pleft - 1;
          }
          ip_x = ip_x - oStartX + iStartX;

          if (i < ptop) {
            ip_y = ptop;
          } else if (i >= ptop && i < iheight + ptop) {
            ip_y = i;
          } else {
            ip_y = iheight + ptop - 1;
          }
          ip_y = ip_y - oStartY + iStartY;

          if (z < pfront) {
            ip_z = pfront;
          } else if (z >= pfront && z < idepth + pfront) {
            ip_z = z;
          } else {
            ip_z = idepth + pfront - 1;
          }
          ip_z = ip_z - oStartZ + iStartZ;

          scalar_t *dest_p = output_p + k * owidth * oheight * odepth +
            z * owidth * oheight + i * owidth + j;
          scalar_t *src_p = input_p + k * iwidth * iheight * idepth +
            ip_z * iwidth * iheight + ip_y * iwidth + ip_x;
          *dest_p = *src_p;
        }
      }
    }
  }
}

void replication_pad3d_out_cpu_template(
    Tensor& output,
    const Tensor& input,
    IntList paddingSize)
{
  int pleft = paddingSize[0];
  int pright = paddingSize[1];
  int ptop = paddingSize[2];
  int pbottom = paddingSize[3];
  int pfront = paddingSize[4];
  int pback = paddingSize[5];
  int dimw = 3;
  int dimh = 2;
  int dimd = 1;
  int dimslices = 0;
  int64_t nbatch = 1;
  int64_t nslices;
  int64_t idepth;
  int64_t iheight;
  int64_t iwidth;
  int64_t odepth;
  int64_t oheight;
  int64_t owidth;

  shapeCheck(input, pleft, pright,
      ptop, pbottom, pfront, pback);

  if (input.dim() == 5)
  {
    nbatch = input.size(0);
    dimw++;
    dimh++;
    dimd++;
    dimslices++;
  }

  /* sizes */
  nslices = input.size(dimslices);
  idepth = input.size(dimd);
  iheight = input.size(dimh);
  iwidth = input.size(dimw);
  odepth = idepth + pfront + pback;
  oheight = iheight + ptop + pbottom;
  owidth  = iwidth + pleft + pright;

  /* get contiguous input */
  auto input_ = input.contiguous();

  /* resize output */
  if (input_.dim() == 4)
  {
    output.resize_({nslices, odepth, oheight, owidth});

    AT_DISPATCH_FLOATING_TYPES(input_.type(), "replication_pad3d", [&] {
        auto input_data = input_.data<scalar_t>();
        auto output_data = output.data<scalar_t>();
        replication_pad3d_out_frame<scalar_t>(
            input_data, output_data, nslices, iwidth, iheight, idepth,
            owidth, oheight, odepth, pleft, pright, ptop, pbottom, pfront,
            pback);
        }
        );
  }
  else
  {
    int64_t p;

    output.resize_({nbatch, nslices, odepth, oheight, owidth});

#pragma omp parallel for private(p)
    for (p = 0; p < nbatch; p++)
    {
      AT_DISPATCH_FLOATING_TYPES(input_.type(), "replication_pad3d", [&] {
          auto input_data = input_.data<scalar_t>();
          auto output_data = output.data<scalar_t>();
          replication_pad3d_out_frame<scalar_t>(
              input_data + p * nslices * iwidth * iheight * idepth,
              output_data + p * nslices * owidth * oheight * odepth,
              nslices,
              iwidth, iheight, idepth,
              owidth, oheight, odepth,
              pleft, pright,
              ptop, pbottom,
              pfront, pback);
          }
          );
    }
  }
}

template <typename scalar_t>
static void replication_pad3d_backward_out_frame(
    scalar_t *ginput_p, scalar_t *goutput_p,
    int64_t nslices,
    int64_t iwidth, int64_t iheight, int64_t idepth,
    int64_t owidth, int64_t oheight, int64_t odepth,
    int pleft, int pright,
    int ptop, int pbottom,
    int pfront, int pback)
{
  int iStartX = std::max(0, -pleft);
  int iStartY = std::max(0, -ptop);
  int iStartZ = std::max(0, -pfront);
  int oStartX = std::max(0, pleft);
  int oStartY = std::max(0, ptop);
  int oStartZ = std::max(0, pfront);

  int64_t k, ip_x, ip_y, ip_z;
#pragma omp parallel for private(k, ip_x, ip_y, ip_z)
  for (k = 0; k < nslices; k++) {
    for (int64_t z = 0; z < odepth; z++) {
      for (int64_t i = 0; i < oheight; i++) {
        for (int64_t j = 0; j < owidth; j++) {
          if (j < pleft) {
            ip_x = pleft;
          } else if (j >= pleft && j < iwidth + pleft) {
            ip_x = j;
          } else {
            ip_x = iwidth + pleft - 1;
          }
          ip_x = ip_x - oStartX + iStartX;

          if (i < ptop) {
            ip_y = ptop;
          } else if (i >= ptop && i < iheight + ptop) {
            ip_y = i;
          } else {
            ip_y = iheight + ptop - 1;
          }
          ip_y = ip_y - oStartY + iStartY;

          if (z < pfront) {
            ip_z = pfront;
          } else if (z >= pfront && z < idepth + pfront) {
            ip_z = z;
          } else {
            ip_z = idepth + pfront - 1;
          }
          ip_z = ip_z - oStartZ + iStartZ;

          scalar_t *src_p = goutput_p + k * owidth * oheight * odepth +
            z * owidth * oheight + i * owidth + j;
          scalar_t *dest_p = ginput_p + k * iwidth * iheight * idepth +
            ip_z * iwidth * iheight + ip_y * iwidth + ip_x;
          *dest_p += *src_p;
        }
      }
    }
  }
}

Tensor& replication_pad3d_backward_out_cpu_template(
    Tensor& gradInput,
    const Tensor& gradOutput_,
    const Tensor& input,
    IntList paddingSize)
{
  int pleft = paddingSize[0];
  int pright = paddingSize[1];
  int ptop = paddingSize[2];
  int pbottom = paddingSize[3];
  int pfront = paddingSize[4];
  int pback = paddingSize[5];
  int dimw = 3;
  int dimh = 2;
  int dimd = 1;
  int dimslices = 0;
  int64_t nbatch = 1;
  int64_t nslices;
  int64_t idepth;
  int64_t iheight;
  int64_t iwidth;
  int64_t odepth;
  int64_t oheight;
  int64_t owidth;

  if (input.dim() == 5)
  {
    nbatch = input.size(0);
    dimw++;
    dimh++;
    dimd++;
    dimslices++;
  }

  /* sizes */
  nslices = input.size(dimslices);
  idepth = input.size(dimd);
  iheight = input.size(dimh);
  iwidth = input.size(dimw);
  odepth = idepth + pfront + pback;
  oheight = iheight + ptop + pbottom;
  owidth  = iwidth + pleft + pright;


  shapeCheck(input, pleft, pright,
      ptop, pbottom, pfront, pback);

  /* get contiguous gradOutput */
  auto gradOutput = gradOutput_.contiguous();

  /* resize */
  gradInput.resize_as_(input);
  gradInput.zero_();

  /* backprop */
  if (input.dim() == 4) {
    AT_DISPATCH_FLOATING_TYPES(
        input.type(), "replication_pad3d_backward", [&] {
        replication_pad3d_backward_out_frame<scalar_t> (
            gradInput.data<scalar_t>(),
            gradOutput.data<scalar_t>(),
            nslices,
            iwidth, iheight, idepth,
            owidth, oheight, odepth,
            pleft, pright,
            ptop, pbottom,
            pfront, pback);
        }
        );
  } else {
    int64_t p;
#pragma omp parallel for private(p)
    for (p = 0; p < nbatch; p++) {
      AT_DISPATCH_FLOATING_TYPES(
          input.type(), "replication_pad3d_backward", [&] {
          replication_pad3d_backward_out_frame<scalar_t>(
              gradInput.data<scalar_t>() + p * nslices * idepth * iheight * iwidth,
              gradOutput.data<scalar_t>() + p * nslices * odepth * oheight * owidth,
              nslices,
              iwidth, iheight, idepth,
              owidth, oheight, odepth,
              pleft, pright,
              ptop, pbottom,
              pfront, pback);
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
    const Tensor& input,
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

Tensor& replication_pad2d_out_cpu(
    Tensor& output,
    const Tensor& input,
    IntList paddingSize)
{
  replication_pad2d_out_cpu_template(
      output, input, paddingSize);
  return output;
}

Tensor replication_pad2d_cpu(
    const Tensor& input,
    IntList paddingSize)
{
  auto output = at::empty({0}, input.options());
  replication_pad2d_out_cpu_template(
      output, input, paddingSize);
  return output;
}

Tensor& replication_pad2d_backward_out_cpu(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input,
    IntList paddingSize)
{
  replication_pad2d_backward_out_cpu_template(
      gradInput, gradOutput, input, paddingSize);
  return gradInput;
}

Tensor replication_pad2d_backward_cpu(
    const Tensor& gradOutput,
    const Tensor& input,
    IntList paddingSize)
{
  auto gradInput = at::zeros_like(input);
  replication_pad2d_backward_out_cpu_template(
      gradInput, gradOutput, input, paddingSize);
  return gradInput;
}

Tensor& replication_pad3d_out_cpu(
    Tensor& output,
    const Tensor& input,
    IntList paddingSize)
{
  replication_pad3d_out_cpu_template(
      output, input, paddingSize);
  return output;
}

Tensor replication_pad3d_cpu(
    const Tensor& input,
    IntList paddingSize)
{
  auto output = at::empty({0}, input.options());
  replication_pad3d_out_cpu_template(
      output, input, paddingSize);
  return output;
}

Tensor& replication_pad3d_backward_out_cpu(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input,
    IntList paddingSize)
{
  replication_pad3d_backward_out_cpu_template(
      gradInput, gradOutput, input, paddingSize);
  return gradInput;
}

Tensor replication_pad3d_backward_cpu(
    const Tensor& gradOutput,
    const Tensor& input,
    IntList paddingSize)
{
  auto gradInput = at::zeros_like(input);
  replication_pad3d_backward_out_cpu_template(
      gradInput, gradOutput, input, paddingSize);
  return gradInput;
}

} // at::native
} // at
