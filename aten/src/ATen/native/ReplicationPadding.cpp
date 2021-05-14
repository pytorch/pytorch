#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <algorithm>

namespace at {

namespace meta {

TORCH_META_FUNC(replication_pad1d) (
  const Tensor& input, IntArrayRef paddingSize  // no out argument!
) {

  int64_t dimw = 1;
  int64_t dimslices = 0;
  int64_t nbatch = 1;

  TORCH_CHECK(paddingSize.size() == 2, "padding size is expected to be 2");

  int64_t pad_l = paddingSize[0];
  int64_t pad_r = paddingSize[1];

  // allow empty batch size but not other dimensions.
  TORCH_CHECK((input.dim() == 2 && input.size(0) != 0 && input.size(1) != 0) ||
              (input.dim() == 3 && input.size(1) != 0 && input.size(2) != 0),
              "Expected 2D or 3D (batch mode) tensor with possibly 0 batch size and other non-zero dimensions for input, but got: ",
              input.sizes());

  if (input.ndimension() == 3) {
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

  if (input.ndimension() == 2) {
    set_output({nslices, owidth}, input.options());
  } else {
    set_output({nbatch, nslices, owidth}, input.options());
  }
}

TORCH_META_FUNC(replication_pad1d_backward) (
  const Tensor& gradOutput,
  const Tensor& input,
  IntArrayRef paddingSize
) {
  int64_t dimw = 1;
  int64_t dimslices = 0;
  int64_t nbatch = 1;
  TORCH_CHECK(paddingSize.size() == 2, "padding size is expected to be 2");
  int64_t pad_l = paddingSize[0];
  int64_t pad_r = paddingSize[1];

  if (input.ndimension() == 3)
  {
    // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
    nbatch = input.size(0);
    dimw++;
    dimslices++;
  }

  /* sizes */
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores,clang-diagnostic-unused-variable)
  int64_t nslices = input.size(dimslices);
  int64_t iwidth = input.size(dimw);
  int64_t owidth  = iwidth + pad_l + pad_r;

  TORCH_CHECK(owidth == gradOutput.size(dimw),
      "gradOutput width unexpected. Expected: ", owidth,
      " Got: ", gradOutput.size(dimw));

  set_output(input.sizes(), input.options());
}

TORCH_META_FUNC(replication_pad2d) (
  const Tensor& input, IntArrayRef paddingSize
) {
  TORCH_CHECK(paddingSize.size() == 4, "padding size is expected to be 4");
  int64_t pad_l = paddingSize[0];
  int64_t pad_r = paddingSize[1];
  int64_t pad_t = paddingSize[2];
  int64_t pad_b = paddingSize[3];
  int64_t dimw = 2;
  int64_t dimh = 1;
  int64_t dimslices = 0;
  int64_t nbatch = 1;

  // allow 0 dim batch size and nothing else.
  bool valid_dims = input.size(1) != 0 && input.size(2) != 0;
  TORCH_CHECK(
      (input.dim() == 3 && input.size(0) != 0 && valid_dims) ||
      (input.dim() == 4 && valid_dims && input.size(3) != 0),
      "Expected 3D or 4D (batch mode) tensor with possibly 0 batch size and other non-zero dimensions for input, but got: ",
      input.sizes());

  if (input.dim() == 4)
  {
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

  if (input.dim() == 3) {
    set_output({nslices, oheight, owidth}, input.options());
  } else {
    set_output({nbatch, nslices, oheight, owidth}, input.options());
  }
}

} // namespace meta


static inline void shapeCheck3d(
    const Tensor& input,
    int pleft, int pright,
    int ptop, int pbottom,
    int pfront, int pback) {
  int dimw = 3;
  int dimh = 2;
  int dimd = 1;
  int dimslices = 0;

  // allow batch size of 0-dim.
  bool valid_dims = input.size(1) != 0 && input.size(2) != 0 && input.size(3) != 0;
  TORCH_CHECK(
      (input.dim() == 4 && input.size(0) != 0 && valid_dims) ||
      (input.dim() == 5 && valid_dims && input.size(4) != 0),
      "Expected 4D or 5D (batch mode) tensor with possibly 0 batch size and other non-zero dimensions for input, but got: ",
      input.sizes());

  if (input.dim() == 5)
  {
    dimw++;
    dimh++;
    dimd++;
    dimslices++;
  }

  /* sizes */
  // int64_t nslices = input.size(dimslices);
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

}

namespace meta {

TORCH_META_FUNC(replication_pad3d) (
  const Tensor& input, IntArrayRef paddingSize
) {
  TORCH_CHECK(paddingSize.size() == 6, "padding size is expected to be 6");
  int64_t pleft = paddingSize[0];
  int64_t pright = paddingSize[1];
  int64_t ptop = paddingSize[2];
  int64_t pbottom = paddingSize[3];
  int64_t pfront = paddingSize[4];
  int64_t pback = paddingSize[5];
  int64_t dimw = 3;
  int64_t dimh = 2;
  int64_t dimd = 1;
  int64_t dimslices = 0;
  int64_t nbatch = 1;

  shapeCheck3d(input, pleft, pright, ptop, pbottom, pfront, pback);

  if (input.dim() == 5)
  {
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

  /* resize output */
  if (input.dim() == 4) {
    set_output({nslices, odepth, oheight, owidth}, input.options());
  } else {
    set_output({nbatch, nslices, odepth, oheight, owidth}, input.options());
  }
}

} // namespace meta

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

  at::parallel_for(0, nslices, 0, [&](int64_t start, int64_t end) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    long ip_x;
    for (auto k = start; k < end; k++)
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
  });
}

template <typename scalar_t>
static void replication_pad1d_out_batch(
    scalar_t *input_data, scalar_t *output_data,
    long nslices,
    long iwidth,
    long owidth,
    int pad_l, int pad_r,
    int nbatch)
{
  at::parallel_for(0, nbatch, 0, [&](int64_t start, int64_t end) {
    for (auto p = start; p < end; p++)
    {
      scalar_t *input_p = input_data+p*nslices*iwidth;
      scalar_t *output_p = output_data+p*nslices*owidth;
      replication_pad1d_out_frame(input_p, output_p, nslices, iwidth, owidth, pad_l, pad_r);
    }
  });
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

  at::parallel_for(0, nslices, 0, [&](int64_t start, int64_t end) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    long ip_x;
    for (auto k = start; k < end; k++)
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
  });
}

template <typename scalar_t>
static void replication_pad1d_backward_out_batch(
    scalar_t *ginput_data, scalar_t *goutput_data,
    long nslices,
    long iwidth,
    long owidth,
    int pad_l, int pad_r,
    int nbatch)
{
  at::parallel_for(0, nbatch, 0, [&](int64_t start, int64_t end) {
    for (auto p = start; p < end; p++)
    {
      scalar_t *ginput_p = ginput_data + p * nslices * iwidth;
      scalar_t *goutput_p = goutput_data + p * nslices * owidth;
      replication_pad1d_backward_out_frame(ginput_p, goutput_p,
        nslices, iwidth, owidth, pad_l, pad_r);
    }
  });
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

  at::parallel_for(0, nslices, 0, [&](int64_t start, int64_t end) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int64_t ip_x, ip_y;
    for (auto k = start; k < end; k++)
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
  });
}

template <typename scalar_t>
static void replication_pad2d_out_batch(
    scalar_t *input_data, scalar_t *output_data,
    int64_t nslices,
    int64_t iwidth, int64_t iheight,
    int64_t owidth, int64_t oheight,
    int pad_l, int pad_r,
    int pad_t, int pad_b,
    int nbatch)
{
  at::parallel_for(0, nbatch, 0, [&](int64_t start, int64_t end) {
    for (auto p = start; p < end; p++)
    {
      scalar_t *input_p = input_data+p*nslices*iwidth*iheight;
      scalar_t *output_p = output_data+p*nslices*owidth*oheight;
      replication_pad2d_out_frame(input_p, output_p, nslices,
          iwidth, iheight, owidth, oheight, pad_l, pad_r, pad_t, pad_b);
    }
  });
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

  at::parallel_for(0, nslices, 0, [&](int64_t start, int64_t end) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int64_t ip_x, ip_y;
    for (auto k = start; k < end; k++)
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
  });
}

template <typename scalar_t>
static void replication_pad2d_backward_out_batch(
    scalar_t *ginput_data, scalar_t *goutput_data,
    int64_t nslices,
    int64_t iwidth, int64_t iheight,
    int64_t owidth, int64_t oheight,
    int pad_l, int pad_r,
    int pad_t, int pad_b,
    int nbatch)
{
  at::parallel_for(0, nbatch, 0, [&](int64_t start, int64_t end) {
    for (auto p = start; p < end; p++)
    {
      scalar_t *ginput_p = ginput_data + p * nslices * iheight * iwidth;
      scalar_t *goutput_p = goutput_data + p * nslices * oheight * owidth;
      replication_pad2d_backward_out_frame(ginput_p, goutput_p, nslices,
          iwidth, iheight, owidth, oheight, pad_l, pad_r, pad_t, pad_b);
    }
  });
}

Tensor& replication_pad2d_backward_out_cpu_template(
    Tensor& gradInput,
    const Tensor& gradOutput_,
    const Tensor& input,
    IntArrayRef paddingSize)
{
  TORCH_CHECK(paddingSize.size() == 4, "padding size is expected to be 4");
  int pad_l = paddingSize[0];
  int pad_r = paddingSize[1];
  int pad_t = paddingSize[2];
  int pad_b = paddingSize[3];
  int dimw = 2;
  int dimh = 1;
  int dimslices = 0;
  int64_t nbatch = 1;

  if (input.dim() == 4)
  {
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

  TORCH_CHECK(owidth == gradOutput_.size(dimw),
      "gradOutput width unexpected. Expected: ", owidth, ", Got: ",
      gradOutput_.size(dimw));
  TORCH_CHECK(oheight == gradOutput_.size(dimh),
      "gradOutput height unexpected. Expected: ", oheight, ", Got: ",
      gradOutput_.size(dimh));

  /* get contiguous gradOutput */
  auto gradOutput = gradOutput_.contiguous();

  /* resize */
  gradInput.resize_as_(input);
  if (gradInput.numel() == 0) {
    return gradInput;
  }

  gradInput.zero_();

  /* backprop */
  if (input.dim() == 3)
  {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      input.scalar_type(), "replication_pad2d_backward_cpu", [&] {
      replication_pad2d_backward_out_frame<scalar_t>(
        gradInput.data_ptr<scalar_t>(),
        gradOutput.data_ptr<scalar_t>(),
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
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      input.scalar_type(), "replication_pad2d_backward_cpu", [&] {
      replication_pad2d_backward_out_batch<scalar_t>(
        gradInput.data_ptr<scalar_t>(),
        gradOutput.data_ptr<scalar_t>(),
        nslices,
        iwidth, iheight,
        owidth, oheight,
        pad_l, pad_r,
        pad_t, pad_b,
        nbatch);
      }
    );
  }
  return gradInput;
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

  at::parallel_for(0, nslices, 0, [&](int64_t start, int64_t end) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int64_t ip_x, ip_y, ip_z;
    for (auto k = start; k < end; k++) {
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
  });
}

template <typename scalar_t>
static void replication_pad3d_out_batch(
    scalar_t *input_data, scalar_t *output_data,
    int64_t nslices,
    int64_t iwidth, int64_t iheight, int64_t idepth,
    int64_t owidth, int64_t oheight, int64_t odepth,
    int pleft, int pright,
    int ptop, int pbottom,
    int pfront, int pback,
    int nbatch)
{
  at::parallel_for(0, nbatch, 0, [&](int64_t start, int64_t end) {
    for (auto p = start; p < end; p++)
    {
      scalar_t *input_p = input_data + p * nslices * iwidth * iheight * idepth;
      scalar_t *output_p = output_data + p * nslices * owidth * oheight * odepth;
      replication_pad3d_out_frame(input_p, output_p, nslices,
          iwidth, iheight, idepth, owidth, oheight, odepth,
          pleft, pright, ptop, pbottom, pfront, pback);
    }
  });
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

  at::parallel_for(0, nslices, 0, [&](int64_t start, int64_t end) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int64_t ip_x, ip_y, ip_z;
    for (auto k = start; k < end; k++) {
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
  });
}

template <typename scalar_t>
static void replication_pad3d_backward_out_batch(
    scalar_t *ginput_data, scalar_t *goutput_data,
    int64_t nslices,
    int64_t iwidth, int64_t iheight, int64_t idepth,
    int64_t owidth, int64_t oheight, int64_t odepth,
    int pleft, int pright,
    int ptop, int pbottom,
    int pfront, int pback,
    int nbatch)
{
  at::parallel_for(0, nbatch, 0, [&](int64_t start, int64_t end) {
    for (auto p = start; p < end; p++)
    {
      scalar_t *ginput_p = ginput_data + p * nslices * idepth * iheight * iwidth;
      scalar_t *goutput_p = goutput_data + p * nslices * odepth * oheight * owidth;
      replication_pad3d_backward_out_frame(ginput_p, goutput_p, nslices,
          iwidth, iheight, idepth, owidth, oheight, odepth,
          pleft, pright, ptop, pbottom, pfront, pback);
    }
  });
}

Tensor& replication_pad3d_backward_out_cpu_template(
    Tensor& gradInput,
    const Tensor& gradOutput_,
    const Tensor& input,
    IntArrayRef paddingSize)
{
  TORCH_CHECK(paddingSize.size() == 6, "padding size is expected to be 6");
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

  if (input.dim() == 5)
  {
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


  shapeCheck3d(input, pleft, pright,
      ptop, pbottom, pfront, pback);

  /* get contiguous gradOutput */
  auto gradOutput = gradOutput_.contiguous();

  /* resize */
  gradInput.resize_as_(input);
  if (gradInput.numel() == 0) {
    return gradInput;
  }
  gradInput.zero_();

  /* backprop */
  if (input.dim() == 4)
  {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      input.scalar_type(), "replication_pad3d_backward_cpu", [&] {
      replication_pad3d_backward_out_frame<scalar_t> (
        gradInput.data_ptr<scalar_t>(),
        gradOutput.data_ptr<scalar_t>(),
        nslices,
        iwidth, iheight, idepth,
        owidth, oheight, odepth,
        pleft, pright,
        ptop, pbottom,
        pfront, pback);
      }
    );
  }
  else
  {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      input.scalar_type(), "replication_pad3d_backward_cpu", [&] {
      replication_pad3d_backward_out_batch<scalar_t> (
        gradInput.data_ptr<scalar_t>(),
        gradOutput.data_ptr<scalar_t>(),
        nslices,
        iwidth, iheight, idepth,
        owidth, oheight, odepth,
        pleft, pright,
        ptop, pbottom,
        pfront, pback,
        nbatch);
      }
    );
  }
  return gradInput;
}
} // namespace

TORCH_IMPL_FUNC(replication_pad1d_out_cpu) (
  const Tensor& input_, IntArrayRef paddingSize, const Tensor& output
) {
  constexpr int64_t dimw = -1;
  constexpr int64_t dimslices = -2;

  int64_t pad_l = paddingSize[0];
  int64_t pad_r = paddingSize[1];

  /* get contiguous input */
  auto input = input_.contiguous();

  int64_t nbatch = 1;
  if (input.ndimension() == 3) {
    nbatch = input.size(0);
  }

  /* sizes */
  long nslices = input.size(dimslices);
  long iwidth = input.size(dimw);
  long owidth = output.size(dimw);

  if (input.ndimension() == 2)
  {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "replication_pad1d_cpu", [&] {
      auto input_data = input.data_ptr<scalar_t>();
      auto output_data = output.data_ptr<scalar_t>();
      replication_pad1d_out_frame<scalar_t>(
        input_data,
        output_data,
        nslices,
        iwidth,
        owidth,
        pad_l, pad_r);
      }
    );
  }
  else
  {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "replication_pad1d_cpu", [&] {
      auto input_data = input.data_ptr<scalar_t>();
      auto output_data = output.data_ptr<scalar_t>();
      replication_pad1d_out_batch<scalar_t>(
        input_data,
        output_data,
        nslices,
        iwidth,
        owidth,
        pad_l, pad_r,
        nbatch);
      }
    );
  }
}

TORCH_IMPL_FUNC(replication_pad1d_backward_out_cpu) (
  const Tensor& gradOutput_, const Tensor& input, IntArrayRef paddingSize, const Tensor& gradInput
) {
  int64_t dimw = 1;
  int64_t dimslices = 0;
  int64_t nbatch = 1;
  int64_t pad_l = paddingSize[0];
  int64_t pad_r = paddingSize[1];

  if (input.ndimension() == 3)
  {
    nbatch = input.size(0);
    dimw++;
    dimslices++;
  }

  /* get contiguous gradOutput */
  auto gradOutput = gradOutput_.contiguous();

  /* sizes */
  int64_t nslices = input.size(dimslices);
  int64_t iwidth  = input.size(dimw);
  int64_t owidth  = gradOutput.size(dimw);

  TORCH_CHECK(owidth == gradOutput.size(dimw),
      "gradOutput width unexpected. Expected: ", owidth,
      " Got: ", gradOutput_.size(dimw));

  if (gradInput.numel() == 0) {
    return;
  }

  gradInput.zero_();

  /* backprop */
  if (input.ndimension() == 2)
  {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      input.scalar_type(), "replication_pad1d_backward_cpu", [&] {
      scalar_t *gradInput_data = gradInput.data_ptr<scalar_t>();
      scalar_t *gradOutput_data = gradOutput.data_ptr<scalar_t>();

      replication_pad1d_backward_out_frame<scalar_t> (
        gradInput_data,
        gradOutput_data,
        nslices,
        iwidth,
        owidth,
        pad_l, pad_r);
      }
    );
  }
  else
  {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      input.scalar_type(), "replication_pad1d_backward_cpu", [&] {
      scalar_t *gradInput_data = gradInput.data_ptr<scalar_t>();
      scalar_t *gradOutput_data = gradOutput.data_ptr<scalar_t>();

      replication_pad1d_backward_out_batch<scalar_t> (
        gradInput_data,
        gradOutput_data,
        nslices,
        iwidth,
        owidth,
        pad_l, pad_r,
        nbatch);
      }
    );
  }
}

TORCH_IMPL_FUNC(replication_pad2d_out_cpu) (
  const Tensor& input_, IntArrayRef paddingSize, const Tensor& output
) {
  int64_t pad_l = paddingSize[0];
  int64_t pad_r = paddingSize[1];
  int64_t pad_t = paddingSize[2];
  int64_t pad_b = paddingSize[3];
  int64_t dimw = 2;
  int64_t dimh = 1;
  int64_t dimslices = 0;
  int64_t nbatch = 1;
  if (input_.dim() == 4) {
    nbatch = input_.size(0);
    dimw++;
    dimh++;
    dimslices++;
  }

  int64_t nslices = input_.size(dimslices);
  int64_t iheight = input_.size(dimh);
  int64_t iwidth = input_.size(dimw);
  int64_t oheight = iheight + pad_t + pad_b;
  int64_t owidth  = iwidth + pad_l + pad_r;

  /* get contiguous input */
  auto input = input_.contiguous();

  /* resize output */
  if (input.dim() == 3)
  {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "replication_pad2d_cpu", [&] {
      auto input_data = input.data_ptr<scalar_t>();
      auto output_data = output.data_ptr<scalar_t>();
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
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "replication_pad2d_cpu", [&] {
      auto input_data = input.data_ptr<scalar_t>();
      auto output_data = output.data_ptr<scalar_t>();
      replication_pad2d_out_batch<scalar_t> (input_data, output_data,
        nslices,
        iwidth, iheight,
        owidth, oheight,
        pad_l, pad_r,
        pad_t, pad_b,
        nbatch);
      }
    );
  }
}

Tensor& replication_pad2d_backward_out_cpu(const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef paddingSize,
    Tensor& gradInput)
{
  replication_pad2d_backward_out_cpu_template(
      gradInput, gradOutput, input, paddingSize);
  return gradInput;
}

Tensor replication_pad2d_backward_cpu(
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef paddingSize)
{
  auto gradInput = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  replication_pad2d_backward_out_cpu_template(
      gradInput, gradOutput, input, paddingSize);
  return gradInput;
}

TORCH_IMPL_FUNC(replication_pad3d_out_cpu) (
  const Tensor& input_, IntArrayRef paddingSize, const Tensor& output
) {
  int64_t pleft = paddingSize[0];
  int64_t pright = paddingSize[1];
  int64_t ptop = paddingSize[2];
  int64_t pbottom = paddingSize[3];
  int64_t pfront = paddingSize[4];
  int64_t pback = paddingSize[5];
  int64_t dimw = 3;
  int64_t dimh = 2;
  int64_t dimd = 1;
  int64_t dimslices = 0;
  int64_t nbatch = 1;

  /* get contiguous input */
  auto input = input_.contiguous();

  if (input.dim() == 5) {
    nbatch = input.size(0);
    dimw++;
    dimh++;
    dimd++;
    dimslices++;
  }

  /* sizes */
  int64_t nslices = input.size(dimslices);
  int64_t idepth  = input.size(dimd);
  int64_t iheight = input.size(dimh);
  int64_t iwidth  = input.size(dimw);
  int64_t odepth  = output.size(dimd);
  int64_t oheight = output.size(dimh);
  int64_t owidth  = output.size(dimw);

  /* resize output */
  if (input.dim() == 4) {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "replication_pad3d_cpu", [&] {
      auto input_data = input.data_ptr<scalar_t>();
      auto output_data = output.data_ptr<scalar_t>();
      replication_pad3d_out_frame<scalar_t>(
        input_data, output_data, nslices, iwidth, iheight, idepth,
        owidth, oheight, odepth, pleft, pright, ptop, pbottom, pfront,
        pback);
      }
    );
  }
  else
  {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "replication_pad3d_cpu", [&] {
      auto input_data = input.data_ptr<scalar_t>();
      auto output_data = output.data_ptr<scalar_t>();
      replication_pad3d_out_batch<scalar_t>(
        input_data, output_data, nslices, iwidth, iheight, idepth,
        owidth, oheight, odepth, pleft, pright, ptop, pbottom, pfront,
        pback,
        nbatch);
      }
    );
  }
}

Tensor& replication_pad3d_backward_out_cpu(const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef paddingSize,
    Tensor& gradInput)
{
  replication_pad3d_backward_out_cpu_template(
      gradInput, gradOutput, input, paddingSize);
  return gradInput;
}

Tensor replication_pad3d_backward_cpu(
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef paddingSize)
{
  auto gradInput = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  replication_pad3d_backward_out_cpu_template(
      gradInput, gradOutput, input, paddingSize);
  return gradInput;
}
} // at::native
} // at
