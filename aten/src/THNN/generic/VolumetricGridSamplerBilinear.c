#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/VolumetricGridSamplerBilinear.c"
#else

#undef MIN
#define MIN(a,b) ( ((a)<(b)) ? (a) : (b) )
#undef MAX
#define MAX(a,b) ( ((a)>(b)) ? (a) : (b) )

#undef MODE_BORDER
#define MODE_BORDER 1

static inline void THNN_(VolumetricGridSamplerBilinear_shapeCheck)
     (THTensor *input, THTensor *grid, THTensor *gradOutput) {
  THNN_ARGCHECK(input->nDimension == 5, 2, input,
    "5D input tensor expected but got: %s");
  THNN_ARGCHECK(grid->nDimension == 5, 2, grid,
    "5D grid tensor expected but got: %s");

  int nbatch   = THTensor_(size)(input, 0);
  int channels = THTensor_(size)(input, 1);
  int odepth    = THTensor_(size)(grid, 1);
  int oheight   = THTensor_(size)(grid, 2);
  int owidth    = THTensor_(size)(grid, 3);

  THNN_CHECK_DIM_SIZE(grid, 5, 0, nbatch);
  THNN_CHECK_DIM_SIZE(grid, 5, 4, 3);

  if (gradOutput != NULL) {
    THNN_CHECK_DIM_SIZE(gradOutput, 5, 0, nbatch);
    THNN_CHECK_DIM_SIZE(gradOutput, 5, 1, channels);
    THNN_CHECK_DIM_SIZE(gradOutput, 5, 2, odepth);
    THNN_CHECK_DIM_SIZE(gradOutput, 5, 3, oheight);
    THNN_CHECK_DIM_SIZE(gradOutput, 5, 4, owidth);
  }
}

#define SAFE_GET(input, x, y, z, n, c, D, H, W) \
  x >= 0 && x < W && y >=0 && y < H && z >= 0 && z < D \
    ? THTensor_fastGet5d(input, n, c, z, y, x) : 0

#define CLIP_COORDINATES(in, out, clip_limit) out = MIN((clip_limit-1), MAX(in, 0))

TH_API void THNN_(VolumetricGridSamplerBilinear_updateOutput)(
    THNNState *state,
    THTensor *input,
    THTensor *grid,
    THTensor *output,
    int padding_mode) {

  THNN_(VolumetricGridSamplerBilinear_shapeCheck)(input, grid, NULL);
  int N = THTensor_(size)(input, 0);
  int C = THTensor_(size)(input, 1);
  int ID = THTensor_(size)(input, 2);
  int IH = THTensor_(size)(input, 3);
  int IW = THTensor_(size)(input, 4);
  int D = THTensor_(size)(grid, 1);
  int H = THTensor_(size)(grid, 2);
  int W = THTensor_(size)(grid, 3);

  // resize output to the same shape as input
  THTensor_(resize5d)(output, N, C, D, H, W);

  // loop over each output pixel
  int n, d, h, w, c;
#pragma omp parallel for private(n, d, h, w, c)
  for (n = 0; n < N; ++n) {
    for (d = 0; d < D; ++d) {
      for (h = 0; h < H; ++h) {
        for (w = 0; w < W; ++w) {
          // get the corresponding input x, y, z co-ordinates from grid
          real ix = THTensor_fastGet5d(grid, n, d, h, w, 0);
          real iy = THTensor_fastGet5d(grid, n, d, h, w, 1);
          real iz = THTensor_fastGet5d(grid, n, d, h, w, 2);

          // normalize ix, iy, iz from [-1, 1] to [0, IW-1] & [0, IH-1] & [0, ID-1]
          ix = ((ix + 1) / 2) * (IW-1);
          iy = ((iy + 1) / 2) * (IH-1);
          iz = ((iz + 1) / 2) * (ID-1);

          // get corner pixel values from (x, y, z)
          // for 4d, we used north-east-south-west
          // for 5d, we add top-bottom
          int ix_tnw = floor(ix);
          int iy_tnw = floor(iy);
          int iz_tnw = floor(iz);

          int ix_tne = ix_tnw + 1;
          int iy_tne = iy_tnw;
          int iz_tne = iz_tnw;

          int ix_tsw = ix_tnw;
          int iy_tsw = iy_tnw + 1;
          int iz_tsw = iz_tnw;

          int ix_tse = ix_tnw + 1;
          int iy_tse = iy_tnw + 1;
          int iz_tse = iz_tnw;

          int ix_bnw = ix_tnw;
          int iy_bnw = iy_tnw;
          int iz_bnw = iz_tnw + 1;

          int ix_bne = ix_tnw + 1;
          int iy_bne = iy_tnw;
          int iz_bne = iz_tnw + 1;

          int ix_bsw = ix_tnw;
          int iy_bsw = iy_tnw + 1;
          int iz_bsw = iz_tnw + 1;

          int ix_bse = ix_tnw + 1;
          int iy_bse = iy_tnw + 1;
          int iz_bse = iz_tnw + 1;

          // get surfaces to each neighbor:
          real tnw = (ix_bse - ix)    * (iy_bse - iy)    * (iz_bse - iz);
          real tne = (ix    - ix_bsw) * (iy_bsw - iy)    * (iz_bsw - iz);
          real tsw = (ix_bne - ix)    * (iy    - iy_bne) * (iz_bne - iz);
          real tse = (ix    - ix_bnw) * (iy    - iy_bnw) * (iz_bnw - iz);
          real bnw = (ix_tse - ix)    * (iy_tse - iy)    * (iz - iz_tse);
          real bne = (ix    - ix_tsw) * (iy_tsw - iy)    * (iz - iz_tsw);
          real bsw = (ix_tne - ix)    * (iy    - iy_tne) * (iz - iz_tne);
          real bse = (ix    - ix_tnw) * (iy    - iy_tnw) * (iz - iz_tnw);

          if (padding_mode==MODE_BORDER){
            // clip coordinates to image borders
            CLIP_COORDINATES(ix_tnw, ix_tnw, IW);
            CLIP_COORDINATES(iy_tnw, iy_tnw, IH);
            CLIP_COORDINATES(iz_tnw, iz_tnw, ID);
            CLIP_COORDINATES(ix_tne, ix_tne, IW);
            CLIP_COORDINATES(iy_tne, iy_tne, IH);
            CLIP_COORDINATES(iz_tne, iz_tne, ID);
            CLIP_COORDINATES(ix_tsw, ix_tsw, IW);
            CLIP_COORDINATES(iy_tsw, iy_tsw, IH);
            CLIP_COORDINATES(iz_tsw, iz_tsw, ID);
            CLIP_COORDINATES(ix_tse, ix_tse, IW);
            CLIP_COORDINATES(iy_tse, iy_tse, IH);
            CLIP_COORDINATES(iz_tse, iz_tse, ID);
            CLIP_COORDINATES(ix_bnw, ix_bnw, IW);
            CLIP_COORDINATES(iy_bnw, iy_bnw, IH);
            CLIP_COORDINATES(iz_bnw, iz_bnw, ID);
            CLIP_COORDINATES(ix_bne, ix_bne, IW);
            CLIP_COORDINATES(iy_bne, iy_bne, IH);
            CLIP_COORDINATES(iz_bne, iz_bne, ID);
            CLIP_COORDINATES(ix_bsw, ix_bsw, IW);
            CLIP_COORDINATES(iy_bsw, iy_bsw, IH);
            CLIP_COORDINATES(iz_bsw, iz_bsw, ID);
            CLIP_COORDINATES(ix_bse, ix_bse, IW);
            CLIP_COORDINATES(iy_bse, iy_bse, IH);
            CLIP_COORDINATES(iz_bse, iz_bse, ID);
          }

          // calculate bilinear weighted pixel value and set output pixel
          for (c = 0; c < C; ++c) {
            //   (c, iy_nw, ix_nw) * nw + (c, iy_ne, ix_ne) * ne
            // + (c, iy_sw, ix_sw) * sw + (c, iy_se, ix_se) * se
            real tnw_val = SAFE_GET(input, ix_tnw, iy_tnw, iz_tnw, n, c, ID, IH, IW);
            real tne_val = SAFE_GET(input, ix_tne, iy_tne, iz_tne, n, c, ID, IH, IW);
            real tsw_val = SAFE_GET(input, ix_tsw, iy_tsw, iz_tsw, n, c, ID, IH, IW);
            real tse_val = SAFE_GET(input, ix_tse, iy_tse, iz_tse, n, c, ID, IH, IW);
            real bnw_val = SAFE_GET(input, ix_bnw, iy_bnw, iz_bnw, n, c, ID, IH, IW);
            real bne_val = SAFE_GET(input, ix_bne, iy_bne, iz_bne, n, c, ID, IH, IW);
            real bsw_val = SAFE_GET(input, ix_bsw, iy_bsw, iz_bsw, n, c, ID, IH, IW);
            real bse_val = SAFE_GET(input, ix_bse, iy_bse, iz_bse, n, c, ID, IH, IW);
            real out_val = tnw_val * tnw + tne_val * tne + tsw_val * tsw + tse_val * tse +
              bnw_val * bnw + bne_val * bne + bsw_val * bsw + bse_val * bse;
            THTensor_fastSet5d(output, n, c, d, h, w, out_val);
          }
        }
      }
    }
  }
}

#define SAFE_ADD(input, x, y, z, n, c, D, H, W, value)  \
  do {                                                                  \
    if (x >= 0 && x < W && y >=0 && y < H && z >=0 && z < D) {          \
      real old_value = THTensor_fastGet5d(input, n, c, z, y, x);        \
      THTensor_fastSet5d(input, n, c, z, y, x, value + old_value);      \
    }                                                                   \
  } while(0)

TH_API void THNN_(VolumetricGridSamplerBilinear_updateGradInput)(
    THNNState *state,
    THTensor *input, THTensor *gradInput,
    THTensor *grid, THTensor *gradGrid,
    THTensor *gradOutput,
    int padding_mode) {

  THNN_(VolumetricGridSamplerBilinear_shapeCheck)(input, grid, gradOutput);
  int N = THTensor_(size)(input, 0);
  int C = THTensor_(size)(input, 1);
  int ID = THTensor_(size)(input, 2);
  int IH = THTensor_(size)(input, 3);
  int IW = THTensor_(size)(input, 4);
  int D = THTensor_(size)(grid, 1);
  int H = THTensor_(size)(grid, 2);
  int W = THTensor_(size)(grid, 3);

  THTensor_(resize5d)(gradInput, N, C, ID, IH, IW);
  THTensor_(resize5d)(gradGrid, N, D, H, W, 3);
  THTensor_(zero)(gradInput);
  THTensor_(zero)(gradGrid);

  // loop over each output pixel
  int n, d, h, w;
//#pragma omp parallel for private(n, d, h, w)
  for (n = 0; n < N; ++n) {
    for (d = 0; d < D; ++d) {
      for (h = 0; h < H; ++h) {
        for (w = 0; w < W; ++w) {
          // get the corresponding input x, y, z co-ordinates from grid
          real ix = THTensor_fastGet5d(grid, n, d, h, w, 0);
          real iy = THTensor_fastGet5d(grid, n, d, h, w, 1);
          real iz = THTensor_fastGet5d(grid, n, d, h, w, 2);

          real gix = 0;
          real giy = 0;
          real giz = 0;

          // normalize ix, iy, iz from [-1, 1] to [0, W-1] & [0, H-1] & [0, D-1]
          ix = ((ix + 1) / 2) * (IW-1);
          iy = ((iy + 1) / 2) * (IH-1);
          iz = ((iz + 1) / 2) * (ID-1);

          // get corner pixel values from (x, y, z)
          // for 4d, we used north-east-south-west
          // for 5d, we add top-bottom
          int ix_tnw = floor(ix);
          int iy_tnw = floor(iy);
          int iz_tnw = floor(iz);

          int ix_tne = ix_tnw + 1;
          int iy_tne = iy_tnw;
          int iz_tne = iz_tnw;

          int ix_tsw = ix_tnw;
          int iy_tsw = iy_tnw + 1;
          int iz_tsw = iz_tnw;

          int ix_tse = ix_tnw + 1;
          int iy_tse = iy_tnw + 1;
          int iz_tse = iz_tnw;

          int ix_bnw = ix_tnw;
          int iy_bnw = iy_tnw;
          int iz_bnw = iz_tnw + 1;

          int ix_bne = ix_tnw + 1;
          int iy_bne = iy_tnw;
          int iz_bne = iz_tnw + 1;

          int ix_bsw = ix_tnw;
          int iy_bsw = iy_tnw + 1;
          int iz_bsw = iz_tnw + 1;

          int ix_bse = ix_tnw + 1;
          int iy_bse = iy_tnw + 1;
          int iz_bse = iz_tnw + 1;

          // get surfaces to each neighbor:
          real tnw = (ix_bse - ix)    * (iy_bse - iy)    * (iz_bse - iz);
          real tne = (ix    - ix_bsw) * (iy_bsw - iy)    * (iz_bsw - iz);
          real tsw = (ix_bne - ix)    * (iy    - iy_bne) * (iz_bne - iz);
          real tse = (ix    - ix_bnw) * (iy    - iy_bnw) * (iz_bnw - iz);
          real bnw = (ix_tse - ix)    * (iy_tse - iy)    * (iz - iz_tse);
          real bne = (ix    - ix_tsw) * (iy_tsw - iy)    * (iz - iz_tsw);
          real bsw = (ix_tne - ix)    * (iy    - iy_tne) * (iz - iz_tne);
          real bse = (ix    - ix_tnw) * (iy    - iy_tnw) * (iz - iz_tnw);

          int ix_tnw_cl, iy_tnw_cl, iz_tnw_cl, ix_tne_cl, iy_tne_cl, iz_tne_cl;
          int ix_tsw_cl, iy_tsw_cl, iz_tsw_cl, ix_tse_cl, iy_tse_cl, iz_tse_cl;
          int ix_bnw_cl, iy_bnw_cl, iz_bnw_cl, ix_bne_cl, iy_bne_cl, iz_bne_cl;
          int ix_bsw_cl, iy_bsw_cl, iz_bsw_cl, ix_bse_cl, iy_bse_cl, iz_bse_cl;

          if (padding_mode==MODE_BORDER){
            // clip coordinates to image borders
            CLIP_COORDINATES(ix_tnw, ix_tnw_cl, IW);
            CLIP_COORDINATES(iy_tnw, iy_tnw_cl, IH);
            CLIP_COORDINATES(iz_tnw, iz_tnw_cl, ID);
            CLIP_COORDINATES(ix_tne, ix_tne_cl, IW);
            CLIP_COORDINATES(iy_tne, iy_tne_cl, IH);
            CLIP_COORDINATES(iz_tne, iz_tne_cl, ID);
            CLIP_COORDINATES(ix_tsw, ix_tsw_cl, IW);
            CLIP_COORDINATES(iy_tsw, iy_tsw_cl, IH);
            CLIP_COORDINATES(iz_tsw, iz_tsw_cl, ID);
            CLIP_COORDINATES(ix_tse, ix_tse_cl, IW);
            CLIP_COORDINATES(iy_tse, iy_tse_cl, IH);
            CLIP_COORDINATES(iz_tse, iz_tse_cl, ID);
            CLIP_COORDINATES(ix_bnw, ix_bnw_cl, IW);
            CLIP_COORDINATES(iy_bnw, iy_bnw_cl, IH);
            CLIP_COORDINATES(iz_bnw, iz_bnw_cl, ID);
            CLIP_COORDINATES(ix_bne, ix_bne_cl, IW);
            CLIP_COORDINATES(iy_bne, iy_bne_cl, IH);
            CLIP_COORDINATES(iz_bne, iz_bne_cl, ID);
            CLIP_COORDINATES(ix_bsw, ix_bsw_cl, IW);
            CLIP_COORDINATES(iy_bsw, iy_bsw_cl, IH);
            CLIP_COORDINATES(iz_bsw, iz_bsw_cl, ID);
            CLIP_COORDINATES(ix_bse, ix_bse_cl, IW);
            CLIP_COORDINATES(iy_bse, iy_bse_cl, IH);
            CLIP_COORDINATES(iz_bse, iz_bse_cl, ID);
          }
          else {
            ix_tnw_cl = ix_tnw;
            iy_tnw_cl = iy_tnw;
            iz_tnw_cl = iz_tnw;
            ix_tne_cl = ix_tne;
            iy_tne_cl = iy_tne;
            iz_tne_cl = iz_tne;
            ix_tsw_cl = ix_tsw;
            iy_tsw_cl = iy_tsw;
            iz_tsw_cl = iz_tsw;
            ix_tse_cl = ix_tse;
            iy_tse_cl = iy_tse;
            iz_tse_cl = iz_tse;
            ix_bnw_cl = ix_bnw;
            iy_bnw_cl = iy_bnw;
            iz_bnw_cl = iz_bnw;
            ix_bne_cl = ix_bne;
            iy_bne_cl = iy_bne;
            iz_bne_cl = iz_bne;
            ix_bsw_cl = ix_bsw;
            iy_bsw_cl = iy_bsw;
            iz_bsw_cl = iz_bsw;
            ix_bse_cl = ix_bse;
            iy_bse_cl = iy_bse;
            iz_bse_cl = iz_bse;
          }

          for (int c = 0; c < C; ++c) {
            real gradout = THTensor_fastGet5d(gradOutput, n, c, d, h, w);

            // calculate and set gradInput
            SAFE_ADD(gradInput, ix_tnw_cl, iy_tnw_cl, iz_tnw_cl, n, c, ID, IH, IW, tnw * gradout);
            SAFE_ADD(gradInput, ix_tne_cl, iy_tne_cl, iz_tne_cl, n, c, ID, IH, IW, tne * gradout);
            SAFE_ADD(gradInput, ix_tsw_cl, iy_tsw_cl, iz_tsw_cl, n, c, ID, IH, IW, tsw * gradout);
            SAFE_ADD(gradInput, ix_tse_cl, iy_tse_cl, iz_tse_cl, n, c, ID, IH, IW, tse * gradout);
            SAFE_ADD(gradInput, ix_bnw_cl, iy_bnw_cl, iz_bnw_cl, n, c, ID, IH, IW, bnw * gradout);
            SAFE_ADD(gradInput, ix_bne_cl, iy_bne_cl, iz_bne_cl, n, c, ID, IH, IW, bne * gradout);
            SAFE_ADD(gradInput, ix_bsw_cl, iy_bsw_cl, iz_bsw_cl, n, c, ID, IH, IW, bsw * gradout);
            SAFE_ADD(gradInput, ix_bse_cl, iy_bse_cl, iz_bse_cl, n, c, ID, IH, IW, bse * gradout);

            // calculate gradGrid
            real tnw_val = SAFE_GET(input, ix_tnw_cl, iy_tnw_cl, iz_tnw_cl, n, c, ID, IH, IW);
            real tne_val = SAFE_GET(input, ix_tne_cl, iy_tne_cl, iz_tne_cl, n, c, ID, IH, IW);
            real tsw_val = SAFE_GET(input, ix_tsw_cl, iy_tsw_cl, iz_tsw_cl, n, c, ID, IH, IW);
            real tse_val = SAFE_GET(input, ix_tse_cl, iy_tse_cl, iz_tse_cl, n, c, ID, IH, IW);
            real bnw_val = SAFE_GET(input, ix_bnw_cl, iy_bnw_cl, iz_bnw_cl, n, c, ID, IH, IW);
            real bne_val = SAFE_GET(input, ix_bne_cl, iy_bne_cl, iz_bne_cl, n, c, ID, IH, IW);
            real bsw_val = SAFE_GET(input, ix_bsw_cl, iy_bsw_cl, iz_bsw_cl, n, c, ID, IH, IW);
            real bse_val = SAFE_GET(input, ix_bse_cl, iy_bse_cl, iz_bse_cl, n, c, ID, IH, IW);

            gix -= tnw_val * (iy_bse - iy) * (iz_bse - iz) * gradout;
            gix += tne_val * (iy_bsw - iy) * (iz_bsw - iz) * gradout;
            gix -= tsw_val * (iy - iy_bne) * (iz_bne - iz) * gradout;
            gix += tse_val * (iy - iy_bnw) * (iz_bnw - iz) * gradout;
            gix -= bnw_val * (iy_tse - iy) * (iz - iz_tse) * gradout;
            gix += bne_val * (iy_tsw - iy) * (iz - iz_tsw) * gradout;
            gix -= bsw_val * (iy - iy_tne) * (iz - iz_tne) * gradout;
            gix += bse_val * (iy - iy_tnw) * (iz - iz_tnw) * gradout;


            giy -= tnw_val * (ix_bse - ix)    * (iz_bse - iz) * gradout;
            giy -= tne_val * (ix    - ix_bsw) * (iz_bsw - iz) * gradout;
            giy += tsw_val * (ix_bne - ix)    * (iz_bne - iz) * gradout;
            giy += tse_val * (ix    - ix_bnw) * (iz_bnw - iz) * gradout;
            giy -= bnw_val * (ix_tse - ix)    * (iz - iz_tse) * gradout;
            giy -= bne_val * (ix    - ix_tsw) * (iz - iz_tsw) * gradout;
            giy += bsw_val * (ix_tne - ix)    * (iz - iz_tne) * gradout;
            giy += bse_val * (ix    - ix_tnw) * (iz - iz_tnw) * gradout;

            giz -= tnw_val * (ix_bse - ix)    * (iy_bse - iy)    * gradout;
            giz -= tne_val * (ix    - ix_bsw) * (iy_bsw - iy)    * gradout;
            giz -= tsw_val * (ix_bne - ix)    * (iy    - iy_bne) * gradout;
            giz -= tse_val * (ix    - ix_bnw) * (iy    - iy_bnw) * gradout;
            giz += bnw_val * (ix_tse - ix)    * (iy_tse - iy)    * gradout;
            giz += bne_val * (ix    - ix_tsw) * (iy_tsw - iy)    * gradout;
            giz += bsw_val * (ix_tne - ix)    * (iy    - iy_tne) * gradout;
            giz += bse_val * (ix    - ix_tnw) * (iy    - iy_tnw) * gradout;

          }

          // un-normalize gradGrid values back to [-1, 1] constraints
          gix = gix * (IW - 1) / 2;
          giy = giy * (IH - 1) / 2;
          giz = giz * (ID - 1) / 2;

          real gix_old = THTensor_fastGet5d(gradGrid, n, d, h, w, 0);
          real giy_old = THTensor_fastGet5d(gradGrid, n, d, h, w, 1);
          real giz_old = THTensor_fastGet5d(gradGrid, n, d, h, w, 2);

          THTensor_fastSet5d(gradGrid, n, d, h, w, 0, gix_old + gix);
          THTensor_fastSet5d(gradGrid, n, d, h, w, 1, giy_old + giy);
          THTensor_fastSet5d(gradGrid, n, d, h, w, 2, giz_old + giz);
        }
      }
    }
  }
}

#undef MIN
#undef MAX
#undef SAFE_GET
#undef CLIP_COORDINATES
#undef SAFE_ADD
#undef MODE_BORDER

#endif
