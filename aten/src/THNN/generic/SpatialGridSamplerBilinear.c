#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialGridSamplerBilinear.c"
#else

#undef MIN
#define MIN(a,b) ( ((a)<(b)) ? (a) : (b) )
#undef MAX
#define MAX(a,b) ( ((a)>(b)) ? (a) : (b) )

#undef MODE_BORDER
#define MODE_BORDER 1

static inline void THNN_(SpatialGridSamplerBilinear_shapeCheck)
     (THTensor *input, THTensor *grid, THTensor *gradOutput) {
  THNN_ARGCHECK(input->nDimension == 4, 2, input,
    "4D input tensor expected but got: %s");
  THNN_ARGCHECK(grid->nDimension == 4, 2, grid,
    "4D grid tensor expected but got: %s");

  int nbatch   = THTensor_(size)(input, 0);
  int channels = THTensor_(size)(input, 1);
  int oheight   = THTensor_(size)(grid, 1);
  int owidth    = THTensor_(size)(grid, 2);

  THNN_CHECK_DIM_SIZE(grid, 4, 0, nbatch);
  THNN_CHECK_DIM_SIZE(grid, 4, 3, 2);

  if (gradOutput != NULL) {
    THNN_CHECK_DIM_SIZE(gradOutput, 4, 0, nbatch);
    THNN_CHECK_DIM_SIZE(gradOutput, 4, 1, channels);
    THNN_CHECK_DIM_SIZE(gradOutput, 4, 2, oheight);
    THNN_CHECK_DIM_SIZE(gradOutput, 4, 3, owidth);
  }
}

#define SAFE_GET(input, x, y, n, c, H, W) x >= 0 && x < W && y >=0 \
    && y < H ? THTensor_fastGet4d(input, n, c, y, x) : 0

#define CLIP_COORDINATES(in, out, clip_limit) out = MIN((clip_limit-1), MAX(in, 0))

TH_API void THNN_(SpatialGridSamplerBilinear_updateOutput)(
    THNNState *state,
    THTensor *input,
    THTensor *grid,
    THTensor *output,
    int padding_mode) {

  THNN_(SpatialGridSamplerBilinear_shapeCheck)(input, grid, NULL);
  int N = THTensor_(size)(input, 0);
  int C = THTensor_(size)(input, 1);
  int IH = THTensor_(size)(input, 2);
  int IW = THTensor_(size)(input, 3);
  int H = THTensor_(size)(grid, 1);
  int W = THTensor_(size)(grid, 2);

  // resize output to the same shape as input
  THTensor_(resize4d)(output, N, C, H, W);

  // loop over each output pixel
  int n, h, w, c;
#pragma omp parallel for private(n, h, w, c)
  for (n = 0; n < N; ++n) {
    for (h = 0; h < H; ++h) {
      for (w = 0; w < W; ++w) {
        // get the corresponding input x, y co-ordinates from grid
        real ix = THTensor_fastGet4d(grid, n, h, w, 0);
        real iy = THTensor_fastGet4d(grid, n, h, w, 1);

        // normalize ix, iy from [-1, 1] to [0, IH-1] & [0, IW-1]
        ix = ((ix + 1) / 2) * (IW-1);
        iy = ((iy + 1) / 2) * (IH-1);

        // get NE, NW, SE, SW pixel values from (x, y)
        int ix_nw = floor(ix);
        int iy_nw = floor(iy);
        int ix_ne = ix_nw + 1;
        int iy_ne = iy_nw;
        int ix_sw = ix_nw;
        int iy_sw = iy_nw + 1;
        int ix_se = ix_nw + 1;
        int iy_se = iy_nw + 1;

        // get surfaces to each neighbor:
        real nw = (ix_se - ix)    * (iy_se - iy);
        real ne = (ix    - ix_sw) * (iy_sw - iy);
        real sw = (ix_ne - ix)    * (iy    - iy_ne);
        real se = (ix    - ix_nw) * (iy    - iy_nw);

        if (padding_mode==MODE_BORDER){
          // clip coordinates to image borders
          CLIP_COORDINATES(ix_nw, ix_nw, IW);
          CLIP_COORDINATES(iy_nw, iy_nw, IH);
          CLIP_COORDINATES(ix_ne, ix_ne, IW);
          CLIP_COORDINATES(iy_ne, iy_ne, IH);
          CLIP_COORDINATES(ix_sw, ix_sw, IW);
          CLIP_COORDINATES(iy_sw, iy_sw, IH);
          CLIP_COORDINATES(ix_se, ix_se, IW);
          CLIP_COORDINATES(iy_se, iy_se, IH);
        }

        // calculate bilinear weighted pixel value and set output pixel
        for (c = 0; c < C; ++c) {
          //   (c, iy_nw, ix_nw) * nw + (c, iy_ne, ix_ne) * ne
          // + (c, iy_sw, ix_sw) * sw + (c, iy_se, ix_se) * se
          real nw_val = SAFE_GET(input, ix_nw, iy_nw, n, c, IH, IW);
          real ne_val = SAFE_GET(input, ix_ne, iy_ne, n, c, IH, IW);
          real sw_val = SAFE_GET(input, ix_sw, iy_sw, n, c, IH, IW);
          real se_val = SAFE_GET(input, ix_se, iy_se, n, c, IH, IW);
          real out_val = nw_val * nw + ne_val * ne + sw_val * sw + se_val * se;
          THTensor_fastSet4d(output, n, c, h, w, out_val);
        }
      }
    }
  }
}

#define SAFE_ADD(input, x, y, n, c, H, W, value)    \
  do {                \
    if (x >= 0 && x < W && y >=0 && y < H) {      \
      real old_value = THTensor_fastGet4d(input, n, c, y, x); \
      THTensor_fastSet4d(input, n, c, y, x, value + old_value); \
    }               \
  } while(0)

TH_API void THNN_(SpatialGridSamplerBilinear_updateGradInput)(
    THNNState *state,
    THTensor *input, THTensor *gradInput,
    THTensor *grid, THTensor *gradGrid,
    THTensor *gradOutput,
    int padding_mode) {

  THNN_(SpatialGridSamplerBilinear_shapeCheck)(input, grid, gradOutput);
  int N = THTensor_(size)(input, 0);
  int C = THTensor_(size)(input, 1);
  int IH = THTensor_(size)(input, 2);
  int IW = THTensor_(size)(input, 3);
  int H = THTensor_(size)(grid, 1);
  int W = THTensor_(size)(grid, 2);

  THTensor_(resize4d)(gradInput, N, C, IH, IW);
  THTensor_(resize4d)(gradGrid, N, H, W, 2);
  THTensor_(zero)(gradInput);
  THTensor_(zero)(gradGrid);

  // loop over each output pixel
  int n, h, w;
#pragma omp parallel for private(n, h, w)
  for (n = 0; n < N; ++n) {
    for (h = 0; h < H; ++h) {
      for (w = 0; w < W; ++w) {
        // get the corresponding input x, y co-ordinates from grid
        real ix = THTensor_fastGet4d(grid, n, h, w, 0);
        real iy = THTensor_fastGet4d(grid, n, h, w, 1);

        real gix = 0;
        real giy = 0;

        // normalize ix, iy from [-1, 1] to [0, H-1] & [0, W-1]
        ix = ((ix + 1) / 2) * (IW-1);
        iy = ((iy + 1) / 2) * (IH-1);

        // get NE, NW, SE, SW pixel values from (x, y)
        int ix_nw = floor(ix);
        int iy_nw = floor(iy);
        int ix_ne = ix_nw + 1;
        int iy_ne = iy_nw;
        int ix_sw = ix_nw;
        int iy_sw = iy_nw + 1;
        int ix_se = ix_nw + 1;
        int iy_se = iy_nw + 1;

        // get surfaces to each neighbor:
        real nw = (ix_se - ix)    * (iy_se - iy);
        real ne = (ix    - ix_sw) * (iy_sw - iy);
        real sw = (ix_ne - ix)    * (iy    - iy_ne);
        real se = (ix    - ix_nw) * (iy    - iy_nw);

        int ix_nw_cl, iy_nw_cl, ix_ne_cl, iy_ne_cl, ix_sw_cl, iy_sw_cl, ix_se_cl, iy_se_cl;

        if (padding_mode==MODE_BORDER){
          // get clipped NE, NW, SE, SW pixel values from (x, y)
          CLIP_COORDINATES(ix_nw, ix_nw_cl, IW);
          CLIP_COORDINATES(iy_nw, iy_nw_cl, IH);
          CLIP_COORDINATES(ix_ne, ix_ne_cl, IW);
          CLIP_COORDINATES(iy_ne, iy_ne_cl, IH);
          CLIP_COORDINATES(ix_sw, ix_sw_cl, IW);
          CLIP_COORDINATES(iy_sw, iy_sw_cl, IH);
          CLIP_COORDINATES(ix_se, ix_se_cl, IW);
          CLIP_COORDINATES(iy_se, iy_se_cl, IH);
        }
        else {
          ix_nw_cl = ix_nw;
          iy_nw_cl = iy_nw;
          ix_ne_cl = ix_ne;
          iy_ne_cl = iy_ne;
          ix_sw_cl = ix_sw;
          iy_sw_cl = iy_sw;
          ix_se_cl = ix_se;
          iy_se_cl = iy_se;
        }

        for (int c = 0; c < C; ++c) {
          real gradout = THTensor_fastGet4d(gradOutput, n, c, h, w);

          // calculate and set gradInput
          SAFE_ADD(gradInput, ix_nw_cl, iy_nw_cl, n, c, IH, IW, nw * gradout);
          SAFE_ADD(gradInput, ix_ne_cl, iy_ne_cl, n, c, IH, IW, ne * gradout);
          SAFE_ADD(gradInput, ix_sw_cl, iy_sw_cl, n, c, IH, IW, sw * gradout);
          SAFE_ADD(gradInput, ix_se_cl, iy_se_cl, n, c, IH, IW, se * gradout);

          // calculate gradGrid
          real nw_val = SAFE_GET(input, ix_nw_cl, iy_nw_cl, n, c, IH, IW);
          real ne_val = SAFE_GET(input, ix_ne_cl, iy_ne_cl, n, c, IH, IW);
          real sw_val = SAFE_GET(input, ix_sw_cl, iy_sw_cl, n, c, IH, IW);
          real se_val = SAFE_GET(input, ix_se_cl, iy_se_cl, n, c, IH, IW);

          gix -= nw_val * (iy_se - iy) * gradout;
          gix += ne_val * (iy_sw - iy) * gradout;
          gix -= sw_val * (iy - iy_ne) * gradout;
          gix += se_val * (iy - iy_nw) * gradout;

          giy -= nw_val * (ix_se - ix) * gradout;
          giy -= ne_val * (ix - ix_sw) * gradout;
          giy += sw_val * (ix_ne - ix) * gradout;
          giy += se_val * (ix - ix_nw) * gradout;
        }

        // un-normalize gradGrid values back to [-1, 1] constraints
        gix = gix * (IW - 1) / 2;
        giy = giy * (IH - 1) / 2;

        real gix_old = THTensor_fastGet4d(gradGrid, n, h, w, 0);
        real giy_old = THTensor_fastGet4d(gradGrid, n, h, w, 1);

        THTensor_fastSet4d(gradGrid, n, h, w, 0, gix_old + gix);
        THTensor_fastSet4d(gradGrid, n, h, w, 1, giy_old + giy);
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
