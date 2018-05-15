#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialGridSamplerGaussian.c"
#else

#undef MIN
#define MIN(a,b) ( ((a)<(b)) ? (a) : (b) )
#undef MAX
#define MAX(a,b) ( ((a)>(b)) ? (a) : (b) )

#undef MODE_BORDER
#define MODE_BORDER 1


static inline void THNN_(SpatialGridSamplerGaussian_shapeCheck)
     (THTensor *input, THTensor *grid, THTensor *gradOutput) {
  THNN_ARGCHECK(input->nDimension == 4, 2, input,
    "4D input tensor expected but got: %s");
  THNN_ARGCHECK(grid->nDimension == 4, 2, grid,
    "4D grid tensor expected but got: %s");
  int nbatch      = THTensor_(size)(input, 0);
  int channels    = THTensor_(size)(input, 1);
  int oheight     = THTensor_(size)(grid, 1);
  int owidth      = THTensor_(size)(grid, 2);

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

#define NORMAL_CDF(val, mu, sigma) 0.5 * erfc(-((val) - (mu)) / ((M_SQRT2) * (sigma)))

#define NORMAL_PDF(val, mu, sigma) exp(- 0.5 * pow(((val) - (mu)) / (sigma), 2)) / (sqrt(2 * (M_PI)) * (sigma))

#define SAFE_ADD(input, x, y, n, c, H, W, value)    \
  do {                \
    if (x >= 0 && x < W && y >=0 && y < H) {      \
      real old_value = THTensor_fastGet4d(input, n, c, y, x); \
      THTensor_fastSet4d(input, n, c, y, x, value + old_value); \
    }               \
  } while(0)

TH_API void THNN_(SpatialGridSamplerGaussian_updateOutput)(
    THNNState *state,
    THTensor *input,
    THTensor *grid,
    THTensor *output,
    int kernel_size,
    real kernel_std,
    int padding_mode) {

  THNN_(SpatialGridSamplerGaussian_shapeCheck)(input, grid, NULL);
  int N = THTensor_(size)(input, 0);
  int C = THTensor_(size)(input, 1);
  int IH = THTensor_(size)(input, 2);
  int IW = THTensor_(size)(input, 3);
  int H = THTensor_(size)(grid, 1);
  int W = THTensor_(size)(grid, 2);


  THTensor_(resize4d)(output, N, C, H, W);
  THTensor_(zero)(output);

  // loop over each output pixel
  int n, h, w, c, i, j;
// #pragma omp parallel for private(n, h, w)
  for (n = 0; n < N; ++n) {
    for (h = 0; h < H; ++h) {
      for (w = 0; w < W; ++w) {
        // get the corresponding input x, y co-ordinates from grid
        real ix = THTensor_fastGet4d(grid, n, h, w, 0);
        real iy = THTensor_fastGet4d(grid, n, h, w, 1);

        // normalize ix, iy from [-1, 1] to [0, IH-1] & [0, IW-1]
        ix = ((ix + 1) / 2) * (IW-1);
        iy = ((iy + 1) / 2) * (IH-1);
        // NOTE: (ix, iy) is the antecedant coordinate
        // the coordinate is the exact position of the top left side of the pixel

        // calculating coordinate of nearest pixel with torch.floor(grid + 0.5)
        // calculating coordinate of north west pixel by removing floor((kernel_size - 1) / 2)
        int ix_nw = floor(ix + 0.5) - floor((kernel_size - 1) * 0.5);
        int iy_nw = floor(iy + 0.5) - floor((kernel_size - 1) * 0.5);

        real sumw = 0;

        for(i=0; i < kernel_size; i++){
          for(j=0; j < kernel_size; j++){

            int ix_p = ix_nw + i;
            int iy_p = iy_nw + j;

            int ix_p_cl, iy_p_cl;
            if (padding_mode==MODE_BORDER){
              //  // clip cooridinates to image borders
                CLIP_COORDINATES(ix_p, ix_p_cl, IW);
                CLIP_COORDINATES(iy_p, iy_p_cl, IH);
             }
             else {
              ix_p_cl = ix_p;
              iy_p_cl = iy_p;
             }

            // if antecedant value in image bounds
            if(ix_p_cl >= 0 && ix_p_cl < IW && iy_p_cl >= 0 && iy_p_cl < IH){
              real xw =  NORMAL_CDF(ix_p + 1.0, ix + 0.5, kernel_std) - NORMAL_CDF(ix_p, ix + 0.5, kernel_std);
              real yw =  NORMAL_CDF(iy_p + 1.0, iy + 0.5, kernel_std) - NORMAL_CDF(iy_p, iy + 0.5, kernel_std);

              sumw += xw * yw;
              for (c = 0; c < C; ++c) {
                  real val = THTensor_fastGet4d(input, n, c, iy_p_cl, ix_p_cl);
                  real old_val = THTensor_fastGet4d(output, n, c, h, w);
                  THTensor_fastSet4d(output, n, c, h, w, xw * yw * val + old_val);
              }
            }
          }
        }
        for(c=0; c<C; ++c){
          if(sumw > 0){
            real val = THTensor_fastGet4d(output, n, c, h, w);
            THTensor_fastSet4d(output, n, c, h, w, val / sumw);
          }
        }
      }
    }
  }
}


TH_API void THNN_(SpatialGridSamplerGaussian_updateGradInput)(
    THNNState *state,
    THTensor *input, THTensor *gradInput,
    THTensor *grid, THTensor *gradGrid,
    THTensor *gradOutput,
    int kernel_size,
    real kernel_std,
    int padding_mode) {

  THNN_(SpatialGridSamplerGaussian_shapeCheck)(input, grid, gradOutput);
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
  int n, h, w, c, i, j;
//  #pragma omp parallel for private(n, h, w)
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

        // calculating coordinate of nearest pixel with torch.floor(grid + 0.5)
        // calculating coordinate of north west pixel by removing floor((kernel_size - 1) / 2)
        int ix_nw = floor(ix + 0.5) - floor((kernel_size - 1) * 0.5);
        int iy_nw = floor(iy + 0.5) - floor((kernel_size - 1) * 0.5);

        // calculating sumw
        real sumw = 0;
        for(i=0; i < kernel_size; i++){
          for(j=0; j < kernel_size; j++){
            int ix_p = ix_nw + i;
            int iy_p = iy_nw + j;
            // if antecedant value in image bounds
            if(ix_p >= 0 && ix_p < IW && iy_p >= 0 && iy_p < IH){
              real xw =  NORMAL_CDF(ix_p + 1.0, ix + 0.5, kernel_std) - NORMAL_CDF(ix_p, ix + 0.5, kernel_std);
              real yw =  NORMAL_CDF(iy_p + 1.0, iy + 0.5, kernel_std) - NORMAL_CDF(iy_p, iy + 0.5, kernel_std);
              sumw += xw * yw;
            }
          }
        }

        real sumvw = 0;
        real sumdwdgix = 0;
        real sumdwdgiy = 0;
        real sumvdwdgix = 0;
        real sumvdwdgiy = 0;
        real sumvgo;

        for(i=0; i < kernel_size; i++){
          for(j=0; j < kernel_size; j++){

            int ix_p = ix_nw + i;
            int iy_p = iy_nw + j;

            int ix_p_cl, iy_p_cl;
            if (padding_mode==MODE_BORDER){
              // clip cooridinates to image borders
              CLIP_COORDINATES(ix_p, ix_p_cl, IW);
              CLIP_COORDINATES(iy_p, iy_p_cl, IH);
            }
            else {
              ix_p_cl = ix_p;
              iy_p_cl = iy_p;
            }

            // if antecedant value in image bounds
            if(ix_p_cl >= 0 && ix_p_cl < IW && iy_p_cl >= 0 && iy_p_cl < IH){

              real xw =  NORMAL_CDF(ix_p + 1.0, ix + 0.5, kernel_std) - NORMAL_CDF(ix_p, ix + 0.5, kernel_std);
              real yw =  NORMAL_CDF(iy_p + 1.0, iy + 0.5, kernel_std) - NORMAL_CDF(iy_p, iy + 0.5, kernel_std);

              real dxw = NORMAL_PDF(ix_p, ix + 0.5, kernel_std) - NORMAL_PDF(ix_p + 1, ix + 0.5, kernel_std);
              real dyw = NORMAL_PDF(iy_p, iy + 0.5, kernel_std) - NORMAL_PDF(iy_p + 1, iy + 0.5, kernel_std);

              sumvgo = 0;
              for (c = 0; c < C; ++c) {
                real gradout = THTensor_fastGet4d(gradOutput, n, c, h, w);
                real val = THTensor_fastGet4d(input, n, c, iy_p_cl, ix_p_cl);
                sumvgo += val * gradout;
                if( sumw > 0){
                  real old_value = THTensor_fastGet4d(gradInput, n, c, iy_p_cl, ix_p_cl);
                  THTensor_fastSet4d(gradInput, n, c, iy_p_cl, ix_p_cl,  old_value + (xw * yw * gradout) / sumw);
                }
              }
              sumvw         += xw * yw * sumvgo;
              sumdwdgix     += dxw * yw;
              sumdwdgiy     += dyw * xw;
              sumvdwdgix    += dxw * yw * sumvgo;
              sumvdwdgiy    += xw * dyw * sumvgo;
            }
          }
        }
        if (sumw > 0){
          gix = - (sumdwdgix / (sumw * sumw)) * sumvw + (1. / sumw) * sumvdwdgix;
          giy = - (sumdwdgiy / (sumw * sumw)) * sumvw + (1. / sumw) * sumvdwdgiy;
        }
        // un-normalize gradGrid values back to [-1, 1] constraints
        gix = gix * (IW - 1) / 2.;
        giy = giy * (IH - 1) / 2.;

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