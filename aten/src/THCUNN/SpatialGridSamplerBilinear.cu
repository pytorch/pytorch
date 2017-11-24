#include "THCUNN.h"
#include "common.h"
#include "THCDeviceTensor.cuh"
#include "THCDeviceTensorUtils.cuh"
#include "THCDeviceUtils.cuh"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "THCAtomics.cuh"

#define WITHIN_BOUNDS(x, y, H, W) (x >= 0 && x < W && y >= 0 && y < H)
#define SAFE_ADD(input, x, y, n, c, H, W, value)    \
  do {    \
    if (WITHIN_BOUNDS(x, y, H, W)) {    \
      atomicAdd(&input[n][c][y][x], value);   \
    }   \
  } while(0)

#undef MIN
#define MIN(a,b) ( ((a)<(b)) ? (a) : (b) )
#undef MAX
#define MAX(a,b) ( ((a)>(b)) ? (a) : (b) )
#define CLIP_COORDINATES(in, out, clip_limit) out = MIN((clip_limit-1), MAX(in, 0))

const int MODE_BORDER = 1;


template <typename Dtype>
__launch_bounds__(1024)
__global__ void SpatialGridSamplerBilinear_updateOutput_kernel(
    const int nthreads,
    THCDeviceTensor<Dtype, 4> input,
    THCDeviceTensor<Dtype, 4> grid,
    THCDeviceTensor<Dtype, 4> output,
    const int padding_mode) {

  int N = input.getSize(0);
  int C = input.getSize(1);
  int IH = input.getSize(2);
  int IW = input.getSize(3);
  int H = grid.getSize(1);
  int W = grid.getSize(2);

  CUDA_KERNEL_LOOP(index, nthreads) {

    const int n = index % N;
    const int h = (index / N) % H;
    const int w = (index / (N * H)) % W;
    int c;

    // get the corresponding input x, y co-ordinates from grid
    Dtype ix = grid[n][h][w][0];
    Dtype iy = grid[n][h][w][1];

    // normalize ix, iy from [-1, 1] to [0, IH-1] & [0, IW-1]
    ix = ScalarConvert<float,Dtype>::to(((ix + 1) / 2) * (IW-1));
    iy = ScalarConvert<float,Dtype>::to(((iy + 1) / 2) * (IH-1));

    // get NE, NW, SE, SW pixel values from (x, y)
    int ix_nw = floor(ScalarConvert<Dtype,float>::to(ix));
    int iy_nw = floor(ScalarConvert<Dtype,float>::to(iy));
    int ix_ne = ix_nw + 1;
    int iy_ne = iy_nw;
    int ix_sw = ix_nw;
    int iy_sw = iy_nw + 1;
    int ix_se = ix_nw + 1;
    int iy_se = iy_nw + 1;

    // get surfaces to each neighbor:
    Dtype nw = (ix_se - ix)    * (iy_se - iy);
    Dtype ne = (ix    - ix_sw) * (iy_sw - iy);
    Dtype sw = (ix_ne - ix)    * (iy    - iy_ne);
    Dtype se = (ix    - ix_nw) * (iy    - iy_nw);

    // calculate bilinear weighted pixel value and set output pixel
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

    Dtype out_val;
    for (c = 0; c < C; ++c) {
      out_val = ScalarConvert<int,Dtype>::to(0);
      if (WITHIN_BOUNDS(ix_nw, iy_nw, IH, IW)) {
        out_val += input[n][c][iy_nw][ix_nw] * nw;
      }
      if (WITHIN_BOUNDS(ix_ne, iy_ne, IH, IW)) {
        out_val += input[n][c][iy_ne][ix_ne] * ne;
      }
      if (WITHIN_BOUNDS(ix_sw, iy_sw, IH, IW)) {
        out_val += input[n][c][iy_sw][ix_sw] * sw;
      }
      if (WITHIN_BOUNDS(ix_se, iy_se, IH, IW)) {
        out_val += input[n][c][iy_se][ix_se] * se;
      }
      output[n][c][h][w] = out_val;
    }
  }
}

template <typename Dtype>
__launch_bounds__(1024)
__global__ void SpatialGridSamplerBilinear_updateGradInput_kernel(
    const int nthreads,
    THCDeviceTensor<Dtype, 4> input, THCDeviceTensor<Dtype, 4> gradInput,
    THCDeviceTensor<Dtype, 4> grid, THCDeviceTensor<Dtype, 4> gradGrid,
    THCDeviceTensor<Dtype, 4> gradOutput,
    const int padding_mode) {

  int N = input.getSize(0);
  int C = input.getSize(1);
  int IH = input.getSize(2);
  int IW = input.getSize(3);
  int H = grid.getSize(1);
  int W = grid.getSize(2);

  CUDA_KERNEL_LOOP(index, nthreads) {

    const int n = index % N;
    const int h = (index / N) % H;
    const int w = (index / (N * H)) % W;

    // get the corresponding input x, y co-ordinates from grid
    Dtype ix = grid[n][h][w][0];
    Dtype iy = grid[n][h][w][1];

    Dtype gix = ScalarConvert<int,Dtype>::to(0);
    Dtype giy = ScalarConvert<int,Dtype>::to(0);

    // normalize ix, iy from [-1, 1] to [0, H-1] & [0, W-1]
    ix = ScalarConvert<float,Dtype>::to(((ix + 1) / 2) * (IW-1));
    iy = ScalarConvert<float,Dtype>::to(((iy + 1) / 2) * (IH-1));;

    // get NE, NW, SE, SW pixel values from (x, y)
    int ix_nw = floor(ScalarConvert<Dtype,float>::to(ix));
    int iy_nw = floor(ScalarConvert<Dtype,float>::to(iy));;
    int ix_ne = ix_nw + 1;
    int iy_ne = iy_nw;
    int ix_sw = ix_nw;
    int iy_sw = iy_nw + 1;
    int ix_se = ix_nw + 1;
    int iy_se = iy_nw + 1;

    // get surfaces to each neighbor:
    Dtype nw = (ix_se - ix)    * (iy_se - iy);
    Dtype ne = (ix    - ix_sw) * (iy_sw - iy);
    Dtype sw = (ix_ne - ix)    * (iy    - iy_ne);
    Dtype se = (ix    - ix_nw) * (iy    - iy_nw);

    Dtype gradout;
    Dtype nw_val;
    Dtype ne_val;
    Dtype sw_val;
    Dtype se_val;
    
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
      gradout = gradOutput[n][c][h][w];

      // calculate and set gradInput
      SAFE_ADD(gradInput, ix_nw_cl, iy_nw_cl, n, c, IH, IW, nw * gradout);
      SAFE_ADD(gradInput, ix_ne_cl, iy_ne_cl, n, c, IH, IW, ne * gradout);
      SAFE_ADD(gradInput, ix_sw_cl, iy_sw_cl, n, c, IH, IW, sw * gradout);
      SAFE_ADD(gradInput, ix_se_cl, iy_se_cl, n, c, IH, IW, se * gradout);

      // calculate gradGrid
      nw_val = ScalarConvert<int,Dtype>::to(0);
      if (WITHIN_BOUNDS(ix_nw_cl, iy_nw_cl, IH, IW)) {
        nw_val = input[n][c][iy_nw_cl][ix_nw_cl];
      }
      ne_val = ScalarConvert<int,Dtype>::to(0);
      if (WITHIN_BOUNDS(ix_ne_cl, iy_ne_cl, IH, IW)) {
        ne_val = input[n][c][iy_ne_cl][ix_ne_cl];
      }
      sw_val = ScalarConvert<int,Dtype>::to(0);
      if (WITHIN_BOUNDS(ix_sw_cl, iy_sw_cl, IH, IW)) {
        sw_val = input[n][c][iy_sw_cl][ix_sw_cl];
      }
      se_val = ScalarConvert<int,Dtype>::to(0);
      if (WITHIN_BOUNDS(ix_se_cl, iy_se_cl, IH, IW)) {
        se_val = input[n][c][iy_se_cl][ix_se_cl];
      }

      gix += ScalarConvert<int,Dtype>::to(-1)*(nw_val * (iy_se - iy) * gradout);
      gix += ne_val * (iy_sw - iy) * gradout;
      gix += ScalarConvert<int,Dtype>::to(-1)*(sw_val * (iy - iy_ne) * gradout);
      gix += se_val * (iy - iy_nw) * gradout;

      giy += ScalarConvert<int,Dtype>::to(-1)*(nw_val * (ix_se - ix) * gradout);
      giy += ScalarConvert<int,Dtype>::to(-1)*(ne_val * (ix - ix_sw) * gradout);
      giy += sw_val * (ix_ne - ix) * gradout;
      giy += se_val * (ix - ix_nw) * gradout;
    }

    // un-normalize gradGrid values back to [-1, 1] constraints
    gix = gix * (IW - 1) / 2;
    giy = giy * (IH - 1) / 2;

    Dtype gix_old = gradGrid[n][h][w][0];
    Dtype giy_old = gradGrid[n][h][w][1];

    gradGrid[n][h][w][0] = gix_old + gix;
    gradGrid[n][h][w][1] = giy_old + giy;
  }
}

#undef MIN
#undef MAX
#undef CLIP_COORDINATES
#undef WITHIN_BOUNDS
#undef SAFE_ADD

#include "generic/SpatialGridSamplerBilinear.cu"
#include "THCGenerateFloatTypes.h"
