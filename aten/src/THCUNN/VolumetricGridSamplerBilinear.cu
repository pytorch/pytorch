#include "THCUNN.h"
#include "common.h"
#include "THCDeviceTensor.cuh"
#include "THCDeviceTensorUtils.cuh"
#include "THCDeviceUtils.cuh"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "THCAtomics.cuh"

#define WITHIN_BOUNDS(x, y, z, D, H, W) (x >= 0 && x < W && y >= 0 && y < H && z >= 0 && z < D)
#define SAFE_ADD(input, x, y, z, n, c, D, H, W, value)	\
  do {    \
    if (WITHIN_BOUNDS(x, y, z, D, H, W)) {    \
      atomicAdd(&input[n][c][z][y][x], value);	\
    }						\
  } while(0)

#undef MIN
#define MIN(a,b) ( ((a)<(b)) ? (a) : (b) )
#undef MAX
#define MAX(a,b) ( ((a)>(b)) ? (a) : (b) )
#define CLIP_COORDINATES(in, out, clip_limit) out = MIN((clip_limit-1), MAX(in, 0))

const int MODE_BORDER = 1;


template <typename Dtype>
__launch_bounds__(1024)
__global__ void VolumetricGridSamplerBilinear_updateOutput_kernel(
    const int nthreads,
    THCDeviceTensor<Dtype, 5> input,
    THCDeviceTensor<Dtype, 5> grid,
    THCDeviceTensor<Dtype, 5> output,
    const int padding_mode) {

  int N = input.getSize(0);
  int C = input.getSize(1);
  int ID = input.getSize(2);
  int IH = input.getSize(3);
  int IW = input.getSize(4);
  int D = grid.getSize(1);
  int H = grid.getSize(2);
  int W = grid.getSize(3);

  CUDA_KERNEL_LOOP(index, nthreads) {

    const int n = index % N;
    const int d = (index / N) % D;
    const int h = (index / (N * D)) % H;
    const int w = (index / (N * D * H)) % W;
    int c;

    // get the corresponding input x, y, z co-ordinates from grid
    Dtype ix = grid[n][d][h][w][0];
    Dtype iy = grid[n][d][h][w][1];
    Dtype iz = grid[n][d][h][w][2];

    // normalize ix, iy, iz from [-1, 1] to [0, IW-1] & [0, IH-1] & [0, ID-1]
    ix = ScalarConvert<float,Dtype>::to(((ix + 1) / 2) * (IW-1));
    iy = ScalarConvert<float,Dtype>::to(((iy + 1) / 2) * (IH-1));
    iz = ScalarConvert<float,Dtype>::to(((iz + 1) / 2) * (ID-1));

    // get corner pixel values from (x, y, z)
    // for 4d, we used north-east-south-west
    // for 5d, we add top-bottom
    int ix_tnw = floor(ScalarConvert<Dtype,float>::to(ix));
    int iy_tnw = floor(ScalarConvert<Dtype,float>::to(iy));
    int iz_tnw = floor(ScalarConvert<Dtype,float>::to(iz));
    
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
    Dtype tnw = (ix_bse - ix)    * (iy_bse - iy)    * (iz_bse - iz);
    Dtype tne = (ix    - ix_bsw) * (iy_bsw - iy)    * (iz_bsw - iz);
    Dtype tsw = (ix_bne - ix)    * (iy    - iy_bne) * (iz_bne - iz);
    Dtype tse = (ix    - ix_bnw) * (iy    - iy_bnw) * (iz_bnw - iz);
    Dtype bnw = (ix_tse - ix)    * (iy_tse - iy)    * (iz - iz_tse);
    Dtype bne = (ix    - ix_tsw) * (iy_tsw - iy)    * (iz - iz_tsw);
    Dtype bsw = (ix_tne - ix)    * (iy    - iy_tne) * (iz - iz_tne);
    Dtype bse = (ix    - ix_tnw) * (iy    - iy_tnw) * (iz - iz_tnw);

    // calculate bilinear weighted pixel value and set output pixel
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

    Dtype out_val;
    for (c = 0; c < C; ++c) {
      out_val = ScalarConvert<int,Dtype>::to(0);
      if (WITHIN_BOUNDS(ix_tnw, iy_tnw, iz_tnw, ID, IH, IW)) {
        out_val += input[n][c][iz_tnw][iy_tnw][ix_tnw] * tnw;
      }
      if (WITHIN_BOUNDS(ix_tne, iy_tne, iz_tne, ID, IH, IW)) {
        out_val += input[n][c][iz_tne][iy_tne][ix_tne] * tne;
      }
      if (WITHIN_BOUNDS(ix_tsw, iy_tsw, iz_tsw, ID, IH, IW)) {
        out_val += input[n][c][iz_tsw][iy_tsw][ix_tsw] * tsw;
      }
      if (WITHIN_BOUNDS(ix_tse, iy_tse, iz_tse, ID, IH, IW)) {
        out_val += input[n][c][iz_tse][iy_tse][ix_tse] * tse;
      }
      if (WITHIN_BOUNDS(ix_bnw, iy_bnw, iz_bnw, ID, IH, IW)) {
        out_val += input[n][c][iz_bnw][iy_bnw][ix_bnw] * bnw;
      }
      if (WITHIN_BOUNDS(ix_bne, iy_bne, iz_bne, ID, IH, IW)) {
        out_val += input[n][c][iz_bne][iy_bne][ix_bne] * bne;
      }
      if (WITHIN_BOUNDS(ix_bsw, iy_bsw, iz_bsw, ID, IH, IW)) {
        out_val += input[n][c][iz_bsw][iy_bsw][ix_bsw] * bsw;
      }
      if (WITHIN_BOUNDS(ix_bse, iy_bse, iz_bse, ID, IH, IW)) {
        out_val += input[n][c][iz_bse][iy_bse][ix_bse] * bse;
      }
      output[n][c][d][h][w] = out_val;
    }
  }
}

template <typename Dtype>
__launch_bounds__(1024)
__global__ void VolumetricGridSamplerBilinear_updateGradInput_kernel(
    const int nthreads,
    THCDeviceTensor<Dtype, 5> input, THCDeviceTensor<Dtype, 5> gradInput,
    THCDeviceTensor<Dtype, 5> grid, THCDeviceTensor<Dtype, 5> gradGrid,
    THCDeviceTensor<Dtype, 5> gradOutput,
    const int padding_mode) {

  int N = input.getSize(0);
  int C = input.getSize(1);
  int ID = input.getSize(2);
  int IH = input.getSize(3);
  int IW = input.getSize(4);
  int D = grid.getSize(1);
  int H = grid.getSize(2);
  int W = grid.getSize(3);

  CUDA_KERNEL_LOOP(index, nthreads) {

    const int n = index % N;
    const int d = (index / N) % D;
    const int h = (index / (N * D)) % H;
    const int w = (index / (N * D * H)) % W;

    // get the corresponding input x, y, z co-ordinates from grid
    Dtype ix = grid[n][d][h][w][0];
    Dtype iy = grid[n][d][h][w][1];
    Dtype iz = grid[n][d][h][w][2];

    Dtype gix = ScalarConvert<int,Dtype>::to(0);
    Dtype giy = ScalarConvert<int,Dtype>::to(0);
    Dtype giz = ScalarConvert<int,Dtype>::to(0);

    // normalize ix, iy, iz from [-1, 1] to [0, IW-1] & [0, IH-1] & [0, ID-1]
    ix = ScalarConvert<float,Dtype>::to(((ix + 1) / 2) * (IW-1));
    iy = ScalarConvert<float,Dtype>::to(((iy + 1) / 2) * (IH-1));
    iz = ScalarConvert<float,Dtype>::to(((iz + 1) / 2) * (ID-1));

    // get corner pixel values from (x, y, z)
    // for 4d, we used north-east-south-west
    // for 5d, we add top-bottom
    int ix_tnw = floor(ScalarConvert<Dtype,float>::to(ix));
    int iy_tnw = floor(ScalarConvert<Dtype,float>::to(iy));
    int iz_tnw = floor(ScalarConvert<Dtype,float>::to(iz));
    
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
    Dtype tnw = (ix_bse - ix)    * (iy_bse - iy)    * (iz_bse - iz);
    Dtype tne = (ix    - ix_bsw) * (iy_bsw - iy)    * (iz_bsw - iz);
    Dtype tsw = (ix_bne - ix)    * (iy    - iy_bne) * (iz_bne - iz);
    Dtype tse = (ix    - ix_bnw) * (iy    - iy_bnw) * (iz_bnw - iz);
    Dtype bnw = (ix_tse - ix)    * (iy_tse - iy)    * (iz - iz_tse);
    Dtype bne = (ix    - ix_tsw) * (iy_tsw - iy)    * (iz - iz_tsw);
    Dtype bsw = (ix_tne - ix)    * (iy    - iy_tne) * (iz - iz_tne);
    Dtype bse = (ix    - ix_tnw) * (iy    - iy_tnw) * (iz - iz_tnw);

    Dtype gradout;
    Dtype tnw_val;
    Dtype tne_val;
    Dtype tsw_val;
    Dtype tse_val;
    Dtype bnw_val;
    Dtype bne_val;
    Dtype bsw_val;
    Dtype bse_val;
    
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
      gradout = gradOutput[n][c][d][h][w];

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
      tnw_val = ScalarConvert<int,Dtype>::to(0);
      if (WITHIN_BOUNDS(ix_tnw_cl, iy_tnw_cl, iz_tnw_cl, ID, IH, IW)) {
        tnw_val = input[n][c][iz_tnw_cl][iy_tnw_cl][ix_tnw_cl];
      }
      tne_val = ScalarConvert<int,Dtype>::to(0);
      if (WITHIN_BOUNDS(ix_tne_cl, iy_tne_cl, iz_tne_cl, ID, IH, IW)) {
        tne_val = input[n][c][iz_tne_cl][iy_tne_cl][ix_tne_cl];
      }
      tsw_val = ScalarConvert<int,Dtype>::to(0);
      if (WITHIN_BOUNDS(ix_tsw_cl, iy_tsw_cl, iz_tsw_cl, ID, IH, IW)) {
        tsw_val = input[n][c][iz_tsw_cl][iy_tsw_cl][ix_tsw_cl];
      }
      tse_val = ScalarConvert<int,Dtype>::to(0);
      if (WITHIN_BOUNDS(ix_tse_cl, iy_tse_cl, iz_tse_cl, ID, IH, IW)) {
        tse_val = input[n][c][iz_tse_cl][iy_tse_cl][ix_tse_cl];
      }
      bnw_val = ScalarConvert<int,Dtype>::to(0);
      if (WITHIN_BOUNDS(ix_bnw_cl, iy_bnw_cl, iz_bnw_cl, ID, IH, IW)) {
        bnw_val = input[n][c][iz_bnw_cl][iy_bnw_cl][ix_bnw_cl];
      }
      bne_val = ScalarConvert<int,Dtype>::to(0);
      if (WITHIN_BOUNDS(ix_bne_cl, iy_bne_cl, iz_bne_cl, ID, IH, IW)) {
        bne_val = input[n][c][iz_bne_cl][iy_bne_cl][ix_bne_cl];
      }
      bsw_val = ScalarConvert<int,Dtype>::to(0);
      if (WITHIN_BOUNDS(ix_bsw_cl, iy_bsw_cl, iz_bsw_cl, ID, IH, IW)) {
        bsw_val = input[n][c][iz_bsw_cl][iy_bsw_cl][ix_bsw_cl];
      }
      bse_val = ScalarConvert<int,Dtype>::to(0);
      if (WITHIN_BOUNDS(ix_bse_cl, iy_bse_cl, iz_bse_cl, ID, IH, IW)) {
        bse_val = input[n][c][iz_bse_cl][iy_bse_cl][ix_bse_cl];
      }

      Dtype m1 = ScalarConvert<int,Dtype>::to(-1);
      gix += m1 * tnw_val * (iy_bse - iy) * (iz_bse - iz) * gradout;
      gix += tne_val * (iy_bsw - iy) * (iz_bsw - iz) * gradout;
      gix += m1 * tsw_val * (iy - iy_bne) * (iz_bne - iz) * gradout;
      gix += tse_val * (iy - iy_bnw) * (iz_bnw - iz) * gradout;
      gix += m1 * bnw_val * (iy_tse - iy) * (iz - iz_tse) * gradout;
      gix += bne_val * (iy_tsw - iy) * (iz - iz_tsw) * gradout;
      gix += m1 * bsw_val * (iy - iy_tne) * (iz - iz_tne) * gradout;
      gix += bse_val * (iy - iy_tnw) * (iz - iz_tnw) * gradout;


      giy += m1 * tnw_val * (ix_bse - ix)    * (iz_bse - iz) * gradout;
      giy += m1 * tne_val * (ix    - ix_bsw) * (iz_bsw - iz) * gradout;
      giy += tsw_val * (ix_bne - ix)    * (iz_bne - iz) * gradout;
      giy += tse_val * (ix    - ix_bnw) * (iz_bnw - iz) * gradout;
      giy += m1 * bnw_val * (ix_tse - ix)    * (iz - iz_tse) * gradout;
      giy += m1 * bne_val * (ix    - ix_tsw) * (iz - iz_tsw) * gradout;
      giy += bsw_val * (ix_tne - ix)    * (iz - iz_tne) * gradout;
      giy += bse_val * (ix    - ix_tnw) * (iz - iz_tnw) * gradout;

      giz += m1 * tnw_val * (ix_bse - ix)    * (iy_bse - iy)    * gradout;
      giz += m1 * tne_val * (ix    - ix_bsw) * (iy_bsw - iy)    * gradout;
      giz += m1 * tsw_val * (ix_bne - ix)    * (iy    - iy_bne) * gradout;
      giz += m1 * tse_val * (ix    - ix_bnw) * (iy    - iy_bnw) * gradout;
      giz += bnw_val * (ix_tse - ix)    * (iy_tse - iy)    * gradout;
      giz += bne_val * (ix    - ix_tsw) * (iy_tsw - iy)    * gradout;
      giz += bsw_val * (ix_tne - ix)    * (iy    - iy_tne) * gradout;
      giz += bse_val * (ix    - ix_tnw) * (iy    - iy_tnw) * gradout;
    }

    // un-normalize gradGrid values back to [-1, 1] constraints
    gix = gix * (IW - 1) / 2;
    giy = giy * (IH - 1) / 2;
    giz = giz * (ID - 1) / 2;

    Dtype gix_old = gradGrid[n][d][h][w][0];
    Dtype giy_old = gradGrid[n][d][h][w][1];
    Dtype giz_old = gradGrid[n][d][h][w][2];

    gradGrid[n][d][h][w][0] = gix_old + gix;
    gradGrid[n][d][h][w][1] = giy_old + giy;
    gradGrid[n][d][h][w][2] = giz_old + giz;
  }
}

#undef MIN
#undef MAX
#undef CLIP_COORDINATES
#undef WITHIN_BOUNDS
#undef SAFE_ADD

#include "generic/VolumetricGridSamplerBilinear.cu"
#include "THCGenerateFloatTypes.h"
