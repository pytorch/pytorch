#include "THCUNN.h"
#include "common.h"
#include "THCDeviceTensor.cuh"
#include "THCDeviceTensorUtils.cuh"
#include "THCDeviceUtils.cuh"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "THCAtomics.cuh"

#define M_SQRT2 1.41421356237309504880


#define WITHIN_BOUNDS(x, y, H, W) (x >= 0 && x < W && y >= 0 && y < H)
#define SAFE_ADD(input, x, y, n, c, value)    \
  do {    \
      atomicAdd(&input[n][c][y][x], value);   \
     } while(0)

#undef MIN
#define MIN(a,b) ( ((a)<(b)) ? (a) : (b) )
#undef MAX
#define MAX(a,b) ( ((a)>(b)) ? (a) : (b) )
#define CLIP_COORDINATES(in, out, clip_limit) out = MIN((clip_limit-1), MAX(in, 0))

const int MODE_BORDER = 1;


#define NORMAL_CDF(val, mu, sigma) 0.5 * erfc(-((val) - (mu)) / ((M_SQRT2) * (sigma)))
#define NORMAL_PDF(val, mu, sigma) exp(- 0.5 * pow(((val) - (mu)) / (sigma), 2)) / (sqrt(2 * (M_PI)) * (sigma))


template <typename Dtype>
__launch_bounds__(1024)
__global__ void SpatialGridSamplerGaussian_updateOutput_kernel(
    const int nthreads,
    THCDeviceTensor<Dtype, 4> input,
    THCDeviceTensor<Dtype, 4> grid,
    THCDeviceTensor<Dtype, 4> output,
    const int kernel_size,
    const float kernel_std,
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
    int ix_nw = floor(ScalarConvert<Dtype,float>::to(ix)+0.5) - floor(( kernel_size - 1) * 0.5);
    int iy_nw = floor(ScalarConvert<Dtype,float>::to(iy)+0.5) - floor(( kernel_size - 1) * 0.5);

    int i, j;
    Dtype sumw = ScalarConvert<float, Dtype>::to(0.);
    for(i=0; i < kernel_size; i++){
          for(j=0; j < kernel_size; j++){

            // get surfaces to each neighbor:
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
             float xwf =  NORMAL_CDF(ix_p + 1.0, ix + 0.5, kernel_std) - NORMAL_CDF(ix_p, ix + 0.5, kernel_std);
             float ywf =  NORMAL_CDF(iy_p + 1.0, iy + 0.5, kernel_std) - NORMAL_CDF(iy_p, iy + 0.5, kernel_std);
             Dtype xw = ScalarConvert<float, Dtype>::to(xwf);
             Dtype yw = ScalarConvert<float, Dtype>::to(ywf);

              // calculate Gaussian weighted pixel value and set output pixel
              sumw += xw * yw;

              for (c = 0; c < C; ++c) {
                  Dtype val = input[n][c][iy_p_cl][ix_p_cl];
                  Dtype old_val = output[n][c][h][w];
                  output[n][c][h][w] = old_val +  val * yw * xw;
              }
      }
    }
  }
  for (c = 0; c < C; ++c) {
        if(sumw > 0){
        Dtype old_val = output[n][c][h][w];
        output[n][c][h][w] = old_val / sumw;
      }
    }
  }
}

template <typename Dtype>
__launch_bounds__(1024)
__global__ void SpatialGridSamplerGaussian_updateGradInput_kernel(
    const int nthreads,
    THCDeviceTensor<Dtype, 4> input, THCDeviceTensor<Dtype, 4> gradInput,
    THCDeviceTensor<Dtype, 4> grid, THCDeviceTensor<Dtype, 4> gradGrid,
    THCDeviceTensor<Dtype, 4> gradOutput,
    const  int kernel_size,
    const float kernel_std,
    const int padding_mode) {

  int N = input.getSize(0);
  int C = input.getSize(1);
  int IH = input.getSize(2);
  int IW = input.getSize(3);
  int H = grid.getSize(1);
  int W = grid.getSize(2);

  int N1 = gradInput.getSize(0);
  int N2 = gradInput.getSize(1);
  int N3 = gradInput.getSize(2);
  int N4 = gradInput.getSize(3);
  int N5 = gradInput.getSize(4);

  CUDA_KERNEL_LOOP(index, nthreads) {

    const int n = index % N;
    const int h = (index / N) % H;
    const int w = (index / (N * H)) % W;
    int c;
    int i, j;

    // get the corresponding input x, y co-ordinates from grid
    Dtype ix = grid[n][h][w][0];
    Dtype iy = grid[n][h][w][1];

    Dtype gix = ScalarConvert<int,Dtype>::to(0);
    Dtype giy = ScalarConvert<int,Dtype>::to(0);

    // normalize ix, iy from [-1, 1] to [0, H-1] & [0, W-1]
    ix = ScalarConvert<float,Dtype>::to(((ix + 1) / 2) * (IW-1));
    iy = ScalarConvert<float,Dtype>::to(((iy + 1) / 2) * (IH-1));

    // get NE, NW, SE, SW pixel values from (x, y)
    int ix_nw = floor(ScalarConvert<Dtype,float>::to(ix)+0.5) - floor((kernel_size-1)*0.5);
    int iy_nw = floor(ScalarConvert<Dtype,float>::to(iy)+0.5) - floor((kernel_size-1)*0.5);


    // calculating sumw
    Dtype sumw = ScalarConvert<float, Dtype>::to(0.);
    for(i=0; i < kernel_size; i++){
      for(j=0; j < kernel_size; j++){
         int ix_p = ix_nw + i;
         int iy_p = iy_nw + j;
            // if antecedant value in image bounds
            if(ix_p >= 0 && ix_p < IW && iy_p >= 0 && iy_p < IH){
             float xwf =  NORMAL_CDF(ix_p + 1.0, ix + 0.5, kernel_std) - NORMAL_CDF(ix_p, ix + 0.5, kernel_std);
             float ywf =  NORMAL_CDF(iy_p + 1.0, iy + 0.5, kernel_std) - NORMAL_CDF(iy_p, iy + 0.5, kernel_std);
             Dtype xw = ScalarConvert<float, Dtype>::to(xwf);
             Dtype yw = ScalarConvert<float, Dtype>::to(ywf);
             sumw += xw * yw;
            }
          }
        }


    Dtype sumvw = ScalarConvert<float, Dtype>::to(0.);
    Dtype sumdwdgix = ScalarConvert<float, Dtype>::to(0.);
    Dtype sumdwdgiy = ScalarConvert<float, Dtype>::to(0.);
    Dtype sumvdwdgix = ScalarConvert<float, Dtype>::to(0.);
    Dtype sumvdwdgiy = ScalarConvert<float, Dtype>::to(0.);
    Dtype sumvgo = ScalarConvert<float, Dtype>::to(0.);


    for(i=0; i < kernel_size; i++){
          for(j=0; j < kernel_size; j++){
            // get surfaces to each neighbor:
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

             float xwf =  NORMAL_CDF(ix_p + 1.0, ix + 0.5, kernel_std) - NORMAL_CDF(ix_p, ix + 0.5, kernel_std);
             float ywf =  NORMAL_CDF(iy_p + 1.0, iy + 0.5, kernel_std) - NORMAL_CDF(iy_p, iy + 0.5, kernel_std);

             Dtype xw = ScalarConvert<float, Dtype>::to(xwf);
             Dtype yw = ScalarConvert<float, Dtype>::to(ywf);


             float dxwf = NORMAL_PDF(ix_p, ix + 0.5, kernel_std) - NORMAL_PDF(ix_p + 1, ix + 0.5, kernel_std);
             float dywf = NORMAL_PDF(iy_p, iy + 0.5, kernel_std) - NORMAL_PDF(iy_p + 1, iy + 0.5, kernel_std);

             Dtype dxw = ScalarConvert<float, Dtype>::to(dxwf);
             Dtype dyw = ScalarConvert<float, Dtype>::to(dywf);


             // calculate Gaussian weighted pixel value and set output pixel
             sumvgo = ScalarConvert<float, Dtype>::to(0.);
              for (c = 0; c < C; ++c) {
                Dtype gradout = gradOutput[n][c][h][w];
                Dtype val = input[n][c][iy_p_cl][ix_p_cl];
                sumvgo += val * gradout;
                Dtype old_value = gradInput[n][c][iy_p_cl][ix_p_cl];
                if (sumw > 0){
                SAFE_ADD(gradInput, ix_p_cl, iy_p_cl, n, c, (xw * yw * gradout) / sumw);
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
       gix = - (sumdwdgix / (sumw * sumw)) * sumvw + (ScalarConvert<float, Dtype>::to(1.) / sumw) * sumvdwdgix;
       giy = - (sumdwdgiy / (sumw * sumw)) * sumvw + (ScalarConvert<float, Dtype>::to(1.) / sumw) * sumvdwdgiy;
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

#include "generic/SpatialGridSamplerGaussian.cu"
#include "THCGenerateFloatTypes.h"
