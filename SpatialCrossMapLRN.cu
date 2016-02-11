#include "THCUNN.h"
#include "common.h"

template <typename Dtype>
__global__ void LRNFillScale(const int nthreads, const Dtype* const in,
    const int num, const int channels, const int height,
    const int width, const int size, const Dtype alpha_over_size,
    const Dtype k, Dtype* const scale) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int n = index / width / height;
    const int offset = (n * channels * height + h) * width + w;
    const int step = height * width;
    const Dtype* const in_off = in + offset;
    Dtype* const scale_off = scale + offset;
    int head = 0;
    const int pre_pad = (size - 1) / 2;
    const int post_pad = size - pre_pad - 1;
    Dtype accum_scale = 0;
    // fill the scale at [n, :, h, w]
    // accumulate values
    while (head < post_pad && head < channels) {
      accum_scale += in_off[head * step] * in_off[head * step];
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_scale += in_off[head * step] * in_off[head * step];
      if (head - size >= 0) {
        accum_scale -= in_off[(head - size) * step]
                       * in_off[(head - size) * step];
      }
      scale_off[(head - post_pad) * step] = k + accum_scale * alpha_over_size;
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      if (head - size >= 0) {
        accum_scale -= in_off[(head - size) * step]
                       * in_off[(head - size) * step];
      }
      scale_off[(head - post_pad) * step] = k + accum_scale * alpha_over_size;
      ++head;
    }
  }
}

__global__ void LRNComputeOutput(const int nthreads, const float* in,
    const float* scale, const float negative_beta, float* out) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    out[index] = in[index] * pow(scale[index], negative_beta);
  }
}

template <typename Dtype>
__global__ void LRNComputeDiff(const int nthreads,
    const Dtype* const bottom_data, const Dtype* const top_data,
    const Dtype* const scale, const Dtype* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int size, const Dtype negative_beta,
    const Dtype cache_ratio, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int n = index / width / height;
    const int offset = (n * channels * height + h) * width + w;
    const int step = height * width;
    const Dtype* const bottom_off = bottom_data + offset;
    const Dtype* const top_off = top_data + offset;
    const Dtype* const scale_off = scale + offset;
    const Dtype* const top_diff_off = top_diff + offset;
    Dtype* const bottom_diff_off = bottom_diff + offset;
    int head = 0;
    const int pre_pad = size - (size + 1) / 2;
    const int post_pad = size - pre_pad - 1;
    Dtype accum_ratio = 0;
    // accumulate values
    while (head < post_pad && head < channels) {
      accum_ratio += top_diff_off[head * step] * top_off[head * step] /
          scale_off[head * step];
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_ratio += top_diff_off[head * step] * top_off[head * step] /
          scale_off[head * step];
      if (head - size >= 0) {
        accum_ratio -= top_diff_off[(head - size) * step] *
            top_off[(head - size) * step] / scale_off[(head - size) * step];
      }
      bottom_diff_off[(head - post_pad) * step] =
          top_diff_off[(head - post_pad) * step]
            * pow(scale_off[(head - post_pad) * step], negative_beta)
          - cache_ratio * bottom_off[(head - post_pad) * step] * accum_ratio;
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      if (head - size >= 0) {
        accum_ratio -= top_diff_off[(head - size) * step] *
            top_off[(head - size) * step] / scale_off[(head - size) * step];
      }
      bottom_diff_off[(head - post_pad) * step] =
          top_diff_off[(head - post_pad) * step]
            * pow(scale_off[(head - post_pad) * step], negative_beta)
          - cache_ratio * bottom_off[(head - post_pad) * step] * accum_ratio;
      ++head;
    }
  }
}

extern "C"
void LRNforward(THCState* state, THCudaTensor* input, THCudaTensor* output,
    THCudaTensor* scale, int local_size, float alpha, float beta, float k)
{
  THCudaTensor_resizeAs(state, output, input);
  THCudaTensor_resizeAs(state, scale, input);
  
  int batchSize;
  int nInputPlane;
  int imsize_h;
  int imsize_w;

  if (input->nDimension == 3) {
    batchSize = 1;
    nInputPlane = input->size[0];
    imsize_h = input->size[1];
    imsize_w = input->size[2];
  }
  else
  {
    batchSize = input->size[0];
    nInputPlane = input->size[1];
    imsize_h = input->size[2];
    imsize_w = input->size[3];
  }

  input = THCudaTensor_newContiguous(state, input);

  int n_threads = batchSize * imsize_h * imsize_w;
  LRNFillScale<<<GET_BLOCKS(n_threads), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
      n_threads, THCudaTensor_data(state, input), batchSize, nInputPlane, imsize_h, imsize_w, local_size,
      alpha / local_size, k, THCudaTensor_data(state, scale));
  n_threads *= nInputPlane;
  LRNComputeOutput<<<GET_BLOCKS(n_threads), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
    n_threads, THCudaTensor_data(state, input), THCudaTensor_data(state, scale), -beta, THCudaTensor_data(state, output));

  THCudaTensor_free(state, input);
}


extern "C"
void LRNbackward(THCState* state, THCudaTensor* input, THCudaTensor* output,
    THCudaTensor* gradOutput, THCudaTensor* gradInput, THCudaTensor* scale,
    int local_size, float alpha, float beta, float k)
{
  THCudaTensor_resizeAs(state, gradInput, input);
  
  int batchSize;
  int nInputPlane;
  int imsize_h;
  int imsize_w;

  if (input->nDimension == 3) {
    batchSize = 1;
    nInputPlane = input->size[0];
    imsize_h = input->size[1];
    imsize_w = input->size[2];
  }
  else
  {
    batchSize = input->size[0];
    nInputPlane = input->size[1];
    imsize_h = input->size[2];
    imsize_w = input->size[3];
  }

  input = THCudaTensor_newContiguous(state, input);
  gradOutput = THCudaTensor_newContiguous(state, gradOutput);

  int n_threads = batchSize * imsize_h * imsize_w;
  LRNComputeDiff<<<GET_BLOCKS(n_threads), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
      n_threads, THCudaTensor_data(state, input), THCudaTensor_data(state, output),
      THCudaTensor_data(state, scale), THCudaTensor_data(state, gradOutput), batchSize, nInputPlane, imsize_h, imsize_w,
      local_size, -beta, float(2. * alpha * beta / local_size),
      THCudaTensor_data(state, gradInput));

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, gradOutput);
}

void THNN_CudaSpatialCrossMapLRN_updateOutput(
    THCState *state,
    THCudaTensor *input,
    THCudaTensor *output,
    THCudaTensor *scale,
    int size,
    float alpha,
    float beta,
    float k)
{
  LRNforward(state, input, output, scale, size, alpha, beta, k);
}

void THNN_CudaSpatialCrossMapLRN_updateGradInput(
    THCState *state,
    THCudaTensor *input,
    THCudaTensor *gradOutput,
    THCudaTensor *gradInput,
    THCudaTensor *scale,
    THCudaTensor *output,
    int size,
    float alpha,
    float beta,
    float k)
{
  LRNbackward(state, input, output, gradOutput, gradInput, scale, size, alpha, beta, k);
}
