#include "THCUNN.h"
#include "common.h"

// kernels borrowed from Caffe
template <typename Dtype>
__global__ void MaxPoolForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w, Dtype* top_data,
    Dtype* top_mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height);
    int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    bottom_data += (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (bottom_data[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = bottom_data[maxidx];
        }
      }
    }
    top_data[index] = maxval;
    top_mask[index] = maxidx + 1;
  }
}


template <typename Dtype>
__global__ void MaxPoolBackward(const int nthreads, const Dtype* top_diff,
    const Dtype* top_mask, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    int phstart =
        (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    int phend = min((h + pad_h) / stride_h + 1, pooled_height);
    int pwstart =
        (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    int offset = (n * channels + c) * pooled_height * pooled_width;
    top_diff += offset;
    top_mask += offset;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
	if (top_mask[ph * pooled_width + pw] - 1 == h * width + w) {
	  gradient += top_diff[ph * pooled_width + pw];
	}
      }
    }
    bottom_diff[index] = gradient;
  }
}

void THNN_CudaSpatialMaxPooling_updateOutput(THCState *state, THCudaTensor *input, THCudaTensor *output, THCudaTensor *indices, int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode)
{

  THCUNN_assertSameGPU(state, 3, input, output, indices);
  THArgCheck(input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch) tensor expected");

  long nInputCols, nInputRows, nInputPlane, batchSize;
  long nOutputCols, nOutputRows;

  if (input->nDimension == 3) {
    nInputCols = input->size[2];
    nInputRows = input->size[1];
    nInputPlane = input->size[0];
    batchSize = 1;
  }
  else
  {
    nInputCols = input->size[3];
    nInputRows = input->size[2];
    nInputPlane = input->size[1];
    batchSize = input->size[0];
  }

  THArgCheck(nInputCols >= kW - padW && nInputRows >= kH - padH, 2, "input image smaller than kernel size");
  THArgCheck(kW/2 >= padW && kH/2 >= padH, 2, "pad should be smaller than half of kernel size");

  if(ceil_mode) {
    nOutputCols = ceil(float(nInputCols - kW + 2*padW) / float(dW)) + 1;
    nOutputRows = ceil(float(nInputRows - kH + 2*padH) / float(dH)) + 1;
  }
  else {
    nOutputCols = floor(float(nInputCols - kW + 2*padW) / float(dW)) + 1;
    nOutputRows = floor(float(nInputRows - kH + 2*padH) / float(dH)) + 1;
  }

  if (padW || padH)
  {
    // ensure that the last pooling starts inside the image
    if ((nOutputRows - 1)*dH >= nInputRows + padH)
      --nOutputRows;
    if ((nOutputCols  - 1)*dW >= nInputCols  + padW)
      --nOutputCols;
  }

  input = THCudaTensor_newContiguous(state, input);
  float* input_data = THCudaTensor_data(state, input);

  THCudaTensor_resize4d(state, output, batchSize, nInputPlane, nOutputRows, nOutputCols);
  THCudaTensor_resizeAs(state, indices, output);

  float* indices_data = THCudaTensor_data(state, indices);
  float* output_data = THCudaTensor_data(state, output);

  int count = THCudaTensor_nElement(state, output);

  MaxPoolForward <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>
      (count, input_data,
      batchSize, nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols,
      kH, kW, dH, dW, padH, padW, output_data, indices_data);

  if(input->nDimension == 3)
    THCudaTensor_resize3d(state, output, nInputPlane, nOutputRows, nOutputCols);

  THCudaTensor_free(state, input);

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in SpatialMaxPooling.updateOutput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
}

void THNN_CudaSpatialMaxPooling_updateGradInput(THCState *state, THCudaTensor *input, THCudaTensor *gradOutput, THCudaTensor *gradInput, THCudaTensor *indices, int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode)
{
  THCUNN_assertSameGPU(state, 4, input, gradOutput, indices, gradInput);

  input = THCudaTensor_newContiguous(state, input);
  gradOutput = THCudaTensor_newContiguous(state, gradOutput);

  long nInputCols, nInputRows, nInputPlane, batchSize;
  long nOutputCols, nOutputRows;

  if (input->nDimension == 3) {
    nInputCols = input->size[2];
    nInputRows = input->size[1];
    nInputPlane = input->size[0];
    batchSize = 1;
  }
  else
  {
    nInputCols = input->size[3];
    nInputRows = input->size[2];
    nInputPlane = input->size[1];
    batchSize = input->size[0];
  }

  if(ceil_mode) {
    nOutputCols = ceil(float(nInputCols - kW + 2*padW) / float(dW)) + 1;
    nOutputRows = ceil(float(nInputRows - kH + 2*padH) / float(dH)) + 1;
  }
  else {
    nOutputCols = floor(float(nInputCols - kW + 2*padW) / float(dW)) + 1;
    nOutputRows = floor(float(nInputRows - kH + 2*padH) / float(dH)) + 1;
  }


  gradOutput = THCudaTensor_newContiguous(state, gradOutput);
  THCudaTensor_resizeAs(state, gradInput, input);

  int count = THCudaTensor_nElement(state, input);

  MaxPoolBackward <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>
      (count,
      THCudaTensor_data(state, gradOutput),
      THCudaTensor_data(state, indices),
      batchSize, nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols,
      kH, kW, dH, dW, padH, padW,
      THCudaTensor_data(state, gradInput));

  THCudaTensor_free(state, gradOutput);

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in SpatialMaxPooling.updateGradInput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
  // clean
  THCudaTensor_free(state, input);
  THCudaTensor_free(state, gradOutput);
}

