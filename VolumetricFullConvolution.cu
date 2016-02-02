#include "THCUNN.h"
#include "common.h"

#define CUDA_KERNEL_LOOP(i, n) \
for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
  i < (n); \
  i += blockDim.x * gridDim.x)

#define CUDA_CHECK(condition) \
/* Code block avoids redefinition of cudaError_t error */ \
 do { \
   cudaError_t error = condition; \
   THArgCheck(error == cudaSuccess, 2, cudaGetErrorString(error)); \
 } while (0)

template <typename Dtype>
__global__ void vol2col_kernel(const int n, const Dtype* data_im,
    const int length, const int height, const int width,
    const int ksize, const int kdepth, const int pad,
    const int temporal_pad, const int stride, const int temporal_stride,
    const int length_col, const int height_col, const int width_col,
    Dtype* data_col)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    int w_out = index % width_col;
    int h_out = (index / width_col ) % height_col;
    int l_out = (index / width_col / height_col) % length_col;
    int channel_in = index / width_col / height_col / length_col;
    int channel_out = channel_in * kdepth * ksize * ksize;
    int h_in = h_out * stride - pad;
    int w_in = w_out * stride - pad;
    int l_in = l_out * temporal_stride - temporal_pad;

    data_col += ((channel_out * length_col + l_out) * height_col + h_out) * width_col + w_out;
    data_im += ((channel_in * length + l_in) * height + h_in) * width + w_in;
    for (int k = 0; k < kdepth; ++k)
    {
      for (int i = 0; i < ksize; ++i)
      {
        for (int j = 0; j < ksize; ++j)
        {
          int l = l_in + k;
          int h = h_in + i;
          int w = w_in + j;
          *data_col = (l >= 0 && h >= 0 && w >= 0 && h < height && w < width && l < length) ?
              data_im[(k * height + i) * width + j] : 0;
          data_col += length_col * height_col * width_col;
        }
      }
    }
  }
}

template <typename Dtype>
void vol2col(const Dtype* data_im, const int channels, const int length,
    const int height, const int width, const int ksize, const int kdepth, const int pad,
    const int temporal_pad, const int stride, const int temporal_stride, Dtype* data_col)
{

  int length_col = (length + 2 * temporal_pad - kdepth) / temporal_stride + 1;
  int height_col = (height + 2 * pad - ksize) / stride + 1;
  int width_col = (width + 2 * pad - ksize) / stride + 1;
  int num_kernels = channels * length_col * height_col * width_col;

  vol2col_kernel<Dtype><<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
      num_kernels, data_im, length, height, width, ksize, kdepth, pad, temporal_pad, stride, temporal_stride,
      length_col, height_col, width_col, data_col);

}

// Explicit instantiation
template void vol2col<float>(const float* data_im, const int channels, const int length,
    const int height, const int width, const int ksize, const int kdepth, const int pad,
    const int temporal_pad, const int stride, const int temporal_stride, float* data_col);
template void vol2col<double>(const double* data_im, const int channels, const int length,
    const int height, const int width, const int ksize, const int kdepth, const int pad,
    const int temporal_pad, const int stride, const int temporal_stride, double* data_col);

template <typename Dtype>
__global__ void col2vol_kernel(const int n, const Dtype* data_col,
    const int length, const int height, const int width, const int channels,
    const int ksize, const int kdepth,
    const int pad, const int temporal_pad, const int stride, const int temporal_stride,
    const int length_col, const int height_col, const int width_col,
    Dtype* data_im)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    Dtype val = 0;
    int w = index % width + pad;
    int h = (index / width) % height + pad;
    int l = (index / width / height) % length + temporal_pad;
    int c = index / (width * height * length);
    // compute the start and end of the output
    int w_col_start = (w < ksize) ? 0 : (w - ksize) / stride + 1;
    int w_col_end = min(w / stride + 1, width_col);
    int h_col_start = (h < ksize) ? 0 : (h - ksize) / stride + 1;
    int h_col_end = min(h / stride + 1, height_col);
    int l_col_start = (l < kdepth) ? 0 : (l - kdepth) / temporal_stride + 1;
    int l_col_end = min(l / temporal_stride + 1, length_col);

    int offset = (c * kdepth * ksize * ksize + l * ksize * ksize + h * ksize + w) * length_col * height_col * width_col;

    int coeff_l_col = (1 - temporal_stride * ksize * ksize * length_col) * height_col * width_col;
    int coeff_h_col = (1 - stride * ksize * length_col * height_col) * width_col;
    int coeff_w_col = (1 - stride * length_col * height_col * width_col);

    for (int l_col = l_col_start; l_col < l_col_end; ++l_col) {
      for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
        for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
          val += data_col[offset + l_col * coeff_l_col + h_col * coeff_h_col + w_col * coeff_w_col];
        }
      }
    }
    data_im[index] = val;
  }
}

template <typename Dtype>
void col2vol(const Dtype* data_col, const int channels, const int length,
    const int height, const int width, const int ksize, const int kdepth, const int pad,
    const int temporal_pad, const int stride, const int temporal_stride, Dtype* data_im)
{
  int length_col = (length + 2 * temporal_pad - kdepth) / temporal_stride + 1;
  int height_col = (height + 2 * pad - ksize) / stride + 1;
  int width_col = (width + 2 * pad - ksize) / stride + 1;
  int num_kernels = channels * length * height * width;

  col2vol_kernel<Dtype><<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
      num_kernels, data_col, length, height, width, channels, ksize, kdepth, pad, temporal_pad, stride, temporal_stride,
      length_col, height_col, width_col, data_im);
}

// Explicit instantiation
template void col2vol<float>(const float* data_col, const int channels, const int length,
    const int height, const int width, const int ksize, const int kdepth, const int pad,
    const int temporal_pad, const int stride, const int temporal_stride, float* data_im);
template void col2vol<double>(const double* data_col, const int channels, const int length,
    const int height, const int width, const int ksize, const int kdepth, const int pad,
    const int temporal_pad, const int stride, const int temporal_stride, double* data_im);

void THNN_CudaVolumetricFullConvolution_updateOutput(
  THCState *state,
  THCudaTensor *input,
  THCudaTensor *output,
  THCudaTensor *weight,
  THCudaTensor *bias,
  THCudaTensor *finput,
  THCudaTensor *fgradInput,
  int dT, int dW, int dH,
  int pT, int pW, int pH)
{
  // number of input/output planes and kernel size is indirectly defined by the weight tensor
  THArgCheck(weight->nDimension == 5, 4,
    "5D weight tensor is expected (nOutputPlane x nInputPlane x kT x kH x kW)"
  );

  int nOutputPlane = (int)weight->size[0];
  int nInputPlane  = (int)weight->size[1];
  int kT           = (int)weight->size[2];
  int kW           = (int)weight->size[3];
  int kH           = (int)weight->size[4];

  THCudaTensor *columns = finput;
  THCudaTensor *ones = fgradInput;

  int inputDepth   = input->size[2];
  int inputHeight  = input->size[3];
  int inputWidth   = input->size[4];

  int outputDepth  = (inputDepth - 1)  * dT - 2 * pT + kT;
  int outputHeight = (inputHeight - 1) * dH - 2 * pH + kH;
  int outputWidth  = (inputWidth - 1)  * dW - 2 * pW + kW;

  THAssert(THCudaTensor_checkGPU(state, 6, input, output, weight, bias, columns, ones));
  THArgCheck(input->nDimension == 5, 2, "5D (batch mode) tensor is expected");
  THArgCheck(kH == kW && pH == pW, 2, "kH == kW && pH == pW is expected");

  // Batch size
  long batchSize = input->size[0];

  // Figure out the dimensions for individual gemms.
  int M_ = nInputPlane;
  int K_ = nOutputPlane * kT * kH * kW;
  int N_ = inputDepth * inputHeight * inputWidth;
  int N0_ = outputDepth * outputHeight * outputWidth;

  // Resize output
  THCudaTensor_resize5d(state, output, batchSize, nOutputPlane, outputDepth,
                        outputHeight, outputWidth);

  // Resize temporary columns
  THCudaTensor_resize5d(state, columns, 1, nOutputPlane * kT * kH * kW, inputDepth, inputHeight, inputWidth);

  // Define a buffer of ones, for bias accumulation
  // Note: this buffer can be shared with other modules, it only ever gets increased,
  // and always contains ones.
  if (ones->nDimension != 3 ||
    ones->size[0] * ones->size[1] * ones->size[2] < outputDepth * outputHeight * outputWidth)
  {
    // Resize plane and fill with ones...
    THCudaTensor_resize3d(state, ones, outputDepth, outputHeight, outputWidth);
    THCudaTensor_fill(state, ones, 1);
  }

  // Helpers
  THCudaTensor *input_n = THCudaTensor_new(state);
  THCudaTensor *output_n = THCudaTensor_new(state);

  for (int n = 0; n < batchSize; ++n)
  {
    THCudaTensor_select(state, input_n, input, 0, n);
    THCudaTensor_select(state, output_n, output, 0, n);

    // do gemm
    THCudaBlas_gemm(state, 'n', 't', N_, K_, M_,
    1, THCudaTensor_data(state, input_n), N_,
    THCudaTensor_data(state, weight), K_,
    0, THCudaTensor_data(state, columns), N_);

    // col2vol from columns -> output
    col2vol<float>(THCudaTensor_data(state, columns), nOutputPlane, outputDepth, outputHeight, outputWidth,
    kH, kT, pH, pT, dH, dT,
    THCudaTensor_data(state, output_n));

    // third, add bias
    THCudaBlas_gemm(state, 'n', 'n', N0_, nOutputPlane,
    1, 1,
    THCudaTensor_data(state, ones), N0_,
    THCudaTensor_data(state, bias), 1,
    1, THCudaTensor_data(state, output_n), N0_);

  }
  THCudaTensor_free(state, input_n);
  THCudaTensor_free(state, output_n);
}

void THNN_CudaVolumetricFullConvolution_updateGradInput(
  THCState *state,
  THCudaTensor *input,
  THCudaTensor *gradOutput,
  THCudaTensor *gradInput,
  THCudaTensor *weight,
  THCudaTensor *finput,
  THCudaTensor *fgradInput,
  int dT, int dW, int dH,
  int pT, int pW, int pH)
{
  // number of input/output planes and kernel size is indirectly defined by the weight tensor
  THArgCheck(weight->nDimension == 5, 4,
    "5D weight tensor is expected (nOutputPlane x nInputPlane x kT x kH x kW)"
  );

  int nOutputPlane = (int)weight->size[0];
  int nInputPlane  = (int)weight->size[1];
  int kT           = (int)weight->size[2];
  int kW           = (int)weight->size[3];
  int kH           = (int)weight->size[4];

  THCudaTensor *gradColumns = finput;

  THAssert(THCudaTensor_checkGPU(state, 5, input, gradOutput, weight,
                                 gradColumns, gradInput));
  THArgCheck(input->nDimension == 5, 2, "5D (batch mode) tensor is expected");
  THArgCheck(kH == kW && pH == pW, 2, "kH == kW && pH == pW is expected");

  int inputDepth   = input->size[2];
  int inputHeight  = input->size[3];
  int inputWidth   = input->size[4];

  int outputDepth  = (inputDepth - 1) * dT - 2 * pT + kT;
  int outputHeight = (inputHeight - 1) * dH - 2 * pH + kH;
  int outputWidth  = (inputWidth - 1) * dW - 2 * pW + kW;

  // Batch size
  int batchSize = input->size[0];

  // Figure out the dimensions for individual gemms.
  int M_ = nInputPlane;
  int K_ = nOutputPlane * kT * kH * kW;
  int N_ = inputDepth * inputHeight * inputWidth;

  // Resize output
  THCudaTensor_resize5d(state, gradInput, batchSize, nInputPlane, inputDepth, inputHeight, inputWidth);

  // Resize temporary columns
  THCudaTensor_resize5d(state, gradColumns, 1, nOutputPlane * kT * kH * kW, inputDepth, inputHeight, inputWidth);

  // Helpers
  THCudaTensor *gradInput_n = THCudaTensor_new(state);
  THCudaTensor *gradOutput_n = THCudaTensor_new(state);

  // For each n in batch, do:
  for (int n = 0; n < batchSize; n++)
  {
    THCudaTensor_select(state, gradInput_n, gradInput, 0, n);
    THCudaTensor_select(state, gradOutput_n, gradOutput, 0, n);

    // vol2col from gradOutput to gradColumns
    vol2col<float>(
      THCudaTensor_data(state, gradOutput_n),
      nOutputPlane,
      outputDepth, outputHeight, outputWidth,
      kH, kT, pH,
      pT, dH, dT,
      THCudaTensor_data(state, gradColumns)
    );

    // gemm to compute gradInput
    THCudaBlas_gemm(
      state, 'n', 'n', N_, M_, K_,
      1, THCudaTensor_data(state, gradColumns), N_,
      THCudaTensor_data(state, weight), K_,
      0, THCudaTensor_data(state, gradInput_n), N_
    );
  }

  // Free
  THCudaTensor_free(state, gradInput_n);
  THCudaTensor_free(state, gradOutput_n);
}

void THNN_CudaVolumetricFullConvolution_accGradParameters(
  THCState *state,
  THCudaTensor *input,
  THCudaTensor *gradOutput,
  THCudaTensor *gradWeight,
  THCudaTensor *gradBias,
  THCudaTensor *finput,
  THCudaTensor *fgradInput,
  int dT, int dW, int dH,
  int pT, int pW, int pH,
  float scale)
{
  // number of input/output planes and kernel size is indirectly defined by the gradWeight tensor
  THArgCheck(gradWeight->nDimension == 5, 4,
    "5D gradWeight tensor is expected (nOutputPlane x nInputPlane x kT x kH x kW)"
  );

  int nOutputPlane = (int)gradWeight->size[0];
  int nInputPlane  = (int)gradWeight->size[1];
  int kT           = (int)gradWeight->size[2];
  int kW           = (int)gradWeight->size[3];
  int kH           = (int)gradWeight->size[4];

  THCudaTensor *gradColumns = finput;
  THCudaTensor *ones = fgradInput;

  THAssert(THCudaTensor_checkGPU(state, 6, input, gradOutput, gradWeight,
                                 gradBias, gradColumns, ones));
  THArgCheck(input->nDimension == 5, 2, "5D (batch mode) tensor is expected");
  THArgCheck(kH == kW && pH == pW, 2, "kH == kW && pH == pW is expected");

  THCudaTensor_resize1d(state, gradBias, nOutputPlane);
  THCudaTensor_resize5d(state, gradWeight, nOutputPlane, nInputPlane, kT, kH, kW);

  int inputDepth   = input->size[2];
  int inputHeight  = input->size[3];
  int inputWidth   = input->size[4];

  int outputDepth  = (inputDepth - 1) * dT - 2 * pT + kT;
  int outputHeight = (inputHeight - 1) * dH - 2 * pH + kH;
  int outputWidth  = (inputWidth - 1) * dW - 2 * pW + kW;

  // Batch size
  long batchSize = input->size[0];

  // Figure out the dimensions for individual gemms.
  int M_ = nInputPlane;
  int K_ = nOutputPlane * kT * kH * kW;
  int N_ = inputDepth * inputHeight * inputWidth;
  int N0_ = outputDepth * outputHeight * outputWidth;

  // Resize temporary columns
  THCudaTensor_resize5d(state, gradColumns, 1, nOutputPlane * kT * kH * kW, inputDepth, inputHeight, inputWidth);

  if (ones->nDimension != 3 ||
    ones->size[0] * ones->size[1] * ones->size[2] < outputDepth * outputHeight * outputWidth)
  {
    // Resize plane and fill with ones...
    THCudaTensor_resize3d(state, ones, outputDepth, outputHeight, outputWidth);
    THCudaTensor_fill(state, ones, 1);
  }

  // Helpers
  THCudaTensor *input_n = THCudaTensor_new(state);
  THCudaTensor *gradOutput_n = THCudaTensor_new(state);

  // reset gradBias = 0
  CUDA_CHECK(cudaMemset(THCudaTensor_data(state, gradBias), 0,
      sizeof(float) * nOutputPlane));

  // reset gradWeight = 0
  CUDA_CHECK(cudaMemset(THCudaTensor_data(state, gradWeight), 0,
              sizeof(float) * M_ * K_));

  // For each n in batch, do:
  for (int n = 0; n < batchSize; n++)
  {
    THCudaTensor_select(state, input_n, input, 0, n);
    THCudaTensor_select(state, gradOutput_n, gradOutput, 0, n);

    // accumulate gradBias
    THCudaBlas_gemv(
      state, 't', N0_, nOutputPlane, 1,
      THCudaTensor_data(state, gradOutput_n), N0_,
      THCudaTensor_data(state, ones), 1,
      1,
      THCudaTensor_data(state, gradBias), 1
    );

    vol2col<float>(
      THCudaTensor_data(state, gradOutput_n),
      nOutputPlane,
      outputDepth, outputHeight, outputWidth,
      kH, kT, pH,
      pT, dH, dT,
      THCudaTensor_data(state, gradColumns)
    );

    // accummulate gradWeight
    THCudaBlas_gemm(
      state, 't', 'n', K_, M_, N_,
      1, THCudaTensor_data(state, gradColumns), N_,
      THCudaTensor_data(state, input_n), N_,
      1, THCudaTensor_data(state, gradWeight), K_
    );
  }

  // Free
  THCudaTensor_free(state, input_n);
  THCudaTensor_free(state, gradOutput_n);
}
