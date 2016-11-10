#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/SpatialUpSamplingBilinear.cu"
#else

void THNN_(SpatialUpSamplingBilinear_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           int outputHeight,
           int outputWidth)
{
  // TODO: check argument shapes
  input = THCTensor_(newContiguous)(state, input);
  output = THCTensor_(newContiguous)(state, output);
  THCUNN_assertSameGPU(state, 2, input, output);
  THCTensor_(zero)(state, output);
  THCDeviceTensor<real, 4> idata = toDeviceTensor<real, 4>(state, input);
  THCDeviceTensor<real, 4> odata = toDeviceTensor<real, 4>(state, output);
  int height1 = idata.getSize(2);
  int width1 = idata.getSize(3);
  int height2 = odata.getSize(2);
  int width2 = odata.getSize(3);
  assert( height1 > 0 && width1 > 0 && height2 > 0 && width2 > 0);
  const accreal rheight= (height2 > 1) ? (accreal)(height1 - 1)/(height2 - 1) : accreal(0);
  const accreal rwidth = (width2 > 1) ? (accreal)(width1 - 1)/(width2 - 1) : accreal(0);
  const int num_kernels = height2 * width2;
  const int num_threads =
    THCState_getCurrentDeviceProperties(state)->maxThreadsPerBlock;
  cudaStream_t stream = THCState_getCurrentStream(state);
  caffe_gpu_interp2_kernel<real, accreal> <<<THCCeilDiv(num_kernels, num_threads), num_threads ,
   0 , stream>>>(num_kernels, rheight, rwidth, idata, odata);
  THCudaCheck(cudaGetLastError());
  THCTensor_(free)(state, input);
  THCTensor_(free)(state, output);
}


void THNN_(SpatialUpSamplingBilinear_updateGradInput)(
           THCState *state,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           int nbatch,
           int nchannels,
           int inputHeight,
           int inputWidth,
           int outputHeight,
           int outputWidth)
{
  // TODO: check argument shapes
  gradInput = THCTensor_(newContiguous)(state, gradInput);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);
  THCUNN_assertSameGPU(state, 2, gradOutput, gradInput);
  THCTensor_(zero)(state, gradInput);
  THCDeviceTensor<real, 4> data1 = toDeviceTensor<real, 4>(state, gradInput);
  THCDeviceTensor<real, 4> data2 = toDeviceTensor<real, 4>(state, gradOutput);
  int height1 = data1.getSize(2);
  int width1 = data1.getSize(3);
  int height2 = data2.getSize(2);
  int width2 = data2.getSize(3);
  assert(height1 > 0 && width1 > 0 && height2 > 0 && width2 > 0);
  const accreal rheight= (height2 > 1) ? (accreal)(height1 - 1)/(height2 - 1) : accreal(0);
  const accreal rwidth = (width2 > 1) ? (accreal)(width1 - 1) / (width2 - 1) : accreal(0);
  const int num_kernels = height2 * width2;
  const int num_threads =
    THCState_getCurrentDeviceProperties(state)->maxThreadsPerBlock;
  cudaStream_t stream = THCState_getCurrentStream(state);
  caffe_gpu_interp2_kernel_backward<real ,accreal> <<<THCCeilDiv(num_kernels, num_threads),
  num_threads, 0, stream>>>(num_kernels, rheight, rwidth, data1, data2);
  THCudaCheck(cudaGetLastError());
  THCTensor_(free)(state, gradInput);
  THCTensor_(free)(state, gradOutput);
}

#endif
