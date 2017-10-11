#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/DepthwiseConvolution.cu"
#else

static inline void THNN_(DepthwiseConvolution_shapeCheck)(
                         THCState *state,
                         THCTensor *input, THCTensor *gradOutput,
                         THCTensor *weight, THCTensor *bias,
                         int kW, int kH,
                         int dW, int dH,
                         int padW, int padH,
                         int dilationW, int dilationH) {

}

void THNN_(DepthwiseConvolution_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCTensor *weight,
                  int kW, int kH,
                  int dW, int dH,
                  int padW, int padH,
                  int dilationW, int dilationH)
{
  THCUNN_assertSameGPU(state, 3, input, output, weight);

  // Only handle 4D Input Tensors for now
  assert(THCTensor_(nDimension)(state, input) == 4);
  assert(THCTensor_(nDimension)(state, weight) == 4);

  // Calculate size of output and reshape
  int batchSize = input->size[0];

  // For now, we limit depthwise conv to 1-to-1 ratio between input
  // and output channels, i.e. there is no depthwise multiplier
  int inputChannels = input->size[1];

  int height = input->size[2];
  int width = input->size[3];

  /* int outputWidth = (width + 2*padW - (dilationW * (kW - 1) + 1)) / dW + 1; */
  /* int outputHeight = (height + 2*padH - (dilationH * (kH - 1) + 1)) / dH + 1; */

  /* int outputHeight = (height + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1; */
  /* int outputWidth = (width + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1; */

  int outputChannels = output->size[1];
  int outputHeight = output->size[2];
  int outputWidth = output->size[3];

  int depthwiseMultiplier = outputChannels / inputChannels;

  printf("output ch %d, output h %d, output w %d, depth multiplier %d\n", outputChannels, outputHeight, outputWidth, depthwiseMultiplier);

  /* THCTensor_(resize4d)(state, output, batchSize, inputChannels, outputHeight, outputWidth); */

  THCDeviceTensor<real, 4> dInput = toDeviceTensor<real, 4>(state, input);
  THCDeviceTensor<real, 4> dWeight = toDeviceTensor<real, 4>(state, weight);
  THCDeviceTensor<real, 4> dOutput = toDeviceTensor<real, 4>(state, output);

  // Just have enough blocks to handle all of the outputs...
  int n = THCTensor_(nElement)(state, output);
  int blocks = GET_BLOCKS(n);
  dim3 grid(blocks);
  dim3 block(CUDA_NUM_THREADS);

  /* dim3 grid(1); */
  /* dim3 block(1); */

  depthwiseConvolutionUpdateOutput<<<grid, block, 0, THCState_getCurrentStream(state)>>>(
    dInput, dOutput, dWeight, n, outputChannels, depthwiseMultiplier, width, height, outputWidth, outputHeight,
    kW, kH, dW, dH, padW, padH, dilationW, dilationH);

  THCudaCheck(cudaGetLastError());
}

void THNN_(DepthwiseConvolution_updateGradInput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradInput,
                  THCTensor *weight,
                  int kW, int kH,
                  int dW, int dH,
                  int padW, int padH,
                  int dilationW, int dilationH)
{
  THCUNN_assertSameGPU(state, 4, input, gradOutput, gradInput, weight);

  // Only handle 4D Input Tensors for now
  assert(THCTensor_(nDimension)(state, input) == 4);
  assert(THCTensor_(nDimension)(state, weight) == 4);
  assert(THCTensor_(nDimension)(state, gradOutput) == 4);

  // Assert GradOutput is contiguous
  assert(THCTensor_(isContiguous)(state, gradOutput));

  // Resize GradInput
  THCTensor_(resizeAs)(state, gradInput, input);

  int inputChannels = input->size[1];
  int height = input->size[2];
  int width = input->size[3];

  int outputChannels = gradOutput->size[1];
  int outputHeight = gradOutput->size[2];
  int outputWidth = gradOutput->size[3];

  int depthwiseMultiplier = outputChannels / inputChannels;

  THCDeviceTensor<real, 4> dGradOutput = toDeviceTensor<real, 4>(state, gradOutput);
  THCDeviceTensor<real, 4> dGradInput = toDeviceTensor<real, 4>(state, gradInput);
  THCDeviceTensor<real, 4> dWeight = toDeviceTensor<real, 4>(state, weight);

  int n = THCTensor_(nElement)(state, gradInput);
  int blocks = GET_BLOCKS(n);
  dim3 grid(blocks);
  dim3 block(CUDA_NUM_THREADS);

  /* dim3 grid(1); */
  /* dim3 block(1); */

  depthwiseConvolutionUpdateGradInput<<<grid, block, 0, THCState_getCurrentStream(state)>>>(
    dGradOutput, dGradInput, dWeight, n, inputChannels, depthwiseMultiplier, width, height, outputWidth,
    outputHeight, kW, kH, dW, dH, padW, padH, dilationW, dilationH);

  THCudaCheck(cudaGetLastError());
}

void THNN_(DepthwiseConvolution_accGradParameters)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *gradOutput,
                  THCTensor *gradWeight,
                  int kW, int kH,
                  int dW, int dH,
                  int padW, int padH,
                  int dilationW, int dilationH)
{
  THCUNN_assertSameGPU(state, 3, input, gradOutput, gradWeight);

  // Only handle 4D Input Tensors for now
  assert(THCTensor_(nDimension)(state, input) == 4);
  assert(THCTensor_(nDimension)(state, gradOutput) == 4);
  assert(THCTensor_(nDimension)(state, gradWeight) == 4);

  // Assert GradOutput is contiguous
  assert(THCTensor_(isContiguous)(state, gradOutput));

  // No stride, padding, dilation, yet...

  int batchSize = input->size[0];
  int inputChannels = input->size[1];
  int height = input->size[2];
  int width = input->size[3];

  int outputChannels = gradOutput->size[1];
  int outputHeight = gradOutput->size[2];
  int outputWidth = gradOutput->size[3];

  int depthwiseMultiplier = outputChannels / inputChannels;

  THCDeviceTensor<real, 4> dGradOutput = toDeviceTensor<real, 4>(state, gradOutput);
  THCDeviceTensor<real, 4> dInput = toDeviceTensor<real, 4>(state, input);
  THCDeviceTensor<real, 4> dGradWeight = toDeviceTensor<real, 4>(state, gradWeight);

  // We parallelize so that each block computes a single value in gradWeight
  int blocks = outputChannels * kH * kW;

  // Because each weight position is a function of convolving the gradOutput over
  // the input, we need batchSize * outputHeight * outputWidth individual calculations
  int n = batchSize * outputHeight * outputWidth;

  dim3 grid(blocks);

  // Probably can be smarter about picking the number of threads
  dim3 block(CUDA_NUM_THREADS);
  /* dim3 grid(1); */
  /* dim3 block(1); */

  int smem = CUDA_NUM_THREADS * sizeof(real);

  /* printf("blocks: %d, threads: %d\n", blocks, block.x); */

  depthwiseConvolutionAccGradParameters<<<grid, block, smem, THCState_getCurrentStream(state)>>>(
      dGradOutput, dInput, dGradWeight, batchSize, outputChannels, depthwiseMultiplier, n, width, height,
      outputWidth, outputHeight, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
  THCudaCheck(cudaGetLastError());
}

#endif
