#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/SpatialDepthwiseConvolution.cu"
#else

void THNN_(SpatialDepthwiseConvolution_updateOutput)(
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

  // We assume that the input and weight Tensors are shaped properly by
  // the caller, so we verify that here to some extent

  // Weight Tensor is shape (output_channels, 1, kH, kW)
  assert(weight->size[1] == 1);

  // Input Tensor is shape (N, input_channels, H, W)
  // We verify that the # of output_channels is a multiple of input_channels
  assert(weight->size[0] % input->size[1] == 0);

  // Following the behvaior of other THCUNN functions, we shape the output
  // Tensor ourselves

  int batchSize = input->size[0];
  int height = input->size[2];
  int width = input->size[3];
  int outputHeight = (height + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
  int outputWidth = (width + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  int outputChannels = weight->size[0];

  THCTensor_(resize4d)(state, output, batchSize, outputChannels, outputHeight, outputWidth);

  THCDeviceTensor<real, 4> dInput = toDeviceTensor<real, 4>(state, input);
  THCDeviceTensor<real, 4> dWeight = toDeviceTensor<real, 4>(state, weight);
  THCDeviceTensor<real, 4> dOutput = toDeviceTensor<real, 4>(state, output);

  // Kernel currently relies upon all the Tensors to be contiguous
  assert(dInput.isContiguous());
  assert(dWeight.isContiguous());
  assert(dOutput.isContiguous());

  int inputChannels = input->size[1];
  int depthwiseMultiplier = outputChannels / inputChannels;

  // One thread per output value
  int n = THCTensor_(nElement)(state, output);
  int blocks = GET_BLOCKS(n);
  dim3 grid(blocks);
  dim3 block(CUDA_NUM_THREADS);

  spatialDepthwiseConvolutionUpdateOutput<<<grid, block, 0, THCState_getCurrentStream(state)>>>(
    dInput, dOutput, dWeight, n, outputChannels, depthwiseMultiplier,
    width, height, outputWidth, outputHeight,
    kW, kH, dW, dH, padW, padH, dilationW, dilationH);

  THCudaCheck(cudaGetLastError());
}

void THNN_(SpatialDepthwiseConvolution_updateGradInput)(
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

  // Minimal shape checking, as above
  // Same # of elements in batch
  assert(input->size[0] == gradOutput->size[0]);
  // Same # of filters as outputChannels
  assert(weight->size[0] == gradOutput->size[1]);

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

  // Kernel currently relies upon all the Tensors to be contiguous
  assert(dGradOutput.isContiguous());
  assert(dGradInput.isContiguous());
  assert(dWeight.isContiguous());

  // One thread per gradInput value
  int n = THCTensor_(nElement)(state, gradInput);
  int blocks = GET_BLOCKS(n);
  dim3 grid(blocks);
  dim3 block(CUDA_NUM_THREADS);

  spatialDepthwiseConvolutionUpdateGradInput<<<grid, block, 0, THCState_getCurrentStream(state)>>>(
    dGradOutput, dGradInput, dWeight, n, inputChannels, depthwiseMultiplier, outputChannels, width,
    height, outputWidth, outputHeight, kW, kH, dW, dH, padW, padH, dilationW, dilationH);

  THCudaCheck(cudaGetLastError());
}

void THNN_(SpatialDepthwiseConvolution_accGradParameters)(
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

  // Minimal shape checking as above
  // Same # of elements in batch
  assert(input->size[0] == gradOutput->size[0]);
  // Same # of filters as outputChannels
  assert(gradWeight->size[0] == gradOutput->size[1]);

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

  // Kernel currently relies upon all the Tensors to be contiguous
  assert(dGradOutput.isContiguous());
  assert(dInput.isContiguous());
  assert(dGradWeight.isContiguous());

  // We parallelize so that each block computes a single value in gradWeight
  int blocks = outputChannels * kH * kW;

  // Because each weight position is a function of convolving the gradOutput over
  // the input, we need batchSize * outputHeight * outputWidth individual calculations
  int n = batchSize * outputHeight * outputWidth;

  // Make sure we have enough threads to perform the reduction, and use this number
  // to create the shared memory size for the reduction
  dim3 grid(blocks);
  dim3 block(std::min(nextHighestPowerOf2(n), (unsigned int64_t) CUDA_NUM_THREADS));
  int smem = block.x * sizeof(real);

  spatialDepthwiseConvolutionAccGradParameters<<<grid, block, smem, THCState_getCurrentStream(state)>>>(
      dGradOutput, dInput, dGradWeight, batchSize, inputChannels, outputChannels, depthwiseMultiplier, n,
      width, height, outputWidth, outputHeight, kW, kH, dW, dH, padW, padH, dilationW, dilationH);

  THCudaCheck(cudaGetLastError());
}

#endif
