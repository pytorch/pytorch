#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/SpatialDepthwiseConvolution.cu"
#else

void THNN_(SpatialDepthwiseConvolution_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCTensor *weight,
                  THCTensor *bias,
                  int kW, int kH,
                  int dW, int dH,
                  int padW, int padH,
                  int dilationW, int dilationH)
{
  THCUNN_assertSameGPU(state, 3, input, output, weight);

  // Only handle 4D Input Tensors for now
  THAssert(!input->is_empty() && THCTensor_(nDimensionLegacyNoScalars)(state, input) == 4);
  THAssert(!weight->is_empty() && THCTensor_(nDimensionLegacyNoScalars)(state, weight) == 4);

  // We assume that the input and weight Tensors are shaped properly by
  // the caller, so we verify that here to some extent

  // Weight Tensor is shape (output_channels, 1, kH, kW)
  THAssert(weight->size(1) == 1);

  // Input Tensor is shape (N, input_channels, H, W)
  // We verify that the # of output_channels is a multiple of input_channels
  THAssert(weight->size(0) % input->size(1) == 0);

  // Bias has same # of channels as output
  if (bias) {
    THAssert(THTensor_sizeLegacyNoScalars(bias, 0) == weight->size(0));
  }

  input = THCTensor_(newContiguous)(state, input);
  weight = THCTensor_(newContiguous)(state, weight);
  bias = bias ? THCTensor_(newContiguous)(state, bias) : bias;

  // Following the behvaior of other THCUNN functions, we shape the output
  // Tensor ourselves

  int batchSize = input->size(0);
  int height = input->size(2);
  int width = input->size(3);
  int outputHeight = (height + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
  int outputWidth = (width + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  int outputChannels = weight->size(0);

  THCTensor_(resize4d)(state, output, batchSize, outputChannels, outputHeight, outputWidth);

  // Create THCDeviceTensor
  // Kernel currently relies upon all the Tensors to be contiguous, but we made
  // them contiguous above
  THCDeviceTensor<scalar_t, 4> dInput = toDeviceTensor<scalar_t, 4>(state, input);
  THCDeviceTensor<scalar_t, 4> dWeight = toDeviceTensor<scalar_t, 4>(state, weight);
  THCDeviceTensor<scalar_t, 4> dOutput = toDeviceTensor<scalar_t, 4>(state, output);
  THCDeviceTensor<scalar_t, 1> dBias;
  if (bias) {
    dBias = toDeviceTensor<scalar_t, 1>(state, bias);
  }

  int inputChannels = input->size(1);
  int depthwiseMultiplier = outputChannels / inputChannels;

  // One thread per output value
  int n = THCTensor_(nElement)(state, output);
  int blocks = GET_BLOCKS(n);
  dim3 grid(blocks);
  dim3 block(CUDA_NUM_THREADS);
  if (kW == 3 && kH == 3) {
  spatialDepthwiseConvolutionUpdateOutput<scalar_t, accreal, unsigned int, 3><<<grid, block, 0, THCState_getCurrentStream(state)>>>(
    dInput, dOutput, dWeight, dBias, bias != NULL, n, outputChannels, depthwiseMultiplier,
    width, height, outputWidth, outputHeight,
    kW, kH, dW, dH, padW, padH, dilationW, dilationH);
  } else if (kW == 1 && kH == 1) {
  spatialDepthwiseConvolutionUpdateOutput<scalar_t, accreal, unsigned int, 1><<<grid, block, 0, THCState_getCurrentStream(state)>>>(
    dInput, dOutput, dWeight, dBias, bias != NULL, n, outputChannels, depthwiseMultiplier,
    width, height, outputWidth, outputHeight,
    kW, kH, dW, dH, padW, padH, dilationW, dilationH);
  } else {
  spatialDepthwiseConvolutionUpdateOutput<scalar_t, accreal, unsigned int, 0><<<grid, block, 0, THCState_getCurrentStream(state)>>>(
    dInput, dOutput, dWeight, dBias, bias != NULL, n, outputChannels, depthwiseMultiplier,
    width, height, outputWidth, outputHeight,
    kW, kH, dW, dH, padW, padH, dilationW, dilationH);
  }

  THCudaCheck(cudaGetLastError());

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, weight);
  if (bias) THCTensor_(free)(state, bias);
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
  THCUNN_assertSameGPU(state, 3, gradOutput, gradInput, weight);

  // Only handle 4D Input Tensors for now
  THAssert(!input->is_empty() && THCTensor_(nDimensionLegacyNoScalars)(state, input) == 4);
  THAssert(!weight->is_empty() && THCTensor_(nDimensionLegacyNoScalars)(state, weight) == 4);
  THAssert(!gradOutput->is_empty() && THCTensor_(nDimensionLegacyNoScalars)(state, gradOutput) == 4);

  // Minimal shape checking, as above
  // Same # of elements in batch
  THAssert(input->size(0) == gradOutput->size(0));
  // Same # of filters as outputChannels
  THAssert(weight->size(0) == gradOutput->size(1));

  // Resize GradInput
  THCTensor_(resizeAs)(state, gradInput, input);

  int inputChannels = input->size(1);
  int height = input->size(2);
  int width = input->size(3);

  int outputChannels = gradOutput->size(1);
  int outputHeight = gradOutput->size(2);
  int outputWidth = gradOutput->size(3);

  int depthwiseMultiplier = outputChannels / inputChannels;

  THCDeviceTensor<scalar_t, 4> dGradOutput = toDeviceTensor<scalar_t, 4>(state, gradOutput);
  THCDeviceTensor<scalar_t, 4> dGradInput = toDeviceTensor<scalar_t, 4>(state, gradInput);
  THCDeviceTensor<scalar_t, 4> dWeight = toDeviceTensor<scalar_t, 4>(state, weight);

  // Kernel currently relies upon all the Tensors to be contiguous
  THAssert(dGradOutput.isContiguous());
  THAssert(dGradInput.isContiguous());
  THAssert(dWeight.isContiguous());

  // One thread per gradInput value
  int n = THCTensor_(nElement)(state, gradInput);
  int blocks = GET_BLOCKS(n);
  dim3 grid(blocks);
  dim3 block(CUDA_NUM_THREADS);
  if (kW == 3 && kH == 3)
    if (dW == 1 && dH == 1){
      spatialDepthwiseConvolutionUpdateGradInput<scalar_t, accreal, unsigned int, 3, 1><<<grid, block, 0, THCState_getCurrentStream(state)>>>(
      dGradOutput, dGradInput, dWeight, n, inputChannels, depthwiseMultiplier, outputChannels, width,
      height, outputWidth, outputHeight, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
    } else if (dW == 2 && dH == 2) {
      spatialDepthwiseConvolutionUpdateGradInput<scalar_t, accreal, unsigned int, 3, 2><<<grid, block, 0, THCState_getCurrentStream(state)>>>(
      dGradOutput, dGradInput, dWeight, n, inputChannels, depthwiseMultiplier, outputChannels, width,
      height, outputWidth, outputHeight, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
    } else {
      spatialDepthwiseConvolutionUpdateGradInput<scalar_t, accreal, unsigned int, 3, 0><<<grid, block, 0, THCState_getCurrentStream(state)>>>(
      dGradOutput, dGradInput, dWeight, n, inputChannels, depthwiseMultiplier, outputChannels, width,
      height, outputWidth, outputHeight, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
    }
  else if (kW == 1 && kH == 1)
    if (dW == 1 && dH == 1){
      spatialDepthwiseConvolutionUpdateGradInput<scalar_t, accreal, unsigned int, 1, 1><<<grid, block, 0, THCState_getCurrentStream(state)>>>(
      dGradOutput, dGradInput, dWeight, n, inputChannels, depthwiseMultiplier, outputChannels, width,
      height, outputWidth, outputHeight, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
    } else if (dW == 2 && dH == 2) {
      spatialDepthwiseConvolutionUpdateGradInput<scalar_t, accreal, unsigned int, 1, 2><<<grid, block, 0, THCState_getCurrentStream(state)>>>(
      dGradOutput, dGradInput, dWeight, n, inputChannels, depthwiseMultiplier, outputChannels, width,
      height, outputWidth, outputHeight, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
    } else {
      spatialDepthwiseConvolutionUpdateGradInput<scalar_t, accreal, unsigned int, 1, 0><<<grid, block, 0, THCState_getCurrentStream(state)>>>(
      dGradOutput, dGradInput, dWeight, n, inputChannels, depthwiseMultiplier, outputChannels, width,
      height, outputWidth, outputHeight, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
    }
  else
    if (dW == 1 && dH == 1){
      spatialDepthwiseConvolutionUpdateGradInput<scalar_t, accreal, unsigned int, 0, 1><<<grid, block, 0, THCState_getCurrentStream(state)>>>(
      dGradOutput, dGradInput, dWeight, n, inputChannels, depthwiseMultiplier, outputChannels, width,
      height, outputWidth, outputHeight, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
    } else if (dW == 2 && dH == 2) {
      spatialDepthwiseConvolutionUpdateGradInput<scalar_t, accreal, unsigned int, 0, 2><<<grid, block, 0, THCState_getCurrentStream(state)>>>(
      dGradOutput, dGradInput, dWeight, n, inputChannels, depthwiseMultiplier, outputChannels, width,
      height, outputWidth, outputHeight, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
    } else {
      spatialDepthwiseConvolutionUpdateGradInput<scalar_t, accreal, unsigned int, 0, 0><<<grid, block, 0, THCState_getCurrentStream(state)>>>(
      dGradOutput, dGradInput, dWeight, n, inputChannels, depthwiseMultiplier, outputChannels, width,
      height, outputWidth, outputHeight, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
    }


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
  THAssert(!input->is_empty() && THCTensor_(nDimensionLegacyNoScalars)(state, input) == 4);
  THAssert(!gradOutput->is_empty() && THCTensor_(nDimensionLegacyNoScalars)(state, gradOutput) == 4);
  THAssert(!gradWeight->is_empty() && THCTensor_(nDimensionLegacyNoScalars)(state, gradWeight) == 4);

  // Minimal shape checking as above
  // Same # of elements in batch
  THAssert(input->size(0) == gradOutput->size(0));
  // Same # of filters as outputChannels
  THAssert(gradWeight->size(0) == gradOutput->size(1));

  int batchSize = input->size(0);
  int inputChannels = input->size(1);
  int height = input->size(2);
  int width = input->size(3);

  int outputChannels = gradOutput->size(1);
  int outputHeight = gradOutput->size(2);
  int outputWidth = gradOutput->size(3);

  int depthwiseMultiplier = outputChannels / inputChannels;

  THCDeviceTensor<scalar_t, 4> dGradOutput = toDeviceTensor<scalar_t, 4>(state, gradOutput);
  THCDeviceTensor<scalar_t, 4> dInput = toDeviceTensor<scalar_t, 4>(state, input);
  THCDeviceTensor<scalar_t, 4> dGradWeight = toDeviceTensor<scalar_t, 4>(state, gradWeight);

  // Kernel currently relies upon all the Tensors to be contiguous
  THAssert(dGradOutput.isContiguous());
  THAssert(dInput.isContiguous());
  THAssert(dGradWeight.isContiguous());

  // We parallelize so that each block computes a single value in gradWeight
  int blocks = outputChannels * kH * kW;


  // Make sure we have enough threads to perform the reduction, and use this number
  // to create the shared memory size for the reduction
  dim3 grid(blocks);
  dim3 block(getGradParamsNumThreads(batchSize));
  int smem = block.x * sizeof(accreal);

  spatialDepthwiseConvolutionAccGradParameters<scalar_t, accreal, unsigned int><<<grid, block, smem, THCState_getCurrentStream(state)>>>(
      dGradOutput, dInput, dGradWeight, batchSize, inputChannels, outputChannels, depthwiseMultiplier,
      width, height, outputWidth, outputHeight, kW, kH, dW, dH, padW, padH, dilationW, dilationH);

  THCudaCheck(cudaGetLastError());
}

void THNN_(SpatialDepthwiseConvolutionTBC_updateOutput)(
                  THCState *state,
                  THCTensor *input,
                  THCTensor *output,
                  THCTensor *weight,
                  THCTensor *bias,
                  int kW, int kH,
                  int dW, int dH,
                  int padW, int padH,
                  int dilationW, int dilationH)
{
  THCUNN_assertSameGPU(state, 3, input, output, weight);

  // Only handle 4D Input Tensors for now
  THAssert(!input->is_empty() && THCTensor_(nDimensionLegacyNoScalars)(state, input) == 4);
  THAssert(!weight->is_empty() && THCTensor_(nDimensionLegacyNoScalars)(state, weight) == 4);

  // We assume that the input and weight Tensors are shaped properly by
  // the caller, so we verify that here to some extent

  // Weight Tensor is shape (kH, kW, 1, output_channels)
  THAssert(weight->size(2) == 1);

  // Input Tensor is shape (H, W, N, input_channels)
  // We verify that the # of output_channels is a multiple of input_channels
  THAssert(weight->size(3) % input->size(3) == 0);

  // Bias has same # of channels as output
  if (bias) {
    THAssert(THTensor_sizeLegacyNoScalars(bias, 0) == weight->size(3));
  }

  input = THCTensor_(newContiguous)(state, input);
  weight = THCTensor_(newContiguous)(state, weight);
  bias = bias ? THCTensor_(newContiguous)(state, bias) : bias;

  // Following the behvaior of other THCUNN functions, we shape the output
  // Tensor ourselves

  int batchSize = input->size(2);
  int height = input->size(0);
  int width = input->size(1);
  int outputHeight = (height + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
  int outputWidth = (width + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  int outputChannels = weight->size(3);

  THCTensor_(resize4d)(state, output, outputHeight, outputWidth, batchSize, outputChannels);

  // Create THCDeviceTensor
  // Kernel currently relies upon all the Tensors to be contiguous, but we made
  // them contiguous above
  THCDeviceTensor<scalar_t, 4> dInput = toDeviceTensor<scalar_t, 4>(state, input);
  THCDeviceTensor<scalar_t, 4> dWeight = toDeviceTensor<scalar_t, 4>(state, weight);
  THCDeviceTensor<scalar_t, 4> dOutput = toDeviceTensor<scalar_t, 4>(state, output);
  THCDeviceTensor<scalar_t, 1> dBias;
  if (bias) {
    dBias = toDeviceTensor<scalar_t, 1>(state, bias);
  }

  int inputChannels = input->size(3);
  int depthwiseMultiplier = outputChannels / inputChannels;

  // One thread per output value
  int n = THCTensor_(nElement)(state, output);
  int blocks = GET_BLOCKS(n);
  dim3 grid(blocks);
  dim3 block(CUDA_NUM_THREADS);
  if (kW == 3 && kH == 3) {
  spatialDepthwiseConvolutionTBCUpdateOutput<scalar_t, accreal, unsigned int, 3><<<grid, block, 0, THCState_getCurrentStream(state)>>>(
    dInput, dOutput, dWeight, dBias, bias != NULL, n, outputChannels, depthwiseMultiplier,
    width, height, outputWidth, outputHeight,
    kW, kH, dW, dH, padW, padH, dilationW, dilationH);
  } else if (kW == 1 && kH == 1) {
  spatialDepthwiseConvolutionTBCUpdateOutput<scalar_t, accreal, unsigned int, 1><<<grid, block, 0, THCState_getCurrentStream(state)>>>(
    dInput, dOutput, dWeight, dBias, bias != NULL, n, outputChannels, depthwiseMultiplier,
    width, height, outputWidth, outputHeight,
    kW, kH, dW, dH, padW, padH, dilationW, dilationH);
  } else {
  spatialDepthwiseConvolutionTBCUpdateOutput<scalar_t, accreal, unsigned int, 0><<<grid, block, 0, THCState_getCurrentStream(state)>>>(
    dInput, dOutput, dWeight, dBias, bias != NULL, n, outputChannels, depthwiseMultiplier,
    width, height, outputWidth, outputHeight,
    kW, kH, dW, dH, padW, padH, dilationW, dilationH);
  }

  THCudaCheck(cudaGetLastError());

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, weight);
  if (bias) THCTensor_(free)(state, bias);
}

void THNN_(SpatialDepthwiseConvolutionTBC_updateGradInput)(
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
  THCUNN_assertSameGPU(state, 3, gradOutput, gradInput, weight);

  // Only handle 4D Input Tensors for now
  THAssert(!input->is_empty() && THCTensor_(nDimensionLegacyNoScalars)(state, input) == 4);
  THAssert(!weight->is_empty() && THCTensor_(nDimensionLegacyNoScalars)(state, weight) == 4);
  THAssert(!gradOutput->is_empty() && THCTensor_(nDimensionLegacyNoScalars)(state, gradOutput) == 4);

  // input :  H,  W, N, iC
  // output: oH, oW, N, oC
  // weight: kH, kW, 1, oC
  // bias  : oC

  // Minimal shape checking, as above
  // Same # of elements in batch
  THAssert(input->size(2) == gradOutput->size(2));
  // Same # of filters as outputChannels
  THAssert(weight->size(3) == gradOutput->size(3));

  // Resize GradInput
  THCTensor_(resizeAs)(state, gradInput, input);

  int inputChannels = input->size(3);
  int height = input->size(0);
  int width = input->size(1);

  int outputChannels = gradOutput->size(3);
  int outputHeight = gradOutput->size(0);
  int outputWidth = gradOutput->size(1);

  int depthwiseMultiplier = outputChannels / inputChannels;

  THCDeviceTensor<scalar_t, 4> dGradOutput = toDeviceTensor<scalar_t, 4>(state, gradOutput);
  THCDeviceTensor<scalar_t, 4> dGradInput = toDeviceTensor<scalar_t, 4>(state, gradInput);
  THCDeviceTensor<scalar_t, 4> dWeight = toDeviceTensor<scalar_t, 4>(state, weight);

  // Kernel currently relies upon all the Tensors to be contiguous
  THAssert(dGradOutput.isContiguous());
  THAssert(dGradInput.isContiguous());
  THAssert(dWeight.isContiguous());

  // One thread per gradInput value
  int n = THCTensor_(nElement)(state, gradInput);
  int blocks = GET_BLOCKS(n);
  dim3 grid(blocks);
  dim3 block(CUDA_NUM_THREADS);
  if (kW == 3 && kH == 3)
    if (dW == 1 && dH == 1){
      spatialDepthwiseConvolutionTBCUpdateGradInput<scalar_t, accreal, unsigned int, 3, 1><<<grid, block, 0, THCState_getCurrentStream(state)>>>(
      dGradOutput, dGradInput, dWeight, n, inputChannels, depthwiseMultiplier, outputChannels, width,
      height, outputWidth, outputHeight, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
    } else if (dW == 2 && dH == 2) {
      spatialDepthwiseConvolutionTBCUpdateGradInput<scalar_t, accreal, unsigned int, 3, 2><<<grid, block, 0, THCState_getCurrentStream(state)>>>(
      dGradOutput, dGradInput, dWeight, n, inputChannels, depthwiseMultiplier, outputChannels, width,
      height, outputWidth, outputHeight, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
    } else {
      spatialDepthwiseConvolutionTBCUpdateGradInput<scalar_t, accreal, unsigned int, 3, 0><<<grid, block, 0, THCState_getCurrentStream(state)>>>(
      dGradOutput, dGradInput, dWeight, n, inputChannels, depthwiseMultiplier, outputChannels, width,
      height, outputWidth, outputHeight, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
    }
  else if (kW == 1 && kH == 1)
    if (dW == 1 && dH == 1){
      spatialDepthwiseConvolutionTBCUpdateGradInput<scalar_t, accreal, unsigned int, 1, 1><<<grid, block, 0, THCState_getCurrentStream(state)>>>(
      dGradOutput, dGradInput, dWeight, n, inputChannels, depthwiseMultiplier, outputChannels, width,
      height, outputWidth, outputHeight, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
    } else if (dW == 2 && dH == 2) {
      spatialDepthwiseConvolutionTBCUpdateGradInput<scalar_t, accreal, unsigned int, 1, 2><<<grid, block, 0, THCState_getCurrentStream(state)>>>(
      dGradOutput, dGradInput, dWeight, n, inputChannels, depthwiseMultiplier, outputChannels, width,
      height, outputWidth, outputHeight, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
    } else {
      spatialDepthwiseConvolutionTBCUpdateGradInput<scalar_t, accreal, unsigned int, 1, 0><<<grid, block, 0, THCState_getCurrentStream(state)>>>(
      dGradOutput, dGradInput, dWeight, n, inputChannels, depthwiseMultiplier, outputChannels, width,
      height, outputWidth, outputHeight, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
    }
  else
    if (dW == 1 && dH == 1){
      spatialDepthwiseConvolutionTBCUpdateGradInput<scalar_t, accreal, unsigned int, 0, 1><<<grid, block, 0, THCState_getCurrentStream(state)>>>(
      dGradOutput, dGradInput, dWeight, n, inputChannels, depthwiseMultiplier, outputChannels, width,
      height, outputWidth, outputHeight, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
    } else if (dW == 2 && dH == 2) {
      spatialDepthwiseConvolutionTBCUpdateGradInput<scalar_t, accreal, unsigned int, 0, 2><<<grid, block, 0, THCState_getCurrentStream(state)>>>(
      dGradOutput, dGradInput, dWeight, n, inputChannels, depthwiseMultiplier, outputChannels, width,
      height, outputWidth, outputHeight, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
    } else {
      spatialDepthwiseConvolutionTBCUpdateGradInput<scalar_t, accreal, unsigned int, 0, 0><<<grid, block, 0, THCState_getCurrentStream(state)>>>(
      dGradOutput, dGradInput, dWeight, n, inputChannels, depthwiseMultiplier, outputChannels, width,
      height, outputWidth, outputHeight, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
    }


  THCudaCheck(cudaGetLastError());
}

void THNN_(SpatialDepthwiseConvolutionTBC_accGradParameters)(
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

  // input :  H,  W, N, iC
  // output: oH, oW, N, oC
  // weight: kH, kW, 1, oC
  // bias  : oC

  // Only handle 4D Input Tensors for now
  THAssert(!input->is_empty() && THCTensor_(nDimensionLegacyNoScalars)(state, input) == 4);
  THAssert(!gradOutput->is_empty() && THCTensor_(nDimensionLegacyNoScalars)(state, gradOutput) == 4);
  THAssert(!gradWeight->is_empty() && THCTensor_(nDimensionLegacyNoScalars)(state, gradWeight) == 4);

  // Minimal shape checking as above
  // Same # of elements in batch
  THAssert(input->size(2) == gradOutput->size(2));
  // Same # of filters as outputChannels
  THAssert(gradWeight->size(3) == gradOutput->size(3));

  int batchSize = input->size(2);
  int inputChannels = input->size(3);
  int height = input->size(0);
  int width = input->size(1);

  int outputChannels = gradOutput->size(3);
  int outputHeight = gradOutput->size(0);
  int outputWidth = gradOutput->size(1);

  int depthwiseMultiplier = outputChannels / inputChannels;

  THCDeviceTensor<scalar_t, 4> dGradOutput = toDeviceTensor<scalar_t, 4>(state, gradOutput);
  THCDeviceTensor<scalar_t, 4> dInput = toDeviceTensor<scalar_t, 4>(state, input);
  THCDeviceTensor<scalar_t, 4> dGradWeight = toDeviceTensor<scalar_t, 4>(state, gradWeight);

  // Kernel currently relies upon all the Tensors to be contiguous
  THAssert(dGradOutput.isContiguous());
  THAssert(dInput.isContiguous());
  THAssert(dGradWeight.isContiguous());

  // One thread per gradWeight value
  int n = THCTensor_(nElement)(state, gradWeight);
  int blocks = GET_BLOCKS(n);
  dim3 grid(blocks);
  dim3 block(CUDA_NUM_THREADS);

  spatialDepthwiseConvolutionTBCAccGradParameters<scalar_t, accreal, unsigned int><<<grid, block, 0, THCState_getCurrentStream(state)>>>(
      dGradOutput, dInput, dGradWeight, batchSize, inputChannels, outputChannels, depthwiseMultiplier,
      width, height, outputWidth, outputHeight, kW, kH, dW, dH, padW, padH, dilationW, dilationH);

  THCudaCheck(cudaGetLastError());
}

#endif
