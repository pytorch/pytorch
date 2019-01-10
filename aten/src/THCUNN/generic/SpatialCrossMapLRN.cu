#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/SpatialCrossMapLRN.cu"
#else

void LRNforward(THCState* state, THCTensor* input, THCTensor* output,
    THCTensor* scale, int local_size, accreal alpha_, accreal beta_, accreal k_)
{
  real alpha = ScalarConvert<accreal, real>::to(alpha_);
  real beta = ScalarConvert<accreal, real>::to(beta_);
  real k = ScalarConvert<accreal, real>::to(k_);

  THCTensor_(resizeAs)(state, output, input);
  THCTensor_(resizeAs)(state, scale, input);

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

  input = THCTensor_(newContiguous)(state, input);

  int n_threads = batchSize * imsize_h * imsize_w;
  LRNFillScale<real, accreal> <<<GET_BLOCKS(n_threads), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
      n_threads, THCTensor_(data)(state, input), batchSize, nInputPlane, imsize_h, imsize_w, local_size,
      alpha / local_size, k, THCTensor_(data)(state, scale));
  n_threads *= nInputPlane;
  THCudaCheck(cudaGetLastError());
  LRNComputeOutput<<<GET_BLOCKS(n_threads), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
    n_threads, THCTensor_(data)(state, input), THCTensor_(data)(state, scale), -beta, THCTensor_(data)(state, output));
  THCudaCheck(cudaGetLastError());

  THCTensor_(free)(state, input);
}


void LRNbackward(THCState* state, THCTensor* input, THCTensor* output,
    THCTensor* gradOutput, THCTensor* gradInput, THCTensor* scale,
    int local_size, accreal alpha_, accreal beta_, accreal k_)
{
  real alpha = ScalarConvert<accreal, real>::to(alpha_);
  real beta = ScalarConvert<accreal, real>::to(beta_);
  real k = ScalarConvert<accreal, real>::to(k_);
  (void) k;
  THCTensor_(resizeAs)(state, gradInput, input);

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

  input = THCTensor_(newContiguous)(state, input);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);

  int n_threads = batchSize * imsize_h * imsize_w;
  LRNComputeDiff<real, accreal> <<<GET_BLOCKS(n_threads), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
      n_threads, THCTensor_(data)(state, input), THCTensor_(data)(state, output),
      THCTensor_(data)(state, scale), THCTensor_(data)(state, gradOutput), batchSize, nInputPlane, imsize_h, imsize_w,
      local_size, -beta, ScalarConvert<int, real>::to(2) * alpha * beta / local_size,
      THCTensor_(data)(state, gradInput));
  THCudaCheck(cudaGetLastError());

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, gradOutput);
}

void THNN_(SpatialCrossMapLRN_updateOutput)(
    THCState *state,
    THCTensor *input,
    THCTensor *output,
    THCTensor *scale,
    int size,
    accreal alpha,
    accreal beta,
    accreal k)
{
  LRNforward(state, input, output, scale, size, alpha, beta, k);
}

void THNN_(SpatialCrossMapLRN_updateGradInput)(
    THCState *state,
    THCTensor *input,
    THCTensor *gradOutput,
    THCTensor *gradInput,
    THCTensor *scale,
    THCTensor *output,
    int size,
    accreal alpha,
    accreal beta,
    accreal k)
{
  LRNbackward(state, input, output, gradOutput, gradInput, scale, size, alpha, beta, k);
}

#endif
