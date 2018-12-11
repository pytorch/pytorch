#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THCUNN/generic/SpatialCrossMapLRN.cu"
#else

void THNN_(LRNforward)(THCState* state, THCTensor* input, THCTensor* output,
    THCTensor* scale, int local_size, accreal alpha_, accreal beta_, accreal k_)
{
  scalar_t alpha = ScalarConvert<accreal, scalar_t>::to(alpha_);
  scalar_t beta = ScalarConvert<accreal, scalar_t>::to(beta_);
  scalar_t k = ScalarConvert<accreal, scalar_t>::to(k_);

  THCTensor_(resizeAs)(state, output, input);
  THCTensor_(resizeAs)(state, scale, input);

  int batchSize;
  int nInputPlane;
  int imsize_h;
  int imsize_w;

  if (input->dim() == 3) {
    batchSize = 1;
    nInputPlane = input->size(0);
    imsize_h = input->size(1);
    imsize_w = input->size(2);
  }
  else
  {
    batchSize = input->size(0);
    nInputPlane = input->size(1);
    imsize_h = input->size(2);
    imsize_w = input->size(3);
  }

  input = THCTensor_(newContiguous)(state, input);

  int n_threads = batchSize * imsize_h * imsize_w;
  LRNFillScale<scalar_t, accreal> <<<GET_BLOCKS(n_threads), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
      n_threads, THCTensor_(data)(state, input), batchSize, nInputPlane, imsize_h, imsize_w, local_size,
      alpha / local_size, k, THCTensor_(data)(state, scale));
  n_threads *= nInputPlane;
  THCudaCheck(cudaGetLastError());
  LRNComputeOutput<<<GET_BLOCKS(n_threads), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
    n_threads, THCTensor_(data)(state, input), THCTensor_(data)(state, scale), -beta, THCTensor_(data)(state, output));
  THCudaCheck(cudaGetLastError());

  THCTensor_(free)(state, input);
}


void THNN_(LRNbackward)(THCState* state, THCTensor* input, THCTensor* output,
    THCTensor* gradOutput, THCTensor* gradInput, THCTensor* scale,
    int local_size, accreal alpha_, accreal beta_, accreal k_)
{
  scalar_t alpha = ScalarConvert<accreal, scalar_t>::to(alpha_);
  scalar_t beta = ScalarConvert<accreal, scalar_t>::to(beta_);
  scalar_t k = ScalarConvert<accreal, scalar_t>::to(k_);
  (void) k;
  THCTensor_(resizeAs)(state, gradInput, input);

  int batchSize;
  int nInputPlane;
  int imsize_h;
  int imsize_w;

  if (input->dim() == 3) {
    batchSize = 1;
    nInputPlane = input->size(0);
    imsize_h = input->size(1);
    imsize_w = input->size(2);
  }
  else
  {
    batchSize = input->size(0);
    nInputPlane = input->size(1);
    imsize_h = input->size(2);
    imsize_w = input->size(3);
  }

  input = THCTensor_(newContiguous)(state, input);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);

  int n_threads = batchSize * imsize_h * imsize_w;
  LRNComputeDiff<scalar_t, accreal> <<<GET_BLOCKS(n_threads), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
      n_threads, THCTensor_(data)(state, input), THCTensor_(data)(state, output),
      THCTensor_(data)(state, scale), THCTensor_(data)(state, gradOutput), batchSize, nInputPlane, imsize_h, imsize_w,
      local_size, -beta, ScalarConvert<int, scalar_t>::to(2) * alpha * beta / local_size,
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
  THNN_(LRNforward)(state, input, output, scale, size, alpha, beta, k);
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
  THNN_(LRNbackward)(state, input, output, gradOutput, gradInput, scale, size, alpha, beta, k);
}

#endif
