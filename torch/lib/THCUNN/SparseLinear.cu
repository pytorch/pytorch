#include "THCUNN.h"
#include <cusparse.h>
#include <thrust/device_vector.h>

static cusparseHandle_t cusparse_handle = 0;

static void init_cusparse() {
  if (cusparse_handle == 0) {
    cusparseStatus_t status = cusparseCreate(&cusparse_handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      THError("CUSPARSE Library initialization failed");
    }
  }
}

static bool checkInput(THCudaTensor* t)
{
  return t->nDimension == 2 && t->size[1] == 3;
}

static bool checkSize2D(THCudaTensor* t, long size0, long size1)
{
  return t->nDimension == 2 && t->size[0] == size0 && t->size[1] == size1;
}

static bool checkSize1D(THCudaTensor* t, long size0)
{
  return t->nDimension == 1 && t->size[0] == size0;
}

void THNN_CudaSparseLinear_updateOutput(THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          THCudaTensor *weight,
          THCudaTensor *bias)
{
  THAssert(THCudaTensor_checkGPU(state, 4, input, output, weight, bias));

  long h;
  long outDim = THCudaTensor_size(state, weight, 0);
  long inDim = THCudaTensor_size(state, weight, 1);

  THArgCheck(checkInput(input), 2, "input size must be nnz x 3");
  THArgCheck(THCudaTensor_nDimension(state, output) == 2, 3, "output must be batchsize x outputsize");
  THArgCheck(checkSize1D(bias, outDim), 5, "bias size wrong");

  long batchnum = THCudaTensor_size(state, output, 0);
  long nnz = THCudaTensor_size(state, input, 0);

  THCudaTensor *buffer = THCudaTensor_new(state);
  THCudaTensor *sel = THCudaTensor_new(state);
  THCudaTensor *values = THCudaTensor_new(state);
  THCudaIntTensor *rowbuf = THCudaIntTensor_new(state);
  THCudaIntTensor *csrPtrs = THCudaIntTensor_new(state);
  THCudaIntTensor *colInds = THCudaIntTensor_new(state);

  THCudaTensor_resize1d(state, values, nnz);
  THCudaIntTensor_resize1d(state, rowbuf, nnz);
  THCudaIntTensor_resize1d(state, colInds, nnz);
  THCudaIntTensor_resize1d(state, csrPtrs, batchnum+1);

  // Get data ready for cusparse, need CudaInt buffers
  // We do not need to sort, since rows are already in order
  // If rows might get out of order in future implementations, or if cusparse
  //    complains with an illegal memory access, sort like we do in AccGradParameters
  THCudaTensor_select(state, sel, input, 1, 0);
  THCudaIntTensor_copyCudaFloat(state, rowbuf, sel);
  THCudaTensor_select(state, sel, input, 1, 1);
  THCudaIntTensor_copyCudaFloat(state, colInds, sel);
  THCudaTensor_select(state, sel, input, 1, 2);
  THCudaTensor_copyCuda(state, values, sel);

  init_cusparse();
  cusparseXcoo2csr(cusparse_handle,
      THCudaIntTensor_data(state, rowbuf), nnz, batchnum, 
      THCudaIntTensor_data(state, csrPtrs), CUSPARSE_INDEX_BASE_ONE);

  // output = bias
  THCudaTensor_resize2d(state, buffer, outDim, batchnum);
  THCudaTensor_zero(state, buffer);
  for (h=0; h<batchnum; h++) {
    THCudaTensor_select(state, sel, buffer, 1, h);
    THCudaTensor_copy(state, sel, bias);
  }
  
  // output = W * x
  float one = 1;
  cusparseMatDescr_t descr = 0;
  cusparseCreateMatDescr(&descr);
  cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ONE);  
  cusparseScsrmm(cusparse_handle,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      batchnum, outDim, inDim, nnz,
      &one,
      descr,
      THCudaTensor_data(state, values),
      THCudaIntTensor_data(state, csrPtrs),
      THCudaIntTensor_data(state, colInds),
      THCudaTensor_data(state, weight), inDim,
      &one, THCudaTensor_data(state, buffer), batchnum
  );
  THCudaTensor_transpose(state, buffer, NULL, 0, 1);

  // We do work in the buffer to keep the output contiguous
  THCudaTensor_copy(state, output, buffer);

  cusparseDestroyMatDescr(descr); 
  descr = 0;
  THCudaTensor_free(state, buffer);
  THCudaTensor_free(state, sel);
  THCudaTensor_free(state, values);
  THCudaIntTensor_free(state, rowbuf);
  THCudaIntTensor_free(state, colInds);
  THCudaIntTensor_free(state, csrPtrs);
}

void THNN_CudaSparseLinear_accGradParameters(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradWeight,
          THCudaTensor *gradBias,
          THCudaTensor *weight,
          THCudaTensor *bias,
          double weightDecay,
          double scale)
{
  long outDim = THCudaTensor_size(state, weight, 0);
  long inDim = THCudaTensor_size(state, weight, 1);

  THArgCheck(checkInput(input), 2, "input size must be batchsize x nnz x 2");
  THArgCheck(checkSize2D(gradWeight, outDim, inDim), 4, "gradWeight size wrong");
  THArgCheck(checkSize1D(gradBias, outDim), 5, "gradBias size wrong");
  THArgCheck(THCudaTensor_isContiguous(state, gradOutput), 3,
             "gradOutput must be contiguous");

  long nnz = THCudaTensor_size(state, input, 0);
  long batchnum = THCudaTensor_size(state, gradOutput, 0);

  THCudaTensor *buf = THCudaTensor_new(state);
  THCudaTensor *cols = THCudaTensor_new(state);
  THCudaTensor *sel = THCudaTensor_new(state);
  THCudaTensor *inds = THCudaTensor_new(state);
  THCudaTensor *values = THCudaTensor_new(state);
  THCudaIntTensor *colbuf = THCudaIntTensor_new(state);
  THCudaIntTensor *colPtrs = THCudaIntTensor_new(state);
  THCudaIntTensor *rowInds = THCudaIntTensor_new(state);

  THCudaTensor_select(state, sel, input, 1, 0); // rowInds
  THCudaTensor_select(state, cols, input, 1, 1); // colInds
  THCudaTensor_cadd(state, buf, sel, batchnum, cols); // colInds * buatchdim + rowInds
  THCudaTensor_sort(state, buf, inds, buf, 0, 0); // Indicies are now in ind
  THCudaTensor_indexSelect(state, buf, input, 0, inds);

  THCudaTensor_resize1d(state, values, nnz);
  THCudaIntTensor_resize1d(state, colbuf, nnz);
  THCudaIntTensor_resize1d(state, rowInds, nnz);
  THCudaIntTensor_resize1d(state, colPtrs, inDim+1);

  // Get data ready for cusparse, need CudaInt buffers
  THCudaTensor_select(state, sel, buf, 1, 0);
  THCudaIntTensor_copyCudaFloat(state, rowInds, sel);
  THCudaTensor_select(state, sel, buf, 1, 1);
  THCudaIntTensor_copyCudaFloat(state, colbuf, sel);
  THCudaTensor_select(state, sel, buf, 1, 2);
  THCudaTensor_copyCuda(state, values, sel);

  init_cusparse();
  // Secretly coo2csc
  cusparseXcoo2csr(cusparse_handle,
      THCudaIntTensor_data(state, colbuf), nnz, inDim, 
      THCudaIntTensor_data(state, colPtrs), CUSPARSE_INDEX_BASE_ONE);

  // FORTRAN expects contiguous col-major matricies
  THCudaTensor_transpose(state, gradOutput, NULL, 0, 1);
  THCudaTensor_resize2d(state, buf, batchnum, outDim);
  THCudaTensor_copy(state, buf, gradOutput);
  THCudaTensor_transpose(state, gradOutput, NULL, 0, 1); // Restore gradOutput

  float one = 1;
  cusparseMatDescr_t descr = 0;
  cusparseCreateMatDescr(&descr);
  cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ONE);  
  cusparseScsrmm(cusparse_handle,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      inDim, outDim, batchnum, nnz,
      &one,
      descr,
      THCudaTensor_data(state, values),
      THCudaIntTensor_data(state, colPtrs),
      THCudaIntTensor_data(state, rowInds),
      THCudaTensor_data(state, buf), batchnum,
      &one, THCudaTensor_data(state, gradWeight), inDim
  );

  THCudaTensor_sum(state, buf, gradOutput, 0);
  THCudaTensor_resize1d(state, buf, outDim);
  THCudaTensor_cadd(state, gradBias, gradBias, scale, buf);

  if (weightDecay != 0)
  {
    THCudaTensor_cadd(state, gradWeight, gradWeight, weightDecay, weight);
    THCudaTensor_cadd(state, gradBias, gradBias, weightDecay, bias);
  }

  THCudaTensor_free(state, buf);
  THCudaTensor_free(state, sel);
  THCudaTensor_free(state, cols);
  THCudaTensor_free(state, inds);
  THCudaTensor_free(state, values);
  THCudaIntTensor_free(state, colbuf);
  THCudaIntTensor_free(state, rowInds);
  THCudaIntTensor_free(state, colPtrs);
}

void THNN_CudaSparseLinear_legacyUpdateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          THCudaTensor *weight,
          THCudaTensor *bias) {
  THError("CUDA does not support legacy input format, please use a table of nnz x 2 vectors");
}
void THNN_CudaSparseLinear_legacyAccGradParameters(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradWeight,
          THCudaTensor *gradBias,
          THCudaTensor *weight,
          THCudaTensor *bias,
          double weightDecay,
          double scale) {
  THError("CUDA does not support legacy input format, please use a table of nnz x 2 vectors");
}

// Dense updates are pretty fast on the GPU
void THNN_CudaSparseLinear_zeroGradParameters(
          THCState *state,
          THCudaTensor *gradWeight,
          THCudaTensor *gradBias,
          THCudaTensor *lastInput) {
  THCudaTensor_zero(state, gradWeight);
  THCudaTensor_zero(state, gradBias);
}

TH_API void THNN_CudaSparseLinear_updateParameters(
          THCState *state,
          THCudaTensor *weight,
          THCudaTensor *bias,
          THCudaTensor *gradWeight,
          THCudaTensor *gradBias,
          THCudaTensor *lastInput,
          double learningRate) {
  THCudaTensor_cadd(state, weight, weight, -learningRate, gradWeight);
  THCudaTensor_cadd(state, bias, bias, -learningRate, gradBias);
}

void THNN_CudaSparseLinear_cudaClearState(THCState *state) {
}

