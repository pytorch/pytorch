#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/SpatialMaxUnpooling.cu"
#else

void THNN_(SpatialMaxUnpooling_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           THCIndexTensor *indices,
           int owidth, int oheight)
{
  THCUNN_assertSameGPU(state, 3, input, output, indices);
  THCUNN_argCheck(state, input->nDimension == 3 || input->nDimension == 4, 2, input,
                  "3D or 4D (batch mode) tensor expected for input, but got: %s");
  THCUNN_check_shape_indices(state, indices, input);

  int64_t nInputCols, nInputRows, nInputPlane, batchSize;

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

  input = THCTensor_(newContiguous)(state, input);
  indices = THCIndexTensor_(newContiguous)(state, indices);
  THCTensor_(resize4d)(state, output, batchSize, nInputPlane, oheight, owidth);
  THCTensor_(zero)(state, output);

  int count = THCTensor_(nElement)(state, input);

  MaxUnpoolForward <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>
      (count, THCTensor_(data)(state, input), THCIndexTensor_(data)(state, indices),
      batchSize, nInputPlane, nInputRows, nInputCols, oheight, owidth, THCTensor_(data)(state, output));
  THCudaCheck(cudaGetLastError());

  if(input->nDimension == 3)
    THCTensor_(resize3d)(state, output, nInputPlane, oheight, owidth);

  THCTensor_(free)(state, input);
  THCIndexTensor_(free)(state, indices);
}

void THNN_(SpatialMaxUnpooling_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           THCIndexTensor *indices,
           int owidth, int oheight)
{
  THCUNN_assertSameGPU(state, 4, input, gradOutput, indices, gradInput);
  THCUNN_check_shape_indices(state, indices, input);

  int64_t nInputCols, nInputRows, nInputPlane, batchSize;
  int dimw = 2;
  int dimh = 1;

  if (input->nDimension == 3) {
    nInputPlane = input->size[0];
    batchSize = 1;
  }
  else
  {
    ++dimw;
    ++dimh;
    nInputPlane = input->size[1];
    batchSize = input->size[0];
  }
  nInputCols = input->size[dimw];
  nInputRows = input->size[dimh];

  if(owidth!=gradOutput->size[dimw] || oheight!=gradOutput->size[dimh]){
     THError("Inconsistent gradOutput size. oheight= %d, owidth= %d, gradOutput: %dx%d",
             oheight, owidth,gradOutput->size[dimh],gradOutput->size[dimw]);
  }

  input = THCTensor_(newContiguous)(state, input);
  indices = THCIndexTensor_(newContiguous)(state, indices);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);
  THCTensor_(resizeAs)(state, gradInput, input);

  int count = THCTensor_(nElement)(state, input);

  MaxUnpoolBackward <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>
      (count, THCTensor_(data)(state, gradOutput), THCIndexTensor_(data)(state, indices),
      batchSize, nInputPlane, nInputRows, nInputCols, oheight, owidth, THCTensor_(data)(state, gradInput));
  THCudaCheck(cudaGetLastError());

  // clean
  THCTensor_(free)(state, input);
  THCIndexTensor_(free)(state, indices);
  THCTensor_(free)(state, gradOutput);
}

#endif
