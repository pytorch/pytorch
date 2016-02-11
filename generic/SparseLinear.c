#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SparseLinear.c"
#else

#ifdef _OPENMP
#include <omp.h>
#endif

static bool THNN_(checkInput)(THTensor* t)
{
  return t->nDimension == 2 && t->size[1] == 2;
}

static bool THNN_(checkSize2D)(THTensor* t, long size0, long size1)
{
  return t->nDimension == 2 && t->size[0] == size0 && t->size[1] == size1;
}

static bool THNN_(checkSize1D)(THTensor* t, long size0)
{
  return t->nDimension == 1 && t->size[0] == size0;
}

void THNN_(SparseLinear_updateOutput)(THNNState *state, THTensor *input, THTensor *output, THTensor *weight, THTensor *bias, THTensor *shardBuffer)
{
  long i;
  long outDim = weight->size[0];
  long inDim = weight->size[1];

  THArgCheck(THNN_(checkInput)(input), 2, "input size must be nnz x 2");
  THArgCheck(THNN_(checkSize1D)(output, outDim), 3, "output size wrong");
  THArgCheck(THNN_(checkSize1D)(bias, outDim), 5, "bias size wrong");

  if (shardBuffer != NULL)
  {
    long num_shards = shardBuffer->size[1];
    THArgCheck(
      shardBuffer->nDimension == 2 && shardBuffer->size[0] == outDim && num_shards > 0,
      6,
      "shardBuffer size wrong"
    );

    THTensor_(zero)(shardBuffer);
    #pragma omp parallel for private(i) schedule(static) num_threads(num_shards)
    for (i = 0; i < input->size[0]; i++)
    {
#ifdef _OPENMP
      int shardId = omp_get_thread_num();
#else
      int shardId = 1;
#endif

      long offset = (long)(THTensor_(get2d)(input, i, 0)) - 1;
      if (offset >= 0 && offset < inDim)
      {
        THBlas_(axpy)(
          outDim,
          THTensor_(get2d)(input, i, 1),
          THTensor_(data)(weight) + offset * weight->stride[1],
          weight->stride[0],
          THTensor_(data)(shardBuffer) + shardId * shardBuffer->stride[1],
          shardBuffer->stride[0]
        );
      }
      else
      {
        THError("index out of bound. updateOutput: \
%ld not between 1 and %ld", offset + 1, inDim);
      }
    }

    THTensor_(sum)(output, shardBuffer, 1);
    THTensor_(cadd)(output, bias, 1.0, output);

    return;
  }

  THTensor_(copy)(output, bias);
  for (i = 0; i < input->size[0]; i++)
  {
    long offset = (long)(THTensor_(get2d)(input, i, 0)) - 1;
    if (offset >= 0 && offset < inDim) // make sure indices are in bounds..
    {
      real val = THTensor_(get2d)(input, i, 1);
      THBlas_(axpy)(
        output->size[0],
        val,
        THTensor_(data)(weight)+offset*weight->stride[1],
        weight->stride[0],
        THTensor_(data)(output),
        output->stride[0]
      );
    }
    else
    {
      THError("index out of bound. updateOutput: \
%ld not between 1 and %ld", offset + 1, inDim);
    }
  }
}

void THNN_(SparseLinear_accGradParameters)(
  THNNState *state,
  THTensor *input,
  THTensor *gradOutput,
  THTensor *gradWeight,
  THTensor *gradBias,
  THTensor *weight,
  THTensor *bias,
  real weightDecay,
  real scale)
{
  long i;
  long nnz = input->size[0];
  long outDim = weight->size[0];
  long inDim = weight->size[1];

  THArgCheck(THNN_(checkInput)(input), 2, "input size must be nnz x 2");
  THArgCheck(THNN_(checkSize1D)(gradOutput, outDim), 3, "gradOutput size wrong");
  THArgCheck(THNN_(checkSize2D)(gradWeight, outDim, inDim), 4, "gradWeight size wrong");
  THArgCheck(THNN_(checkSize1D)(gradBias, outDim), 5, "gradBias size wrong");

  #pragma omp parallel for private(i) schedule(static) if(outDim * nnz > 100000)
  for (i = 0; i < nnz; i++)
  {
      long offset = (long)(THTensor_(get2d)(input, i, 0)) - 1;

      if (offset >= 0 && offset < inDim) // make sure indices are in bounds..
      {
        real val = scale*THTensor_(get2d)(input, i, 1);

        THBlas_(axpy)(
          outDim,
          val,
          THTensor_(data)(gradOutput),
          gradOutput->stride[0],
          THTensor_(data)(gradWeight)+offset*gradWeight->stride[1],
          gradWeight->stride[0]
        );
      }
      else
      {
        THError("index out of bound. accGradParameters: \
%ld not between 1 and %ld", offset + 1, inDim);
      }
  }

  THTensor_(cadd)(gradBias, gradBias, scale, gradOutput);

  if (weightDecay != 0)
  {
    #pragma omp parallel for private(i) schedule(static) if(outDim * nnz > 100000)
    for (i = 0; i < nnz; i++)
    {
      long offset = (long)(THTensor_(get2d)(input, i, 0)) - 1;
      THBlas_(axpy)(
        outDim,
        weightDecay,
        THTensor_(data)(weight) + offset*weight->stride[1],
        weight->stride[0],
        THTensor_(data)(gradWeight)+offset*gradWeight->stride[1],
        gradWeight->stride[0]
      );
    }
    THTensor_(cadd)(gradBias, gradBias, weightDecay, bias);
  }
}

void THNN_(SparseLinear_updateParameters)(
  THNNState *state,
  THTensor *weight,
  THTensor *bias,
  THTensor *gradWeight,
  THTensor *gradBias,
  THTensor *lastInput,
  real learningRate)
{
  long i;
  long nnz = lastInput->size[0];
  long outDim = weight->size[0];
  long inDim = weight->size[1];

  THArgCheck(THNN_(checkSize2D)(gradWeight, outDim, inDim), 4, "gradWeight size wrong");
  THArgCheck(THNN_(checkSize1D)(bias, outDim), 3, "bias size wrong");
  THArgCheck(THNN_(checkSize1D)(gradBias, outDim), 5, "gradBias size wrong");

  THTensor_(cadd)(bias, bias, -learningRate, gradBias);

  #pragma omp parallel for private(i) schedule(static) if(outDim * nnz > 50000)
  for (i = 0; i < nnz; i++)
  {
    long offset = (long)(THTensor_(get2d)(lastInput, i, 0)) - 1;

    if (offset >= 0 && offset < inDim) // make sure indices are in bounds..
    {
      real* pGradWeight =
        THTensor_(data)(gradWeight)+offset*gradWeight->stride[1];
      THBlas_(axpy)(
        outDim,
        -learningRate,
        pGradWeight,
        gradWeight->stride[0],
        THTensor_(data)(weight)+offset*weight->stride[1],
        weight->stride[0]
      );
    }
    else
    {
      THError("index out of bound. updateParameters: \
%ld not between 1 and %ld", offset + 1, inDim);
    }
  }
}

void THNN_(SparseLinear_zeroGradParameters)(THNNState *state, THTensor *gradWeight, THTensor *gradBias, THTensor *lastInput)
{
  long i;
  long nnz = lastInput->size[0];
  long outDim = gradWeight->size[0];
  long inDim = gradWeight->size[1];

  THArgCheck(THNN_(checkSize1D)(gradBias, outDim), 3, "gradBias size wrong");

  THTensor_(zero)(gradBias);
  #pragma omp parallel for private(i) schedule(static) if(outDim * nnz > 50000)
  for (i = 0; i < nnz; i++)
  {
    long offset = (long)(THTensor_(get2d)(lastInput, i, 0)) - 1;

    if(offset >= 0 && offset < inDim) // make sure indices are in bounds..
    {
      real* pGradWeight = THTensor_(data)(gradWeight) + offset * gradWeight->stride[1];
      if (gradWeight->stride[0] == 1)
      {
          THVector_(fill)(pGradWeight, 0, outDim);
      }
      else
      {
        long j;
        for (j = 0; j < outDim; ++j)
        {
          pGradWeight[j * gradWeight->stride[0]] = 0;
        }
      }
    }
    else
    {
      THError("index out of bound. zeroGradParameters: \
%ld not between 1 and %ld", offset + 1, inDim);
    }
  }
}

void THNN_(SparseLinear_updateGradInput)(
  THNNState *state,
  THTensor *input,
  THTensor *gradOutput,
  THTensor *gradInput,
  THTensor *weight)
{
  long i;
  long nnz = input->size[0];
  long outDim = weight->size[0];
  long inDim = weight->size[1];

  THArgCheck(THNN_(checkInput)(input), 2, "input must be an nnz x 2 tensor");
  THArgCheck(THNN_(checkSize1D)(gradOutput, outDim), 3, "gradOutput size wrong");

  THTensor_(resize2d)(gradInput, input->size[0], input->size[1]);

  #pragma omp parallel for private(i) schedule(static) if(outDim * nnz > 100000)
  for (i = 0; i < nnz; ++i)
  {
    long offset = (long)(THTensor_(get2d)(input, i, 0)) - 1;
    THTensor_(set2d)(gradInput, i, 0, offset + 1);

    if (offset >= 0 && offset < inDim)
    {
      real val = 
        THBlas_(dot)(
          outDim,
          THTensor_(data)(gradOutput),
          gradOutput->stride[0],
          THTensor_(data)(weight) + offset * weight->stride[1],
          weight->stride[0]
        );
      THTensor_(set2d)(gradInput, i, 1, val);
    }
    else
    {
      THError("index out of bound. updateGradInput: \
%ld not between 1 and %ld", offset + 1, inDim);
    }
  }
}

#endif
