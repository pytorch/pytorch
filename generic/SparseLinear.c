#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SparseLinear.c"
#else

#ifdef _OPENMP
#include <omp.h>
#endif

static int nn_(checkInput)(THTensor* t) {
  return t->nDimension == 2 && t->size[1] == 2;
}

static int nn_(checkSize2D)(THTensor* t, long size0, long size1) {
  return t->nDimension == 2 && t->size[0] == size0 && t->size[1] == size1;
}

static int nn_(checkSize1D)(THTensor* t, long size0) {
  return t->nDimension == 1 && t->size[0] == size0;
}

static int nn_(SparseLinear_updateOutput)(lua_State *L)
{
  long i;
  THTensor * input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor * weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor * bias = luaT_getfieldcheckudata(L, 1, "bias", torch_Tensor);
  THTensor * output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

  long outDim = weight->size[0];
  long inDim = weight->size[1];

  luaL_argcheck(L, nn_(checkInput)(input), 2, "input size must be nnz x 2");
  luaL_argcheck(L, nn_(checkSize1D)(output, outDim), 1, "output size wrong");
  luaL_argcheck(L, nn_(checkSize1D)(bias, outDim), 1, "bias size wrong");

  lua_getfield(L, 1, "shardBuffer");
  if (!lua_isnil(L, -1)) {
    THTensor *buffer =
      luaT_getfieldcheckudata(L, 1, "shardBuffer", torch_Tensor);
    long num_shards = buffer->size[1];
    luaL_argcheck(L,
                  buffer->nDimension == 2 && buffer->size[0] == outDim &&
                      num_shards > 0,
                  1,
                  "shardBuffer size wrong");

    THTensor_(zero)(buffer);
    #pragma omp parallel for private(i) schedule(static) num_threads(num_shards)
    for (i = 0; i < input->size[0]; i++) {
#ifdef _OPENMP
      int shardId = omp_get_thread_num();
#else
      int shardId = 1;
#endif
      long offset = (long)(THTensor_(get2d)(input, i, 0)) - 1;

      if (offset >= 0 && offset < inDim) {
        THBlas_(axpy)(outDim,
                      THTensor_(get2d)(input, i, 1),
                      THTensor_(data)(weight) + offset * weight->stride[1],
                      weight->stride[0],
                      THTensor_(data)(buffer) + shardId * buffer->stride[1],
                      buffer->stride[0]);
      } else {
        luaL_error(L, "index out of bound. updateOutput: \
%ld not between 1 and %ld", offset + 1, inDim);
      }
    }

    THTensor_(sum)(output, buffer, 1);
    THTensor_(cadd)(output, bias, 1.0, output);

    lua_getfield(L, 1, "output");
    return 1;
  }

  THTensor_(copy)(output, bias);
  for(i = 0; i < input->size[0]; i++)
  {
    long offset = (long)(THTensor_(get2d)(input, i, 0)) - 1;
    if(offset >= 0 && offset < inDim) /* make sure indices are in bounds.. */
    {
        real val = THTensor_(get2d)(input, i, 1);
        THBlas_(axpy)(output->size[0],
                      val,
                      THTensor_(data)(weight)+offset*weight->stride[1],
                      weight->stride[0],
                      THTensor_(data)(output),
                      output->stride[0]);
    }
    else {
        luaL_error(L, "index out of bound. updateOutput: \
%ld not between 1 and %ld", offset + 1, inDim);
    }
  }

  lua_getfield(L, 1, "output");
  return 1;
}

static int nn_(SparseLinear_accGradParameters)(lua_State *L)
{
  long i;
  THTensor * input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor * gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  real scale = luaL_optnumber(L, 4, 1);
  THTensor * bias = luaT_getfieldcheckudata(L, 1, "bias", torch_Tensor);
  THTensor * weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor * gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_Tensor);
  THTensor * gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_Tensor);
  real weightDecay = luaT_getfieldchecknumber(L, 1, "weightDecay");

  long nnz = input->size[0];
  long outDim = weight->size[0];
  long inDim = weight->size[1];

  luaL_argcheck(L, nn_(checkInput)(input), 2, "input size must be nnz x 2");
  luaL_argcheck(
    L, nn_(checkSize1D)(gradOutput, outDim), 3, "gradOutput size wrong");
  luaL_argcheck(
    L, nn_(checkSize2D)(gradWeight, outDim, inDim), 1, "gradWeight size wrong");
  luaL_argcheck(
    L, nn_(checkSize1D)(gradBias, outDim), 1, "gradBias size wrong");

  #pragma omp parallel for private(i) schedule(static) if(outDim * nnz > 100000)
  for(i = 0; i < nnz; i++)
  {
      long offset = (long)(THTensor_(get2d)(input, i, 0)) - 1;

      if(offset >= 0 && offset < inDim) /* make sure indices are in bounds.. */
      {
          real val = scale*THTensor_(get2d)(input, i, 1);

          THBlas_(axpy)(outDim,
                        val,
                        THTensor_(data)(gradOutput),
                        gradOutput->stride[0],
                        THTensor_(data)(gradWeight)+offset*gradWeight->stride[1],
                        gradWeight->stride[0]);
      }
      else {
          luaL_error(L, "index out of bound. accGradParameters: \
%ld not between 1 and %ld", offset + 1, inDim);
      }
  }

  THTensor_(cadd)(gradBias, gradBias, scale, gradOutput);

  if(weightDecay != 0) {
    #pragma omp parallel for private(i) schedule(static) if(outDim * nnz > 100000)
    for(i = 0; i < nnz; i++) {
      long offset = (long)(THTensor_(get2d)(input, i, 0)) - 1;
      THBlas_(axpy)(outDim,
                    weightDecay,
                    THTensor_(data)(weight) + offset*weight->stride[1],
                    weight->stride[0],
                    THTensor_(data)(gradWeight)+offset*gradWeight->stride[1],
                    gradWeight->stride[0]);
    }
    THTensor_(cadd)(gradBias, gradBias, weightDecay, bias);
  }

  return 0;
}

int nn_(SparseLinear_updateParameters)(lua_State *L)
{
  long i;
  real learningRate = luaL_checknumber(L, 2);
  THTensor * weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor * bias = luaT_getfieldcheckudata(L, 1, "bias", torch_Tensor);
  THTensor * gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_Tensor);
  THTensor * gradWeight = luaT_getfieldcheckudata(
    L, 1, "gradWeight", torch_Tensor);
  THTensor * lastInput = luaT_getfieldcheckudata(
    L, 1, "lastInput", torch_Tensor);

  long nnz = lastInput->size[0];
  long outDim = weight->size[0];
  long inDim = weight->size[1];

  luaL_argcheck(
    L, nn_(checkSize2D)(gradWeight, outDim, inDim), 1, "gradWeight size wrong");
  luaL_argcheck(
    L, nn_(checkSize1D)(bias, outDim), 1, "bias size wrong");
  luaL_argcheck(
    L, nn_(checkSize1D)(gradBias, outDim), 1, "gradBias size wrong");

  THTensor_(cadd)(bias, bias, -learningRate, gradBias);

  #pragma omp parallel for private(i) schedule(static) if(outDim * nnz > 50000)
  for(i = 0; i < nnz; i++)
  {
      long offset = (long)(THTensor_(get2d)(lastInput, i, 0)) - 1;

      if(offset >= 0 && offset < inDim) /* make sure indices are in bounds.. */
      {
          real* pGradWeight =
            THTensor_(data)(gradWeight)+offset*gradWeight->stride[1];
          THBlas_(axpy)(outDim,
                        -learningRate,
                        pGradWeight,
                        gradWeight->stride[0],
                        THTensor_(data)(weight)+offset*weight->stride[1],
                        weight->stride[0]);
      }
      else {
          luaL_error(L, "index out of bound. updateParameters: \
%ld not between 1 and %ld", offset + 1, inDim);
      }
  }
  return 0;
}

int nn_(SparseLinear_zeroGradParameters)(lua_State *L)
{
  long i;
  THTensor * gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_Tensor);
  THTensor * gradWeight = luaT_getfieldcheckudata(
    L, 1, "gradWeight", torch_Tensor);
  THTensor * lastInput = luaT_getfieldcheckudata(
    L, 1, "lastInput", torch_Tensor);

  long nnz = lastInput->size[0];
  long outDim = gradWeight->size[0];
  long inDim = gradWeight->size[1];

  luaL_argcheck(
    L, nn_(checkSize1D)(gradBias, outDim), 1, "gradBias size wrong");

  THTensor_(zero)(gradBias);
  #pragma omp parallel for private(i) schedule(static) if(outDim * nnz > 50000)
  for(i = 0; i < nnz; i++)
  {
      long offset = (long)(THTensor_(get2d)(lastInput, i, 0)) - 1;

      if(offset >= 0 && offset < inDim) /* make sure indices are in bounds.. */
      {
          real* pGradWeight =
            THTensor_(data)(gradWeight)+offset*gradWeight->stride[1];
          if(gradWeight->stride[0] == 1) {
              THVector_(fill)(pGradWeight, 0, outDim);
          } else {
              long j;
              for(j = 0; j < outDim; ++j) {
                  pGradWeight[j * gradWeight->stride[0]] = 0;
              }
          }
      }
      else {
          luaL_error(L, "index out of bound. zeroGradParameters: \
%ld not between 1 and %ld", offset + 1, inDim);
      }
  }
  return 0;
}

static int nn_(SparseLinear_updateGradInput)(lua_State *L) {
  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor *gradInput =
      luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);

  long i;
  long nnz = input->size[0];
  long outDim = weight->size[0];
  long inDim = weight->size[1];

  luaL_argcheck(
    L, nn_(checkInput)(input), 2, "input must be an nnz x 2 tensor");
  luaL_argcheck(
    L, nn_(checkSize1D)(gradOutput, outDim), 3, "gradOutput size wrong");

  THTensor_(resize2d)(gradInput, input->size[0], input->size[1]);

  #pragma omp parallel for private(i) schedule(static) if(outDim * nnz > 100000)
  for (i = 0; i < nnz; ++i) {
    long offset = (long)(THTensor_(get2d)(input, i, 0)) - 1;
    THTensor_(set2d)(gradInput, i, 0, offset + 1);

    if (offset >= 0 && offset < inDim) {
      real val =
          THBlas_(dot)(outDim,
                       THTensor_(data)(gradOutput),
                       gradOutput->stride[0],
                       THTensor_(data)(weight) + offset * weight->stride[1],
                       weight->stride[0]);
      THTensor_(set2d)(gradInput, i, 1, val);
    } else {
      luaL_error(L, "index out of bound. updateGradInput: \
%ld not between 1 and %ld", offset + 1, inDim);
    }
  }
  return 0;
}

static const struct luaL_Reg nn_(SparseLinear__) [] = {
  {"SparseLinear_updateOutput", nn_(SparseLinear_updateOutput)},
  {"SparseLinear_accGradParameters", nn_(SparseLinear_accGradParameters)},
  {"SparseLinear_updateParameters", nn_(SparseLinear_updateParameters)},
  {"SparseLinear_zeroGradParameters", nn_(SparseLinear_zeroGradParameters)},
  {"SparseLinear_updateGradInput", nn_(SparseLinear_updateGradInput)},
  {NULL, NULL}
};

void nn_(SparseLinear_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(SparseLinear__), "nn");
  lua_pop(L,1);
}

#endif
