#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialFractionalMaxPooling.c"
#else

static long* nn_(SpatialFractionalMaxPooling_generateIntervals)(
  real sample,
  long inputSize,
  long outputSize,
  int poolSize) {
  real alpha = (real) (inputSize - poolSize) / (real) (outputSize - 1);
  long* sequence = (long*) THAlloc(sizeof(long) * outputSize);

  long i;
  for (i = 0; i < outputSize - 1; ++i) {
    sequence[i] =
      (long) ((i + sample) * alpha) - (long) (sample * alpha);
  }
  sequence[outputSize - 1] = inputSize - poolSize;

  return sequence;
}

static void nn_(SpatialFractionalMaxPooling_updateOutput_frame)(
  real* input,
  real* output,
  real* indices,
  real* randomSamples,
  long numPlanes,
  long inputW, long inputH,
  long outputW, long outputH,
  int poolSizeW, int poolSizeH) {
  long plane;
#pragma omp parallel for private(plane)
  for (plane = 0; plane < numPlanes; ++plane) {
    /* each plane contains 2 random samples, one for W and one for H */
    real* randomSamplesForPlane = randomSamples + plane * 2;

    /* Generate interval sequence */
    long* sequenceW =
      nn_(SpatialFractionalMaxPooling_generateIntervals)(
        randomSamplesForPlane[0], inputW, outputW, poolSizeW);
    long* sequenceH =
      nn_(SpatialFractionalMaxPooling_generateIntervals)(
        randomSamplesForPlane[1], inputH, outputH, poolSizeH);

    /* loop over output */
    long h, w;

    real* inputForPlane = input + plane * inputW * inputH;
    real* outputForPlane = output + plane * outputW * outputH;
    real* indicesForPlane = indices + plane * outputW * outputH;

    for (h = 0; h < outputH; ++h) {
      long inputHStart = sequenceH[h];

      for (w = 0; w < outputW; ++w) {
        long inputWStart = sequenceW[w];

        real maxVal = -THInf;
        long maxIndex = -1;

        long h2, w2;
        for (h2 = inputHStart; h2 < inputHStart + poolSizeH; ++h2) {
          for (w2 = inputWStart; w2 < inputWStart + poolSizeW; ++w2) {
            THAssert(h2 >= 0 && h2 < inputH);
            THAssert(w2 >= 0 && w2 < inputW);

            long planeIndex = h2 * inputW + w2;
            real val = inputForPlane[planeIndex];
            if (val > maxVal) {
              maxVal = val;
              maxIndex = planeIndex;
            }
          }
        }

        THAssert(maxVal != -THInf);
        THAssert(maxIndex != -1);

        outputForPlane[h * outputW + w] = maxVal;
        /* +1 to lua index */
        indicesForPlane[h * outputW + w] = (real) maxIndex + 1;
      }
    }

    THFree(sequenceW);
    THFree(sequenceH);
  }
}

static int nn_(SpatialFractionalMaxPooling_updateOutput)(lua_State *L) {
  THTensor* output = luaT_checkudata(L, 1, torch_Tensor);
  THTensor* input = luaT_checkudata(L, 2, torch_Tensor);
  int outputW = luaL_checknumber(L, 3);
  int outputH = luaL_checknumber(L, 4);
  int poolSizeW = luaL_checknumber(L, 5);
  int poolSizeH = luaL_checknumber(L, 6);
  THTensor* indices = luaT_checkudata(L, 7, torch_Tensor);
  THTensor* randomSamples = luaT_checkudata(L, 8, torch_Tensor);

  long numBatch = 1;
  int planeDim = 0;
  int heightDim = 1;
  int widthDim = 2;

  long numInputDims = THTensor_(nDimension)(input);
  luaL_argcheck(L, numInputDims == 3 || numInputDims == 4, 2,
                "3D or 4D (batch mode) tensor expected");

  if (numInputDims == 4) {
    numBatch = THTensor_(size)(input, 0);
    planeDim++;
    heightDim++;
    widthDim++;
  }

  /* sizes */
  long numPlanes = THTensor_(size)(input, planeDim);
  long inputH = THTensor_(size)(input, heightDim);
  long inputW = THTensor_(size)(input, widthDim);

  luaL_argcheck(L, outputH + poolSizeH - 1 < inputH, 6,
                "poolSizeH too large relative to input height");
  luaL_argcheck(L, outputW + poolSizeW - 1 < inputW, 5,
                "poolSizeW too large relative to input width");

  /* get contiguous input */
  input = THTensor_(newContiguous)(input);

  if (numInputDims == 3) {
    /* resize output */
    THTensor_(resize3d)(output, numPlanes, outputH, outputW);
    /* indices will contain the locations for each output point */
    THTensor_(resize3d)(indices, numPlanes, outputH, outputW);

    nn_(SpatialFractionalMaxPooling_updateOutput_frame)(
      THTensor_(data)(input),
      THTensor_(data)(output),
      THTensor_(data)(indices),
      THTensor_(data)(randomSamples),
      numPlanes, inputW, inputH, outputW, outputH, poolSizeW, poolSizeH);
  } else {
    THTensor_(resize4d)(output, numBatch, numPlanes, outputH, outputW);
    /* indices will contain the locations for each output point */
    THTensor_(resize4d)(indices, numBatch, numPlanes, outputH, outputW);

    long batch;
#pragma omp parallel for private(batch)
    for (batch = 0; batch < numBatch; ++batch) {
      nn_(SpatialFractionalMaxPooling_updateOutput_frame)(
        THTensor_(data)(input) + batch * numPlanes * inputH * inputW,
        THTensor_(data)(output) + batch * numPlanes * outputH * outputW,
        THTensor_(data)(indices) + batch * numPlanes * outputH * outputW,
        THTensor_(data)(randomSamples) + batch * numPlanes * 2,
        numPlanes, inputW, inputH, outputW, outputH, poolSizeW, poolSizeH);
    }
  }

  /* cleanup */
  THTensor_(free)(input);

  return 0;
}

static void nn_(SpatialFractionalMaxPooling_updateGradInput_frame)(
  real* gradInput,
  real* gradOutput,
  real* indices,
  long numPlanes,
  long inputW, long inputH,
  long outputW, long outputH) {
  long plane;
#pragma omp parallel for private(plane)
  for (plane = 0; plane < numPlanes; plane++) {
    real* gradInputForPlane = gradInput + plane * inputW * inputH;
    real* gradOutputForPlane = gradOutput + plane * outputW * outputH;
    real* indicesForPlane = indices + plane * outputW * outputH;

    long h, w;
    for (h = 0; h < outputH; ++h) {
      for (w = 0; w < outputW; ++w) {
        long outputIndex = h * outputW + w;
        long index = indicesForPlane[outputIndex] - 1;
        THAssert(index >= 0 && index < inputW * inputH);

        gradInputForPlane[index] += gradOutputForPlane[outputIndex];
      }
    }
  }
}

static int nn_(SpatialFractionalMaxPooling_updateGradInput)(lua_State *L) {
  THTensor* gradInput = luaT_checkudata(L, 1, torch_Tensor);
  THTensor* input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor* gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  long outputW = luaL_checknumber(L, 4);
  long outputH = luaL_checknumber(L, 5);
  int poolSizeW = luaL_checknumber(L, 6);
  int poolSizeH = luaL_checknumber(L, 7);
  THTensor* indices = luaT_checkudata(L, 8, torch_Tensor);

  long numBatch = 1;
  int planeDim = 0;
  int heightDim = 1;
  int widthDim = 2;

  long numInputDims = THTensor_(nDimension)(input);
  if (numInputDims == 4) {
    numBatch = THTensor_(size)(input, 0);
    planeDim = 1;
    heightDim++;
    widthDim++;
  }

  /* sizes */
  long numPlanes = THTensor_(size)(input, planeDim);
  long inputH = THTensor_(size)(input, heightDim);
  long inputW = THTensor_(size)(input, widthDim);

  luaL_argcheck(L, outputW == THTensor_(size)(gradOutput, widthDim), 3,
                "gradOutput width unexpected");
  luaL_argcheck(L, outputH == THTensor_(size)(gradOutput, heightDim), 3,
                "gradOutput height unexpected");

  /* get contiguous gradOutput */
  gradOutput = THTensor_(newContiguous)(gradOutput);

  /* resize */
  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);

  /* backprop */
  if (numInputDims == 3) {
    nn_(SpatialFractionalMaxPooling_updateGradInput_frame)(
      THTensor_(data)(gradInput),
      THTensor_(data)(gradOutput),
      THTensor_(data)(indices),
      numPlanes, inputW, inputH, outputW, outputH);
  } else {
    long batch;
#pragma omp parallel for private(batch)
    for (batch = 0; batch < numBatch; ++batch) {
      nn_(SpatialFractionalMaxPooling_updateGradInput_frame)(
        THTensor_(data)(gradInput) + batch * numPlanes * inputH * inputW,
        THTensor_(data)(gradOutput) + batch * numPlanes * outputH * outputW,
        THTensor_(data)(indices) + batch * numPlanes * outputH * outputW,
        numPlanes, inputW, inputH, outputW, outputH);
    }
  }

  /* cleanup */
  THTensor_(free)(gradOutput);

  return 0;
}

static const struct luaL_Reg nn_(SpatialFractionalMaxPooling__) [] = {
  {"SpatialFractionalMaxPooling_updateOutput", nn_(SpatialFractionalMaxPooling_updateOutput)},
  {"SpatialFractionalMaxPooling_updateGradInput", nn_(SpatialFractionalMaxPooling_updateGradInput)},
  {NULL, NULL}
};

static void nn_(SpatialFractionalMaxPooling_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(SpatialFractionalMaxPooling__), "nn");
  lua_pop(L,1);
}

#endif
