#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialFractionalMaxPooling.c"
#else

static long* THNN_(SpatialFractionalMaxPooling_generateIntervals)(
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

static void THNN_(SpatialFractionalMaxPooling_updateOutput_frame)(
  real* input,
  real* output,
  THIndex_t* indices,
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
      THNN_(SpatialFractionalMaxPooling_generateIntervals)(
        randomSamplesForPlane[0], inputW, outputW, poolSizeW);
    long* sequenceH =
      THNN_(SpatialFractionalMaxPooling_generateIntervals)(
        randomSamplesForPlane[1], inputH, outputH, poolSizeH);

    /* loop over output */
    long h, w;

    real* inputForPlane = input + plane * inputW * inputH;
    real* outputForPlane = output + plane * outputW * outputH;
    THIndex_t* indicesForPlane = indices + plane * outputW * outputH;

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
        indicesForPlane[h * outputW + w] = maxIndex + TH_INDEX_BASE;
      }
    }

    THFree(sequenceW);
    THFree(sequenceH);
  }
}

void THNN_(SpatialFractionalMaxPooling_updateOutput)(
    THNNState *state,
    THTensor *input,
    THTensor *output,
    int outputW, int outputH,
    int poolSizeW, int poolSizeH,
    THIndexTensor *indices,
    THTensor *randomSamples) {

  long numBatch = 1;
  int planeDim = 0;
  int heightDim = 1;
  int widthDim = 2;

  long numInputDims = THTensor_(nDimension)(input);
  THNN_ARGCHECK(numInputDims == 3 || numInputDims == 4, 2, input,
		"3D or 4D (batch mode) tensor expected for input, but got: %s");

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

  THArgCheck(outputH + poolSizeH - 1 < inputH, 7,
             "poolSizeH (%d) too large relative to input height (%d)",
	     poolSizeH, inputH);
  THArgCheck(outputW + poolSizeW - 1 < inputW, 6,
             "poolSizeW (%d) too large relative to input width (%d)",
	     poolSizeW, inputW);

  /* get contiguous input */
  input = THTensor_(newContiguous)(input);

  if (numInputDims == 3) {
    /* resize output */
    THTensor_(resize3d)(output, numPlanes, outputH, outputW);
    /* indices will contain the locations for each output point */
    THIndexTensor_(resize3d)(indices, numPlanes, outputH, outputW);

    THNN_(SpatialFractionalMaxPooling_updateOutput_frame)(
      THTensor_(data)(input),
      THTensor_(data)(output),
      THIndexTensor_(data)(indices),
      THTensor_(data)(randomSamples),
      numPlanes, inputW, inputH, outputW, outputH, poolSizeW, poolSizeH);
  } else {
    THTensor_(resize4d)(output, numBatch, numPlanes, outputH, outputW);
    /* indices will contain the locations for each output point */
    THIndexTensor_(resize4d)(indices, numBatch, numPlanes, outputH, outputW);

    long batch;
#pragma omp parallel for private(batch)
    for (batch = 0; batch < numBatch; ++batch) {
      THNN_(SpatialFractionalMaxPooling_updateOutput_frame)(
        THTensor_(data)(input) + batch * numPlanes * inputH * inputW,
        THTensor_(data)(output) + batch * numPlanes * outputH * outputW,
        THIndexTensor_(data)(indices) + batch * numPlanes * outputH * outputW,
        THTensor_(data)(randomSamples) + batch * numPlanes * 2,
        numPlanes, inputW, inputH, outputW, outputH, poolSizeW, poolSizeH);
    }
  }

  /* cleanup */
  THTensor_(free)(input);
}

static void THNN_(SpatialFractionalMaxPooling_updateGradInput_frame)(
  real* gradInput,
  real* gradOutput,
  THIndex_t* indices,
  long numPlanes,
  long inputW, long inputH,
  long outputW, long outputH) {
  long plane;
#pragma omp parallel for private(plane)
  for (plane = 0; plane < numPlanes; plane++) {
    real* gradInputForPlane = gradInput + plane * inputW * inputH;
    real* gradOutputForPlane = gradOutput + plane * outputW * outputH;
    THIndex_t* indicesForPlane = indices + plane * outputW * outputH;

    long h, w;
    for (h = 0; h < outputH; ++h) {
      for (w = 0; w < outputW; ++w) {
        long outputIndex = h * outputW + w;
        long index = indicesForPlane[outputIndex] - TH_INDEX_BASE;
        THAssert(index >= 0 && index < inputW * inputH);

        gradInputForPlane[index] += gradOutputForPlane[outputIndex];
      }
    }
  }
}

void THNN_(SpatialFractionalMaxPooling_updateGradInput)(
    THNNState *state,
    THTensor *input,
    THTensor *gradOutput,
    THTensor *gradInput,
    int outputW, int outputH,
    int poolSizeW, int poolSizeH,
    THIndexTensor *indices) {

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

  THArgCheck(outputW == THTensor_(size)(gradOutput, widthDim), 3,
             "gradOutput width unexpected");
  THArgCheck(outputH == THTensor_(size)(gradOutput, heightDim), 3,
             "gradOutput height unexpected");

  /* get contiguous gradOutput */
  gradOutput = THTensor_(newContiguous)(gradOutput);

  /* resize */
  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);

  /* backprop */
  if (numInputDims == 3) {
    THNN_(SpatialFractionalMaxPooling_updateGradInput_frame)(
      THTensor_(data)(gradInput),
      THTensor_(data)(gradOutput),
      THIndexTensor_(data)(indices),
      numPlanes, inputW, inputH, outputW, outputH);
  } else {
    long batch;
#pragma omp parallel for private(batch)
    for (batch = 0; batch < numBatch; ++batch) {
      THNN_(SpatialFractionalMaxPooling_updateGradInput_frame)(
        THTensor_(data)(gradInput) + batch * numPlanes * inputH * inputW,
        THTensor_(data)(gradOutput) + batch * numPlanes * outputH * outputW,
        THIndexTensor_(data)(indices) + batch * numPlanes * outputH * outputW,
        numPlanes, inputW, inputH, outputW, outputH);
    }
  }

  /* cleanup */
  THTensor_(free)(gradOutput);
}

#endif
