#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/VolumetricFractionalMaxPooling.c"
#else

static long* THNN_(VolumetricFractionalMaxPooling_generateIntervals)(
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

static void THNN_(VolumetricFractionalMaxPooling_updateOutput_frame)(
  real* input,
  real* output,
  THIndex_t* indices,
  real* randomSamples,
  long numPlanes,
  long inputT, long inputW, long inputH,
  long outputT, long outputW, long outputH,
  int poolSizeT, int poolSizeW, int poolSizeH) {
  long plane;
#pragma omp parallel for private(plane)
  for (plane = 0; plane < numPlanes; ++plane) {
    /* each plane contains 3 random samples, one for T, one for W, and one for H */
    real* randomSamplesForPlane = randomSamples + plane * 3;

    /* Generate interval sequence */
    long* sequenceT =
      THNN_(VolumetricFractionalMaxPooling_generateIntervals)(
        randomSamplesForPlane[0], inputT, outputT, poolSizeT);
    long* sequenceW =
      THNN_(VolumetricFractionalMaxPooling_generateIntervals)(
        randomSamplesForPlane[1], inputW, outputW, poolSizeW);
    long* sequenceH =
      THNN_(VolumetricFractionalMaxPooling_generateIntervals)(
        randomSamplesForPlane[2], inputH, outputH, poolSizeH);

    /* loop over output */
    long h, w, t;

    real* inputForPlane = input + plane * inputT * inputW * inputH;
    real* outputForPlane = output + plane * outputT * outputW * outputH;
    THIndex_t* indicesForPlane = indices + plane * outputT * outputW * outputH;

    for (h = 0; h < outputH; ++h) {
      long inputHStart = sequenceH[h];

      for (w = 0; w < outputW; ++w) {
        long inputWStart = sequenceW[w];

        for (t = 0; t < outputT; ++t) {
          long inputTStart = sequenceT[t];

          real maxVal = -THInf;
          long maxIndex = -1;

          long h2, w2, t2;
          for (h2 = inputHStart; h2 < inputHStart + poolSizeH; ++h2) {
            for (w2 = inputWStart; w2 < inputWStart + poolSizeW; ++w2) {
              for (t2 = inputTStart; t2 < inputTStart + poolSizeT; ++t2) {
                THAssert(h2 >= 0 && h2 < inputH);
                THAssert(w2 >= 0 && w2 < inputW);
                THAssert(t2 >= 0 && t2 < inputT);

                long planeIndex = h2 * inputW * inputT + w2 * inputT + t2;
                real val = inputForPlane[planeIndex];
                if (val > maxVal) {
                  maxVal = val;
                  maxIndex = planeIndex;
                }
              }
            }
          }

          THAssert(maxVal != -THInf);
          THAssert(maxIndex != -1);

          outputForPlane[h * outputW * outputT + w * outputT + t] = maxVal;
          /* +1 to lua index */
          indicesForPlane[h * outputW * outputT + w * outputT + t] = maxIndex + TH_INDEX_BASE;
        }
      }
    }

    THFree(sequenceT);
    THFree(sequenceW);
    THFree(sequenceH);
  }
}

void THNN_(VolumetricFractionalMaxPooling_updateOutput)(
    THNNState *state,
    THTensor *input,
    THTensor *output,
    int outputT, int outputW, int outputH,
    int poolSizeT, int poolSizeW, int poolSizeH,
    THIndexTensor *indices,
    THTensor *randomSamples) {

  long numBatch = 1;
  int planeDim = 0;
  int heightDim = 1;
  int widthDim = 2;
  int timeDim = 3;

  long numInputDims = THTensor_(nDimension)(input);
  THNN_ARGCHECK(numInputDims == 4 || numInputDims == 5, 2, input,
		"4D or 5D (batch mode) tensor expected for input, but got: %s");

  if (numInputDims == 5) {
    numBatch = THTensor_(size)(input, 0);
    planeDim++;
    heightDim++;
    widthDim++;
    timeDim++;
  }

  /* sizes */
  long numPlanes = THTensor_(size)(input, planeDim);
  long inputH = THTensor_(size)(input, heightDim);
  long inputW = THTensor_(size)(input, widthDim);
  long inputT = THTensor_(size)(input, timeDim);

  THArgCheck(outputH + poolSizeH - 1 < inputH, 9,
             "poolSizeH (%d) too large relative to input height (%d)",
	     poolSizeH, inputH);
  THArgCheck(outputW + poolSizeW - 1 < inputW, 8,
             "poolSizeW (%d) too large relative to input width (%d)",
	     poolSizeW, inputW);
  THArgCheck(outputT + poolSizeT - 1 < inputT, 7,
             "poolSizeT (%d) too large relative to input time (%d)",
	     poolSizeT, inputT);

  /* get contiguous input */
  input = THTensor_(newContiguous)(input);

  if (numInputDims == 4) {
    /* resize output */
    THTensor_(resize4d)(output, numPlanes, outputH, outputW, outputT);
    /* indices will contain the locations for each output point */
    THIndexTensor_(resize4d)(indices, numPlanes, outputH, outputW, outputT);

    THNN_(VolumetricFractionalMaxPooling_updateOutput_frame)(
      THTensor_(data)(input),
      THTensor_(data)(output),
      THIndexTensor_(data)(indices),
      THTensor_(data)(randomSamples),
      numPlanes, inputT, inputW, inputH,
      outputT, outputW, outputH, poolSizeT, poolSizeW, poolSizeH);
  } else {
    THTensor_(resize5d)(output, numBatch, numPlanes, outputH, outputW, outputT);
    /* indices will contain the locations for each output point */
    THIndexTensor_(resize5d)(indices, numBatch, numPlanes, outputH, outputW, outputT);

    long batch;
#pragma omp parallel for private(batch)
    for (batch = 0; batch < numBatch; ++batch) {
      THNN_(VolumetricFractionalMaxPooling_updateOutput_frame)(
        THTensor_(data)(input) + batch * numPlanes * inputH * inputW * inputT,
        THTensor_(data)(output) + batch * numPlanes * outputH * outputW * outputT,
        THIndexTensor_(data)(indices) + batch * numPlanes * outputH * outputW * outputT,
        THTensor_(data)(randomSamples) + batch * numPlanes * 3,
        numPlanes, inputT, inputW, inputH,
        outputT, outputW, outputH, poolSizeT, poolSizeW, poolSizeH);
    }
  }

  /* cleanup */
  THTensor_(free)(input);
}

static void THNN_(VolumetricFractionalMaxPooling_updateGradInput_frame)(
  real* gradInput,
  real* gradOutput,
  THIndex_t* indices,
  long numPlanes,
  long inputT, long inputW, long inputH,
  long outputT, long outputW, long outputH) {
  long plane;
#pragma omp parallel for private(plane)
  for (plane = 0; plane < numPlanes; plane++) {
    real* gradInputForPlane = gradInput + plane * inputT * inputW * inputH;
    real* gradOutputForPlane = gradOutput + plane * outputT * outputW * outputH;
    THIndex_t* indicesForPlane = indices + plane * outputT * outputW * outputH;

    long h, w, t;
    for (h = 0; h < outputH; ++h) {
      for (w = 0; w < outputW; ++w) {
        for (t = 0; t < outputT; ++t) {
          long outputIndex = h * outputW * outputT + w * outputT + t;
          long index = indicesForPlane[outputIndex] - TH_INDEX_BASE;
          THAssert(index >= 0 && index < inputT * inputW * inputH);

          gradInputForPlane[index] += gradOutputForPlane[outputIndex];
        }
      }
    }
  }
}

void THNN_(VolumetricFractionalMaxPooling_updateGradInput)(
    THNNState *state,
    THTensor *input,
    THTensor *gradOutput,
    THTensor *gradInput,
    int outputT, int outputW, int outputH,
    int poolSizeT, int poolSizeW, int poolSizeH,
    THIndexTensor *indices) {

  long numBatch = 1;
  int planeDim = 0;
  int heightDim = 1;
  int widthDim = 2;
  int timeDim = 3;

  long numInputDims = THTensor_(nDimension)(input);
  if (numInputDims == 5) {
    numBatch = THTensor_(size)(input, 0);
    planeDim = 1;
    heightDim++;
    widthDim++;
    timeDim++;
  }

  /* sizes */
  long numPlanes = THTensor_(size)(input, planeDim);
  long inputH = THTensor_(size)(input, heightDim);
  long inputW = THTensor_(size)(input, widthDim);
  long inputT = THTensor_(size)(input, timeDim);

  THArgCheck(outputT == THTensor_(size)(gradOutput, timeDim), 3,
             "gradOutput time unexpected");
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
  if (numInputDims == 4) {
    THNN_(VolumetricFractionalMaxPooling_updateGradInput_frame)(
      THTensor_(data)(gradInput),
      THTensor_(data)(gradOutput),
      THIndexTensor_(data)(indices),
      numPlanes, inputT, inputW, inputH, outputT, outputW, outputH);
  } else {
    long batch;
#pragma omp parallel for private(batch)
    for (batch = 0; batch < numBatch; ++batch) {
      THNN_(VolumetricFractionalMaxPooling_updateGradInput_frame)(
        THTensor_(data)(gradInput) + batch * numPlanes * inputH * inputW * inputT,
        THTensor_(data)(gradOutput) + batch * numPlanes * outputH * outputW * outputT,
        THIndexTensor_(data)(indices) + batch * numPlanes * outputH * outputW * outputT,
        numPlanes, inputT, inputW, inputH, outputT, outputW, outputH);
    }
  }

  /* cleanup */
  THTensor_(free)(gradOutput);
}

#endif
