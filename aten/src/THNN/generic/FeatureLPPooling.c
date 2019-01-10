#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/FeatureLPPooling.c"
#else

#ifndef FEATURE_LP_DEFS
#define FEATURE_LP_DEFS

#ifdef _MSC_VER
  #define FEATURE_LP_SIZE_TYPE int64_t
  #define FEATURE_LP_CAST_TYPE (int64_t)
#else
  #define FEATURE_LP_SIZE_TYPE size_t
  #define FEATURE_LP_CAST_TYPE
#endif

typedef struct {
  size_t size[4];
  size_t stride[4];
} FeatureLPPoolingSizes;

static inline size_t flpGetOffset(FeatureLPPoolingSizes* s,
                           FEATURE_LP_SIZE_TYPE batch,
                           FEATURE_LP_SIZE_TYPE feature,
                           FEATURE_LP_SIZE_TYPE opt1,
                           FEATURE_LP_SIZE_TYPE opt2) {
  return s->stride[0] * batch +
    s->stride[1] * feature +
    s->stride[2] * opt1 +
    s->stride[3] * opt2;
}

static inline size_t flpOutputSize(FEATURE_LP_SIZE_TYPE inputSize,
                            FEATURE_LP_SIZE_TYPE width,
                            FEATURE_LP_SIZE_TYPE stride) {
  return ((inputSize - width) / stride) + 1;
}

#endif // FEATURE_LP_DEFS

FeatureLPPoolingSizes
THNN_(FeatureLPPooling_upcastCPU)(THTensor* t, bool batchMode) {
  int dim = THTensor_(nDimension)(t);

  // Upcast to [batch dim][feature dim][opt dim 1][opt dim 2]
  FeatureLPPoolingSizes s;
  for (int i = 0; i < 4; ++i) {
    s.size[i] = 1;
    s.stride[i] = 1;
  }

  if (dim == 1) {
    THAssert(!batchMode);
    // [feature dim]
    s.size[1] = THTensor_(size)(t, 0);
    s.stride[1] = THTensor_(stride)(t, 0);
  } else if (dim == 2) {
    if (batchMode) {
      // [batch dim][feature dim]
      for (int i = 0; i < 2; ++i) {
        s.size[i] = THTensor_(size)(t, i);
        s.stride[i] = THTensor_(stride)(t, i);
      }
    } else {
      // [feature dim][opt dim 1]
      s.size[1] = THTensor_(size)(t, 0);
      s.stride[1] = THTensor_(stride)(t, 0);
      s.size[2] = THTensor_(size)(t, 1);
      s.stride[2] = THTensor_(stride)(t, 1);
    }
  } else if (dim == 3) {
    if (batchMode) {
      // [batch dim][feature dim][opt dim 1]
      for (int i = 0; i < 3; ++i) {
        s.size[i] = THTensor_(size)(t, i);
        s.stride[i] = THTensor_(stride)(t, i);
      }
    } else {
      // [feature dim][opt dim 1][opt dim 2]
      for (int i = 1; i < 4; ++i) {
        s.size[i] = THTensor_(size)(t, i - 1);
        s.stride[i] = THTensor_(stride)(t, i - 1);
      }
    }
  } else if (dim == 4) {
    // [batch dim][feature dim][opt dim 1][opt dim 2]
    THAssert(batchMode);
    for (int i = 0; i < 4; ++i) {
      s.size[i] = THTensor_(size)(t, i);
      s.stride[i] = THTensor_(stride)(t, i);
    }
  }

  return s;
}

void
THNN_(FeatureLPPooling_resizeForOutputCPU)(THTensor* toResize,
                                           THTensor* input,
                                           bool batchMode,
                                           int width,
                                           int stride) {
  int inputDim = THTensor_(nDimension)(input);
  THAssert(inputDim >= 1 && inputDim <= 4);

  int64_t outSize =
    flpOutputSize(THTensor_(size)(input, 0), width, stride);
  if (batchMode) {
    THAssert(inputDim > 1);
    outSize =
      flpOutputSize(THTensor_(size)(input, 1), width, stride);
  } else {
    THAssert(inputDim < 4);
  }

  if (inputDim == 1) {
    THTensor_(resize1d)(toResize, outSize);
  } else if (inputDim == 2) {
    if (batchMode) {
      THTensor_(resize2d)(toResize,
                          THTensor_(size)(input, 0),
                          outSize);
    } else {
      THTensor_(resize2d)(toResize,
                          outSize,
                          THTensor_(size)(input, 1));
    }
  } else if (inputDim == 3) {
    if (batchMode) {
      THTensor_(resize3d)(toResize,
                          THTensor_(size)(input, 0), outSize,
                          THTensor_(size)(input, 2));
    } else {
      THTensor_(resize3d)(toResize,
                          outSize, THTensor_(size)(input, 1),
                          THTensor_(size)(input, 2));
    }
  } else if (inputDim == 4) {
    THTensor_(resize4d)(toResize,
                        THTensor_(size)(input, 0),
                        outSize,
                        THTensor_(size)(input, 2),
                        THTensor_(size)(input, 3));
  }
}

// Makes `toResize` the same size/dimensionality as `src`
void
THNN_(FeatureLPPooling_resizeCPU)(THTensor* toResize,
                                  THTensor* src) {
  int inputDim = THTensor_(nDimension)(src);
  THAssert(inputDim >= 1 && inputDim <= 4);

  if (inputDim == 1) {
    THTensor_(resize1d)(toResize,
                        THTensor_(size)(src, 0));
  } else if (inputDim == 2) {
    THTensor_(resize2d)(
      toResize,
      THTensor_(size)(src, 0),
      THTensor_(size)(src, 1));
  } else if (inputDim == 3) {
    THTensor_(resize3d)(
      toResize,
      THTensor_(size)(src, 0),
      THTensor_(size)(src, 1),
      THTensor_(size)(src, 2));
  } else if (inputDim == 4) {
    THTensor_(resize4d)(
      toResize,
      THTensor_(size)(src, 0),
      THTensor_(size)(src, 1),
      THTensor_(size)(src, 2),
      THTensor_(size)(src, 3));
  }
}

void
THNN_(FeatureLPPooling_updateOutput)(
  THNNState *state,
  THTensor *input,
  THTensor *output,
  accreal power,
  int width,
  int stride,
  bool batchMode) {
  int inputDim = THTensor_(nDimension)(input);

  if (batchMode) {
    THArgCheck(inputDim >= 2 && inputDim <= 4, 2,
               "input must be 2-4 dimensions for batch mode");
  } else {
    THArgCheck(inputDim >= 1 && inputDim <= 3, 2,
               "input must be 1-3 dimensions for non-batch mode");
  }

  FeatureLPPoolingSizes inputDesc =
    THNN_(FeatureLPPooling_upcastCPU)(input, batchMode);

  // Make sure the feature dimension is properly sized
  THArgCheck(inputDesc.size[1] >= (FEATURE_LP_SIZE_TYPE) width, 3,
             "input: feature dimension must be >= width");

  // Make sure that width and stride are within range
  THArgCheck(width >= 2 && width <= 16, 5,
             "width must be between 2 - 16");

  THArgCheck(stride >= 1 && stride <= 4, 6,
             "stride must be between 1 - 4");

  // Resize output

  THNN_(FeatureLPPooling_resizeForOutputCPU)(
    output, input, batchMode, width, stride);

  FeatureLPPoolingSizes outputDesc =
    THNN_(FeatureLPPooling_upcastCPU)(output, batchMode);

  real* inputP = THTensor_(data)(input);
  real* outputP = THTensor_(data)(output);

  FEATURE_LP_SIZE_TYPE batch, opt1, opt2, outputFeature, i;

#pragma omp parallel for
  for (batch = 0; batch < FEATURE_LP_CAST_TYPE inputDesc.size[0]; ++batch) {
    for (opt1 = 0; opt1 < FEATURE_LP_CAST_TYPE inputDesc.size[2]; ++opt1) {
      for (opt2 = 0; opt2 < FEATURE_LP_CAST_TYPE inputDesc.size[3]; ++opt2) {
        for (outputFeature = 0;
             outputFeature < FEATURE_LP_CAST_TYPE outputDesc.size[1]; ++outputFeature) {

          accreal v = (accreal) 0;
          for (i = 0; i < (FEATURE_LP_SIZE_TYPE) width; ++i) {
            FEATURE_LP_SIZE_TYPE inputFeature = outputFeature * stride + i;
            if (inputFeature >= FEATURE_LP_CAST_TYPE inputDesc.size[1]) {
              break;
            }

            v +=
              pow(inputP[flpGetOffset(&inputDesc,
                                      batch,
                                      inputFeature,
                                      opt1,
                                      opt2)], power);
          }

          outputP[flpGetOffset(&outputDesc, batch, outputFeature, opt1, opt2)] =
            pow(v, (accreal) 1 / power);
        }
      }
    }
  }
}

void
THNN_(FeatureLPPooling_updateGradInput)(
  THNNState *state,
  THTensor* gradOutput,
  THTensor* input,
  THTensor* output,
  THTensor* gradInput,
  accreal power,
  int width,
  int stride,
  bool batchMode) {
  int inputDim = THTensor_(nDimension)(input);

  if (batchMode) {
    THArgCheck(inputDim >= 2 && inputDim <= 4, 3,
               "input must be 2-4 dimensions for batch mode");
  } else {
    THArgCheck(inputDim >= 1 && inputDim <= 3, 3,
               "input must be 1-3 dimensions for non-batch mode");
  }

  FeatureLPPoolingSizes inputDesc =
    THNN_(FeatureLPPooling_upcastCPU)(input, batchMode);
  FeatureLPPoolingSizes gradOutputDesc =
    THNN_(FeatureLPPooling_upcastCPU)(gradOutput, batchMode);
  FeatureLPPoolingSizes outputDesc =
    THNN_(FeatureLPPooling_upcastCPU)(output, batchMode);

  // Make sure the feature dimension is properly sized
  THArgCheck(inputDesc.size[1] >= (FEATURE_LP_SIZE_TYPE) width, 3,
             "input: feature dimension must be >= width");

  // Make sure that width and stride are within range
  THArgCheck(width >= 2 && width <= 16, 7,
             "width must be between 2 - 16");

  THArgCheck(stride >= 1 && stride <= 4, 8,
             "stride must be between 1 - 4");

  for (int i = 0; i < 4; ++i) {
    THAssertMsg(outputDesc.size[i] == gradOutputDesc.size[i],
                "output and gradOutput sizes do not match");
  }

  // Make sure that the input sizes produce the output sizes
  THArgCheck(flpOutputSize(FEATURE_LP_CAST_TYPE inputDesc.size[1], width, stride) ==
             outputDesc.size[1], 3,
             "input and output sizes do not match with respect to "
             "width and stride");

  // Resize `gradInput` based on `input`
  THNN_(FeatureLPPooling_resizeCPU)(gradInput, input);

  // Zero gradInput for accumulation
  THTensor_(zero)(gradInput);

  FeatureLPPoolingSizes gradInputDesc =
    THNN_(FeatureLPPooling_upcastCPU)(gradInput, batchMode);

  real* gradOutputP = THTensor_(data)(gradOutput);
  real* gradInputP = THTensor_(data)(gradInput);
  real* outputP = THTensor_(data)(output);
  real* inputP = THTensor_(data)(input);

  FEATURE_LP_SIZE_TYPE batch, opt1, opt2, outputFeature, i;

#pragma omp parallel for
  for (batch = 0; batch < FEATURE_LP_CAST_TYPE inputDesc.size[0]; ++batch) {
    for (opt1 = 0; opt1 < FEATURE_LP_CAST_TYPE inputDesc.size[2]; ++opt1) {
      for (opt2 = 0; opt2 < FEATURE_LP_CAST_TYPE inputDesc.size[3]; ++opt2) {
        for (outputFeature = 0;
             outputFeature < FEATURE_LP_CAST_TYPE outputDesc.size[1]; ++outputFeature) {

          // Load output (f(x_is)). It is possible that this is zero, in
          // which case we'll ignore this point.
          real outputV =
            outputP[
              flpGetOffset(&outputDesc, batch, outputFeature, opt1, opt2)];

          if (outputV == (real) 0) {
            continue;
          }

          for (i = 0; i < (FEATURE_LP_SIZE_TYPE) width; ++i) {
            FEATURE_LP_SIZE_TYPE inputFeature = outputFeature * stride + i;
            THAssert(inputFeature < inputDesc.size[1]);

            real gradOutputV =
              gradOutputP[
                flpGetOffset(&gradOutputDesc, batch, outputFeature, opt1, opt2)];
            real inputV =
              inputP[
                flpGetOffset(&inputDesc, batch, inputFeature, opt1, opt2)];

            // Calculate grad * (x_i / f(x_is))^(p - 1)
            real v = gradOutputV * pow(inputV / outputV, power - (accreal) 1);

            gradInputP[
              flpGetOffset(&gradInputDesc, batch, inputFeature, opt1, opt2)]
              += v;
          }
        }
      }
    }
  }
}

#endif
