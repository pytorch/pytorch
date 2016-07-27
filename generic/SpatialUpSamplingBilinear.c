// Adapted from interp.cpp from Caffe util by Pauline Luc
// Originally developed by George Papandreou

#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialUpSamplingBilinear.c"
#else

void THNN_(SpatialUpSamplingBilinear_updateOutput)(
    THNNState *state,
    THTensor *input,
    THTensor *output){
  input = THTensor_(newContiguous)(input);
  output = THTensor_(newContiguous)(output);
  THTensor_(zero)(output);
  real *idata = THTensor_(data)(input);
  real *odata = THTensor_(data)(output);
  int channels = THTensor_(size)(input, 0) * THTensor_(size)(input, 1);
  int height1 = THTensor_(size)(input, 2);
  int width1 = THTensor_(size)(input, 3);
  int height2 = THTensor_(size)(output, 2);
  int width2 = THTensor_(size)(output, 3);
  THAssert(height1 > 0 && width1 > 0 && height2 > 0 && width2 > 0);
  // special case: just copy
  if (height1 == height2 && width1 == width2) {
    for (int h2 = 0; h2 < height2; ++h2) {
      const int h1 = h2;
      for (int w2 = 0; w2 < width2; ++w2) {
        const int w1 = w2;
        const real* pos1 = &idata[h1 * width1 + w1];
        real* pos2 = &odata[h2 * width2 + w2];
        for (int c = 0; c < channels; ++c) {
          pos2[0] = pos1[0];
          pos1 += width1 * height1;
          pos2 += width2 * height2;
        }
      }
    }
    return;
  }
  const float rheight =(height2 > 1) ? (float)(height1 - 1)/(height2 - 1) : 0.f;
  const float rwidth = (width2 > 1) ? (float)(width1 - 1) / (width2 - 1) : 0.f;
  for (int h2 = 0; h2 < height2; ++h2) {
    const float h1r = rheight * h2;
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const real h1lambda = h1r - h1;
    const real h0lambda = (real)1. - h1lambda;
    for (int w2 = 0; w2 < width2; ++w2) {
      const float w1r = rwidth * w2;
      const int w1 = w1r;
      const int w1p = (w1 < width1 - 1) ? 1 : 0;
      const real w1lambda = w1r - w1;
      const real w0lambda = (real)1. - w1lambda;
      const real* pos1 = &idata[h1 * width1 + w1];
      real* pos2 = &odata[h2 * width2 + w2];
      for (int c = 0; c < channels; ++c) {
        pos2[0] = h0lambda * (w0lambda * pos1[0]+ w1lambda * pos1[w1p])
                  + h1lambda * (w0lambda * pos1[h1p * width1]
                  + w1lambda * pos1[h1p * width1 + w1p]);
        pos1 += width1 * height1;
        pos2 += width2 * height2;
      }
    }
  }
}

void THNN_(SpatialUpSamplingBilinear_updateGradInput)(
    THNNState *state,
    THTensor *gradOutput,
    THTensor *gradInput){
  gradInput = THTensor_(newContiguous)(gradInput);
  gradOutput = THTensor_(newContiguous)(gradOutput);
  THTensor_(zero)(gradInput);
  real *data1 = THTensor_(data)(gradInput);
  real *data2 = THTensor_(data)(gradOutput);
  int channels = THTensor_(size)(gradInput, 0) * THTensor_(size)(gradInput, 1);
  int height1 = THTensor_(size)(gradInput, 2);
  int width1 = THTensor_(size)(gradInput, 3);
  int height2 = THTensor_(size)(gradOutput, 2);
  int width2 = THTensor_(size)(gradOutput, 3);
  THAssert(height1 > 0 && width1 > 0 && height2 > 0 && width2 > 0);
  // special case: same-size matching grids
  if (height1 == height2 && width1 == width2) {
    for (int h2 = 0; h2 < height2; ++h2) {
      const int h1 = h2;
      for (int w2 = 0; w2 < width2; ++w2) {
        const int w1 = w2;
        real* pos1 = &data1[h1 * width1 + w1];
        const real* pos2 = &data2[h2 * width2 + w2];
        for (int c = 0; c < channels; ++c) {
          pos1[0] += pos2[0];
          pos1 += width1 * height1;
          pos2 += width2 * height2;
        }
      }
    }
    return;
  }
  const float rheight =(height2 > 1) ? (float)(height1 - 1)/(height2 - 1) : 0.f;
  const float rwidth = (width2 > 1) ? (float)(width1 - 1)/(width2 - 1) : 0.f;
  for (int h2 = 0; h2 < height2; ++h2) {
    const float h1r = rheight * h2;
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const real h1lambda = h1r - h1;
    const real h0lambda = (real)1. - h1lambda;
    for (int w2 = 0; w2 < width2; ++w2) {
      const float w1r = rwidth * w2;
      const int w1 = w1r;
      const int w1p = (w1 < width1 - 1) ? 1 : 0;
      const real w1lambda = w1r - w1;
      const real w0lambda = (real)1. - w1lambda;
      real* pos1 = &data1[h1 * width1 + w1];
      const real* pos2 = &data2[h2 * width2 + w2];
      for (int c = 0; c < channels; ++c) {
        pos1[0] += h0lambda * w0lambda * pos2[0];
        pos1[w1p] += h0lambda * w1lambda * pos2[0];
        pos1[h1p * width1] += h1lambda * w0lambda * pos2[0];
        pos1[h1p * width1 + w1p] += h1lambda * w1lambda * pos2[0];
        pos1 += width1 * height1;
        pos2 += width2 * height2;
      }
    }
  }
}

#endif
