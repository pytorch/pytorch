#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "THNN/generic/SpatialConvolutionMM.c"
#else

#include <ATen/div_rtn.h>
#include <ATen/Parallel.h>

static inline void THNN_(SpatialConvolutionMM_shapeCheck)(
        THTensor *input, THTensor *gradOutput,
        THTensor *weight, THTensor *bias,
        int kH, int kW, int dH, int dW, int padH, int padW, int weight_nullable) {

  THArgCheck(kW > 0 && kH > 0, 9,
               "kernel size should be greater than zero, but got kH: %d kW: %d", kH, kW);
  THArgCheck(dW > 0 && dH > 0, 11,
             "stride should be greater than zero, but got dH: %d dW: %d", dH, dW);

  if (weight != NULL) {
    THNN_ARGCHECK(!weight->is_empty() && (weight->dim() == 2 || weight->dim() == 4), 5, weight,
                    "non-empty 2D or 4D weight tensor expected, but got: %s");
    if (bias != NULL) {
      THNN_CHECK_DIM_SIZE(bias, 1, 0, weight->size(0));
    }
  } else if (!weight_nullable) {
    THError("weight tensor is expected to be non-nullable");
  }

  int ndim = input->dim();
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;

  if (ndim == 4) {
    dimf++;
    dimh++;
    dimw++;
  }

  // Allow for empty batch size but not other dimensions
  bool valid_empty = false;
  if (ndim == 3) {
    valid_empty = input->size(0) == 0 && input->size(1) != 0 && input->size(2) != 0;
  } else if (ndim == 4) {
    valid_empty = input->size(0) == 0 && input->size(1) != 0 && input->size(2) != 0 && input->size(3) != 0;
  }

  THNN_ARGCHECK((!input->is_empty() || valid_empty) && (ndim == 3 || ndim == 4), 2, input,
                "non-empty 3D or 4D input tensor expected but got: %s");

  int64_t inputHeight  = input->size(dimh);
  int64_t inputWidth   = input->size(dimw);

  int64_t exactInputHeight = inputHeight + 2 * padH;
  int64_t exactInputWidth = inputWidth + 2 * padW;

  if (exactInputHeight < kH || exactInputWidth < kW) {
    THError("Calculated padded input size per channel: (%ld x %ld). "
      "Kernel size: (%d x %d). Kernel size can't be greater than actual input size",
      exactInputHeight, exactInputWidth, kH, kW);
  }

  int64_t outputHeight = div_rtn<int64_t>(exactInputHeight - kH, dH) + 1;
  int64_t outputWidth  = div_rtn<int64_t>(exactInputWidth - kW, dW) + 1;

  if (outputWidth < 1 || outputHeight < 1) {
    THError("Given input size per channel: (%ld x %ld). "
      "Calculated output size per channel: (%ld x %ld). Output size is too small",
      inputHeight, inputWidth, outputHeight, outputWidth);
  }

  if (weight != NULL) {
    int64_t nInputPlane = weight->size(1);
    if (weight->dim() == 2) {
      nInputPlane /= (kH * kW);
    }
    THNN_CHECK_DIM_SIZE(input, ndim, dimf, nInputPlane);
  }

  if (gradOutput != NULL) {
    if (weight != NULL) {
      int64_t nOutputPlane = weight->size(0);
      THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimf, nOutputPlane);
    } else if (bias != NULL) {
      int64_t nOutputPlane = THTensor_sizeLegacyNoScalars(bias, 0);
      THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimf, nOutputPlane);
    }
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimh, outputHeight);
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimw, outputWidth);
  }
}

static THTensor* THNN_(newViewWeightMM2d)(THTensor *weight) {
  weight = THTensor_(newContiguous)(weight);
  if (weight->dim() == 4) {
    int64_t s1 = weight->size(0);
    int64_t s2 = weight->size(1) * weight->size(2) * weight->size(3);
    THTensor *old_weight = weight;
    weight = THTensor_(newWithStorage2d)(THTensor_getStoragePtr(weight), weight->storage_offset(),
                                         s1, -1, s2, -1);
        c10::raw::intrusive_ptr::decref(old_weight);
  }
  return weight;
}

static void THNN_(SpatialConvolutionMM_updateOutput_frame)(
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *finput,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH,
          int64_t nInputPlane,
          int64_t inputWidth,
          int64_t inputHeight,
          int64_t nOutputPlane,
          int64_t outputWidth,
          int64_t outputHeight)
{
  int64_t i;
  THTensor *output2d;

  THNN_(unfolded_copy)(finput, input, kW, kH, dW, dH, padW, padH,
                       nInputPlane, inputWidth, inputHeight,
                       outputWidth, outputHeight);

  output2d = THTensor_(newWithStorage2d)(THTensor_getStoragePtr(output), output->storage_offset(),
                                         nOutputPlane, -1,
                                         outputHeight*outputWidth, -1);
  if (bias) {
    for(i = 0; i < nOutputPlane; i++)
        THVector_(fill)
          (THStorage_(data)(THTensor_getStoragePtr(output)) + output->storage_offset() + output->stride(0) * i,
           THTensor_(get1d)(bias, i), outputHeight*outputWidth);
  } else {
    THTensor_(zero)(output);
  }

  THTensor_(addmm)(output2d, output2d, weight, finput, 1, 1);

  c10::raw::intrusive_ptr::decref(output2d);
}

void THNN_(SpatialConvolutionMM_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *finput,
          THTensor *fgradInput,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH)
{
  weight = THNN_(newViewWeightMM2d)(weight);

  THNN_(SpatialConvolutionMM_shapeCheck)
    (input, NULL, weight, bias, kH, kW, dH, dW, padH, padW, 0);

  input = THTensor_(newContiguous)(input);
  int ndim = input->dim();
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;

  if (ndim == 4) {
    dimf++;
    dimh++;
    dimw++;
  }

  int64_t nInputPlane = input->size(dimf);
  int64_t inputHeight  = input->size(dimh);
  int64_t inputWidth   = input->size(dimw);
  int64_t nOutputPlane = weight->size(0);
  int64_t outputHeight = (inputHeight + 2*padH - kH) / dH + 1;
  int64_t outputWidth  = (inputWidth + 2*padW - kW) / dW + 1;

  if(input->dim() == 3)
  {
    THTensor_(resize2d)(finput, kW*kH*nInputPlane, outputHeight*outputWidth);
    THTensor_(resize3d)(output, nOutputPlane, outputHeight, outputWidth);

    THNN_(SpatialConvolutionMM_updateOutput_frame)
      (input, output, weight, bias, finput,
       kW, kH, dW, dH, padW, padH,
       nInputPlane, inputWidth, inputHeight,
       nOutputPlane, outputWidth, outputHeight);
  }
  else
  {
    int64_t T = input->size(0);

    THTensor_(resize3d)(finput, T, kW*kH*nInputPlane, outputHeight*outputWidth);
    THTensor_(resize4d)(output, T, nOutputPlane, outputHeight, outputWidth);

    at::parallel_for(0, T, 0, [&](int64_t start, int64_t end) {
      for(auto t = start; t < end; t++)
      {
        THTensor *input_t = THTensor_(newSelect)(input, 0, t);
        THTensor *output_t = THTensor_(newSelect)(output, 0, t);
        THTensor *finput_t = THTensor_(newSelect)(finput, 0, t);

        THNN_(SpatialConvolutionMM_updateOutput_frame)
          (input_t, output_t, weight, bias, finput_t,
           kW, kH, dW, dH, padW, padH,
           nInputPlane, inputWidth, inputHeight,
           nOutputPlane, outputWidth, outputHeight);

        c10::raw::intrusive_ptr::decref(input_t);
        c10::raw::intrusive_ptr::decref(output_t);
        c10::raw::intrusive_ptr::decref(finput_t);
      }
    });
  }

  c10::raw::intrusive_ptr::decref(input);
  c10::raw::intrusive_ptr::decref(weight);
}

#if !defined(TH_REAL_IS_LONG)

static void THNN_(SpatialConvolutionMM_updateGradInput_frame)(
          THTensor *gradInput,
          THTensor *gradOutput,
          THTensor *weight,
          THTensor *fgradInput,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH)
{
  THTensor *gradOutput2d = THTensor_(newWithStorage2d)
    (THTensor_getStoragePtr(gradOutput), gradOutput->storage_offset(),
     gradOutput->size(0), -1,
     gradOutput->size(1)*gradOutput->size(2), -1);
  THTensor_(addmm)(fgradInput, fgradInput, weight, gradOutput2d, 0, 1);
  c10::raw::intrusive_ptr::decref(gradOutput2d);

  THTensor_(zero)(gradInput);

  THNN_(unfolded_acc)(fgradInput, gradInput, kW, kH, dW, dH,
                      padW, padH,
                      gradInput->size(0), gradInput->size(2), gradInput->size(1),
                      gradOutput->size(2), gradOutput->size(1));
}

void THNN_(SpatialConvolutionMM_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *finput,
          THTensor *fgradInput,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH)
{
  weight = THNN_(newViewWeightMM2d)(weight);

  THNN_(SpatialConvolutionMM_shapeCheck)
    (input, gradOutput, weight, NULL, kH, kW, dH, dW, padH, padW, 0);

  input = THTensor_(newContiguous)(input);
  gradOutput = THTensor_(newContiguous)(gradOutput);

  THTensor_(resizeAs)(gradInput, input);
  THTensor_(resizeAs)(fgradInput, finput);

  // depending on the BLAS library, fgradInput (result tensor) might
  // be left uninitialized on zero alpha, which might lead to weird behavior
  // hence, to be safe, zero it
  THTensor_(zero)(fgradInput);
  THTensor *tweight = THTensor_(new)();
  THTensor_(transpose)(tweight, weight, 0, 1);

  if(input->dim() == 3)
  {
    THNN_(SpatialConvolutionMM_updateGradInput_frame)(gradInput, gradOutput,
                                                      tweight, fgradInput,
                                                      kW, kH, dW, dH, padW, padH);
  }
  else
  {
    int64_t T = input->size(0);

    at::parallel_for(0, T, 0, [&](int64_t start, int64_t end) {
      for (auto t = start; t < end; t++)
      {
        THTensor *gradInput_t = THTensor_(newSelect)(gradInput, 0, t);
        THTensor *gradOutput_t = THTensor_(newSelect)(gradOutput, 0, t);
        THTensor *fgradInput_t = THTensor_(newSelect)(fgradInput, 0, t);

        THNN_(SpatialConvolutionMM_updateGradInput_frame)(gradInput_t, gradOutput_t,
                                                          tweight, fgradInput_t,
                                                          kW, kH, dW, dH, padW, padH);

        c10::raw::intrusive_ptr::decref(gradInput_t);
        c10::raw::intrusive_ptr::decref(gradOutput_t);
        c10::raw::intrusive_ptr::decref(fgradInput_t);
      }
    });
  }

  c10::raw::intrusive_ptr::decref(tweight);
  c10::raw::intrusive_ptr::decref(input);
  c10::raw::intrusive_ptr::decref(gradOutput);
  c10::raw::intrusive_ptr::decref(weight);
}

static void THNN_(SpatialConvolutionMM_accGradParameters_frame)(
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *finput,
          scalar_t scale)
{
  int64_t i;
  THTensor *gradOutput2d = THTensor_(newWithStorage2d)
    (THTensor_getStoragePtr(gradOutput), gradOutput->storage_offset(),
     gradOutput->size(0), -1,
     gradOutput->size(1)*gradOutput->size(2), -1);

  if (gradWeight) {
    THTensor *tfinput = THTensor_(new)();
    THTensor_(transpose)(tfinput, finput, 0, 1);
    THTensor_(addmm)(gradWeight, gradWeight, gradOutput2d, tfinput, 1, scale);
    c10::raw::intrusive_ptr::decref(tfinput);
  }

  if (gradBias) {
    for(i = 0; i < THTensor_sizeLegacyNoScalars(gradBias, 0); i++)
    {
      int64_t k;
      scalar_t sum = 0;
      scalar_t *data = THStorage_(data)(THTensor_getStoragePtr(gradOutput2d)) + gradOutput2d->storage_offset() + i*gradOutput2d->stride(0);
      for(k = 0; k < gradOutput2d->size(1); k++)
        sum += data[k];
      (THStorage_(data)(THTensor_getStoragePtr(gradBias)) + gradBias->storage_offset())[i] += scale*sum;
    }
  }

  c10::raw::intrusive_ptr::decref(gradOutput2d);
}

void THNN_(SpatialConvolutionMM_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *finput,  // can be NULL if gradWeight = NULL
          THTensor *fgradInput,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH,
          accreal scale_)
{
  scalar_t scale = TH_CONVERT_ACCREAL_TO_REAL(scale_);
  if (gradWeight) {
    THArgCheck(THTensor_(isContiguous)(gradWeight), 4, "gradWeight needs to be contiguous");
    gradWeight = THNN_(newViewWeightMM2d)(gradWeight);
  }
  if (gradBias) {
    THArgCheck(THTensor_(isContiguous)(gradBias), 5, "gradBias needs to be contiguous");
  }

  THNN_(SpatialConvolutionMM_shapeCheck)
    (input, gradOutput, gradWeight, gradBias, kH, kW, dH, dW, padH, padW, 1);

  input = THTensor_(newContiguous)(input);
  gradOutput = THTensor_(newContiguous)(gradOutput);

  if(input->dim() == 3)
  {
    THNN_(SpatialConvolutionMM_accGradParameters_frame)(gradOutput, gradWeight,
                                                        gradBias, finput, scale);
  }
  else
  {
    int64_t T = input->size(0);
    int64_t t;

    for(t = 0; t < T; t++)
    {
      THTensor *gradOutput_t = THTensor_(newSelect)(gradOutput, 0, t);
      THTensor *finput_t = NULL;
      if (gradWeight) {
        finput_t = THTensor_(newSelect)(finput, 0, t);
      }

      THNN_(SpatialConvolutionMM_accGradParameters_frame)(gradOutput_t, gradWeight,
                                                          gradBias, finput_t, scale);

      c10::raw::intrusive_ptr::decref(gradOutput_t);
      if (gradWeight) {
        c10::raw::intrusive_ptr::decref(finput_t);
      }
    }
  }

  c10::raw::intrusive_ptr::decref(input);
  c10::raw::intrusive_ptr::decref(gradOutput);
  if (gradWeight) {
    c10::raw::intrusive_ptr::decref(gradWeight);
  }
}
#endif
#endif
