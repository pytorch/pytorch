#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "THNN/generic/VolumetricConvolutionMM.c"
#else

#include <ATen/div_rtn.h>
#include <ATen/Parallel.h>

#define CONV3D_OMP_THRESHOLD 20

static void inline THNN_(VolumetricConvolutionMM_shapeCheck)(
                         THNNState *state,
                         THTensor *input,
                         THTensor *gradOutput,
                         THTensor *weight,
                         THTensor *bias,
                         int kT,
                         int kW,
                         int kH,
                         int dT,
                         int dW,
                         int dH,
                         int pT,
                         int pW,
                         int pH,
                         int weight_nullable) {
  bool valid_empty = false;
  if (input->dim() == 4) {
    valid_empty = input->size(0) == 0 && input->size(1) != 0 && input->size(2) != 0 && input->size(3) != 0;
  } else if (input->dim() == 5) {
    valid_empty = input->size(0) == 0 && input->size(1) != 0 && input->size(2) != 0 && input->size(3) != 0 && input->size(4) != 0;
  }
  THNN_ARGCHECK((!input->is_empty() || valid_empty) && (input->dim() == 4 || input->dim() == 5), 2, input,
                "non-empty 4D or 5D (batch mode) tensor expected for input, but got: %s");
  THArgCheck(kT > 0 && kW > 0 && kH > 0, 8,
             "kernel size should be greater than zero, but got kT: %d kH: %d kW: %d", kT, kH, kW);
  THArgCheck(dT > 0 && dW > 0 && dH > 0, 11,
             "stride should be greater than zero, but got dT: %d dH: %d dW: %d", dT, dH, dW);

  if (weight != NULL) {
    THNN_ARGCHECK(!weight->is_empty() && (weight->dim() == 2 || weight->dim() == 5), 5, weight,
                    "non-empty 2D or 5D weight tensor expected, but got: %s");
    if (bias != NULL) {
      THNN_CHECK_DIM_SIZE(bias, 1, 0, weight->size(0));
    }
  } else if (!weight_nullable) {
    THError("weight tensor is expected to be non-nullable");
  }

  int ndim = input->dim();
  int dimf = 0;
  int dimt = 1;
  int dimh = 2;
  int dimw = 3;

  if (ndim == 5)
  {
    dimf++;
    dimt++;
    dimh++;
    dimw++;
  }

  int64_t inputDepth;
  int64_t inputHeight;
  int64_t inputWidth;

  int64_t exactInputDepth;
  int64_t exactInputHeight;
  int64_t exactInputWidth;
  int64_t outputDepth;
  int64_t outputHeight;
  int64_t outputWidth;

  inputDepth = input->size(dimt);
  inputHeight  = input->size(dimh);
  inputWidth   = input->size(dimw);

  exactInputDepth = inputDepth + 2*pT;
  exactInputHeight = inputHeight + 2*pH;
  exactInputWidth = inputWidth + 2*pW;

  if (exactInputDepth < kT || exactInputHeight < kH || exactInputWidth < kW) {
    THError("Calculated padded input size per channel: (%ld x %ld x %ld). "
      "Kernel size: (%d x %d x %d). Kernel size can't be greater than actual input size",
      exactInputDepth, exactInputHeight, exactInputWidth, kT, kH, kW);
  }

  outputDepth  = div_rtn<int64_t>(exactInputDepth - kT, dT) + 1;
  outputHeight = div_rtn<int64_t>(exactInputHeight - kH, dH) + 1;
  outputWidth  = div_rtn<int64_t>(exactInputWidth - kW, dW) + 1;


  if (outputDepth < 1 || outputWidth < 1 || outputHeight < 1) {
    THError("Given input size per channel: (%ld x %ld x %ld). "
      "Calculated output size per channel: (%ld x %ld x %ld). Output size is too small",
      inputDepth, inputHeight, inputWidth, outputDepth, outputHeight, outputWidth);
  }

  if (weight != NULL) {
    int64_t nInputPlane = weight->size(1);
    if (weight->dim() == 2) {
      nInputPlane /= (kT * kH * kW);
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
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimt, outputDepth);
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimh, outputHeight);
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimw, outputWidth);
  }
}

static THTensor* THNN_(newViewWeight)(THTensor *weight)
{
  weight = THTensor_(newContiguous)(weight);
  if (weight->dim() == 5) {
    int64_t s1 = weight->size(0);
    int64_t s2 = weight->size(1) * weight->size(2) * weight->size(3) * weight->size(4);
    THTensor *old_weight = weight;
    weight = THTensor_(newWithStorage2d)(THTensor_getStoragePtr(weight), weight->storage_offset(),
                                         s1, -1, s2, -1);
    c10::raw::intrusive_ptr::decref(old_weight);
  }
  return weight;
}

/*
  Modified from the version of CUDA implementation, but the loop iterations is larger than that one.
  The larger loop could lower the proportion of openmp overhead. And the inner part in loop is simpler.
  The naive code is below:

  scalar_t *input_data = input->data<scalar_t>();
  scalar_t *finput_data = finput->data<scalar_t>();

  int64_t n = nInputPlane*kT*kH*kW*outputDepth*outputWidth*outputHeight;
  #pragma omp parallel for firstprivate(finput_data, input_data, outputWidth, outputHeight, outputDepth, kW, kH, kT, dW, dH, dT, pW, pH, pT, inputHeight, inputWidth, inputDepth)
  for (int64_t idx = 0; idx < n ; ++idx) {
    int64_t w_out = line_index_offset % outputWidth;
    int64_t remained = line_index_offset / outputWidth;
    int64_t h_out = remained % outputHeight;
    remained /= outputHeight;
    int64_t d_out = remained % outputDepth;
    remained /= outputDepth;
    int k = remained % kW;
    remained /= kW;
    int j = remained % kH;
    remained /= kH;
    int i = remained % kT;
    int64_t nip = remained / kT;

    int64_t d = d_out * dT - pT + i;
    int64_t h = h_out * dH - pH + j;
    int64_t w = w_out * dW - pW + k;

    finput_data[idx] = (h >= 0 && w >= 0 && d >= 0 && h < inputHeight && w < inputWidth && d < inputDepth) ?
      input_data[nip*inputDepth*inputWidth*inputHeight+ d*inputHeight*inputWidth + h*inputWidth + w] : 0;
  }

  However, there are 6 quotient and 6 module operations which are very time-consuming. So we choose relatively
  more complex but more efficient pattern.
*/
static void THNN_(unfolded_copy_vol)(
          THTensor *finput,
          THTensor *input,
          int kT,
          int kW,
          int kH,
          int dT,
          int dW,
          int dH,
          int pT,
          int pW,
          int pH,
          int64_t nInputPlane,
          int64_t inputDepth,
          int64_t inputWidth,
          int64_t inputHeight,
          int64_t outputDepth,
          int64_t outputWidth,
          int64_t outputHeight)
{
  scalar_t *input_data = input->data<scalar_t>();
  scalar_t *finput_data = finput->data<scalar_t>();

  int64_t n = nInputPlane*kT*kH*kW*outputDepth*outputWidth*outputHeight;
  at::parallel_for(0, n, 0, [&](int64_t start, int64_t end) {
    int64_t line_index_offset = start;
    int64_t line_seg_len = (end - start);

    int64_t w_out = line_index_offset % outputWidth;
    int64_t remained = line_index_offset / outputWidth;
    int64_t h_out = remained % outputHeight;
    remained /= outputHeight;
    int64_t d_out = remained % outputDepth;
    remained /= outputDepth;
    int k = remained % kW;
    remained /= kW;
    int j = remained % kH;
    remained /= kH;
    int i = remained % kT;
    int64_t nip = remained / kT;

    int64_t count = 0;
    scalar_t* dst = finput_data + line_index_offset;
    int64_t inputHW = inputHeight*inputWidth;
    int64_t inputDHW = inputHW*inputDepth;

    while (count < line_seg_len) {
      int64_t w = w_out * dW - pW + k;
      int64_t h = h_out * dH - pH + j;
      int64_t d = d_out * dT - pT + i;


      *dst = (h >= 0 && w >= 0 && d >= 0 && h < inputHeight && w < inputWidth && d < inputDepth) ?
        input_data[nip*inputDHW+ d*inputHW + h*inputWidth + w] : scalar_t(0);

      count++;
      if (count < line_seg_len) {
        dst++;
        w_out++;
        if (w_out == outputWidth) {
          w_out = 0;
          h_out++;
          if (h_out == outputHeight) {
            h_out = 0;
            d_out++;
            if (d_out == outputDepth) {
              d_out = 0;
              k++;
              if(k == kW) {
                k = 0;
                j++;
                if(j == kH) {
                  j = 0;
                  i++;
                  if(i == kT) {
                    i = 0;
                    nip++;
                  }
                }
              }
            }
          }
        }
      }
    }
  });
}

static void THNN_(VolumetricConvolutionMM_updateOutput_frame)(
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *finput,
          int kT,
          int kW,
          int kH,
          int dT,
          int dW,
          int dH,
          int pT,
          int pW,
          int pH,
          int64_t nInputPlane,
          int64_t inputDepth,
          int64_t inputWidth,
          int64_t inputHeight,
          int64_t nOutputPlane,
          int64_t outputDepth,
          int64_t outputWidth,
          int64_t outputHeight)
{
  int64_t i;
  THTensor *output2d;

  THNN_(unfolded_copy_vol)(
    finput, input,
    kT, kW, kH,
    dT, dW, dH,
    pT, pW, pH,
    nInputPlane,
    inputDepth, inputWidth, inputHeight,
    outputDepth, outputWidth, outputHeight
  );

  output2d = THTensor_(newWithStorage2d)(
    THTensor_getStoragePtr(output), output->storage_offset(), nOutputPlane, -1,
    outputDepth*outputHeight*outputWidth, -1
  );

  if (bias) {
      for (i = 0; i < nOutputPlane; i++)
      {
        THVector_(fill)(
          THStorage_(data)(THTensor_getStoragePtr(output))+output->storage_offset()+output->stride(0)*i,
          THTensor_(get1d)(bias, i),
          outputDepth*outputHeight*outputWidth
        );
      }
  } else {
    THTensor_(zero)(output);
  }

  THTensor_(addmm)(output2d, output2d, weight, finput, 1, 1);

  c10::raw::intrusive_ptr::decref(output2d);
}

void THNN_(VolumetricConvolutionMM_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *finput,
          THTensor *fgradInput, // unused
          int kT,
          int kW,
          int kH,
          int dT,
          int dW,
          int dH,
          int pT,
          int pW,
          int pH)
{
  int dimf = 0;
  int dimt = 1;
  int dimh = 2;
  int dimw = 3;

  int64_t nInputPlane;
  int64_t inputDepth;
  int64_t inputHeight;
  int64_t inputWidth;
  int64_t nOutputPlane;
  int64_t outputDepth;
  int64_t outputHeight;
  int64_t outputWidth;

  THNN_(VolumetricConvolutionMM_shapeCheck)(
        state, input, NULL, weight, bias,
        kT, kW, kH, dT, dW, dH, pT, pW, pH, 0);
  input = THTensor_(newContiguous)(input);

  if (input->dim() == 5)
  {
    dimf++;
    dimt++;
    dimh++;
    dimw++;
  }

  nInputPlane = input->size(dimf);
  inputDepth = input->size(dimt);
  inputHeight  = input->size(dimh);
  inputWidth   = input->size(dimw);
  nOutputPlane = weight->size(0);
  outputDepth  = (inputDepth + 2*pT - kT) / dT + 1;
  outputHeight = (inputHeight + 2*pH - kH) / dH + 1;
  outputWidth  = (inputWidth + 2*pW - kW) / dW + 1;

  weight = THNN_(newViewWeight)(weight);

  if (input->dim() == 4)
  {
    THTensor_(resize2d)(finput, kT*kW*kH*nInputPlane, outputDepth*outputHeight*outputWidth);
    THTensor_(resize4d)(output, nOutputPlane, outputDepth, outputHeight, outputWidth);

    THNN_(VolumetricConvolutionMM_updateOutput_frame)(
      input, output, weight, bias, finput,
      kT, kW, kH,
      dT, dW, dH,
      pT, pW, pH,
      nInputPlane, inputDepth, inputWidth, inputHeight,
      nOutputPlane, outputDepth, outputWidth, outputHeight
    );
  }
  else
  {
    int64_t T = input->size(0);

    THTensor_(resize3d)(finput, T, kT*kW*kH*nInputPlane, outputDepth*outputHeight*outputWidth);
    THTensor_(resize5d)(output, T, nOutputPlane, outputDepth, outputHeight, outputWidth);
    at::parallel_for(0, T, CONV3D_OMP_THRESHOLD, [&](int64_t start, int64_t end) {
      for (auto t = start; t < end; t++)
      {
        THTensor *input_t = THTensor_(newSelect)(input, 0, t);
        THTensor *output_t = THTensor_(newSelect)(output, 0, t);
        THTensor *finput_t = THTensor_(newSelect)(finput, 0, t);

        THNN_(VolumetricConvolutionMM_updateOutput_frame)(
          input_t, output_t, weight, bias, finput_t,
          kT, kW, kH,
          dT, dW, dH,
          pT, pW, pH,
          nInputPlane, inputDepth, inputWidth, inputHeight,
          nOutputPlane, outputDepth, outputWidth, outputHeight
        );

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

// Kernel for fast unfold+copy
// Borrowed from Theano
// Authors: Arjun Jain, Frédéric Bastien, Jan Schlüter, Nicolas Ballas

static void THNN_(unfolded_acc_vol)(
          THTensor *finput,
          THTensor *input,
          int kT,
          int kW,
          int kH,
          int dT,
          int dW,
          int dH,
          int pT,
          int pW,
          int pH,
          int64_t nInputPlane,
          int64_t inputDepth,
          int64_t inputWidth,
          int64_t inputHeight,
          int64_t outputDepth,
          int64_t outputWidth,
          int64_t outputHeight)
{
  scalar_t *input_data = input->data<scalar_t>();
  scalar_t *finput_data = finput->data<scalar_t>();

  int64_t n = nInputPlane * inputHeight * inputWidth * inputDepth;
  at::parallel_for(0, n, 0, [&](int64_t start, int64_t end) {
    int64_t line_index_offset = start;
    int64_t line_seg_len = (end - start);

    int64_t w = line_index_offset % inputWidth + pW;
    int64_t h_index = line_index_offset / inputWidth;
    int64_t h = h_index % inputHeight + pH;
    int64_t d_index = h_index / inputHeight;
    int64_t d = d_index % inputDepth + pT;
    int64_t c = d_index / inputDepth;

    int64_t outputHW = outputHeight * outputWidth;
    int64_t outputDHW = outputDepth * outputHW;
    int64_t kHkW = kH*kW;
    int64_t kTkHkW = kT*kHkW;

    int64_t coeff_d_col = outputHW - dT * kHkW * outputDHW;
    int64_t coeff_h_col = outputWidth - dH * kW * outputDHW;
    int64_t coeff_w_col = (1 - dW * outputDHW);

    int64_t count = 0;
    while (count < line_seg_len) {
      // compute the start and end of the output
      int64_t w_col_start = (w < kW) ? 0 : (w - kW) / dW + 1;
      int64_t w_col_tmp = w / dW + 1;
      int64_t w_col_end = w_col_tmp < outputWidth? w_col_tmp : outputWidth;

      int64_t h_col_start = (h < kH) ? 0 : (h - kH) / dH + 1;
      int64_t h_col_tmp = h / dH + 1;
      int64_t h_col_end = h_col_tmp < outputHeight? h_col_tmp : outputHeight;

      int64_t d_col_start = (d < kT) ? 0 : (d - kT) / dT + 1;
      int64_t d_col_tmp = d / dT + 1;
      int64_t d_col_end = d_col_tmp < outputDepth? d_col_tmp : outputDepth;

      scalar_t val = 0;
      int64_t offset = (c * kTkHkW + d * kHkW + h * kW + w) * outputDHW;

      int64_t offset_w_col_start = w_col_start * coeff_w_col;
      int64_t offset_d_col_start = d_col_start * coeff_d_col;
      int64_t offset_h_col_start = h_col_start * coeff_h_col;
      int64_t offset_w_col = offset_w_col_start + offset;
      int64_t offset_d_col;
      int64_t offset_h_col;
      int64_t w_col, d_col, h_col;
      for (w_col = w_col_start; w_col < w_col_end; ++w_col) {
        offset_d_col = offset_d_col_start + offset_w_col;
        for (d_col = d_col_start; d_col < d_col_end; ++d_col) {
          offset_h_col = offset_h_col_start + offset_d_col;
          for (h_col = h_col_start; h_col < h_col_end; ++h_col) {
            val += finput_data[offset_h_col];
            offset_h_col += coeff_h_col;
          }
          offset_d_col += coeff_d_col;
        }
        offset_w_col += coeff_w_col;
      }

      input_data[line_index_offset+count] = val;
      count++;

      if (count < line_seg_len) {
        if (w - pW + 1 == inputWidth) {
          w = pW;
          if (h - pH + 1 == inputHeight) {
            h = pH;
            if (d - pT + 1 == inputDepth) {
              d = pT;
              c++;
            }
            else d++;
          }
          else h++;
        }
        else w++;
      }
    }
  });
}


static void THNN_(VolumetricConvolutionMM_updateGradInput_frame)(
          THTensor *gradInput,
          THTensor *gradOutput,
          THTensor *weight,
          THTensor *fgradInput,
          int kT,
          int kW,
          int kH,
          int dT,
          int dW,
          int dH,
          int pT,
          int pW,
          int pH)
{
  THTensor *gradOutput2d = THTensor_(newWithStorage2d)(
    THTensor_getStoragePtr(gradOutput), gradOutput->storage_offset(),
    gradOutput->size(0), -1,
    gradOutput->size(1)*gradOutput->size(2)*gradOutput->size(3), -1
  );

  THTensor_(addmm)(fgradInput, fgradInput, weight, gradOutput2d, 0, 1);
  c10::raw::intrusive_ptr::decref(gradOutput2d);

  THTensor_(zero)(gradInput);

  THNN_(unfolded_acc_vol)(
    fgradInput, gradInput,
    kT, kW, kH,
    dT, dW, dH,
    pT, pW, pH,
    gradInput->size(0), gradInput->size(1), gradInput->size(3), gradInput->size(2),
    gradOutput->size(1), gradOutput->size(3), gradOutput->size(2)
  );
}

void THNN_(VolumetricConvolutionMM_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *finput,
          THTensor *fgradInput,
          int kT,
          int kW,
          int kH,
          int dT,
          int dW,
          int dH,
          int pT,
          int pW,
          int pH)
{
  THNN_(VolumetricConvolutionMM_shapeCheck)(
        state, input, gradOutput, weight, NULL,
        kT, kW, kH, dT, dW, dH, pT, pW, pH, 0);
  input = THTensor_(newContiguous)(input);
  gradOutput = THTensor_(newContiguous)(gradOutput);

  weight = THNN_(newViewWeight)(weight);

  THTensor_(resizeAs)(gradInput, input);
  THTensor_(resizeAs)(fgradInput, finput);
  // depending on the BLAS library, fgradInput (result tensor) might
  // be left uninitialized on zero alpha, which might lead to weird behavior
  // hence, to be safe, zero it
  THTensor_(zero)(fgradInput);
  THTensor *tweight = THTensor_(new)();
  THTensor_(transpose)(tweight, weight, 0, 1);

  if (input->dim() == 4)
  {
    THNN_(VolumetricConvolutionMM_updateGradInput_frame)(
      gradInput, gradOutput, tweight, fgradInput,
      kT, kW, kH,
      dT, dW, dH,
      pT, pW, pH
    );
  }
  else
  {
    int64_t T = input->size(0);
    at::parallel_for(0, T, CONV3D_OMP_THRESHOLD, [&](int64_t start, int64_t end) {
      for (auto t = start; t < end; t++)
      {
        THTensor *gradInput_t = THTensor_(newSelect)(gradInput, 0, t);
        THTensor *gradOutput_t = THTensor_(newSelect)(gradOutput, 0, t);
        THTensor *fgradInput_t = THTensor_(newSelect)(fgradInput, 0, t);

        THNN_(VolumetricConvolutionMM_updateGradInput_frame)(
          gradInput_t, gradOutput_t, tweight, fgradInput_t,
          kT, kW, kH,
          dT, dW, dH,
          pT, pW, pH
        );

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

static void THNN_(VolumetricConvolutionMM_accGradParameters_frame)(
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *finput,  // can be NULL if gradWeight = NULL
          scalar_t scale)
{
  int64_t i;
  THTensor *gradOutput2d = THTensor_(newWithStorage2d)(
    THTensor_getStoragePtr(gradOutput), gradOutput->storage_offset(),
    gradOutput->size(0), -1,
    gradOutput->size(1)*gradOutput->size(2)*gradOutput->size(3), -1
  );

  if (gradWeight){
    THTensor *tfinput = THTensor_(new)();
    THTensor_(transpose)(tfinput, finput, 0, 1);
    THTensor_(addmm)(gradWeight, gradWeight, gradOutput2d, tfinput, 1, scale);
    c10::raw::intrusive_ptr::decref(tfinput);
  }

  if (gradBias) {
    for (i = 0; i < THTensor_sizeLegacyNoScalars(gradBias, 0); i++)
    {
      int64_t k;
      scalar_t sum = 0;
      scalar_t *data = THStorage_(data)(THTensor_getStoragePtr(gradOutput2d)) + gradOutput2d->storage_offset() + i*gradOutput2d->stride(0);
      for (k = 0; k < gradOutput2d->size(1); k++)
        sum += data[k];

      (THStorage_(data)(THTensor_getStoragePtr(gradBias)) + gradBias->storage_offset())[i] += scale * sum;
    }
  }

  c10::raw::intrusive_ptr::decref(gradOutput2d);
}

void THNN_(VolumetricConvolutionMM_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *finput,
          THTensor *fgradInput,
          int kT, int kW, int kH,
          int dT, int dW, int dH,
          int pT, int pW, int pH,
          accreal scale_)
{
  scalar_t scale = TH_CONVERT_ACCREAL_TO_REAL(scale_);

  THNN_(VolumetricConvolutionMM_shapeCheck)(
        state, input, gradOutput, gradWeight, gradBias,
        kT, kW, kH, dT, dW, dH, pT, pW, pH, 1);
  input = THTensor_(newContiguous)(input);
  gradOutput = THTensor_(newContiguous)(gradOutput);

  if (gradWeight) {
    gradWeight = THNN_(newViewWeight)(gradWeight);
  }

  if (input->dim() == 4)   // non-batch mode
  {
    THNN_(VolumetricConvolutionMM_accGradParameters_frame)(gradOutput, gradWeight, gradBias, finput, scale);
  }
  else  // batch mode
  {
    int64_t T = input->size(0);

    at::parallel_for(0, T, CONV3D_OMP_THRESHOLD, [&](int64_t start, int64_t end) {
      for (auto t = start; t < end; t++)
      {
        THTensor *gradOutput_t = THTensor_(newSelect)(gradOutput, 0, t);
        THTensor *finput_t = NULL;
        if (gradWeight) {
          finput_t = THTensor_(newSelect)(finput, 0, t);
        }

        THNN_(VolumetricConvolutionMM_accGradParameters_frame)(gradOutput_t, gradWeight, gradBias, finput_t, scale);

        c10::raw::intrusive_ptr::decref(gradOutput_t);
        if (gradWeight) {
          c10::raw::intrusive_ptr::decref(finput_t);
        }
      }
    });
  }

  c10::raw::intrusive_ptr::decref(input);
  c10::raw::intrusive_ptr::decref(gradOutput);
  if (gradWeight) {
    c10::raw::intrusive_ptr::decref(gradWeight);
  }
}

#endif
#endif
