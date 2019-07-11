#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "THNN/generic/SpatialAveragePooling.c"
#else

#include <THNN/generic/pooling_shape.h>
#include <algorithm>

#include <ATen/Parallel.h>

static inline void THNN_(SpatialAveragePooling_shapeCheck)(
        THTensor *input, THTensor *gradOutput,
        int kH, int kW, int dH, int dW, int padH, int padW,
        bool ceil_mode) {

  THArgCheck(kW > 0 && kH > 0, 5,
             "kernel size should be greater than zero, but got kH: %d kW: %d", kH, kW);
  THArgCheck(dW > 0 && dH > 0, 8,
             "stride should be greater than zero, but got dH: %d dW: %d", dH, dW);

  int ndim = input->dim();
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;

  if (ndim == 4) {
    dimf++;
    dimh++;
    dimw++;
  }

  THNN_ARGCHECK(!input->is_empty() && (ndim == 3 || ndim == 4), 2, input,
                "non-empty 3D or 4D input tensor expected but got: %s");

  THArgCheck(kW/2 >= padW && kH/2 >= padH, 2,
             "pad should be smaller than half of kernel size, but got "
             "padW = %d, padH = %d, kW = %d, kH = %d",
             padW, padH, kW, kH);

  int64_t nInputPlane = input->size(dimh-1);
  int64_t inputHeight = input->size(dimh);
  int64_t inputWidth = input->size(dimw);
  int64_t nOutputPlane = nInputPlane;

  int64_t outputHeight = pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, 1, ceil_mode);
  int64_t outputWidth = pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, 1, ceil_mode);

  if (outputWidth < 1 || outputHeight < 1)
    THError("Given input size: (%dx%dx%d). "
            "Calculated output size: (%dx%dx%d). Output size is too small",
            nInputPlane,inputHeight,inputWidth,nInputPlane,outputHeight,outputWidth);

  if (gradOutput != NULL) {
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimf, nOutputPlane);
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimh, outputHeight);
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimw, outputWidth);
  }
}

void THNN_(SpatialAveragePooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH,
          bool ceil_mode,
          bool count_include_pad)
{
  scalar_t *output_data;
  scalar_t *input_data;

  int dimw = 2;
  int dimh = 1;
  int dimc = 0;
  int64_t nbatch = 1;

  int64_t inputWidth;
  int64_t inputHeight;
  int64_t outputWidth;
  int64_t outputHeight;
  int64_t nInputPlane; // number of channels (or colors)

  THNN_(SpatialAveragePooling_shapeCheck)
    (input, NULL, kH, kW, dH, dW, padH, padW, ceil_mode);

  if (input->dim() == 4) {
    nbatch = input->size(0);
    dimw++;
    dimh++;
    dimc++;
  }

  inputWidth = input->size(dimw);
  inputHeight = input->size(dimh);
  nInputPlane = input->size(dimc);

  outputWidth = pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, 1, ceil_mode);
  outputHeight = pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, 1, ceil_mode);

  if (input->dim() == 3)
    THTensor_(resize3d)(output, nInputPlane, outputHeight, outputWidth);
  else
    THTensor_(resize4d)(output, input->size(0), nInputPlane, outputHeight, outputWidth);

  input = THTensor_(newContiguous)(input);
  THArgCheck(THTensor_(isContiguous)(output), 3, "output must be contiguous");
  input_data = input->data<scalar_t>();
  output_data = output->data<scalar_t>();

  at::parallel_for(0, nInputPlane, 0, [&](int64_t start, int64_t end) {
    for (auto k = start; k < end; k++)
    {
      int64_t p;
      for(p = 0; p < nbatch; p++)
      {
        int64_t xx, yy;
        /* For all output pixels... */
        scalar_t *ptr_output = output_data + p*nInputPlane*outputWidth*outputHeight + k*outputWidth*outputHeight;
        scalar_t *ptr_input = input_data + p*nInputPlane*inputWidth*inputHeight + k*inputWidth*inputHeight;
        int64_t i;
        for(i = 0; i < outputWidth*outputHeight; i++)
          ptr_output[i] = 0;

        for(yy = 0; yy < outputHeight; yy++)
        {
          for(xx = 0; xx < outputWidth; xx++)
          {
            /* Compute the mean of the input image... */
            int64_t hstart = yy * dH - padH;
            int64_t wstart = xx * dW - padW;
            int64_t hend = std::min(hstart + kH, inputHeight + padH);
            int64_t wend = std::min(wstart + kW, inputWidth + padW);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = std::max(hstart, (int64_t) 0);
            wstart = std::max(wstart, (int64_t) 0);
            hend = std::min(hend, inputHeight);
            wend = std::min(wend, inputWidth);

            scalar_t sum = 0;

            int divide_factor;
            if(count_include_pad)
              divide_factor = pool_size;
            else
              divide_factor = (hend - hstart) * (wend - wstart);

            int64_t kx, ky;

            for(ky = hstart; ky < hend; ky++)
            {
              for(kx = wstart; kx < wend; kx++)
                sum += ptr_input[ky*inputWidth + kx];
            }
            /* Update output */
            *ptr_output++ += sum/divide_factor;
          }
        }
      }
    }
  });
  c10::raw::intrusive_ptr::decref(input);
}

void THNN_(SpatialAveragePooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH,
          bool ceil_mode,
          bool count_include_pad)
{
  int dimw = 2;
  int dimh = 1;
  int dimc = 0;
  int64_t nbatch = 1;
  int64_t ndim = 3;

  int64_t inputWidth;
  int64_t inputHeight;
  int64_t outputWidth;
  int64_t outputHeight;
  int64_t nInputPlane; // number of channels (or colors)

  scalar_t *gradOutput_data;
  scalar_t *gradInput_data;

  THNN_(SpatialAveragePooling_shapeCheck)
    (input, gradOutput, kH, kW, dH, dW, padH, padW, ceil_mode);


  if (input->dim() == 4) {
    nbatch = input->size(0);
    dimw++;
    dimh++;
    dimc++;
    ndim = 4;
  }

  inputWidth = input->size(dimw);
  inputHeight = input->size(dimh);
  nInputPlane = input->size(dimc);

  outputWidth = pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, 1, ceil_mode);
  outputHeight = pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, 1, ceil_mode);

  THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimh, outputHeight);
  THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimw, outputWidth);

  THTensor_(resizeAs)(gradInput, input);

  gradOutput = THTensor_(newContiguous)(gradOutput);
  THArgCheck(THTensor_(isContiguous)(gradInput), 4, "gradInput must be contiguous");

  gradInput_data = gradInput->data<scalar_t>();
  gradOutput_data = gradOutput->data<scalar_t>();

  at::parallel_for(0, nInputPlane, 0, [&](int64_t start, int64_t end) {
    for (auto k = start; k < end; k++)
    {
      int64_t p;
      for(p = 0; p < nbatch; p++)
      {
        scalar_t *ptr_gradOutput = gradOutput_data + p*nInputPlane*outputHeight*outputWidth + k*outputWidth*outputHeight;
        int64_t xx, yy;

        scalar_t* ptr_gi = gradInput_data + p*nInputPlane*inputWidth*inputHeight + k*inputWidth*inputHeight;
        scalar_t *ptr_gradInput = gradInput_data + p*nInputPlane*inputWidth*inputHeight + k*inputWidth*inputHeight;

        int64_t i;
        for(i=0; i<inputWidth*inputHeight; i++)
          ptr_gi[i] = 0.0;

        for(yy = 0; yy < outputHeight; yy++)
        {
          for(xx = 0; xx < outputWidth; xx++)
          {
            int64_t hstart = yy * dH - padH;
            int64_t wstart = xx * dW - padW;
            int64_t hend = std::min(hstart + kH, inputHeight + padH);
            int64_t wend = std::min(wstart + kW, inputWidth + padW);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = std::max(hstart, (int64_t) 0);
            wstart = std::max(wstart, (int64_t) 0);
            hend = std::min(hend, inputHeight);
            wend = std::min(wend, inputWidth);

            scalar_t z = *ptr_gradOutput++;

            int divide_factor;
            if(count_include_pad)
              divide_factor = pool_size;
            else
              divide_factor = (hend - hstart) * (wend - wstart);

            int64_t kx, ky;
            for(ky = hstart ; ky < hend; ky++)
            {
              for(kx = wstart; kx < wend; kx++)
                ptr_gradInput[ky*inputWidth + kx] += z/divide_factor;
            }
          }
        }
      }
    }
  });

  c10::raw::intrusive_ptr::decref(gradOutput);
}

#endif
