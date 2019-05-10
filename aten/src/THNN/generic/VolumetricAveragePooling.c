#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "THNN/generic/VolumetricAveragePooling.c"
#else

#include <THNN/generic/pooling_shape.h>
#include <algorithm>

#include <ATen/Parallel.h>

static inline void THNN_(VolumetricAveragePooling_shapeCheck)(
                         THNNState *state,
                         THTensor *input,
                         THTensor *gradOutput,
                         int kT,
                         int kW,
                         int kH,
                         int dT,
                         int dW,
                         int dH,
                         int padT,
                         int padW,
                         int padH,
                         bool ceil_mode)
{
  int64_t nslices;
  int64_t itime;
  int64_t iheight;
  int64_t iwidth;
  int64_t otime;
  int64_t oheight;
  int64_t owidth;
  int ndim = input->dim();
  int dimN = 0;
  int dimt = 1;
  int dimh = 2;
  int dimw = 3;

  if (input->dim() == 5)
  {
    dimN++;
    dimt++;
    dimh++;
    dimw++;
  }

  THArgCheck(kT > 0 && kW > 0 && kH > 0, 5,
             "kernel size should be greater than zero, but got kT: %d kH: %d kW: %d",
             kT, kH, kW);
  THArgCheck(dT > 0 && dW > 0 && dH > 0, 8,
             "stride should be greater than zero, but got dT: %d dH: %d dW: %d",
             dT, dH, dW);
  THNN_ARGCHECK(!input->is_empty() && (input->dim() == 4 || input->dim() == 5), 2, input,
                "non-empty 4D or 5D (batch mode) tensor expected for input, but got: %s");

  THArgCheck(input->size(dimw) >= kW && input->size(dimh) >= kH
             && input->size(dimt) >= kT, 2,
             "input image (T: %d H: %d W: %d) smaller than "
             "kernel size (kT: %d kH: %d kW: %d)",
             input->size(dimt), input->size(dimh), input->size(dimw),
             kT, kH, kW);

  // The second argument is argNumber... here is the index of padH.
  THArgCheck(kT/2 >= padT && kW/2 >= padW && kH/2 >= padH, 11,
            "pad should not be greater than half of kernel size, but got "
            "padT = %d, padW = %d, padH = %d, kT = %d, kW = %d, kH = %d",
            padT, padW, padH, kT, kW, kH);

  /* sizes */
  nslices = input->size(dimN);
  itime   = input->size(dimt);
  iheight = input->size(dimh);
  iwidth  = input->size(dimw);

  otime = pooling_output_shape<int64_t>(itime, kT, padT, dT, 1, ceil_mode);
  oheight = pooling_output_shape<int64_t>(iheight, kH, padH, dH, 1, ceil_mode);
  owidth = pooling_output_shape<int64_t>(iwidth, kW, padW, dW, 1, ceil_mode);

  if (otime < 1 || owidth < 1 || oheight < 1)
    THError("Given input size: (%dx%dx%dx%d). "
            "Calculated output size: (%dx%dx%dx%d). Output size is too small",
            nslices,itime,iheight,iwidth,nslices,otime,oheight,owidth);

  if (gradOutput != NULL) {
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimN, nslices);
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimt, otime);
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimh, oheight);
    THNN_CHECK_DIM_SIZE(gradOutput, ndim, dimw, owidth);
  }
}

static void THNN_(VolumetricAveragePooling_updateOutput_frame)(
          scalar_t *input_p,
          scalar_t *output_p,
          int64_t nslices,
          int64_t itime,
          int64_t iwidth,
          int64_t iheight,
          int64_t otime,
          int64_t owidth,
          int64_t oheight,
          int kT,
          int kW,
          int kH,
          int dT,
          int dW,
          int dH,
          int padT,
          int padW,
          int padH,
          bool count_include_pad)
{
  at::parallel_for(0, nslices, 0, [&](int64_t start, int64_t end) {
    for (auto k = start; k < end; k++)
    {
      int64_t i, j, ti;

      /* local pointers. */
      scalar_t *ip = input_p + k * itime * iwidth * iheight;
      scalar_t *op = output_p + k * otime * owidth * oheight;
      for (i = 0; i < otime * oheight * owidth; ++i)
        *(op + i) = 0;

      /* loop over output */
      for (ti = 0; ti < otime; ti++)
      {
        for (i = 0; i < oheight; i++)
        {
          for (j = 0; j < owidth; j++)
          {
            /* compute pool range. */
            int64_t tstart = ti * dT - padT;
            int64_t hstart = i  * dH - padH;
            int64_t wstart = j  * dW - padW;
            int64_t tend = std::min(tstart + kT, itime + padT);
            int64_t hend = std::min(hstart + kH, iheight + padH);
            int64_t wend = std::min(wstart + kW, iwidth + padW);
            int64_t pool_size = (tend - tstart) * (hend - hstart) * (wend - wstart);
            tstart = std::max(tstart, (int64_t) 0);
            hstart = std::max(hstart, (int64_t) 0);
            wstart = std::max(wstart, (int64_t) 0);
            tend = std::min(tend, itime);
            hend = std::min(hend, iheight);
            wend = std::min(wend, iwidth);

            int divide_factor;
            if (count_include_pad)
              divide_factor = pool_size;
            else
              divide_factor = (tend - tstart) * (hend - hstart) * (wend - wstart);

            /* compute local sum: */
            scalar_t sum = 0.0;
            int64_t x, y, z;

            for (z = tstart; z < tend; z++)
            {
              for (y = hstart; y < hend; y++)
              {
                for (x = wstart; x < wend; x++)
                {
                  sum +=  *(ip + z * iwidth * iheight + y * iwidth + x);
                }
              }
            }

            /* set output to local max */
            *op++ += sum / divide_factor;
          }
        }
      }
    }
  });
}

void THNN_(VolumetricAveragePooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int kT,
          int kW,
          int kH,
          int dT,
          int dW,
          int dH,
          int padT,
          int padW,
          int padH,
          bool ceil_mode,
          bool count_include_pad)
{
  int64_t nslices;
  int64_t itime;
  int64_t iheight;
  int64_t iwidth;
  int64_t otime;
  int64_t oheight;
  int64_t owidth;
  scalar_t *input_data;
  scalar_t *output_data;

  THNN_(VolumetricAveragePooling_shapeCheck)(
        state, input, NULL, kT, kW, kH,
        dT, dW, dH, padT, padW, padH, ceil_mode);

  int dimN = 0;
  int dimt = 1;
  int dimh = 2;
  int dimw = 3;

  if (input->dim() == 5)
  {
    dimN++;
    dimt++;
    dimh++;
    dimw++;
  }

  /* sizes */
  nslices = input->size(dimN);
  itime   = input->size(dimt);
  iheight = input->size(dimh);
  iwidth  = input->size(dimw);
  otime = pooling_output_shape<int64_t>(itime, kT, padT, dT, 1, ceil_mode);
  oheight = pooling_output_shape<int64_t>(iheight, kH, padH, dH, 1, ceil_mode);
  owidth = pooling_output_shape<int64_t>(iwidth, kW, padW, dW, 1, ceil_mode);

  /* get contiguous input */
  input = THTensor_(newContiguous)(input);

  if (input->dim() == 4) /* non-batch mode */
  {
    /* resize output */
    THTensor_(resize4d)(output, nslices, otime, oheight, owidth);

    input_data = input->data<scalar_t>();
    output_data = output->data<scalar_t>();

    THNN_(VolumetricAveragePooling_updateOutput_frame)(
      input_data, output_data, nslices,
      itime, iwidth, iheight,
      otime, owidth, oheight,
      kT, kW, kH,
      dT, dW, dH,
      padT, padW, padH,
      count_include_pad
    );
  }
  else  /* batch mode */
  {
    int64_t nBatch = input->size(0);

    int64_t istride = nslices * itime * iwidth * iheight;
    int64_t ostride = nslices * otime * owidth * oheight;

    /* resize output */
    THTensor_(resize5d)(output, nBatch, nslices, otime, oheight, owidth);

    input_data = input->data<scalar_t>();
    output_data = output->data<scalar_t>();

    at::parallel_for(0, nBatch, 0, [&](int64_t start, int64_t end) {
      for (auto p = start; p < end; p++)
      {
        THNN_(VolumetricAveragePooling_updateOutput_frame)(
          input_data + p * istride, output_data + p * ostride, nslices,
          itime, iwidth, iheight,
          otime, owidth, oheight,
          kT, kW, kH,
          dT, dW, dH,
          padT, padW, padH,
          count_include_pad
        );
      }
    });
  }

  /* cleanup */
  c10::raw::intrusive_ptr::decref(input);
}

static void THNN_(VolumetricAveragePooling_updateGradInput_frame)(
          scalar_t *gradInput_p,
          scalar_t *gradOutput_p,
          int64_t nslices,
          int64_t itime,
          int64_t iwidth,
          int64_t iheight,
          int64_t otime,
          int64_t owidth,
          int64_t oheight,
          int kT,
          int kW,
          int kH,
          int dT,
          int dW,
          int dH,
          int padT,
          int padW,
          int padH,
          bool count_include_pad)
{
  at::parallel_for(0, nslices, 0, [&](int64_t start, int64_t end) {
    for (auto k = start; k < end; k++)
    {
      int64_t i, j, ti;

      /* local pointers */
      scalar_t *ip = gradInput_p + k * itime * iwidth * iheight;
      scalar_t *op = gradOutput_p + k * otime * owidth * oheight;
      for (i = 0; i < itime*iwidth*iheight; i++)
        *(ip + i) = 0;

      /* loop over output */
      for (ti = 0; ti < otime; ti++)
      {
        for (i = 0; i < oheight; i++)
        {
          for (j = 0; j < owidth; j++)
          {
            int64_t tstart = ti * dT - padT;
            int64_t hstart = i  * dH - padH;
            int64_t wstart = j  * dW - padW;
            int64_t tend = std::min(tstart + kT, itime + padT);
            int64_t hend = std::min(hstart + kH, iheight + padH);
            int64_t wend = std::min(wstart + kW, iwidth + padW);
            int64_t pool_size = (tend -tstart) * (hend - hstart) * (wend - wstart);
            tstart = std::max(tstart, (int64_t) 0);
            hstart = std::max(hstart, (int64_t) 0);
            wstart = std::max(wstart, (int64_t) 0);
            tend = std::min(tend, itime);
            hend = std::min(hend, iheight);
            wend = std::min(wend, iwidth);

            int64_t divide_factor;
            if (count_include_pad)
              divide_factor = pool_size;
            else
              divide_factor = (tend - tstart) * (hend - hstart) * (wend - wstart);

            /* scatter gradients out to footprint: */
            scalar_t val  = *op++;

            int64_t x,y,z;
            for (z = tstart; z < tend; z++)
            {
              for (y = hstart; y < hend; y++)
              {
                for (x = wstart; x < wend; x++)
                {
                  *(ip + z * iheight * iwidth + y * iwidth + x) += val / divide_factor;
                }
              }
            }
          }
        }
      }
    }
  });
}

void THNN_(VolumetricAveragePooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          int kT,
          int kW,
          int kH,
          int dT,
          int dW,
          int dH,
          int padT,
          int padW,
          int padH,
          bool ceil_mode,
          bool count_include_pad)
{
  int64_t nslices;
  int64_t itime;
  int64_t iheight;
  int64_t iwidth;
  int64_t otime;
  int64_t oheight;
  int64_t owidth;
  scalar_t *gradInput_data;
  scalar_t *gradOutput_data;

  int dimN = 0;
  int dimt = 1;
  int dimh = 2;
  int dimw = 3;

  THNN_(VolumetricAveragePooling_shapeCheck)(
        state, input, gradOutput, kT, kW, kH,
        dT, dW, dH, padT, padW, padH, ceil_mode);

  /* get contiguous gradOutput */
  gradOutput = THTensor_(newContiguous)(gradOutput);

  /* resize */
  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);

  if (input->dim() == 5)
  {
    dimN++;
    dimt++;
    dimh++;
    dimw++;
  }

  /* sizes */
  nslices = input->size(dimN);
  itime = input->size(dimt);
  iheight = input->size(dimh);
  iwidth = input->size(dimw);
  otime = gradOutput->size(dimt);
  oheight = gradOutput->size(dimh);
  owidth = gradOutput->size(dimw);

  /* get raw pointers */
  gradInput_data = gradInput->data<scalar_t>();
  gradOutput_data = gradOutput->data<scalar_t>();

  /* backprop */
  if (input->dim() == 4) /* non-batch mode*/
  {
    THNN_(VolumetricAveragePooling_updateGradInput_frame)(
      gradInput_data, gradOutput_data, nslices,
      itime, iwidth, iheight,
      otime, owidth, oheight,
      kT, kW, kH,
      dT, dW, dH,
      padT, padW, padH,
      count_include_pad
    );
  }
  else /* batch mode */
  {
    int64_t nBatch = input->size(0);

    int64_t istride = nslices * itime * iwidth * iheight;
    int64_t ostride = nslices * otime * owidth * oheight;

    at::parallel_for(0, nBatch, 0, [&](int64_t start, int64_t end) {
      for (auto p = start; p < end; p++)
      {
        THNN_(VolumetricAveragePooling_updateGradInput_frame)(
          gradInput_data  + p * istride, gradOutput_data + p * ostride, nslices,
          itime, iwidth, iheight,
          otime, owidth, oheight,
          kT, kW, kH,
          dT, dW, dH,
          padT, padW, padH,
          count_include_pad
        );
      }
    });
  }

  /* cleanup */
  c10::raw::intrusive_ptr::decref(gradOutput);
}

#endif
