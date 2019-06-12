#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialReflectionPadding.c"
#else

static void THNN_(SpatialReflectionPadding_updateOutput_frame)(
  scalar_t *input_p, scalar_t *output_p,
  int64_t nslices,
  int64_t iwidth, int64_t iheight,
  int64_t owidth, int64_t oheight,
  int pad_l, int pad_r,
  int pad_t, int pad_b)
{
  int iStartX = fmax(0, -pad_l);
  int iStartY = fmax(0, -pad_t);
  int oStartX = fmax(0, pad_l);
  int oStartY = fmax(0, pad_t);

  int64_t k, ip_x, ip_y;
#pragma omp parallel for private(k, ip_x, ip_y)

  for (k = 0; k < nslices; k++)
  {
    int64_t i, j;
    for (i = 0; i < oheight; i++) {
      for (j = 0; j < owidth; j++) {
        if (j < pad_l) {
          ip_x = pad_l * 2 - j;
        } else if (j >= pad_l && j < iwidth + pad_l) {
          ip_x = j;
        } else {
          ip_x = (iwidth + pad_l - 1) * 2 - j;
        }
        ip_x = ip_x - oStartX + iStartX;

        if (i < pad_t) {
          ip_y = pad_t * 2 - i;
        } else if (i >= pad_t && i < iheight + pad_t) {
          ip_y = i;
        } else {
          ip_y = (iheight + pad_t - 1) * 2 - i;
        }
        ip_y = ip_y - oStartY + iStartY;

        scalar_t *dest_p = output_p + k*owidth*oheight + i * owidth + j;
        scalar_t *src_p = input_p + k*iwidth*iheight + ip_y * iwidth + ip_x;
        *dest_p = *src_p;
      }
    }
  }
}

void THNN_(SpatialReflectionPadding_updateOutput)(THNNState *state,
                                                  THTensor *input,
                                                  THTensor *output,
                                                  int pad_l, int pad_r,
                                                  int pad_t, int pad_b)
{
  int dimw = 2;
  int dimh = 1;
  int dimslices = 0;
  int64_t nbatch = 1;
  int64_t nslices;
  int64_t iheight;
  int64_t iwidth;
  int64_t oheight;
  int64_t owidth;
  scalar_t *input_data;
  scalar_t *output_data;

  THNN_ARGCHECK(!input->is_empty() && (input->dim() == 3 || input->dim() == 4), 2, input,
		"non-empty 3D or 4D (batch mode) tensor expected for input, but got: %s");

  if (input->dim() == 4)
  {
    nbatch = input->size(0);
    dimw++;
    dimh++;
    dimslices++;
  }

  /* input sizes */
  nslices = input->size(dimslices);
  iheight = input->size(dimh);
  iwidth = input->size(dimw);

  AT_CHECK(pad_l < iwidth && pad_r < iwidth,
           "Argument #4: Padding size should be less than the corresponding input dimension, "
           "but got: padding (", pad_l, ", ", pad_r, ") at dimension ", dimw, " of input ", input->sizes());

  AT_CHECK(pad_t < iheight && pad_b < iheight,
           "Argument #6: Padding size should be less than the corresponding input dimension, "
           "but got: padding (", pad_t, ", ", pad_b, ") at dimension ", dimh, " of input ", input->sizes());

  /* output sizes */
  oheight = iheight + pad_t + pad_b;
  owidth  = iwidth + pad_l + pad_r;

  THArgCheck(owidth >= 1 || oheight >= 1 , 2,
	     "input (H: %d, W: %d)is too small."
	     " Calculated output H: %d W: %d",
	     iheight, iwidth, oheight, owidth);

  /* get contiguous input */
  input = THTensor_(newContiguous)(input);

  /* resize output */
  if (input->dim() == 3)
  {
    THTensor_(resize3d)(output, nslices, oheight, owidth);

    input_data = input->data<scalar_t>();
    output_data = output->data<scalar_t>();

    THNN_(SpatialReflectionPadding_updateOutput_frame)(input_data, output_data,
                                                    nslices,
                                                    iwidth, iheight,
                                                    owidth, oheight,
                                                    pad_l, pad_r,
                                                    pad_t, pad_b);
  }
  else
  {
    int64_t p;

    THTensor_(resize4d)(output, nbatch, nslices, oheight, owidth);

    input_data = input->data<scalar_t>();
    output_data = output->data<scalar_t>();

#pragma omp parallel for private(p)
    for (p = 0; p < nbatch; p++)
    {
      THNN_(SpatialReflectionPadding_updateOutput_frame)(
        input_data+p*nslices*iwidth*iheight,
        output_data+p*nslices*owidth*oheight,
        nslices,
        iwidth, iheight,
        owidth, oheight,
        pad_l, pad_r,
        pad_t, pad_b);
    }
  }

  /* cleanup */
  c10::raw::intrusive_ptr::decref(input);
}

static void THNN_(SpatialReflectionPadding_updateGradInput_frame)(
  scalar_t *ginput_p, scalar_t *goutput_p,
  int64_t nslices,
  int64_t iwidth, int64_t iheight,
  int64_t owidth, int64_t oheight,
  int pad_l, int pad_r,
  int pad_t, int pad_b)
{
  int iStartX = fmax(0, -pad_l);
  int iStartY = fmax(0, -pad_t);
  int oStartX = fmax(0, pad_l);
  int oStartY = fmax(0, pad_t);

  int64_t k, ip_x, ip_y;
#pragma omp parallel for private(k, ip_x, ip_y)

  for (k = 0; k < nslices; k++)
  {
    int64_t i, j;
    for (i = 0; i < oheight; i++) {
      for (j = 0; j < owidth; j++) {
        if (j < pad_l) {
          ip_x = pad_l * 2 - j;
        } else if (j >= pad_l && j < iwidth + pad_l) {
          ip_x = j;
        } else {
          ip_x = (iwidth + pad_l - 1) * 2 - j;
        }
        ip_x = ip_x - oStartX + iStartX;

        if (i < pad_t) {
          ip_y = pad_t * 2 - i;
        } else if (i >= pad_t && i < iheight + pad_t) {
          ip_y = i;
        } else {
          ip_y = (iheight + pad_t - 1) * 2 - i;
        }
        ip_y = ip_y - oStartY + iStartY;

        scalar_t *src_p = goutput_p + k*owidth*oheight + i * owidth + j;
        scalar_t *dest_p = ginput_p + k*iwidth*iheight + ip_y * iwidth + ip_x;
        *dest_p += *src_p;
      }
    }
  }
}

void THNN_(SpatialReflectionPadding_updateGradInput)(THNNState *state,
                                                      THTensor *input,
                                                      THTensor *gradOutput,
                                                      THTensor *gradInput,
                                                      int pad_l, int pad_r,
                                                      int pad_t, int pad_b)
{
  int dimw = 2;
  int dimh = 1;
  int dimslices = 0;
  int64_t nbatch = 1;
  int64_t nslices;
  int64_t iheight;
  int64_t iwidth;
  int64_t oheight;
  int64_t owidth;

  if (input->dim() == 4)
  {
    nbatch = input->size(0);
    dimw++;
    dimh++;
    dimslices++;
  }

  /* sizes */
  nslices = input->size(dimslices);
  iheight = input->size(dimh);
  iwidth = input->size(dimw);
  oheight = iheight + pad_t + pad_b;
  owidth  = iwidth + pad_l + pad_r;

  THArgCheck(owidth == THTensor_(size)(gradOutput, dimw), 3,
	     "gradOutput width unexpected. Expected: %d, Got: %d",
	     owidth, THTensor_(size)(gradOutput, dimw));
  THArgCheck(oheight == THTensor_(size)(gradOutput, dimh), 3,
                "gradOutput height unexpected. Expected: %d, Got: %d",
	     oheight, THTensor_(size)(gradOutput, dimh));

  /* get contiguous gradOutput */
  gradOutput = THTensor_(newContiguous)(gradOutput);

  /* resize */
  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);

  /* backprop */
  if (input->dim() == 3) {
    THNN_(SpatialReflectionPadding_updateGradInput_frame)(
      gradInput->data<scalar_t>(),
      gradOutput->data<scalar_t>(),
      nslices,
      iwidth, iheight,
      owidth, oheight,
      pad_l, pad_r,
      pad_t, pad_b);
  } else {
    int64_t p;
#pragma omp parallel for private(p)
    for (p = 0; p < nbatch; p++) {
      THNN_(SpatialReflectionPadding_updateGradInput_frame)(
        gradInput->data<scalar_t>() + p * nslices * iheight * iwidth,
        gradOutput->data<scalar_t>() + p * nslices * oheight * owidth,
        nslices,
        iwidth, iheight,
        owidth, oheight,
        pad_l, pad_r,
        pad_t, pad_b);
    }
  }

  /* cleanup */
  c10::raw::intrusive_ptr::decref(gradOutput);
}

#endif
