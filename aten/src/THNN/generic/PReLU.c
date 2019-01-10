#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/PReLU.c"
#else

void THNN_(PReLU_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight)
{
  THTensor_(resizeAs)(output, input);
  int64_t nOutputPlane = THTensor_(numel)(weight);

  if (nOutputPlane == 1)
  {
    // handle shared parameter case
    real w = *THTensor_(data)(weight);
    TH_TENSOR_APPLY2(real, output, real, input,
      *output_data = (*input_data > 0) ? *input_data : w*(*input_data);
    );
    return;
  }

  input = THTensor_(newContiguous)(input);
  int64_t bs = 1, ks = 1;
  {
    int64_t input_ndim = THTensor_(nDimension)(input);
    if (input->size[input_ndim > 1] != nOutputPlane)
      THError("Wrong number of input planes. Expected %d but got %d.", nOutputPlane, input->size[input_ndim > 1]);

    if (input_ndim > 1) {
        bs = input->size[0];
        for (int d = 2; d < input_ndim; d++) {
            ks *= input->size[d];
        }
    }
  }

  real *output_data = THTensor_(data)(output);
  real *input_data = THTensor_(data)(input);
  real *weight_data = THTensor_(data)(weight);
  THIndex_t i, j, k;
  #pragma omp parallel for private(j,k)
  for (i = 0; i < bs; ++i)
  {
    real* n_input_data = input_data + i*nOutputPlane*ks;
    real* n_output_data = output_data + i*nOutputPlane*ks;
    for (j = 0; j < nOutputPlane; ++j)
    {
      for (k = 0; k < ks; ++k)
        n_output_data[k] = (n_input_data[k] > 0) ? n_input_data[k] : weight_data[j] * n_input_data[k];
      n_input_data += ks;
      n_output_data += ks;
    }
  }
  THTensor_(free)(input);
}

void THNN_(PReLU_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight)
{
  THNN_CHECK_NELEMENT(input, gradOutput);
  THTensor_(resizeAs)(gradInput, input);
  int64_t nOutputPlane = THTensor_(numel)(weight);

  if (nOutputPlane == 1)
  {
    real w = THTensor_(data)(weight)[0];
    TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, input,
       if ((*input_data) > 0)
         *gradInput_data = *gradOutput_data;
       else
         *gradInput_data = w * (*gradOutput_data);
    );
    return;
  }

  input = THTensor_(newContiguous)(input);
  gradOutput = THTensor_(newContiguous)(gradOutput);
  weight = THTensor_(newContiguous)(weight);
  const real *input_data = THTensor_(data)(input);
  const real *gradOutput_data = THTensor_(data)(gradOutput);
  const real *weight_data = THTensor_(data)(weight);
  real *gradInput_data = THTensor_(data)(gradInput);

  int64_t bs = 1, ks = 1;
  {
    int64_t input_ndim = THTensor_(nDimension)(input);
    if (input->size[input_ndim > 1] != nOutputPlane)
      THError("Wrong number of input planes. Expected %d but got %d.", nOutputPlane, input->size[input_ndim > 1]);

    if (input_ndim > 1) {
        bs = input->size[0];
        for (int d = 2; d < input_ndim; d++) {
            ks *= input->size[d];
        }
    }
  }

  THIndex_t i, j, k;
  #pragma omp parallel for private(j,k)
  for (i = 0; i < bs; ++i)
  {
    const real *n_input_data = input_data + i*nOutputPlane*ks;
    const real *n_gradOutput_data = gradOutput_data + i*nOutputPlane*ks;
    real *n_gradInput_data = gradInput_data + i*nOutputPlane*ks;

    for (j = 0; j < nOutputPlane; ++j)
    {
      real w = weight_data[j];
      for (k = 0; k < ks; ++k)
      {
        if (n_input_data[k] > 0)
          n_gradInput_data[k] = n_gradOutput_data[k];
        else
          n_gradInput_data[k] = n_gradOutput_data[k] * w;
      }
      n_input_data += ks;
      n_gradInput_data += ks;
      n_gradOutput_data += ks;
    }
  }
  THTensor_(free)(input);
  THTensor_(free)(gradOutput);
  THTensor_(free)(weight);
}

void THNN_(PReLU_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *gradWeight,
          accreal scale_)
{
  real scale = TH_CONVERT_ACCREAL_TO_REAL(scale_);
  THNN_CHECK_NELEMENT(input, gradOutput);
  int64_t nOutputPlane = THTensor_(numel)(weight);

  if (nOutputPlane == 1)
  {
    real *gradWeight_data = THTensor_(data)(gradWeight);
    real sum = 0;
    TH_TENSOR_APPLY2(real, input, real, gradOutput,
      if ((*input_data) <= 0)
        sum += (*input_data) * (*gradOutput_data);
    );
    gradWeight_data[0] += scale * sum;
    return;
  }

  THArgCheck(THTensor_(isContiguous)(gradWeight), 6, "gradWeight needs to be contiguous");
  input = THTensor_(newContiguous)(input);
  gradOutput = THTensor_(newContiguous)(gradOutput);
  weight = THTensor_(newContiguous)(weight);
  int64_t bs = 1, ks = 1;
  {
    int64_t input_ndim = THTensor_(nDimension)(input);
    if (input->size[input_ndim > 1] != nOutputPlane)
      THError("Wrong number of input planes. Expected %d but got %d.", nOutputPlane, input->size[input_ndim > 1]);

    if (input_ndim > 1) {
        bs = input->size[0];
        for (int d = 2; d < input_ndim; d++) {
          ks *= input->size[d];
        }
    }
  }

  const real *input_data = THTensor_(data)(input);
  const real *gradOutput_data = THTensor_(data)(gradOutput);
  real *gradWeight_data = THTensor_(data)(gradWeight);

  THIndex_t i, j, k;
  for (i = 0; i < bs; ++i)
  {
    const real *n_input_data = input_data + i*nOutputPlane*ks;
    const real *n_gradOutput_data = gradOutput_data + i*nOutputPlane*ks;

    for (j = 0; j < nOutputPlane; ++j)
    {
      real sum = 0;
      for (k = 0; k < ks; ++k)
        if (n_input_data[k] <= 0)
          sum += n_gradOutput_data[k] * n_input_data[k];
      gradWeight_data[j] += scale * sum;
      n_input_data += ks;
      n_gradOutput_data += ks;
    }
  }
  THTensor_(free)(input);
  THTensor_(free)(gradOutput);
  THTensor_(free)(weight);
}

#endif
