#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SoftMax.c"
#else

void THNN_(SoftMax_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output)
{
  real *input_data, *output_data;
  ptrdiff_t nframe = 0, dim = 0, stride = 0;
  ptrdiff_t t;

  if (input->nDimension == 1)
  {
    nframe = 1;
    dim = input->size[0];
    stride = 1;
  }
  else if (input->nDimension == 2)
  {
    nframe = input->size[0];
    dim = input->size[1];
    stride = 1;
  }
  else if (input->nDimension == 3)
  {
    nframe = 1;
    dim = input->size[0];
    stride = input->size[1]*input->size[2];
  }
  else if (input->nDimension == 4)
  {
    nframe = input->size[0];
    dim = input->size[1];
    stride = input->size[2]*input->size[3];
  }
  else
  {
    THArgCheck(0, 2, "1D, 2D, 3D or 4D tensor expected");
  }

  input = THTensor_(newContiguous)(input);
  THTensor_(resizeAs)(output, input);

  input_data = THTensor_(data)(input);
  output_data = THTensor_(data)(output);

#pragma omp parallel for private(t)
  for (t = 0; t < stride*nframe; t++)
  {
    real *input_ptr = input_data + (t/stride)*dim*stride + t % stride;
    real *output_ptr = output_data + (t/stride)*dim*stride + t % stride;

    real inputMax = -THInf;
    accreal sum;

    ptrdiff_t d;
    for (d = 0; d < dim; d++)
    {
      if (input_ptr[d*stride] >= inputMax) inputMax = input_ptr[d*stride];
    }

    sum = 0;
    for (d = 0; d < dim; d++)
    {
      real z = exp(input_ptr[d*stride] - inputMax);
      output_ptr[d*stride] = z;
      sum += z;
    }

    for (d = 0; d < dim; d++)
    {
      output_ptr[d*stride] *= 1/sum;
    }
  }

  THTensor_(free)(input);
}

void THNN_(SoftMax_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output)
{
  THNN_CHECK_SHAPE(input, gradOutput);  
  real *gradInput_data, *gradOutput_data, *output_data;
  ptrdiff_t nframe = 0, dim = 0, stride = 0;
  ptrdiff_t t;

  if (output->nDimension == 1)
  {
    nframe = 1;
    dim = output->size[0];
    stride = 1;
  }
  else if (output->nDimension == 2)
  {
    nframe = output->size[0];
    dim = output->size[1];
    stride = 1;
  }
  else if (output->nDimension == 3)
  {
    nframe = 1;
    dim = output->size[0];
    stride = output->size[1]*output->size[2];
  }
  else if (output->nDimension == 4)
  {
    nframe = output->size[0];
    dim = output->size[1];
    stride = output->size[2]*output->size[3];
  }
  else
  {
    THError("1D, 2D, 3D or 4D tensor expected");
  }

  gradOutput = THTensor_(newContiguous)(gradOutput);
  output = THTensor_(newContiguous)(output);

  THTensor_(resizeAs)(gradInput, output);
  gradInput_data = THTensor_(data)(gradInput);
  output_data = THTensor_(data)(output);
  gradOutput_data = THTensor_(data)(gradOutput);

#pragma omp parallel for private(t)
  for (t = 0; t < stride*nframe; t++)
  {
    real *gradInput_ptr = gradInput_data + (t/stride)*dim*stride + t % stride;
    real *output_ptr = output_data + (t/stride)*dim*stride + t % stride;
    real *gradOutput_ptr = gradOutput_data + (t/stride)*dim*stride + t % stride;

    ptrdiff_t d;
    accreal sum = 0;
    for (d = 0; d < dim; d++)
      sum += (accreal)gradOutput_ptr[d*stride] * output_ptr[d*stride];

    for (d = 0; d < dim; d++)
      gradInput_ptr[d*stride] = output_ptr[d*stride] * (gradOutput_ptr[d*stride] - sum);
  }

  THTensor_(free)(gradOutput);
  THTensor_(free)(output);
}

#endif
