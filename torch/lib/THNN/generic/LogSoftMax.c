#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/LogSoftMax.c"
#else

void THNN_(LogSoftMax_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output)
{
  real *input_data, *output_data;
  ptrdiff_t nframe = 0, dim = 0, stride = 0;
  ptrdiff_t t, d;

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
    THArgCheck(0, 2, "1D, 2D, 3D or 4D tensor expected");

  input = THTensor_(newContiguous)(input);
  THTensor_(resizeAs)(output, input);

  real *input_data0 = THTensor_(data)(input);
  real *output_data0 = THTensor_(data)(output);

  accreal logsum;
  real maxInput;
  #pragma omp parallel for private(t, d, maxInput, logsum, input_data, output_data)
  for (t = 0; t < stride*nframe; t++)
  {
    logsum = 0;
    maxInput = -THInf;
    input_data = input_data0 + (t/stride)*dim*stride + t % stride;
    output_data = output_data0 + (t/stride)*dim*stride + t % stride;

    for (d = 0; d < dim; d++)
      maxInput = THMax(maxInput, input_data[d*stride]);

    for (d = 0; d < dim; d++)
      logsum += exp(input_data[d*stride] - maxInput);
    logsum = maxInput + log(logsum);

    for (d = 0; d < dim; d++)
      output_data[d*stride] = input_data[d*stride] - logsum;
  }

  THTensor_(free)(input);
}

void THNN_(LogSoftMax_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output)
{
  THNN_CHECK_SHAPE(input, gradOutput);
  gradOutput = THTensor_(newContiguous)(gradOutput);
  real *gradInput_data, *gradOutput_data, *output_data;
  ptrdiff_t nframe = 0, dim = 0, stride = 0;
  ptrdiff_t t, d;

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
    THError("1D, 2D, 3D or 4D tensor expected");

  output = THTensor_(newContiguous)(output);
  gradOutput = THTensor_(newContiguous)(gradOutput);

  THTensor_(resizeAs)(gradInput, output);
  real *gradInput_data0 = THTensor_(data)(gradInput);
  real *output_data0 = THTensor_(data)(output);
  real *gradOutput_data0 = THTensor_(data)(gradOutput);
  accreal sum;
  #pragma omp parallel for private(t, sum, d, gradInput_data, output_data, gradOutput_data)
  for (t = 0; t < stride*nframe; t++)
  {
    sum = 0;
    gradInput_data = gradInput_data0 + (t/stride)*dim*stride + t % stride;
    output_data = output_data0 + (t/stride)*dim*stride + t % stride;
    gradOutput_data = gradOutput_data0 + (t/stride)*dim*stride + t % stride;

    for (d = 0; d < dim; d++)
      sum += gradOutput_data[d*stride];

    for (d = 0; d < dim; d++)
      gradInput_data[d*stride] = gradOutput_data[d*stride] - exp(output_data[d*stride])*sum;
  }

  THTensor_(free)(gradOutput);
  THTensor_(free)(output);
}

#endif
