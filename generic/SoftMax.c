#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SoftMax.c"
#else

static int nn_(SoftMax_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  real *input_data, *output_data;
  long nframe = 0, dim = 0, stride = 0;
  long t;

  if(input->nDimension == 1)
  {
    nframe = 1;
    dim = input->size[0];
    stride = 1;
  }
  else if(input->nDimension == 2)
  {
    nframe = input->size[0];
    dim = input->size[1];
    stride = 1;
  }
  else if(input->nDimension == 3)
  {
    nframe = 1;
    dim = input->size[0];
    stride = input->size[1]*input->size[2];
  }
  else if(input->nDimension == 4)
  {
    nframe = input->size[0];
    dim = input->size[1];
    stride = input->size[2]*input->size[3];
  }
  else
    THArgCheck(0, 2, "1D, 2D, 3D or 4D tensor expected");

  input = THTensor_(newContiguous)(input);
  THTensor_(resizeAs)(output, input);

  input_data = THTensor_(data)(input);
  output_data = THTensor_(data)(output);

#pragma omp parallel for private(t)
  for(t = 0; t < stride*nframe; t++)
  {
    real *input_ptr = input_data + (t/stride)*dim*stride + t % stride;
    real *output_ptr = output_data + (t/stride)*dim*stride + t % stride;

    real inputMax = -THInf;
    accreal sum;

    long d;
    for(d = 0; d < dim; d++) {
      if (input_ptr[d*stride] >= inputMax) inputMax = input_ptr[d*stride];
    }

    sum = 0;
    for(d = 0; d < dim; d++) {
      real z = THExpMinusApprox(inputMax - input_ptr[d*stride]);
      output_ptr[d*stride] = z;
      sum += z;
    }

    for(d = 0; d < dim; d++) {
      output_ptr[d*stride] *= 1/sum;
    }
  }

  THTensor_(free)(input);

  return 1;
}

static int nn_(SoftMax_updateGradInput)(lua_State *L)
{
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
  real *gradInput_data, *gradOutput_data, *output_data;
  long nframe = 0, dim = 0, stride = 0;
  long t;

  if(output->nDimension == 1)
  {
    nframe = 1;
    dim = output->size[0];
    stride = 1;
  }
  else if(output->nDimension == 2)
  {
    nframe = output->size[0];
    dim = output->size[1];
    stride = 1;
  }
  else if(output->nDimension == 3)
  {
    nframe = 1;
    dim = output->size[0];
    stride = output->size[1]*output->size[2];
  }
  else if(output->nDimension == 4)
  {
    nframe = output->size[0];
    dim = output->size[1];
    stride = output->size[2]*output->size[3];
  }
  else
    THError("1D, 2D, 3D or 4D tensor expected");

  gradOutput = THTensor_(newContiguous)(gradOutput);
  output = THTensor_(newContiguous)(output);

  THTensor_(resizeAs)(gradInput, output);
  gradInput_data = THTensor_(data)(gradInput);
  output_data = THTensor_(data)(output);
  gradOutput_data = THTensor_(data)(gradOutput);

#pragma omp parallel for private(t)
  for(t = 0; t < stride*nframe; t++)
  {
    real *gradInput_ptr = gradInput_data + (t/stride)*dim*stride + t % stride;
    real *output_ptr = output_data + (t/stride)*dim*stride + t % stride;
    real *gradOutput_ptr = gradOutput_data + (t/stride)*dim*stride + t % stride;

    long d;
    accreal sum = 0;
    for(d = 0; d < dim; d++)
      sum += (accreal)gradOutput_ptr[d*stride] * output_ptr[d*stride];

    for(d = 0; d < dim; d++)
      gradInput_ptr[d*stride] = output_ptr[d*stride] * (gradOutput_ptr[d*stride] - sum);
  }

  THTensor_(free)(gradOutput);
  THTensor_(free)(output);

  return 1;
}

static const struct luaL_Reg nn_(SoftMax__) [] = {
  {"SoftMax_updateOutput", nn_(SoftMax_updateOutput)},
  {"SoftMax_updateGradInput", nn_(SoftMax_updateGradInput)},
  {NULL, NULL}
};

static void nn_(SoftMax_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(SoftMax__), "nn");
  lua_pop(L,1);
}

#endif
