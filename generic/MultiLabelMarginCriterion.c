#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/MultiLabelMarginCriterion.c"
#else

static int nn_(MultiLabelMarginCriterion_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);  
  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  real *input_data, *target_data;
  long nframe, dim;
  long t, d, dt, ddt;
  THTensor *target;
  real sum;

  THArgCheck((input->nDimension == 1) || (input->nDimension == 2), 2, "vector or matrix expected");

  if(input->nDimension == 1)
  {
    nframe = 1;
    dim = input->size[0]; 
    target = luaT_checkudata(L, 3, torch_Tensor);
    THArgCheck((target->nDimension == 1) && (target->size[0] == dim), 3, "inconsistent target size");
  }
  else
  {
    nframe = input->size[0];
    dim = input->size[1];
    target = luaT_checkudata(L, 3, torch_Tensor);
    THArgCheck((target->nDimension == 2) && (target->size[0] == nframe) && (target->size[1] == dim), 3, "inconsistent target size");
  }

  THArgCheck(THTensor_(minall)(target) >= 0, 3, "target out of range");
  THArgCheck(THTensor_(maxall)(target) <= dim, 3, "target out of range");

  target = THTensor_(newContiguous)(target);
  input = THTensor_(newContiguous)(input);
  input_data = THTensor_(data)(input);
  target_data = THTensor_(data)(target);

  sum = 0;
  for(t = 0; t < nframe; t++)
  {
    for(dt = 0; dt < dim; dt++)
    {
      long target_idx = (long)target_data[dt]-1;
      real input_target;
      if(target_idx < 0)
        break;
      
      input_target = input_data[target_idx];
      for(d = 0; d < dim; d++)
      {
        int istarget = 0;
        for(ddt = 0; ddt < dim; ddt++)
        {
          if(!target_data[ddt])
            break;
          if(((long)target_data[ddt])-1 == d)
            istarget = 1;
        }
        
        if(!istarget)
        {
          real z = 1 - input_target + input_data[d];
          if(z > 0)
            sum += z;
        }
      }
    }
    input_data += dim;
    target_data += dim;
  }

  if(sizeAverage)
    sum /= dim;

  lua_pushnumber(L, sum);
  lua_setfield(L, 1, "output");

  THTensor_(free)(input);
  THTensor_(free)(target);
  lua_pushnumber(L, sum);
  return 1;
}

static int nn_(MultiLabelMarginCriterion_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
  real *input_data;
  real *gradInput_data;
  real *target_data;
  long nframe, dim;
  long t, d, dt, ddt;
  THTensor *target;
  real g;

  THArgCheck((input->nDimension == 1) || (input->nDimension == 2), 2, "vector or matrix expected");

  if(input->nDimension == 1)
  {
    nframe = 1;
    dim = input->size[0]; 
    target = luaT_checkudata(L, 3, torch_Tensor);
    THArgCheck((target->nDimension == 1) && (target->size[0] == dim), 3, "inconsistent target size");
  }
  else
  {
    nframe = input->size[0];
    dim = input->size[1];
    target = luaT_checkudata(L, 3, torch_Tensor);
    THArgCheck((target->nDimension == 2) && (target->size[0] == nframe) && (target->size[1] == dim), 3, "inconsistent target size");
  }

  THArgCheck(THTensor_(minall)(target) >= 0, 3, "target out of range");
  THArgCheck(THTensor_(maxall)(target) <= dim, 3, "target out of range");

  target = THTensor_(newContiguous)(target);
  input = THTensor_(newContiguous)(input);
  input_data = THTensor_(data)(input);
  target_data = THTensor_(data)(target);

  g = (sizeAverage ? 1./((real)dim) : 1.);

  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);
  gradInput_data = THTensor_(data)(gradInput);

  for(t = 0; t < nframe; t++)
  {
    for(dt = 0; dt < dim; dt++)
    {
      long target_idx = (long)target_data[dt]-1;
      real input_target;
      if(target_idx < 0)
        break;
      
      input_target = input_data[target_idx];
      for(d = 0; d < dim; d++)
      {
        int istarget = 0;
        for(ddt = 0; ddt < dim; ddt++)
        {
          if(!target_data[ddt])
            break;
          if(((long)target_data[ddt])-1 == d)
            istarget = 1;
        }
        
        if(!istarget)
        {
          real z = 1 - input_target + input_data[d];
          if(z > 0)
          {
            gradInput_data[target_idx] -= g;
            gradInput_data[d] += g;
          }
        }
      }
    }
    input_data += dim;
    target_data += dim;
    gradInput_data += dim;
  }

  THTensor_(free)(input);  
  THTensor_(free)(target);
  return 1;
}

static const struct luaL_Reg nn_(MultiLabelMarginCriterion__) [] = {
  {"MultiLabelMarginCriterion_updateOutput", nn_(MultiLabelMarginCriterion_updateOutput)},
  {"MultiLabelMarginCriterion_updateGradInput", nn_(MultiLabelMarginCriterion_updateGradInput)},
  {NULL, NULL}
};

static void nn_(MultiLabelMarginCriterion_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(MultiLabelMarginCriterion__), "nn");
  lua_pop(L,1);
}

#endif
