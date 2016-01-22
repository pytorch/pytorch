#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/MultiMarginCriterion.c"
#else

static int nn_(MultiMarginCriterion_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);  
  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  int p = luaT_getfieldchecknumber(L, 1, "p");
  real *input_data, *target_data;
  long nframe, dim;
  long t, d;
  real target_;
  THTensor *target;
  real sum;

  THArgCheck((input->nDimension == 1) || (input->nDimension == 2), 2, "vector or matrix expected");

  if(input->nDimension == 1)
  {
    nframe = 1;
    dim = input->size[0]; 
    target_ = luaL_checknumber(L, 3);
    target = THTensor_(newWithSize1d)(1);
    THTensor_(fill)(target, target_);
  }
  else
  {
    nframe = input->size[0];
    dim = input->size[1];
    target = luaT_checkudata(L, 3, torch_Tensor);
    THArgCheck((target->nDimension == 1) && (target->size[0] == nframe), 3, "inconsistent target size");
    target = THTensor_(newContiguous)(target);
  }

  for(t = 0; t < nframe; t++)
  {
    real idx = THTensor_(get1d)(target, t);
    THArgCheck((idx >= 1) && (idx <= dim), 3, "target out of range");
  }

  input = THTensor_(newContiguous)(input);
  input_data = THTensor_(data)(input);
  target_data = THTensor_(data)(target);

  sum = 0;
  for(t = 0; t < nframe; t++)
  {
    long target_idx = (long)(target_data[t]-1);
    real input_target = input_data[target_idx];
    for(d = 0; d < dim; d++)
    {
      real z = 1 - input_target + input_data[d];
      if(d == target_idx)
        continue;
    
      if(z > 0)
        sum += (p==1) ? z : z*z;
    }
    input_data += dim;
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

static int nn_(MultiMarginCriterion_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  int p = luaT_getfieldchecknumber(L, 1, "p");
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
  real *input_data;
  real *gradInput_data;
  real *target_data;
  THTensor *target;
  long nframe, dim;
  long t, d;
  real target_;
  real g;

  THArgCheck((input->nDimension == 1) || (input->nDimension == 2), 2, "vector or matrix expected");

  if(input->nDimension == 1)
  {
    nframe = 1;
    dim = input->size[0]; 
    target_ = luaL_checknumber(L, 3);
    target = THTensor_(newWithSize1d)(1);
    THTensor_(fill)(target, target_);
  }
  else
  {
    nframe = input->size[0];
    dim = input->size[1];
    target = luaT_checkudata(L, 3, torch_Tensor);
    THArgCheck((target->nDimension == 1) && (target->size[0] == nframe), 3, "inconsistent target size");
    target = THTensor_(newContiguous)(target);
  }

  g = (sizeAverage ? 1./((real)dim) : 1.);

  input = THTensor_(newContiguous)(input);
  input_data = THTensor_(data)(input);

  THTensor_(resizeAs)(gradInput, input);
  gradInput_data = THTensor_(data)(gradInput);

  target_data = THTensor_(data)(target);
    
  for(t = 0; t < nframe; t++)
  {
    long target_idx = (long)(target_data[t])-1;
    real input_target = input_data[target_idx];
    real gradInput_target = 0;
    for(d = 0; d < dim; d++)
    {
      real z = 1 - input_target + input_data[d];
      if(d == target_idx)
        continue;
    
      if(z > 0)
      {
        real h = (p == 1) ? g : 2*g*z;
        gradInput_target -= h;
        gradInput_data[d] = h;
      }
      else
        gradInput_data[d] = 0;
    }
    gradInput_data[target_idx] = gradInput_target;
    
    input_data += dim;
    gradInput_data += dim;
  }


  THTensor_(free)(input);  
  THTensor_(free)(target);
  return 1;
}

static const struct luaL_Reg nn_(MultiMarginCriterion__) [] = {
  {"MultiMarginCriterion_updateOutput", nn_(MultiMarginCriterion_updateOutput)},
  {"MultiMarginCriterion_updateGradInput", nn_(MultiMarginCriterion_updateGradInput)},
  {NULL, NULL}
};

static void nn_(MultiMarginCriterion_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(MultiMarginCriterion__), "nn");
  lua_pop(L,1);
}

#endif
