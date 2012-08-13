#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/TensorOperator.c"
#else

static int torch_TensorOperator_(__add__)(lua_State *L)
{
  THTensor *tensor1 = luaT_toudata(L, 1, torch_Tensor);
  THTensor *tensor2 = luaT_toudata(L, 2, torch_Tensor);
  THTensor *r;

  if(!tensor1 && !tensor2)
    luaL_error(L, "expecting two Tensors or one Tensor and one number");
  else
  {
    r = THTensor_(new)();
    luaT_pushudata(L, r, torch_Tensor);
    
    if(!tensor1 && tensor2)
    {
      THTensor_(resizeAs)(r, tensor2);
      THTensor_(copy)(r, tensor2);
      THTensor_(add)(r, r, luaL_checknumber(L, 1));
    }
    else if(tensor1 && !tensor2)
    {
      THTensor_(resizeAs)(r, tensor1);
      THTensor_(copy)(r, tensor1);
      THTensor_(add)(r, r, luaL_checknumber(L, 2));
    }
    else
    {
      THTensor_(resizeAs)(r, tensor1);
      THTensor_(copy)(r, tensor1);
      THTensor_(cadd)(r, r, 1, tensor2);
    }
  }
  return 1;
}

static int torch_TensorOperator_(__sub__)(lua_State *L)
{
  THTensor *tensor1 = luaT_toudata(L, 1, torch_Tensor);
  THTensor *tensor2 = luaT_toudata(L, 2, torch_Tensor);
  THTensor *r;

  if(!tensor1 && !tensor2)
    luaL_error(L, "expecting two Tensors or one Tensor and one number");
  else
  {
    r = THTensor_(new)();
    luaT_pushudata(L, r, torch_Tensor);
    
    if(!tensor1 && tensor2)
    {
      THTensor_(resizeAs)(r, tensor2);
      THTensor_(fill)(r, luaL_checknumber(L, 1));
      THTensor_(cadd)(r, r, -1, tensor2);
    }
    else if(tensor1 && !tensor2)
    {
      THTensor_(resizeAs)(r, tensor1);
      THTensor_(copy)(r, tensor1);
      THTensor_(add)(r, r, -luaL_checknumber(L, 2));
    }
    else
    {
      THTensor_(resizeAs)(r, tensor1);
      THTensor_(copy)(r, tensor1);
      THTensor_(cadd)(r, r, -1, tensor2);
    }
  }
  return 1;
}

static int torch_TensorOperator_(__unm__)(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  THTensor *r;

  r = THTensor_(new)();
  luaT_pushudata(L, r, torch_Tensor);
  THTensor_(resizeAs)(r, tensor);
  THTensor_(copy)(r, tensor);
  THTensor_(mul)(r, r, -1);

  return 1;
}

static int torch_TensorOperator_(__mul__)(lua_State *L)
{
  THTensor *tensor1 = luaT_toudata(L, 1, torch_Tensor);
  THTensor *tensor2 = luaT_toudata(L, 2, torch_Tensor);
  THTensor *r;

  if(!tensor1 && !tensor2)
    luaL_error(L, "expecting two Tensors or one Tensor and one number");
  else
  {
    r = THTensor_(new)();
    luaT_pushudata(L, r, torch_Tensor);
    
    if(!tensor1 && tensor2)
    {
      THTensor_(resizeAs)(r, tensor2);
      THTensor_(copy)(r, tensor2);
      THTensor_(mul)(r, r, luaL_checknumber(L, 1));
    }
    else if(tensor1 && !tensor2)
    {
      THTensor_(resizeAs)(r, tensor1);
      THTensor_(copy)(r, tensor1);
      THTensor_(mul)(r, r, luaL_checknumber(L, 2));
    }
    else
    {
      int dimt = tensor1->nDimension;
      int dims = tensor2->nDimension;
      
      if(dimt == 1 && dims == 1)
        lua_pushnumber(L, THTensor_(dot)(tensor1, tensor2)); /* ok, we wasted r, but who cares */
      else if(dimt == 2 && dims == 1)
      {
        THTensor_(resize1d)(r, tensor1->size[0]);
        THTensor_(zero)(r);
        THTensor_(addmv)(r, 1, r, 1, tensor1, tensor2);
      }
      else if(dimt == 2 && dims == 2)
      {
        THTensor_(resize2d)(r, tensor1->size[0], tensor2->size[1]);
        THTensor_(zero)(r);
        THTensor_(addmm)(r, 1, r, 1, tensor1, tensor2);
      }
      else
        luaL_error(L, "multiplication between %dD and %dD tensors not yet supported", tensor1->nDimension, tensor2->nDimension); 
    }
  }
  return 1;
}

static int torch_TensorOperator_(__div__)(lua_State *L)
{
  THTensor *tensor = luaT_checkudata(L, 1, torch_Tensor);
  THTensor *r;

  luaL_argcheck(L, lua_isnumber(L,2), 2, "number expected");

  r = THTensor_(new)();
  luaT_pushudata(L, r, torch_Tensor);

  THTensor_(resizeAs)(r, tensor);
  THTensor_(copy)(r, tensor);
  THTensor_(mul)(r, r, 1/lua_tonumber(L, 2));

  return 1;
}

static const struct luaL_Reg torch_TensorOperator_(_) [] = {
  {"__add__", torch_TensorOperator_(__add__)},
  {"__sub__", torch_TensorOperator_(__sub__)},
  {"__unm__", torch_TensorOperator_(__unm__)},
  {"__mul__", torch_TensorOperator_(__mul__)},
  {"__div__", torch_TensorOperator_(__div__)},
  {NULL, NULL}
};

void torch_TensorOperator_(init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaL_register(L, NULL, torch_TensorOperator_(_));
  lua_pop(L, 1);
}

#endif
