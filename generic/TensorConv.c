#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/TensorConv.c"
#else

static int torch_(convxcorr2)(lua_State *L, const char* ktype)
{
  int narg = lua_gettop(L);
  THTensor *r_ = NULL;
  THTensor *im = NULL;
  THTensor *ker = NULL;
  char type[2];
  int rgiven = 0;

  type[0] = 'v';
  type[1] = ktype[0];

  if (narg == 2
      && (ker = luaT_toudata(L,2,torch_(Tensor_id)))
      && (im  = luaT_toudata(L,1,torch_(Tensor_id))))
  {
  }
  else if (narg == 3
	   && (lua_type(L,3) == LUA_TSTRING)
	   && (ker = luaT_toudata(L,2,torch_(Tensor_id)))
	   && (im = luaT_toudata(L,1,torch_(Tensor_id))))
  {
    type[0] = *(luaL_checkstring(L,3));
    luaL_argcheck(L, (type[0] == 'v' || type[0] == 'V' || type[0] == 'f' || type[0] == 'F'),
		  3, "[Tensor, ] Tensor, Tensor [, x or c]");
    if (type[0] == 'V') type[0] = 'v';
    if (type[0] == 'F') type[0] = 'f';
  }
  else if (narg == 4
	   && (type[0] = *(luaL_checkstring(L,4)))
	   && (ker = luaT_toudata(L,3,torch_(Tensor_id)))
	   && (im = luaT_toudata(L,2,torch_(Tensor_id)))
	   && (r_ = luaT_toudata(L,1,torch_(Tensor_id))))
  {
    rgiven = 1;
  }
  else
  {
    luaL_error(L,"[Tensor, ] Tensor, Tensor [, x or c]");
  }
  
  if (!r_) r_ = THTensor_(new)();

  if (im->nDimension == 2 && ker->nDimension == 2)
  {
    THTensor_(conv2Dmul)(r_,0.0,1.0,im,ker,1,1,type);
  }
  else if (im->nDimension == 3 && ker->nDimension == 3)
  {
    THTensor_(conv2Dcmul)(r_,0.0,1.0,im,ker,1,1,type);
  }
  else if (im->nDimension == 3 && ker->nDimension == 4)
  {
    THTensor_(conv2Dmv)(r_,0.0,1.0,im,ker,1,1,type);
  }
  else
  {
    luaL_error(L," (2D,2D) or (3D,3D) or (3D,4D) ");
  }

  pushreturn(rgiven, r_, torch_(Tensor_id));

  return 1;
}

static int torch_(convxcorr3)(lua_State *L, char* ktype)
{
  int narg = lua_gettop(L);
  THTensor *r_ = NULL;
  THTensor *im = NULL;
  THTensor *ker = NULL;
  char type[2];
  int rgiven = 0;
  
  type[0] = 'v';
  type[1] = ktype[0];

  if (narg == 2
      && (ker = luaT_toudata(L,2,torch_(Tensor_id)))
      && (im  = luaT_toudata(L,1,torch_(Tensor_id))))
  {
  }
  else if (narg == 3
	   && (lua_type(L,3) == LUA_TSTRING)
	   && (ker = luaT_toudata(L,2,torch_(Tensor_id)))
	   && (im = luaT_toudata(L,1,torch_(Tensor_id))))
  {
    type[0] = *(luaL_checkstring(L,3));
    luaL_argcheck(L, (type[0] == 'v' || type[0] == 'V' || type[0] == 'f' || type[0] == 'F'),
		  3, "[Tensor, ] Tensor, Tensor [, x or c]");
    if (type[0] == 'V') type[0] = 'v';
    if (type[0] == 'F') type[0] = 'f';
  }
  else if (narg == 4
	   && (type[0] = *(luaL_checkstring(L,4)))
	   && (ker = luaT_toudata(L,3,torch_(Tensor_id)))
	   && (im = luaT_toudata(L,2,torch_(Tensor_id)))
	   && (r_ = luaT_toudata(L,1,torch_(Tensor_id))))
  {
    rgiven = 1;
  }
  else
  {
    luaL_error(L,"[Tensor, ] Tensor, Tensor [, x or c]");
  }
  
  if (!r_) r_ = THTensor_(new)();

  if (im->nDimension == 3 && ker->nDimension == 3)
  {
    THTensor_(conv3Dmul)(r_,0.0,1.0,im,ker,1,1,1,type);
  }
  else if (im->nDimension == 4 && ker->nDimension == 4)
  {
    THTensor_(conv3Dcmul)(r_,0.0,1.0,im,ker,1,1,1,type);
  }
  else if (im->nDimension == 4 && ker->nDimension == 5)
  {
    THTensor_(conv3Dmv)(r_,0.0,1.0,im,ker,1,1,1,type);
  }
  else
  {
    luaL_error(L," (3D,3D) or (4D,4D) or (4D,5D) ");
  }

  pushreturn(rgiven, r_, torch_(Tensor_id));

  return 1;
}

static int torch_(conv2)(lua_State *L)
{
  return torch_(convxcorr2)(L,"convolution");
}
static int torch_(xcorr2)(lua_State *L)
{
  return torch_(convxcorr2)(L,"xcorrelation");
}


static int torch_(conv3)(lua_State *L)
{
  return torch_(convxcorr3)(L,"convolution");
}
static int torch_(xcorr3)(lua_State *L)
{
  return torch_(convxcorr3)(L,"xcorrelation");
}

static const struct luaL_Reg torch_(Conv__) [] = {
  {"conv2", torch_(conv2)},
  {"xcorr2", torch_(xcorr2)},
  {"conv3", torch_(conv3)},
  {"xcorr3", torch_(xcorr3)},
  {NULL, NULL}
};

void torch_(Conv_init)(lua_State *L)
{
  torch_(Tensor_id) = luaT_checktypename2id(L, torch_string_(Tensor));

  /* register everything into the "torch" field of the tensor metaclass */
  luaT_pushmetaclass(L, torch_(Tensor_id));
  lua_pushstring(L, "torch");
  lua_rawget(L, -2);
  luaL_register(L, NULL, torch_(Conv__));
  lua_pop(L, 2);
}

#endif

