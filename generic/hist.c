#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/lab.c"
#else

#include "interfaces.c"

static int lab_(histc)(lua_State *L)
{
  THTensor *r = luaT_checkudata(L, 1, torch_(Tensor_id));
  THTensor *h = luaT_checkudata(L, 2, torch_(Tensor_id));
  int nbins = luaL_checknumber(L, 3);
  real *h_data = THTensor_(data)(h);

  TH_TENSOR_APPLY(real, r,                                      \
                  if ((*r_data <= nbins) && (*r_data >= 1)) {   \
                    *(h_data + (int)(*r_data) - 1) += 1;        \
                  })
  return 0;
}

static const struct luaL_Reg lab_(stuff__) [] = {
  {"_histc", lab_(histc)},
#endif
  {NULL, NULL}
};

void lab_(init)(lua_State *L)
{
  torch_(Tensor_id) = luaT_checktypename2id(L, torch_string_(Tensor));
  torch_LongStorage_id = luaT_checktypename2id(L, "torch.LongStorage");

  /* register everything into the "lab" field of the tensor metaclass */
  luaT_pushmetaclass(L, torch_(Tensor_id));
  lua_pushstring(L, "lab");
  lua_newtable(L);
  luaL_register(L, NULL, lab_(stuff__));
  lua_rawset(L, -3);
  lua_pop(L, 1);

/*  luaT_registeratid(L, lab_(stuff__), torch_(Tensor_id)); */
/*  luaL_register(L, NULL, lab_(stuff__)); */  
}

#endif
