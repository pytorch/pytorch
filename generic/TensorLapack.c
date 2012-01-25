#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/TensorLapack.c"
#else

#define pushreturn(i,t,tid) \
  if (!i)					\
    luaT_pushudata(L, t, tid);			\
  else						\
    lua_pushvalue(L,i)			

static int torch_(gesv)(lua_State *L)
{
  int narg = lua_gettop(L);
  THTensor *ra_ = NULL;
  THTensor *rb_ = NULL;
  THTensor *a_ = NULL;
  THTensor *b_ = NULL;
  int ragiven = 0;
  int rbgiven = 0;

  if (narg == 2
      && (a_ = luaT_toudata(L,2,torch_(Tensor_id)))
      && (b_ = luaT_toudata(L,1,torch_(Tensor_id))))
  {
  }
  else if (narg == 3
	   && (a_ = luaT_toudata(L,2,torch_(Tensor_id)))
	   && (b_ = luaT_toudata(L,1,torch_(Tensor_id))))
  {
    if(lua_toboolean(L,3))
    {
      ra_ = a_;
      rb_ = b_;
      a_ = NULL;
      b_ = NULL;
      ragiven = 2;
      rbgiven = 1;
    }
    else
    {
      luaL_error(L,"[Tensor, Tensor], Tensor, Tensor, [,true]");      
    }
  }
  else if (narg == 4
	   && (a_ = luaT_toudata(L,4,torch_(Tensor_id)))
	   && (b_ = luaT_toudata(L,3,torch_(Tensor_id)))
	   && (ra_ = luaT_toudata(L,2,torch_(Tensor_id)))
	   && (rb_ = luaT_toudata(L,1,torch_(Tensor_id))))
  {
    ragiven = 2;
    rbgiven = 1;
  }
  else
  {
    luaL_error(L,"[Tensor, Tensor], Tensor, Tensor, [,true]");      
  }

  if (!ra_) ra_ = THTensor_(new)();
  if (!rb_) rb_ = THTensor_(new)();
  
  THTensor_(gesv)(rb_,ra_,b_,a_);

  pushreturn(rbgiven,rb_,torch_(Tensor_id));
  pushreturn(ragiven,ra_,torch_(Tensor_id));

  return 2;
}

static int torch_(gels)(lua_State *L)
{
  int narg = lua_gettop(L);
  THTensor *ra_ = NULL;
  THTensor *rb_ = NULL;
  THTensor *a_ = NULL;
  THTensor *b_ = NULL;
  int ragiven = 0;
  int rbgiven = 0;

  if (narg == 2
      && (a_ = luaT_toudata(L,2,torch_(Tensor_id)))
      && (b_ = luaT_toudata(L,1,torch_(Tensor_id))))
  {
  }
  else if (narg == 3
	   && (a_ = luaT_toudata(L,2,torch_(Tensor_id)))
	   && (b_ = luaT_toudata(L,1,torch_(Tensor_id))))
  {
    if (lua_toboolean(L,3))
    {
      ra_ = a_;
      rb_ = b_;
      a_ = NULL;
      b_ = NULL;
      ragiven = 2;
      rbgiven = 1;
    }
    else
    {
      luaL_error(L,"[Tensor, Tensor], Tensor, Tensor, [,true]");      
    }
  }
  else if (narg == 4
	   && (a_ = luaT_toudata(L,4,torch_(Tensor_id)))
	   && (b_ = luaT_toudata(L,3,torch_(Tensor_id)))
	   && (ra_ = luaT_toudata(L,2,torch_(Tensor_id)))
	   && (rb_ = luaT_toudata(L,1,torch_(Tensor_id))))
  {
    ragiven = 2;
    rbgiven = 1;
  }
  else
  {
    luaL_error(L,"[Tensor, Tensor], Tensor, Tensor, [,true]");
  }

  if (!ra_) ra_ = THTensor_(new)();
  if (!rb_) rb_ = THTensor_(new)();

  THTensor_(gels)(rb_,ra_,b_,a_);

  pushreturn(rbgiven,rb_,torch_(Tensor_id));
  pushreturn(ragiven,ra_,torch_(Tensor_id));

  return 2;
}

static int torch_(eig)(lua_State *L)
{
  int narg = lua_gettop(L);
  THTensor *re_ = NULL;
  THTensor *rv_ = NULL;
  THTensor *a_ = NULL;
  char type = 'N';
  char uplo = 'U';
  int regiven = 0;
  int rvgiven = 0;

  if (narg == 1
      && (a_ = luaT_toudata(L,1,torch_(Tensor_id))))
  {
  }
  else if (narg == 2
	   && (lua_type(L,2) == LUA_TSTRING)
	   && (a_ = luaT_toudata(L,1,torch_(Tensor_id))))
  {
    type = *(luaL_checkstring(L,2));
    luaL_argcheck(L, (type == 'v' || type == 'V' || type == 'n' || type == 'N'),
		  2, "[Tensor, ] [Tensor, ] Tensor [, N or V]");
    if (type == 'v') type = 'V';
    if (type == 'n') type = 'N';
  }
  else if (narg == 2
	   && (a_  = luaT_toudata(L,2,torch_(Tensor_id)))
	   && (re_ = luaT_toudata(L,1,torch_(Tensor_id))))
  {
    regiven = 1;
  }
  else if (narg == 3
	   && (a_  = luaT_toudata(L,3,torch_(Tensor_id)))
	   && (rv_ = luaT_toudata(L,2,torch_(Tensor_id)))
	   && (re_ = luaT_toudata(L,1,torch_(Tensor_id))))
  {
    regiven = 1;
    rvgiven = 2;
  }
  else if (narg == 4
	   && (type = *(luaL_checkstring(L,4)))
	   && (a_  = luaT_toudata(L,3,torch_(Tensor_id)))
	   && (rv_ = luaT_toudata(L,2,torch_(Tensor_id)))
	   && (re_ = luaT_toudata(L,1,torch_(Tensor_id))))
  {
    regiven = 1;
    rvgiven = 2;
  }
  else
  {
    luaL_error(L,"[Tensor, ] [Tensor, ] Tensor [, N or V]");
  }
  if (!re_) re_ = THTensor_(new)();
  if (!rv_) rv_ = THTensor_(new)();

  THTensor_(syev)(re_,rv_,a_,&type,&uplo);

  pushreturn(regiven, re_, torch_(Tensor_id));
  pushreturn(rvgiven, rv_, torch_(Tensor_id));

  return 2;
}

static int torch_(svd)(lua_State *L)
{
  int narg = lua_gettop(L);
  THTensor *ru_ = NULL;
  THTensor *rs_ = NULL;
  THTensor *rv_ = NULL;
  THTensor *a_ = NULL;
  char type = 'S';
  int rugiven = 0;
  int rsgiven = 0;
  int rvgiven = 0;

  if (narg == 1
      && (a_ = luaT_toudata(L,1,torch_(Tensor_id))))
  {
  }
  else if (narg ==2 
	   && (type = *(luaL_checkstring(L,2)))
	   && (a_ = luaT_toudata(L,1,torch_(Tensor_id))))
  {
    luaL_argcheck(L, (type == 's' || type == 'S' || type == 'a' || type == 'A'),
		  2, "[Tensor, ] [Tensor, ] [Tensor, ] Tensor [, A or S]");
    if (type == 's') type = 'S';
    if (type == 'a') type = 'A';
  }
  else if (narg == 4
	   && (a_  = luaT_toudata(L,4,torch_(Tensor_id)))
	   && (rv_ = luaT_toudata(L,3,torch_(Tensor_id)))
	   && (rs_ = luaT_toudata(L,2,torch_(Tensor_id)))
	   && (ru_ = luaT_toudata(L,1,torch_(Tensor_id))))
  {
    rugiven = 1;
    rsgiven = 2;
    rvgiven = 3;
  }
  else if (narg == 5
	   && (type = *(luaL_checkstring(L,5)))
	   && (a_  = luaT_toudata(L,4,torch_(Tensor_id)))
	   && (rv_ = luaT_toudata(L,3,torch_(Tensor_id)))
	   && (rs_ = luaT_toudata(L,2,torch_(Tensor_id)))
	   && (ru_ = luaT_toudata(L,1,torch_(Tensor_id))))
  {
    rugiven = 1;
    rsgiven = 2;
    rvgiven = 3;
  }
  else
  {
    luaL_error(L,"[Tensor, Tensor, Tensor], Tensor, [, 'A' or 'S' ]");
  }

  if (!ru_) ru_ = THTensor_(new)();
  if (!rs_) rs_ = THTensor_(new)();
  if (!rv_) rv_ = THTensor_(new)();

  THTensor_(gesvd)(ru_,rs_,rv_,a_,&type);

  pushreturn(rugiven,ru_,torch_(Tensor_id));
  pushreturn(rsgiven,rs_,torch_(Tensor_id));
  pushreturn(rvgiven,rv_,torch_(Tensor_id));

  return 3;
}

static const struct luaL_Reg torch_(lapack__) [] = {
  {"gesv", torch_(gesv)},
  {"gels", torch_(gels)},
  {"eig", torch_(eig)},
  {"svd", torch_(svd)},
  {NULL, NULL}
};

void torch_(Lapack_init)(lua_State *L)
{
  torch_(Tensor_id) = luaT_checktypename2id(L, torch_string_(Tensor));

  /* register everything into the "torch" field of the tensor metaclass */
  luaT_pushmetaclass(L, torch_(Tensor_id));
  lua_pushstring(L, "torch");
  lua_rawget(L, -2);
  luaL_register(L, NULL, torch_(lapack__));
  lua_pop(L, 2);
}

#endif
