#include "general.h"
#include "utils.h"

#include <sys/time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static const void* torch_LongStorage_id = NULL;
static const void* torch_default_tensor_id = NULL;

THLongStorage* torch_checklongargs(lua_State *L, int index)
{
  THLongStorage *storage;
  int i;
  int narg = lua_gettop(L)-index+1;

  if(narg == 1 && luaT_toudata(L, index, torch_LongStorage_id))
  {
    THLongStorage *storagesrc = luaT_toudata(L, index, torch_LongStorage_id);
    storage = THLongStorage_newWithSize(storagesrc->size);
    THLongStorage_copy(storage, storagesrc);
  }
  else
  {
    storage = THLongStorage_newWithSize(narg);
    for(i = index; i < index+narg; i++)
    {
      if(!lua_isnumber(L, i))
      {
        THLongStorage_free(storage);
        luaL_argerror(L, i, "number expected");
      }
      THLongStorage_set(storage, i-index, lua_tonumber(L, i));
    }
  }
  return storage;
}

int torch_islongargs(lua_State *L, int index)
{
  int narg = lua_gettop(L)-index+1;

  if(narg == 1 && luaT_toudata(L, index, torch_LongStorage_id))
  {
    return 1;
  }
  else
  {
    int i;

    for(i = index; i < index+narg; i++)
    {
      if(!lua_isnumber(L, i))
        return 0;
    }
    return 1;
  }
  return 0;
}



static int torch_lua_tic(lua_State* L)
{
  struct timeval tv;
  gettimeofday(&tv,NULL);
  double ttime = (double)tv.tv_sec + (double)(tv.tv_usec)/1000000.0;
  lua_pushnumber(L,ttime);
  return 1;
}

static int torch_lua_toc(lua_State* L)
{
  struct timeval tv;
  gettimeofday(&tv,NULL);
  double toctime = (double)tv.tv_sec + (double)(tv.tv_usec)/1000000.0;
  lua_Number tictime = luaL_checknumber(L,1);
  lua_pushnumber(L,toctime-tictime);
  return 1;
}

static int torch_lua_setdefaulttensortype(lua_State *L)
{
  const void *id;

  luaL_checkstring(L, 1);
  
  if(!(id = luaT_typename2id(L, lua_tostring(L, 1))))                  \
    return luaL_error(L, "<%s> is not a string describing a torch object", lua_tostring(L, 1)); \

  torch_default_tensor_id = id;

  return 0;
}

static int torch_lua_getdefaulttensortype(lua_State *L)
{
  lua_pushstring(L, luaT_id2typename(L, torch_default_tensor_id));
  return 1;
}

void torch_setdefaulttensorid(const void* id)
{
  torch_default_tensor_id = id;
}

const void* torch_getdefaulttensorid()
{
  return torch_default_tensor_id;
}

static int torch_getnumthreads(lua_State *L)
{
#ifdef _OPENMP
  lua_pushinteger(L, omp_get_max_threads());
#else
  lua_pushinteger(L, 1);
#endif
  return 1;
}

static int torch_setnumthreads(lua_State *L)
{
  int nth = luaL_checkint(L,1);
#ifdef _OPENMP
  omp_set_num_threads(nth);
#endif
  return 0;
}

static const struct luaL_Reg torch_utils__ [] = {
  {"__setdefaulttensortype", torch_lua_setdefaulttensortype},
  {"getdefaulttensortype", torch_lua_getdefaulttensortype},
  {"tic", torch_lua_tic},
  {"toc", torch_lua_toc},
  {"setnumthreads", torch_setnumthreads},
  {"getnumthreads", torch_getnumthreads},
  {"factory", luaT_lua_factory},
  {"getconstructortable", luaT_lua_getconstructortable},
  {"id", luaT_lua_id},
  {"typename", luaT_lua_typename},
  {"typename2id", luaT_lua_typename2id},
  {"isequal", luaT_lua_isequal},
  {"getenv", luaT_lua_getenv},
  {"setenv", luaT_lua_setenv},
  {"newmetatable", luaT_lua_newmetatable},
  {"setmetatable", luaT_lua_setmetatable},
  {"getmetatable", luaT_lua_getmetatable},
  {"version", luaT_lua_version},
  {"pointer", luaT_lua_pointer},
  {NULL, NULL}
};

void torch_utils_init(lua_State *L)
{
  torch_LongStorage_id = luaT_checktypename2id(L, "torch.LongStorage");
  luaL_register(L, NULL, torch_utils__);
}
