#include "general.h"
#include "utils.h"

#include <sys/time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

THLongStorage* torch_checklongargs(lua_State *L, int index)
{
  THLongStorage *storage;
  int i;
  int narg = lua_gettop(L)-index+1;

  if(narg == 1 && luaT_toudata(L, index, "torch.LongStorage"))
  {
    THLongStorage *storagesrc = luaT_toudata(L, index, "torch.LongStorage");
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

  if(narg == 1 && luaT_toudata(L, index, "torch.LongStorage"))
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

static int torch_lua_getdefaulttensortype(lua_State *L)
{
  const char* tname = torch_getdefaulttensortype(L);
  if(tname)
  {
    lua_pushstring(L, tname);
    return 1;
  }
  return 0;
}

const char* torch_getdefaulttensortype(lua_State *L)
{
  lua_getfield(L, LUA_GLOBALSINDEX, "torch");
  if(lua_istable(L, -1))
  {
    lua_getfield(L, -1, "Tensor");
    if(lua_istable(L, -1))
    {
      if(lua_getmetatable(L, -1))
      {
        lua_pushstring(L, "__index");
        lua_rawget(L, -2);
        if(lua_istable(L, -1))
        {
          lua_rawget(L, LUA_REGISTRYINDEX);
          if(lua_isstring(L, -1))
          {
            const char *tname = lua_tostring(L, -1);
            lua_pop(L, 4);
            return tname;
          }
        }
        else
        {
          lua_pop(L, 4);
          return NULL;
        }
      }
      else
      {
        lua_pop(L, 2);
        return NULL;
      }
    }
    else
    {
      lua_pop(L, 2);
      return NULL;
    }
  }
  else
  {
    lua_pop(L, 1);
    return NULL;
  }
  return NULL;
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
  {"getdefaulttensortype", torch_lua_getdefaulttensortype},
  {"tic", torch_lua_tic},
  {"toc", torch_lua_toc},
  {"setnumthreads", torch_setnumthreads},
  {"getnumthreads", torch_getnumthreads},
  {"factory", luaT_lua_factory},
  {"getconstructortable", luaT_lua_getconstructortable},
  {"typename", luaT_lua_typename},
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
  luaL_register(L, NULL, torch_utils__);
}
