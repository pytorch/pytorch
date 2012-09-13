#include <stdlib.h>
#include <string.h>

#include "luaT.h"

void* luaT_alloc(lua_State *L, long size)
{
  void *ptr;

  if(size == 0)
    return NULL;

  if(size < 0)
    luaL_error(L, "$ Torch: invalid memory size -- maybe an overflow?");

  ptr = malloc(size);
  if(!ptr)
    luaL_error(L, "$ Torch: not enough memory: you tried to allocate %dGB. Buy new RAM!", size/1073741824);

  return ptr;
}

void* luaT_realloc(lua_State *L, void *ptr, long size)
{
  if(!ptr)
    return(luaT_alloc(L, size));

  if(size == 0)
  {
    luaT_free(L, ptr);
    return NULL;
  }

  if(size < 0)
    luaL_error(L, "$ Torch: invalid memory size -- maybe an overflow?");

  ptr = realloc(ptr, size);
  if(!ptr)
    luaL_error(L, "$ Torch: not enough memory: you tried to reallocate %dGB. Buy new RAM!", size/1073741824);
  return ptr;
}

void luaT_free(lua_State *L, void *ptr)
{
  free(ptr);
}

void luaT_stackdump(lua_State *L)
{
  int i;
  const char *tname = NULL;
  int top = lua_gettop(L);
  for(i = 1; i <= top; i++)
  {
    int t = lua_type(L, i);
    printf("%3d. ", i);
    switch(t)
    {
      case LUA_TSTRING:
        printf("'%s'", lua_tostring(L,i));
        break;
      case LUA_TBOOLEAN:
        printf(lua_toboolean(L, i) ? "true" : "false");
        break;
      case LUA_TNUMBER:
        printf("%g", lua_tonumber(L,i));
        break;
      case LUA_TUSERDATA:
        tname = luaT_typename(L, i);
        printf("userdata %lx [%s]", (long)lua_topointer(L, i), (tname ? tname : "not a Torch object"));
        break;
      case LUA_TTABLE:
        lua_pushvalue(L, i);
        lua_rawget(L, LUA_REGISTRYINDEX);
        if(lua_isstring(L, -1))
          tname = lua_tostring(L, -1); /*luaT_typenameid(L, lua_tostring(L, -1)); */
        else
          tname = NULL;
        lua_pop(L, 1);
        if(tname)
          printf("metatable [%s]", tname);
        else
        {
          tname = luaT_typename(L, i);
          printf("table %lx [%s]", (long)lua_topointer(L, i), (tname ? tname : "not a Torch object"));
        }
        break;
      default:
        printf("Lua object type: %s", lua_typename(L,t));
        break;
    }
    printf("\n");
  }
  printf("---------------------------------------------\n");
}

/* metatable operator methods */
static int luaT_mt__index(lua_State *L);
static int luaT_mt__newindex(lua_State *L);
static int luaT_mt__tostring(lua_State *L);
static int luaT_mt__add(lua_State *L);
static int luaT_mt__sub(lua_State *L);
static int luaT_mt__mul(lua_State *L);
static int luaT_mt__div(lua_State *L);
static int luaT_mt__mod(lua_State *L);
static int luaT_mt__pow(lua_State *L);
static int luaT_mt__unm(lua_State *L);
static int luaT_mt__concat(lua_State *L);
static int luaT_mt__len(lua_State *L);
static int luaT_mt__eq(lua_State *L);
static int luaT_mt__lt(lua_State *L);
static int luaT_mt__le(lua_State *L);
static int luaT_mt__call(lua_State *L);

/* Constructor-metatable methods */
static int luaT_cmt__call(lua_State *L);
static int luaT_cmt__newindex(lua_State *L);

const char* luaT_newmetatable(lua_State *L, const char *tname, const char *parenttname,
                              lua_CFunction constructor, lua_CFunction destructor, lua_CFunction factory)
{
  lua_pushcfunction(L, luaT_lua_newmetatable);
  lua_pushstring(L, tname);
  (parenttname ? lua_pushstring(L, parenttname) : lua_pushnil(L));
  (constructor ? lua_pushcfunction(L, constructor) : lua_pushnil(L));
  (destructor ? lua_pushcfunction(L, destructor) : lua_pushnil(L));
  (factory ? lua_pushcfunction(L, factory) : lua_pushnil(L));
  lua_call(L, 5, 1);
  return luaT_typenameid(L, tname);
}

int luaT_pushmetatable(lua_State *L, const char *tname)
{
  lua_getfield(L, LUA_REGISTRYINDEX, tname);
  if(lua_isnil(L, -1))
  {
    lua_pop(L, 1);
    return 0;
  }
  return 1;
}

const char *luaT_typenameid(lua_State *L, const char *tname)
{
  if(luaT_pushmetatable(L, tname))
  {
    const char *tnameid = NULL;
    lua_rawget(L, LUA_REGISTRYINDEX);
    if(lua_isstring(L, -1))
      tnameid = lua_tostring(L, -1);
    lua_pop(L, 1); /* the string/nil */
    return tnameid;
  }
  return NULL;
}

const char* luaT_typename(lua_State *L, int ud)
{
  if(lua_getmetatable(L, ud))
  {
    const char *tname = NULL;
    lua_rawget(L, LUA_REGISTRYINDEX);
    if(lua_isstring(L, -1))
      tname = lua_tostring(L, -1);
    lua_pop(L, 1); /* the string/nil */
    return tname;
  }
  return NULL;
}

void luaT_pushudata(lua_State *L, void *udata, const char *tname)
{
  if(udata)
  {
    void **udata_p = lua_newuserdata(L, sizeof(void*));
    *udata_p = udata;
    if(!luaT_pushmetatable(L, tname))
      luaL_error(L, "Torch internal problem: cannot find metatable for type <%s>", tname);
    lua_setmetatable(L, -2);
  }
  else
    lua_pushnil(L);
}

void *luaT_toudata(lua_State *L, int ud, const char *tname)
{
  void **p = lua_touserdata(L, ud);
  if(p != NULL) /* value is a userdata? */
  {
    if(!luaT_pushmetatable(L, tname))
      luaL_error(L, "Torch internal problem: cannot find metatable for type <%s>", tname);

    /* initialize the table we want to get the metatable on */
    /* note that we have to be careful with indices, as we just inserted stuff */
    lua_pushvalue(L, (ud < 0 ? ud - 1 : ud));
    while(lua_getmetatable(L, -1)) /* get the next metatable */
    {
      lua_remove(L, -2); /* remove the previous metatable [or object, if first time] */
      if(lua_rawequal(L, -1, -2))
      {
        lua_pop(L, 2);  /* remove the two metatables */
        return *p;
      }
    }
    lua_pop(L, 2); /* remove the two metatables */
  }
  return NULL;
}

int luaT_isudata(lua_State *L, int ud, const char *tname)
{
  if(luaT_toudata(L, ud, tname))
    return 1;
  else
    return 0;
}

void *luaT_checkudata(lua_State *L, int ud, const char *tname)
{
  void *p = luaT_toudata(L, ud, tname);
  if(!p)
    luaT_typerror(L, ud, tname);
  return p;
}

void *luaT_getfieldcheckudata(lua_State *L, int ud, const char *field, const char *tname)
{
  void *p;
  lua_getfield(L, ud, field);
  if(lua_isnil(L, -1))
    luaL_error(L, "bad argument #%d (field %s does not exist)", ud, field);
  p = luaT_toudata(L, -1, tname);
  if(!p)
    luaL_error(L, "bad argument #%d (field %s is not a %s)", ud, field, tname);
  return p;
}

void *luaT_getfieldchecklightudata(lua_State *L, int ud, const char *field)
{
  void *p;
  lua_getfield(L, ud, field);
  if(lua_isnil(L, -1))
    luaL_error(L, "bad argument #%d (field %s does not exist)", ud, field);

  if(!lua_islightuserdata(L, -1))
    luaL_error(L, "bad argument #%d (field %s is not a light userdata)", ud, field);

  p = lua_touserdata(L, -1);

  return p;
}

double luaT_getfieldchecknumber(lua_State *L, int ud, const char *field)
{
  lua_getfield(L, ud, field);
  if(lua_isnil(L, -1))
    luaL_error(L, "bad argument #%d (field %s does not exist)", ud, field);
  if(!lua_isnumber(L, -1))
    luaL_error(L, "bad argument #%d (field %s is not a number)", ud, field);
  return lua_tonumber(L, -1);
}

int luaT_getfieldcheckint(lua_State *L, int ud, const char *field)
{
  lua_getfield(L, ud, field);
  if(lua_isnil(L, -1))
    luaL_error(L, "bad argument #%d (field %s does not exist)", ud, field);
  if(!lua_isnumber(L, -1))
    luaL_error(L, "bad argument #%d (field %s is not a number)", ud, field);
  return (int)lua_tonumber(L, -1);
}

const char* luaT_getfieldcheckstring(lua_State *L, int ud, const char *field)
{
  lua_getfield(L, ud, field);
  if(lua_isnil(L, -1))
    luaL_error(L, "bad argument #%d (field %s does not exist)", ud, field);
  if(!lua_isstring(L, -1))
    luaL_error(L, "bad argument #%d (field %s is not a string)", ud, field);
  return lua_tostring(L, -1);
}

int luaT_getfieldcheckboolean(lua_State *L, int ud, const char *field)
{
  lua_getfield(L, ud, field);
  if(lua_isnil(L, -1))
    luaL_error(L, "bad argument #%d (field %s does not exist)", ud, field);
  if(!lua_isboolean(L, -1))
    luaL_error(L, "bad argument #%d (field %s is not a boolean)", ud, field);
  return lua_toboolean(L, -1);
}

void luaT_getfieldchecktable(lua_State *L, int ud, const char *field)
{
  lua_getfield(L, ud, field);
  if(lua_isnil(L, -1))
    luaL_error(L, "bad argument #%d (field %s does not exist)", ud, field);
  if(!lua_istable(L, -1))
    luaL_error(L, "bad argument #%d (field %s is not a table)", ud, field);
}

/**** type checks as in luaL ****/
int luaT_typerror(lua_State *L, int ud, const char *tname)
{
  const char *msg;
  const char *tnameud = luaT_typename(L, ud);

  if(!tnameud)
    tnameud = lua_typename(L, ud);

  msg = lua_pushfstring(L, "%s expected, got %s",
                        tname,
                        (tnameud ? tnameud : "unknown object"));

  return luaL_argerror(L, ud, msg);
}

int luaT_checkboolean(lua_State *L, int ud)
{
  if(!lua_isboolean(L, ud))
    luaT_typerror(L, ud, lua_typename(L, LUA_TBOOLEAN));
  return lua_toboolean(L, ud);
}

int luaT_optboolean(lua_State *L, int ud, int def)
{
  if(lua_isnoneornil(L,ud))
    return def;

  return luaT_checkboolean(L, ud);
}

void luaT_registeratname(lua_State *L, const struct luaL_Reg *methods, const char *name)
{
  int idx = lua_gettop(L);

  luaL_checktype(L, idx, LUA_TTABLE);
  lua_pushstring(L, name);
  lua_rawget(L, idx);

  if(lua_isnil(L, -1))
  {
    lua_pop(L, 1);
    lua_pushstring(L, name);
    lua_newtable(L);
    lua_rawset(L, idx);

    lua_pushstring(L, name);
    lua_rawget(L, idx);
  }

  luaL_register(L, NULL, methods);
  lua_pop(L, 1);
}


/* utility functions */
const char *luaT_classrootname(const char *tname)
{
  int i;
  int sz = strlen(tname);

  for(i = 0; i < sz; i++)
  {
    if(tname[i] == '.')
      return tname+i+1;
  }
  return tname;
}

const char *luaT_classmodulename(const char *tname)
{
  static char luaT_class_module_name[256];
  int i;

  strncpy(luaT_class_module_name, tname, 256);
  for(i = 0; i < 256; i++)
  {
    if(luaT_class_module_name[i] == '\0')
      break;
    if(luaT_class_module_name[i] == '.')
    {
      luaT_class_module_name[i] = '\0';
      return luaT_class_module_name;
    }
  }
  return NULL;
}

/* Lua only functions */
int luaT_lua_newmetatable(lua_State *L)
{
  const char* tname = luaL_checkstring(L, 1);

  lua_settop(L, 5);
  luaL_argcheck(L, lua_isnoneornil(L, 2) || lua_isstring(L, 2), 2, "parent class name or nil expected");
  luaL_argcheck(L, lua_isnoneornil(L, 3) || lua_isfunction(L, 3), 3, "constructor function or nil expected");
  luaL_argcheck(L, lua_isnoneornil(L, 4) || lua_isfunction(L, 4), 4, "destructor function or nil expected");
  luaL_argcheck(L, lua_isnoneornil(L, 5) || lua_isfunction(L, 5), 5, "factory function or nil expected");

  if(luaT_classmodulename(tname))
    lua_getfield(L, LUA_GLOBALSINDEX, luaT_classmodulename(tname));
  else
    lua_pushvalue(L, LUA_GLOBALSINDEX);
  if(!lua_istable(L, 6))
    luaL_error(L, "while creating metatable %s: bad argument #1 (%s is an invalid module name)", tname, luaT_classmodulename(tname));

  /* we first create the new metaclass if we have to */
  if(!luaT_pushmetatable(L, tname))
  {
    /* create the metatable */
    lua_newtable(L);

    /* registry[name] = metatable */
    lua_pushvalue(L, -1);
    lua_setfield(L, LUA_REGISTRYINDEX, tname);

    /* registry[metatable] = tname */
    lua_pushvalue(L, -1);
    lua_pushstring(L, tname);
    lua_rawset(L, LUA_REGISTRYINDEX);

    /* __index handling */
    lua_pushcfunction(L, luaT_mt__index);
    lua_setfield(L, -2, "__index");

    /* __newindex handling */
    lua_pushcfunction(L, luaT_mt__newindex);
    lua_setfield(L, -2, "__newindex");

    /* __typename contains the typename */
    lua_pushstring(L, tname);
    lua_setfield(L, -2, "__typename");

    /* __metatable is self */
    lua_pushvalue(L, -1);
    lua_setfield(L, -2, "__metatable");

    /* by default, __version equals 1 */
    lua_pushnumber(L, 1);
    lua_setfield(L, -2, "__version");

    /* assign default operator functions */
    lua_pushcfunction(L, luaT_mt__tostring);
    lua_setfield(L, -2, "__tostring");

    lua_pushcfunction(L, luaT_mt__add);
    lua_setfield(L, -2, "__add");

    lua_pushcfunction(L, luaT_mt__sub);
    lua_setfield(L, -2, "__sub");

    lua_pushcfunction(L, luaT_mt__mul);
    lua_setfield(L, -2, "__mul");

    lua_pushcfunction(L, luaT_mt__div);
    lua_setfield(L, -2, "__div");

    lua_pushcfunction(L, luaT_mt__mod);
    lua_setfield(L, -2, "__mod");

    lua_pushcfunction(L, luaT_mt__pow);
    lua_setfield(L, -2, "__pow");

    lua_pushcfunction(L, luaT_mt__unm);
    lua_setfield(L, -2, "__unm");

    lua_pushcfunction(L, luaT_mt__concat);
    lua_setfield(L, -2, "__concat");

    lua_pushcfunction(L, luaT_mt__len);
    lua_setfield(L, -2, "__len");

    lua_pushcfunction(L, luaT_mt__eq);
    lua_setfield(L, -2, "__eq");

    lua_pushcfunction(L, luaT_mt__lt);
    lua_setfield(L, -2, "__lt");

    lua_pushcfunction(L, luaT_mt__le);
    lua_setfield(L, -2, "__le");

    lua_pushcfunction(L, luaT_mt__call);
    lua_setfield(L, -2, "__call");
  }

  /* we assign the parent class if necessary */
  if(!lua_isnoneornil(L, 2))
  {
    if(lua_getmetatable(L, -1))
      luaL_error(L, "class %s has been already assigned a parent class\n", tname);
    else
    {
      const char* parenttname = luaL_checkstring(L, 2);
      luaT_pushmetatable(L, parenttname);
      if(lua_isnil(L, -1))
        luaL_error(L, "bad argument #2 (invalid parent class name %s)", parenttname);
      lua_setmetatable(L, -2);
    }
  }

  /* register the destructor function  */
  if(!lua_isnoneornil(L, 4))
  {
    /* does it exists already? */
    lua_pushstring(L, "__gc");
    lua_rawget(L, -2);

    if(lua_isnil(L, -1))
    {
      lua_pop(L, 1); /* pop nil */
      lua_pushstring(L, "__gc");
      lua_pushvalue(L, 4);
      lua_rawset(L, -3);
    }
    else
      luaL_error(L, "%s has been already assigned a destructor", tname);
  }

  /* register the factory function  */
  if(!lua_isnoneornil(L, 5))
  {
    /* does it exists already? */
    lua_pushstring(L, "__factory");
    lua_rawget(L, -2);

    if(lua_isnil(L, -1))
    {
      lua_pop(L, 1); /* pop nil */
      lua_pushstring(L, "__factory");
      lua_pushvalue(L, 5);
      lua_rawset(L, -3);
    }
    else
      luaL_error(L, "%s has been already assigned a factory", tname);
  }

  /******** Constructor table and metatable ********/
  lua_pushstring(L, "__constructor");
  lua_rawget(L, -2);
  if(lua_isnil(L, -1))
  {
    lua_pop(L, 1);                        /* pop nil */
    lua_newtable(L);                      /* fancy table */
    lua_newtable(L);                      /* fancy metatable */

    lua_pushvalue(L, -3);                 /* metatable */
    lua_setfield(L, -2, "__index");       /* so we can get the methods */

    lua_pushcfunction(L, luaT_cmt__newindex);
    lua_setfield(L, -2, "__newindex");    /* so we add new methods */

    lua_pushcfunction(L, luaT_cmt__call);
    lua_setfield(L, -2, "__call");        /* so we can create, we are here for only that */

    lua_pushvalue(L, -3);
    lua_setfield(L, -2, "__metatable");   /* redirect to metatable with methods */

    lua_setmetatable(L, -2);              /* constructor metatable is ... this fancy metatable */

    /* set metatable[__constructor] = constructor-metatable */
    lua_pushstring(L, "__constructor");
    lua_pushvalue(L, -2);
    lua_rawset(L, -4);
  }

  /* register the constructor function  */
  if(!lua_isnoneornil(L, 3))
  {
    /* get constructor metatable */
    lua_getmetatable(L, -1);

    /* does it exists already? */
    lua_pushstring(L, "__new");
    lua_rawget(L, -2);

    if(lua_isnil(L, -1))
    {
      lua_pop(L, 1); /* pop nil */
      lua_pushstring(L, "__new");
      lua_pushvalue(L, 3);
      lua_rawset(L, -3);

      /* set "new" in the metatable too */
      lua_pushstring(L, "new");
      lua_pushvalue(L, 3);
      lua_rawset(L, -5);
    }
    else
      luaL_error(L, "%s has been already assigned a constructor", tname);

    /* pop constructor metatable */
    lua_pop(L, 1);
  }

  /* module.name = constructor metatable */
  lua_setfield(L, 6, luaT_classrootname(tname));

  return 1; /* returns the metatable */
}


/* Lua only utility functions */
int luaT_lua_factory(lua_State *L)
{
  const char* tname = luaL_checkstring(L, 1);
  luaT_pushmetatable(L, tname);
  if(!lua_isnil(L, -1))
  {
    lua_pushstring(L, "__factory");
    lua_rawget(L, -2);
  }
  return 1;
}

int luaT_lua_getconstructortable(lua_State *L)
{
  const char* tname = luaL_checkstring(L, 1);
  if(luaT_pushmetatable(L, tname))
  {
    lua_pushstring(L, "__constructor");
    lua_rawget(L, -2);
    return 1;
  }
  return 0;
}


int luaT_lua_typename(lua_State *L)
{
  const char* tname = NULL;
  luaL_checkany(L, 1);
  if((tname = luaT_typename(L, 1)))
  {
    lua_pushstring(L, tname);
    return 1;
  }
  return 0;
}

int luaT_lua_isequal(lua_State *L)
{
  if(lua_isuserdata(L, 1) && lua_isuserdata(L, 2))
  {
    void **u1, **u2;
    luaL_argcheck(L, luaT_typename(L, 1), 1, "Torch object expected");
    luaL_argcheck(L, luaT_typename(L, 2), 2, "Torch object expected");

    u1 = lua_touserdata(L, 1);
    u2 = lua_touserdata(L, 2);
    if(*u1 == *u2)
      lua_pushboolean(L, 1);
    else
      lua_pushboolean(L, 0);
  }
  else if(lua_istable(L, 1) && lua_istable(L, 2))
    lua_pushboolean(L, lua_rawequal(L, 1, 2));
  else
    lua_pushboolean(L, 0);
  return 1;
}

int luaT_lua_pointer(lua_State *L)
{
  if(lua_isuserdata(L, 1))
  {
    void **ptr;
    luaL_argcheck(L, luaT_typename(L, 1), 1, "Torch object expected");
    ptr = lua_touserdata(L, 1);
    lua_pushnumber(L, (long)(*ptr));
    return 1;
  }
  else if(lua_istable(L, 1) || lua_isthread(L, 1) || lua_isfunction(L, 1))
  {
    const void* ptr = lua_topointer(L, 1);
    lua_pushnumber(L, (long)(ptr));
    return 1;
  }
  else
    luaL_error(L, "Torch object, table, thread or function expected");

  return 0;
}

int luaT_lua_setenv(lua_State *L)
{
  if(!lua_isfunction(L, 1) && !lua_isuserdata(L, 1))
    luaL_typerror(L, 1, "function or userdata");
  luaL_checktype(L, 2, LUA_TTABLE);
  lua_setfenv(L, 1);
  return 0;
}

int luaT_lua_getenv(lua_State *L)
{
  if(!lua_isfunction(L, 1) && !lua_isuserdata(L, 1))
    luaL_typerror(L, 1, "function or userdata");
  lua_getfenv(L, 1);
  return 1;
}

int luaT_lua_getmetatable(lua_State *L)
{
  const char *tname = luaL_checkstring(L, 1);
  if(luaT_pushmetatable(L, tname))
    return 1;
  return 0;
}

int luaT_lua_version(lua_State *L)
{
  luaL_checkany(L, 1);

  if(lua_getmetatable(L, 1))
  {
    lua_pushstring(L, "__version");
    lua_rawget(L, -2);
    return 1;
  }
  return 0;
}

int luaT_lua_setmetatable(lua_State *L)
{
  const char *tname = luaL_checkstring(L, 2);
  luaL_checktype(L, 1, LUA_TTABLE);

  if(!luaT_pushmetatable(L, tname))
    luaL_error(L, "unknown typename %s\n", tname);
  lua_setmetatable(L, 1);

  return 1;
}

/* metatable operator methods */
static int luaT_mt__index(lua_State *L)
{
  if(!lua_getmetatable(L, 1))
    luaL_error(L, "critical internal indexing error: no metatable found");

  if(!lua_istable(L, -1))
    luaL_error(L, "critical internal indexing error: not a metatable");

  /* test for __index__ method first */
  lua_getfield(L, -1, "__index__");
  if(!lua_isnil(L, -1))
  {
    int result;

    if(!lua_isfunction(L, -1))
      luaL_error(L, "critical internal indexing error: __index__ is not a function");

    lua_pushvalue(L, 1);
    lua_pushvalue(L, 2);

    lua_call(L, 2, LUA_MULTRET); /* DEBUG: risque: faut vraiment retourner 1 ou 2 valeurs... */

    result = lua_toboolean(L, -1);
    lua_pop(L, 1);

    if(result)
      return 1;

    /* on the stack: 1. the object 2. the value 3. the metatable */
    /* apparently, __index wants only one element returned */
    /* return lua_gettop(L)-3; */

  }
  else
    lua_pop(L, 1); /* remove nil __index__ on the stack */

  lua_pushvalue(L, 2);
  lua_gettable(L, -2);

  return 1;
}

static int luaT_mt__newindex(lua_State *L)
{
  if(!lua_getmetatable(L, 1))
    luaL_error(L, "critical internal indexing error: no metatable found");

  if(!lua_istable(L, -1))
    luaL_error(L, "critical internal indexing error: not a metatable");

  /* test for __newindex__ method first */
  lua_getfield(L, -1, "__newindex__");
  if(!lua_isnil(L, -1))
  {
    int result;

    if(!lua_isfunction(L, -1))
      luaL_error(L, "critical internal indexing error: __newindex__ is not a function");

    lua_pushvalue(L, 1);
    lua_pushvalue(L, 2);
    lua_pushvalue(L, 3);

    lua_call(L, 3, 1); /* DEBUG: risque: faut vraiment retourner qqch */

    result = lua_toboolean(L, -1);
    lua_pop(L, 1);

    if(result)
      return 0;
  }
  else
    lua_pop(L, 1); /* remove nil __newindex__ on the stack */

  lua_pop(L, 1);    /* pop the metatable */
  if(lua_istable(L, 1))
    lua_rawset(L, 1);
  else
    luaL_error(L, "the class %s cannot be indexed", luaT_typename(L, 1));

  return 0;
}

/* note: check dans metatable pour ca, donc necessaire */
#define MT_DECLARE_OPERATOR(NAME, NIL_BEHAVIOR)                     \
  int luaT_mt__##NAME(lua_State *L)                                 \
  {                                                                 \
    if(!lua_getmetatable(L, 1))                                     \
      luaL_error(L, "internal error in __" #NAME ": no metatable"); \
                                                                    \
    lua_getfield(L, -1, "__" #NAME "__");                           \
    if(lua_isnil(L, -1))                                                \
    {                                                                   \
      NIL_BEHAVIOR;                                                     \
    }                                                                   \
    else                                                                \
    {                                                                   \
      if(lua_isfunction(L, -1))                                         \
      {                                                                 \
        lua_insert(L, 1); /* insert function */                         \
        lua_pop(L, 1); /* remove metatable */                           \
        lua_call(L, lua_gettop(L)-1, LUA_MULTRET); /* we return the result of the call */ \
        return lua_gettop(L);                                           \
      }                                                                 \
      /* we return the thing the user left in __tostring__ */           \
    }                                                                   \
    return 0;                                                           \
  }

MT_DECLARE_OPERATOR(tostring,
                    lua_pushstring(L, luaT_typename(L, 1));
                    return 1;)
MT_DECLARE_OPERATOR(add, luaL_error(L, "%s has no addition operator", luaT_typename(L, 1)))
MT_DECLARE_OPERATOR(sub, luaL_error(L, "%s has no substraction operator", luaT_typename(L, 1)))
MT_DECLARE_OPERATOR(mul, luaL_error(L, "%s has no multiplication operator", luaT_typename(L, 1)))
MT_DECLARE_OPERATOR(div, luaL_error(L, "%s has no division operator", luaT_typename(L, 1)))
MT_DECLARE_OPERATOR(mod, luaL_error(L, "%s has no modulo operator", luaT_typename(L, 1)))
MT_DECLARE_OPERATOR(pow, luaL_error(L, "%s has no power operator", luaT_typename(L, 1)))
MT_DECLARE_OPERATOR(unm, luaL_error(L, "%s has no negation operator", luaT_typename(L, 1)))
MT_DECLARE_OPERATOR(concat, luaL_error(L, "%s has no concat operator", luaT_typename(L, 1)))
MT_DECLARE_OPERATOR(len, luaL_error(L, "%s has no length operator", luaT_typename(L, 1)))
MT_DECLARE_OPERATOR(eq,
                    lua_settop(L, 2);
                    lua_pushcfunction(L, luaT_lua_isequal);
                    lua_insert(L, 1);
                    lua_call(L, 2, 1);
                    return 1;)
MT_DECLARE_OPERATOR(lt, luaL_error(L, "%s has no lower than operator", luaT_typename(L, 1)))
MT_DECLARE_OPERATOR(le, luaL_error(L, "%s has no lower or equal than operator", luaT_typename(L, 1)))
MT_DECLARE_OPERATOR(call, luaL_error(L, "%s has no call operator", luaT_typename(L, 1)))


/* constructor metatable methods */
int luaT_cmt__call(lua_State *L)
{
  if(!lua_istable(L, 1))
    luaL_error(L, "internal error in __call: not a constructor table");

  if(!lua_getmetatable(L, 1))
    luaL_error(L, "internal error in __call: no metatable available");

  lua_pushstring(L, "__new");
  lua_rawget(L, -2);

  if(lua_isnil(L, -1))
    luaL_error(L, "no constructor available");

  lua_remove(L, 1); /* remove constructor atable */
  lua_insert(L, 1); /* insert constructor */
  lua_pop(L, 1);    /* remove fancy metatable */

  lua_call(L, lua_gettop(L)-1, LUA_MULTRET);
  return lua_gettop(L);
}

int luaT_cmt__newindex(lua_State *L)
{
  if(!lua_istable(L, 1))
    luaL_error(L, "internal error in __newindex: not a constructor table");

  if(!lua_getmetatable(L, 1))
    luaL_error(L, "internal error in __newindex: no metatable available");

  lua_pushstring(L, "__metatable");
  lua_rawget(L, -2);

  if(!lua_istable(L, -1))
    luaL_error(L, "internal error in __newindex: no metaclass available");

  lua_insert(L, 2);
  lua_pop(L, 1); /* remove the metatable over the constructor table */

  lua_rawset(L, -3);

  return 0;
}

/******************** deprecated functions ********************/
int luaT_pushmetaclass(lua_State *L, const char *tname)
{
  return luaT_pushmetatable(L, tname);
}

const char* luaT_id(lua_State *L, int ud)
{
  return luaT_typename(L, ud);
}

const char* luaT_id2typename(lua_State *L, const char *id)
{
  return id;
}

const char* luaT_typename2id(lua_State *L, const char *tname)
{
  return luaT_typenameid(L, tname);
}

int luaT_getmetaclass(lua_State *L, int index)
{
  return lua_getmetatable(L, index);
}

const char* luaT_checktypename2id(lua_State *L, const char *tname)
{
  const char* id = luaT_typenameid(L, tname);
  if(!id)
    luaL_error(L, "unknown class <%s>", tname);
  return id;
}

void luaT_registeratid(lua_State *L, const struct luaL_Reg *methods, const char *id)
{
  luaT_registeratname(L, methods, id);
}

/**************************************************************/
