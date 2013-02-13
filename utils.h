#ifndef TORCH_UTILS_INC
#define TORCH_UTILS_INC

#include "luaT.h"
#include "TH.h"

#include <lua.h>
#include <lualib.h>

#ifdef LUA_WIN
#else
#include <unistd.h>
#endif

THLongStorage* torch_checklongargs(lua_State *L, int index);
int torch_islongargs(lua_State *L, int index);

const char* torch_getdefaulttensortype(lua_State *L);

#endif
