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

#ifdef __cplusplus
# define TORCH_EXTERNC extern "C"
#else
# define TORCH_EXTERNC extern
#endif

#ifdef WIN32
# ifdef torch_EXPORTS
#  define TORCH_API TORCH_EXTERNC __declspec(dllexport)
# else
#  define TORCH_API TORCH_EXTERNC __declspec(dllimport)
# endif
#else
# define TORCH_API TORCH_EXTERNC
#endif


TORCH_API THLongStorage* torch_checklongargs(lua_State *L, int index);
TORCH_API int torch_islongargs(lua_State *L, int index);
TORCH_API const char* torch_getdefaulttensortype(lua_State *L);

#endif
