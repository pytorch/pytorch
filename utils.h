#ifndef TORCH_UTILS_INC
#define TORCH_UTILS_INC

#include "luaT.h"
#include "TH.h"

THLongStorage* torch_checklongargs(lua_State *L, int index);
int torch_islongargs(lua_State *L, int index);

void torch_setdefaulttensorid(const void* id);
const void* torch_getdefaulttensorid();

#endif
