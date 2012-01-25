#include "general.h"

static const void* torch_DiskFile_id = NULL;

static int torch_DiskFile_new(lua_State *L)
{
  const char *name = luaL_checkstring(L, 1);
  const char *mode = luaL_optstring(L, 2, "r");
  int isQuiet = luaT_optboolean(L, 3, 0);
  THFile *self = THDiskFile_new(name, mode, isQuiet);

  luaT_pushudata(L, self, torch_DiskFile_id);
  return 1;
}

static int torch_DiskFile_free(lua_State *L)
{
  THFile *self = luaT_checkudata(L, 1, torch_DiskFile_id);
  THFile_free(self);
  return 0;
}

static int torch_DiskFile_isLittleEndianCPU(lua_State *L)
{
  lua_pushboolean(L, THDiskFile_isLittleEndianCPU());
  return 1;
}

static int torch_DiskFile_isBigEndianCPU(lua_State *L)
{
  lua_pushboolean(L, !THDiskFile_isLittleEndianCPU());
  return 1;
}

static int torch_DiskFile_nativeEndianEncoding(lua_State *L)
{
  THFile *self = luaT_checkudata(L, 1, torch_DiskFile_id);
  THDiskFile_nativeEndianEncoding(self);
  lua_settop(L, 1);
  return 1;
}

static int torch_DiskFile_littleEndianEncoding(lua_State *L)
{
  THFile *self = luaT_checkudata(L, 1, torch_DiskFile_id);
  THDiskFile_littleEndianEncoding(self);
  lua_settop(L, 1);
  return 1;
}

static int torch_DiskFile_bigEndianEncoding(lua_State *L)
{
  THFile *self = luaT_checkudata(L, 1, torch_DiskFile_id);
  THDiskFile_bigEndianEncoding(self);
  lua_settop(L, 1);
  return 1;
}

static int torch_DiskFile___tostring__(lua_State *L)
{
  THFile *self = luaT_checkudata(L, 1, torch_DiskFile_id);
  lua_pushfstring(L, "torch.DiskFile on <%s> [status: %s -- mode %c%c]", 
                  THDiskFile_name(self),
                  (THFile_isOpened(self) ? "open" : "closed"),
                  (THFile_isReadable(self) ? 'r' : ' '),
                  (THFile_isWritable(self) ? 'w' : ' '));

  return 1;
}
static const struct luaL_Reg torch_DiskFile__ [] = {
  {"isLittleEndianCPU", torch_DiskFile_isLittleEndianCPU},
  {"isBigEndianCPU", torch_DiskFile_isBigEndianCPU},
  {"nativeEndianEncoding", torch_DiskFile_nativeEndianEncoding},
  {"littleEndianEncoding", torch_DiskFile_littleEndianEncoding},
  {"bigEndianEncoding", torch_DiskFile_bigEndianEncoding},
  {"__tostring__", torch_DiskFile___tostring__},
  {NULL, NULL}
};

void torch_DiskFile_init(lua_State *L)
{
  torch_DiskFile_id = luaT_newmetatable(L, "torch.DiskFile", "torch.File",
                                        torch_DiskFile_new, torch_DiskFile_free, NULL);
  
  luaL_register(L, NULL, torch_DiskFile__);
  lua_pop(L, 1);
}
