#include "general.h"

static int torch_PipeFile_new(lua_State *L)
{
  const char *name = luaL_checkstring(L, 1);
  const char *mode = luaL_optstring(L, 2, "r");
  int isQuiet = luaT_optboolean(L, 3, 0);
  THFile *self = THPipeFile_new(name, mode, isQuiet);

  luaT_pushudata(L, self, "torch.PipeFile");
  return 1;
}

static int torch_PipeFile_free(lua_State *L)
{
  THFile *self = luaT_checkudata(L, 1, "torch.PipeFile");
  THFile_free(self);
  return 0;
}

static int torch_PipeFile___tostring__(lua_State *L)
{
  THFile *self = luaT_checkudata(L, 1, "torch.PipeFile");
  lua_pushfstring(L, "torch.PipeFile on <%s> [status: %s -- mode: %c%c]",
                  THDiskFile_name(self),
                  (THFile_isOpened(self) ? "open" : "closed"),
                  (THFile_isReadable(self) ? 'r' : ' '),
                  (THFile_isWritable(self) ? 'w' : ' '));
  return 1;
}

static const struct luaL_Reg torch_PipeFile__ [] = {
  {"__tostring__", torch_PipeFile___tostring__},
  {NULL, NULL}
};

void torch_PipeFile_init(lua_State *L)
{
  luaT_newmetatable(L, "torch.PipeFile", "torch.DiskFile",
                    torch_PipeFile_new, torch_PipeFile_free, NULL);
  luaL_register(L, NULL, torch_PipeFile__);
  lua_pop(L, 1);
}
