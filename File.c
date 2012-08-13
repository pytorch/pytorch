#include "THFile.h"
#include "luaT.h"

#define IMPLEMENT_TORCH_FILE_FLAG(NAME)                   \
  static int torch_File_##NAME(lua_State *L)              \
  {                                                       \
    THFile *self = luaT_checkudata(L, 1, "torch.File");  \
    lua_pushboolean(L, THFile_##NAME(self));              \
    return 1;                                             \
  }

IMPLEMENT_TORCH_FILE_FLAG(isQuiet)
IMPLEMENT_TORCH_FILE_FLAG(isReadable)
IMPLEMENT_TORCH_FILE_FLAG(isWritable)
IMPLEMENT_TORCH_FILE_FLAG(isBinary)
IMPLEMENT_TORCH_FILE_FLAG(isAutoSpacing)
IMPLEMENT_TORCH_FILE_FLAG(hasError)

#define IMPLEMENT_TORCH_FILE_FUNC(NAME)                   \
  static int torch_File_##NAME(lua_State *L)              \
  {                                                       \
    THFile *self = luaT_checkudata(L, 1, "torch.File");  \
    THFile_##NAME(self);                                  \
    lua_settop(L, 1);                                     \
    return 1;                                             \
  }

IMPLEMENT_TORCH_FILE_FUNC(binary)
IMPLEMENT_TORCH_FILE_FUNC(ascii)
IMPLEMENT_TORCH_FILE_FUNC(autoSpacing)
IMPLEMENT_TORCH_FILE_FUNC(noAutoSpacing)
IMPLEMENT_TORCH_FILE_FUNC(quiet)
IMPLEMENT_TORCH_FILE_FUNC(pedantic)
IMPLEMENT_TORCH_FILE_FUNC(clearError)

IMPLEMENT_TORCH_FILE_FUNC(synchronize)

static int torch_File_seek(lua_State *L)
{
  THFile *self = luaT_checkudata(L, 1, "torch.File");
  long position = luaL_checklong(L, 2)-1;
  THFile_seek(self, position);
  lua_settop(L, 1);
  return 1;
}

IMPLEMENT_TORCH_FILE_FUNC(seekEnd)

static int torch_File_position(lua_State *L)
{
  THFile *self = luaT_checkudata(L, 1, "torch.File");
  lua_pushnumber(L, THFile_position(self)+1);
  return 1;
}

IMPLEMENT_TORCH_FILE_FUNC(close)

#define IMPLEMENT_TORCH_FILE_RW(TYPEC, TYPE)                            \
  static int torch_File_read##TYPEC(lua_State *L)                       \
  {                                                                     \
    THFile *self = luaT_checkudata(L, 1, "torch.File");                \
    int narg = lua_gettop(L);                                           \
                                                                        \
    if(narg == 1)                                                       \
    {                                                                   \
      lua_pushnumber(L, THFile_read##TYPEC##Scalar(self));              \
      return 1;                                                         \
    }                                                                   \
    else if(narg == 2)                                                  \
    {                                                                   \
      if(lua_isnumber(L, 2))                                            \
      {                                                                 \
        long size = lua_tonumber(L, 2);                                 \
        long nread;                                                     \
                                                                        \
        TH##TYPEC##Storage *storage = TH##TYPEC##Storage_newWithSize(size); \
        luaT_pushudata(L, storage, "torch." #TYPEC "Storage");           \
        nread = THFile_read##TYPEC(self, storage);                      \
        if(nread != size)                                               \
          TH##TYPEC##Storage_resize(storage, size);                     \
        return 1;                                                       \
      }                                                                 \
      else if(luaT_toudata(L, 2, "torch." #TYPEC "Storage"))             \
      {                                                                 \
        TH##TYPEC##Storage *storage = luaT_toudata(L, 2, "torch." #TYPEC "Storage"); \
        lua_pushnumber(L, THFile_read##TYPEC(self, storage));           \
        return 1;                                                       \
      }                                                                 \
    }                                                                   \
                                                                        \
    luaL_error(L, "nothing, number, or Storage expected");              \
    return 0;                                                           \
  }                                                                     \
                                                                        \
  static int torch_File_write##TYPEC(lua_State *L)                      \
  {                                                                     \
    THFile *self = luaT_checkudata(L, 1, "torch.File");                \
    int narg = lua_gettop(L);                                           \
                                                                        \
    if(narg == 2)                                                       \
    {                                                                   \
      if(lua_isnumber(L, 2))                                            \
      {                                                                 \
        TYPE value = lua_tonumber(L, 2);                                \
        THFile_write##TYPEC##Scalar(self, (TYPE)value);                 \
        return 0;                                                       \
      }                                                                 \
      else if(luaT_toudata(L, 2, "torch." #TYPEC "Storage"))            \
      {                                                                 \
        TH##TYPEC##Storage *storage = luaT_toudata(L, 2, "torch." #TYPEC "Storage"); \
        lua_pushnumber(L, THFile_write##TYPEC(self, storage));          \
        return 1;                                                       \
      }                                                                 \
    }                                                                   \
                                                                        \
    luaL_error(L, "number, or Storage expected");                       \
    return 0;                                                           \
  }


IMPLEMENT_TORCH_FILE_RW(Byte, unsigned char)
IMPLEMENT_TORCH_FILE_RW(Char, char)
IMPLEMENT_TORCH_FILE_RW(Short, short)
IMPLEMENT_TORCH_FILE_RW(Int, int)
IMPLEMENT_TORCH_FILE_RW(Long, long)
IMPLEMENT_TORCH_FILE_RW(Float, float)
IMPLEMENT_TORCH_FILE_RW(Double, double)

static int torch_File_readString(lua_State *L)
{
  THFile *self = luaT_checkudata(L, 1, "torch.File");
  const char *format = luaL_checkstring(L, 2);
  char *str;
  long size;

  size = THFile_readStringRaw(self, format, &str);
  lua_pushlstring(L, str, size);
  THFree(str);

  return 1;
}

static int torch_File_writeString(lua_State *L)
{
  THFile *self = luaT_checkudata(L, 1, "torch.File");
  const char *str = NULL;
  size_t size;

  luaL_checktype(L, 2, LUA_TSTRING);
  str = lua_tolstring(L, 2, &size);
  lua_pushnumber(L, THFile_writeStringRaw(self, str, (long)size));
  return 1;
}

static const struct luaL_Reg torch_File__ [] = {
  {"isQuiet", torch_File_isQuiet},
  {"isReadable", torch_File_isReadable},
  {"isWritable", torch_File_isWritable},
  {"isBinary", torch_File_isBinary},
  {"isAutoSpacing", torch_File_isAutoSpacing},
  {"hasError", torch_File_hasError},
  {"binary", torch_File_binary},
  {"ascii", torch_File_ascii},
  {"autoSpacing", torch_File_autoSpacing},
  {"noAutoSpacing", torch_File_noAutoSpacing},
  {"quiet", torch_File_quiet},
  {"pedantic", torch_File_pedantic},
  {"clearError", torch_File_clearError},

  /* DEBUG: CHECK DISK FREE & READ/WRITE STRING*/

  {"readByte", torch_File_readByte},
  {"readChar", torch_File_readChar},
  {"readShort", torch_File_readShort},
  {"readInt", torch_File_readInt},
  {"readLong", torch_File_readLong},
  {"readFloat", torch_File_readFloat},
  {"readDouble", torch_File_readDouble},
  {"readString", torch_File_readString},

  {"writeByte", torch_File_writeByte},
  {"writeChar", torch_File_writeChar},
  {"writeShort", torch_File_writeShort},
  {"writeInt", torch_File_writeInt},
  {"writeLong", torch_File_writeLong},
  {"writeFloat", torch_File_writeFloat},
  {"writeDouble", torch_File_writeDouble},
  {"writeString", torch_File_writeString},

  {"synchronize", torch_File_synchronize},
  {"seek", torch_File_seek},
  {"seekEnd", torch_File_seekEnd},
  {"position", torch_File_position},
  {"close", torch_File_close},

  {NULL, NULL}
};

void torch_File_init(lua_State *L)
{
  luaT_newmetatable(L, "torch.File", NULL, NULL, NULL, NULL);
  luaL_register(L, NULL, torch_File__);
  lua_pop(L, 1);
}
