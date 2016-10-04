
#include <TH.h>
#include <luaT.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#if LUA_VERSION_NUM >= 503
#define luaL_checkint(L,n)      ((int)luaL_checkinteger(L, (n)))
#endif

#define PNG_DEBUG 3
#include <png.h>

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch., Real, Tensor)
#define libpng_(NAME) TH_CONCAT_3(libpng_, Real, NAME)

/*
 * Bookkeeping struct for reading png data from memory
 */
typedef struct {
  unsigned char* buffer;
  png_size_t offset;
  png_size_t length;
} libpng_inmem_buffer;

/*
 * Call back for reading png data from memory
 */
static void
libpng_userReadData(png_structp pngPtrSrc, png_bytep dest, png_size_t length)
{
  libpng_inmem_buffer* src = png_get_io_ptr(pngPtrSrc);
  assert(src->offset+length <= src->length);
  memcpy(dest, src->buffer + src->offset, length);
  src->offset += length;
}

/*
 * Error message wrapper (single member struct to preserve `str` size info)
 */
typedef struct {
  char str[256];
} libpng_errmsg;

/*
 * Custom error handling function (see `png_set_error_fn`)
 */
static void
libpng_error_fn(png_structp png_ptr, png_const_charp error_msg)
{
  libpng_errmsg *errmsg = png_get_error_ptr(png_ptr);
  int max = sizeof(errmsg->str) - 1;
  strncpy(errmsg->str, error_msg, max);
  errmsg->str[max] = '\0';
  longjmp(png_jmpbuf(png_ptr), 1);
}

#include "generic/png.c"
#include "THGenerateAllTypes.h"

DLL_EXPORT int luaopen_libpng(lua_State *L)
{
  libpng_FloatMain_init(L);
  libpng_DoubleMain_init(L);
  libpng_ByteMain_init(L);

  lua_newtable(L);
  lua_pushvalue(L, -1);
  lua_setglobal(L, "libpng");

  lua_newtable(L);
  luaT_setfuncs(L, libpng_DoubleMain__, 0);
  lua_setfield(L, -2, "double");

  lua_newtable(L);
  luaT_setfuncs(L, libpng_FloatMain__, 0);
  lua_setfield(L, -2, "float");

  lua_newtable(L);
  luaT_setfuncs(L, libpng_ByteMain__, 0);
  lua_setfield(L, -2, "byte");

  return 1;
}
