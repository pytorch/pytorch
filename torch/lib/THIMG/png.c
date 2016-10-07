#include <TH.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#define PNG_DEBUG 3
#include <png.h>

#include "THIMG.h"


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
