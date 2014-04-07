#include "THFile.h"
#include "THFilePrivate.h"

#define IMPLEMENT_THFILE_RW(TYPEC, TYPE)                          \
  long THFile_read##TYPEC##Raw(THFile *self, TYPE *data, long n)  \
  {                                                               \
    return (*self->vtable->read##TYPEC)(self, data, n);           \
  }                                                               \
                                                                  \
  long THFile_write##TYPEC##Raw(THFile *self, TYPE *data, long n) \
  {                                                               \
    return (*self->vtable->write##TYPEC)(self, data, n);          \
  }
  
IMPLEMENT_THFILE_RW(Byte, unsigned char)
IMPLEMENT_THFILE_RW(Char, char)
IMPLEMENT_THFILE_RW(Short, short)
IMPLEMENT_THFILE_RW(Int, int)
IMPLEMENT_THFILE_RW(Long, long)
IMPLEMENT_THFILE_RW(Float, float)
IMPLEMENT_THFILE_RW(Double, double)

long THFile_readStringRaw(THFile *self, const char *format, char **str_)
{
  return self->vtable->readString(self, format, str_);
}

long THFile_writeStringRaw(THFile *self, const char *str, long size)
{
  return self->vtable->writeString(self, str, size);
}

void THFile_synchronize(THFile *self)
{
  self->vtable->synchronize(self);
}

void THFile_seek(THFile *self, long position)
{
  self->vtable->seek(self, position);
}

void THFile_seekEnd(THFile *self)
{
  self->vtable->seekEnd(self);
}

long THFile_position(THFile *self)
{
  return self->vtable->position(self);
}

void THFile_close(THFile *self)
{
  self->vtable->close(self);
}

void THFile_free(THFile *self)
{
  self->vtable->free(self);
}

int THFile_isOpened(THFile *self)
{
  return self->vtable->isOpened(self);
}

#define IMPLEMENT_THFILE_FLAGS(FLAG) \
  int THFile_##FLAG(THFile *self)    \
  {                                  \
    return self->FLAG;               \
  }

IMPLEMENT_THFILE_FLAGS(isQuiet)
IMPLEMENT_THFILE_FLAGS(isReadable)
IMPLEMENT_THFILE_FLAGS(isWritable)
IMPLEMENT_THFILE_FLAGS(isBinary)
IMPLEMENT_THFILE_FLAGS(isAutoSpacing)
IMPLEMENT_THFILE_FLAGS(hasError)

void THFile_binary(THFile *self)
{
  self->isBinary = 1;
}

void THFile_ascii(THFile *self)
{
  self->isBinary = 0;
}

void THFile_autoSpacing(THFile *self)
{
  self->isAutoSpacing = 1;
}

void THFile_noAutoSpacing(THFile *self)
{
  self->isAutoSpacing = 0;
}

void THFile_quiet(THFile *self)
{
  self->isQuiet = 1;
}

void THFile_pedantic(THFile *self)
{
  self->isQuiet = 0;
}

void THFile_clearError(THFile *self)
{
  self->hasError = 0;
}

#define IMPLEMENT_THFILE_SCALAR(TYPEC, TYPE)                  \
  TYPE THFile_read##TYPEC##Scalar(THFile *self)               \
  {                                                           \
    TYPE scalar;                                              \
    THFile_read##TYPEC##Raw(self, &scalar, 1);                \
    return scalar;                                            \
  }                                                           \
                                                              \
  void THFile_write##TYPEC##Scalar(THFile *self, TYPE scalar) \
  {                                                           \
    THFile_write##TYPEC##Raw(self, &scalar, 1);               \
  }

IMPLEMENT_THFILE_SCALAR(Byte, unsigned char)
IMPLEMENT_THFILE_SCALAR(Char, char)
IMPLEMENT_THFILE_SCALAR(Short, short)
IMPLEMENT_THFILE_SCALAR(Int, int)
IMPLEMENT_THFILE_SCALAR(Long, long)
IMPLEMENT_THFILE_SCALAR(Float, float)
IMPLEMENT_THFILE_SCALAR(Double, double)

#define IMPLEMENT_THFILE_STORAGE(TYPEC, TYPE)                           \
  long THFile_read##TYPEC(THFile *self, TH##TYPEC##Storage *storage)    \
  {                                                                     \
    return THFile_read##TYPEC##Raw(self, storage->data, storage->size); \
  }                                                                     \
                                                                        \
  long THFile_write##TYPEC(THFile *self, TH##TYPEC##Storage *storage)   \
  {                                                                     \
    return THFile_write##TYPEC##Raw(self, storage->data, storage->size); \
  }

IMPLEMENT_THFILE_STORAGE(Byte, unsigned char)
IMPLEMENT_THFILE_STORAGE(Char, char)
IMPLEMENT_THFILE_STORAGE(Short, short)
IMPLEMENT_THFILE_STORAGE(Int, int)
IMPLEMENT_THFILE_STORAGE(Long, long)
IMPLEMENT_THFILE_STORAGE(Float, float)
IMPLEMENT_THFILE_STORAGE(Double, double)
