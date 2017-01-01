#include "THMemoryFile.h"
#include "THFilePrivate.h"
#include "stdint.h"

typedef struct THMemoryFile__
{
    THFile file;
    THCharStorage *storage;
    size_t size;
    size_t position;
	int longSize;

} THMemoryFile;

static int THMemoryFile_isOpened(THFile *self)
{
  THMemoryFile *mfself = (THMemoryFile*)self;
  return (mfself->storage != NULL);
}

static char *THMemoryFile_strnextspace(char *str_, char *c_)
{
  char c;

  while( (c = *str_) )
  {
    if( (c != ' ') && (c != '\n') && (c != ':') && (c != ';') )
      break;
    str_++;
  }

  while( (c = *str_) )
  {
    if( (c == ' ') || (c == '\n') || (c == ':') || (c == ';') )
    {
      *c_ = c;
      *str_ = '\0';
      return(str_);
    }
    str_++;
  }
  return NULL;
}

static void THMemoryFile_grow(THMemoryFile *self, size_t size)
{
  size_t missingSpace;

  if(size <= self->size)
    return;
  else
  {
    if(size < self->storage->size) /* note the "<" and not "<=" */
    {
      self->size = size;
      self->storage->data[self->size] = '\0';
      return;
    }
  }

  missingSpace = size-self->storage->size+1; /* +1 for the '\0' */
  THCharStorage_resize(self->storage, (self->storage->size/2 > missingSpace ?
                                       self->storage->size + (self->storage->size/2)
                                       : self->storage->size + missingSpace));
}

static int THMemoryFile_mode(const char *mode, int *isReadable, int *isWritable)
{
  *isReadable = 0;
  *isWritable = 0;
  if(strlen(mode) == 1)
  {
    if(*mode == 'r')
    {
      *isReadable = 1;
      return 1;
    }
    else if(*mode == 'w')
    {
      *isWritable = 1;
      return 1;
    }
  }
  else if(strlen(mode) == 2)
  {
    if(mode[0] == 'r' && mode[1] == 'w')
    {
      *isReadable = 1;
      *isWritable = 1;
      return 1;
    }
  }
  return 0;
}

/********************************************************/

#define READ_WRITE_METHODS(TYPE, TYPEC, ASCII_READ_ELEM, ASCII_WRITE_ELEM, INSIDE_SPACING) \
  static size_t THMemoryFile_read##TYPEC(THFile *self, TYPE *data, size_t n) \
  {                                                                     \
    THMemoryFile *mfself = (THMemoryFile*)self;                         \
    size_t nread = 0;                                                    \
                                                                        \
    THArgCheck(mfself->storage != NULL, 1, "attempt to use a closed file");     \
    THArgCheck(mfself->file.isReadable, 1, "attempt to read in a write-only file"); \
                                                                        \
    if (n == 0)                                                         \
        return 0;                                                       \
                                                                        \
    if(mfself->file.isBinary)                                           \
    {                                                                   \
      size_t nByte = sizeof(TYPE)*n;                                      \
      size_t nByteRemaining = (mfself->position + nByte <= mfself->size ? nByte : mfself->size-mfself->position); \
      nread = nByteRemaining/sizeof(TYPE);                              \
      memmove(data, mfself->storage->data+mfself->position, nread*sizeof(TYPE)); \
      mfself->position += nread*sizeof(TYPE);                           \
    }                                                                   \
    else                                                                \
    {                                                                   \
      size_t i;                                                           \
      for(i = 0; i < n; i++)                                            \
      {                                                                 \
        size_t nByteRead = 0;                                             \
        char spaceChar = 0;                                             \
        char *spacePtr = THMemoryFile_strnextspace(mfself->storage->data+mfself->position, &spaceChar); \
        ASCII_READ_ELEM;                                                \
        if(ret == EOF)                                                  \
        {                                                               \
          while(mfself->storage->data[mfself->position])                \
            mfself->position++;                                         \
        }                                                               \
        else                                                            \
          mfself->position += nByteRead;                                \
        if(spacePtr)                                                    \
          *spacePtr = spaceChar;                                        \
      }                                                                 \
      if(mfself->file.isAutoSpacing && (n > 0))                         \
      {                                                                 \
        if( (mfself->position < mfself->size) && (mfself->storage->data[mfself->position] == '\n') ) \
          mfself->position++;                                           \
      }                                                                 \
    }                                                                   \
                                                                        \
    if(nread != n)                                                      \
    {                                                                   \
      mfself->file.hasError = 1; /* shouldn't we put hasError to 0 all the time ? */ \
      if(!mfself->file.isQuiet)                                         \
        THError("read error: read %d blocks instead of %d", nread, n);  \
    }                                                                   \
                                                                        \
    return nread;                                                       \
  }                                                                     \
                                                                        \
  static size_t THMemoryFile_write##TYPEC(THFile *self, TYPE *data, size_t n) \
  {                                                                     \
    THMemoryFile *mfself = (THMemoryFile*)self;                         \
                                                                        \
    THArgCheck(mfself->storage != NULL, 1, "attempt to use a closed file");     \
    THArgCheck(mfself->file.isWritable, 1, "attempt to write in a read-only file"); \
                                                                        \
    if (n == 0)                                                         \
        return 0;                                                       \
                                                                        \
    if(mfself->file.isBinary)                                           \
    {                                                                   \
      size_t nByte = sizeof(TYPE)*n;                                      \
      THMemoryFile_grow(mfself, mfself->position+nByte);                \
      memmove(mfself->storage->data+mfself->position, data, nByte);     \
      mfself->position += nByte;                                        \
      if(mfself->position > mfself->size)                               \
      {                                                                 \
        mfself->size = mfself->position;                                \
        mfself->storage->data[mfself->size] = '\0';                     \
      }                                                                 \
    }                                                                   \
    else                                                                \
    {                                                                   \
      size_t i;                                                           \
      for(i = 0; i < n; i++)                                            \
      {                                                                 \
        ssize_t nByteWritten;                                           \
        while (1)                                                       \
        {                                                               \
          ASCII_WRITE_ELEM;                                             \
          if( (nByteWritten > -1) && (nByteWritten < mfself->storage->size-mfself->position) ) \
          {                                                             \
            mfself->position += nByteWritten;                           \
            break;                                                      \
          }                                                             \
          THMemoryFile_grow(mfself, mfself->storage->size + (mfself->storage->size/2) + 2); \
        }                                                               \
        if(mfself->file.isAutoSpacing)                                  \
        {                                                               \
          if(i < n-1)                                                   \
          {                                                             \
            THMemoryFile_grow(mfself, mfself->position+1);              \
            sprintf(mfself->storage->data+mfself->position, " ");       \
            mfself->position++;                                         \
          }                                                             \
          if(i == n-1)                                                  \
          {                                                             \
            THMemoryFile_grow(mfself, mfself->position+1);              \
            sprintf(mfself->storage->data+mfself->position, "\n");      \
            mfself->position++;                                         \
          }                                                             \
        }                                                               \
      }                                                                 \
      if(mfself->position > mfself->size)                               \
      {                                                                 \
        mfself->size = mfself->position;                                \
        mfself->storage->data[mfself->size] = '\0';                     \
      }                                                                 \
    }                                                                   \
                                                                        \
    return n;                                                           \
  }


void THMemoryFile_longSize(THFile *self, int size)
{
  THMemoryFile *dfself = (THMemoryFile*)(self);
  THArgCheck(size == 0 || size == 4 || size == 8, 1, "Invalid long size specified");
  dfself->longSize = size;
}

THCharStorage *THMemoryFile_storage(THFile *self)
{
  THMemoryFile *mfself = (THMemoryFile*)self;
  THArgCheck(mfself->storage != NULL, 1, "attempt to use a closed file");

  THCharStorage_resize(mfself->storage, mfself->size+1);

  return mfself->storage;
}

static void THMemoryFile_synchronize(THFile *self)
{
  THMemoryFile *mfself = (THMemoryFile*)self;
  THArgCheck(mfself->storage != NULL, 1, "attempt to use a closed file");
}

static void THMemoryFile_seek(THFile *self, size_t position)
{
  THMemoryFile *mfself = (THMemoryFile*)self;

  THArgCheck(mfself->storage != NULL, 1, "attempt to use a closed file");
  THArgCheck(position >= 0, 2, "position must be positive");

  if(position <= mfself->size)
    mfself->position = position;
  else
  {
    mfself->file.hasError = 1;
    if(!mfself->file.isQuiet)
      THError("unable to seek at position %zu", position);
  }
}

static void THMemoryFile_seekEnd(THFile *self)
{
  THMemoryFile *mfself = (THMemoryFile*)self;
  THArgCheck(mfself->storage != NULL, 1, "attempt to use a closed file");

  mfself->position = mfself->size;
}

static size_t THMemoryFile_position(THFile *self)
{
  THMemoryFile *mfself = (THMemoryFile*)self;
  THArgCheck(mfself->storage != NULL, 1, "attempt to use a closed file");
  return mfself->position;
}

static void THMemoryFile_close(THFile *self)
{
  THMemoryFile *mfself = (THMemoryFile*)self;
  THArgCheck(mfself->storage != NULL, 1, "attempt to use a closed file");
  THCharStorage_free(mfself->storage);
  mfself->storage = NULL;
}

static void THMemoryFile_free(THFile *self)
{
  THMemoryFile *mfself = (THMemoryFile*)self;

  if(mfself->storage)
    THCharStorage_free(mfself->storage);

  THFree(mfself);
}

/* READ_WRITE_METHODS(bool, Bool, */
/*                    int value = 0; int ret = sscanf(mfself->storage->data+mfself->position, "%d%n", &value, &nByteRead); data[i] = (value ? 1 : 0), */
/*                    int value = (data[i] ? 1 : 0); nByteWritten = snprintf(mfself->storage->data+mfself->position, mfself->storage->size-mfself->position, "%d", value), */
/*                    1) */

READ_WRITE_METHODS(unsigned char, Byte,
                   size_t ret = (mfself->position + n <= mfself->size ? n : mfself->size-mfself->position);  \
                   if(spacePtr) *spacePtr = spaceChar; \
                   nByteRead = ret; \
                   nread = ret; \
                   i = n-1; \
                   memmove(data, mfself->storage->data+mfself->position, nByteRead),
                   nByteWritten = (n < mfself->storage->size-mfself->position ? n : -1); \
                   i = n-1; \
                   if(nByteWritten > -1)
                     memmove(mfself->storage->data+mfself->position, data, nByteWritten),
                   0)

/* DEBUG: we should check if %n is count or not as a element (so ret might need to be ret-- on some systems) */
/* Note that we do a trick for char */
READ_WRITE_METHODS(char, Char,
                   size_t ret = (mfself->position + n <= mfself->size ? n : mfself->size-mfself->position);  \
                   if(spacePtr) *spacePtr = spaceChar; \
                   nByteRead = ret; \
                   nread = ret; \
                   i = n-1; \
                   memmove(data, mfself->storage->data+mfself->position, nByteRead),
                   nByteWritten = (n < mfself->storage->size-mfself->position ? n : -1); \
                   i = n-1; \
                   if(nByteWritten > -1)
                     memmove(mfself->storage->data+mfself->position, data, nByteWritten),
                   0)

READ_WRITE_METHODS(short, Short,
                   int nByteRead_; int ret = sscanf(mfself->storage->data+mfself->position, "%hd%n", &data[i], &nByteRead_); nByteRead = nByteRead_; if(ret <= 0) break; else nread++,
                   nByteWritten = snprintf(mfself->storage->data+mfself->position, mfself->storage->size-mfself->position, "%hd", data[i]),
                   1)

READ_WRITE_METHODS(int, Int,
                   int nByteRead_; int ret = sscanf(mfself->storage->data+mfself->position, "%d%n", &data[i], &nByteRead_); nByteRead = nByteRead_; if(ret <= 0) break; else nread++,
                   nByteWritten = snprintf(mfself->storage->data+mfself->position, mfself->storage->size-mfself->position, "%d", data[i]),
                   1)

READ_WRITE_METHODS(float, Float,
                   int nByteRead_; int ret = sscanf(mfself->storage->data+mfself->position, "%g%n", &data[i], &nByteRead_); nByteRead = nByteRead_; if(ret <= 0) break; else nread++,
                   nByteWritten = snprintf(mfself->storage->data+mfself->position, mfself->storage->size-mfself->position, "%.9g", data[i]),
                   1)

READ_WRITE_METHODS(THHalf, Half,
                   int nByteRead_; float buf; \
                   int ret = sscanf(mfself->storage->data+mfself->position, "%g%n", &buf, &nByteRead_); \
                   data[i] = TH_float2half(buf); nByteRead = nByteRead_; if(ret <= 0) break; else nread++,
                   nByteWritten = snprintf(mfself->storage->data+mfself->position, mfself->storage->size-mfself->position, "%.9g", TH_half2float(data[i])),
                   1)

READ_WRITE_METHODS(double, Double,
                   int nByteRead_; int ret = sscanf(mfself->storage->data+mfself->position, "%lg%n", &data[i], &nByteRead_); nByteRead = nByteRead_; if(ret <= 0) break; else nread++,
                   nByteWritten = snprintf(mfself->storage->data+mfself->position, mfself->storage->size-mfself->position, "%.17g", data[i]),
                   1)

int THDiskFile_isLittleEndianCPU(void);

static size_t THMemoryFile_readLong(THFile *self, long *data, size_t n)
{
  THMemoryFile *mfself = (THMemoryFile*)self;
  size_t nread = 0L;

  THArgCheck(mfself->storage != NULL, 1, "attempt to use a closed file");
  THArgCheck(mfself->file.isReadable, 1, "attempt to read in a write-only file");

  if (n == 0)
    return 0;

  if(mfself->file.isBinary)
  {
    if(mfself->longSize == 0 || mfself->longSize == sizeof(long))
    {
      size_t nByte = sizeof(long)*n;
      size_t nByteRemaining = (mfself->position + nByte <= mfself->size ? nByte : mfself->size-mfself->position);
      nread = nByteRemaining/sizeof(long);
      memmove(data, mfself->storage->data+mfself->position, nread*sizeof(long));
      mfself->position += nread*sizeof(long);
    } else if(mfself->longSize == 4)
    {
      size_t nByte = 4*n;
      size_t nByteRemaining = (mfself->position + nByte <= mfself->size ? nByte : mfself->size-mfself->position);
      int32_t *storage = (int32_t *)(mfself->storage->data + mfself->position);
      nread = nByteRemaining/4;
      size_t i;
      for(i = 0; i < nread; i++)
        data[i] = storage[i];
      mfself->position += nread*4;
    }
    else /* if(mfself->longSize == 8) */
    {
      int big_endian = !THDiskFile_isLittleEndianCPU();
      size_t nByte = 8*n;
      int32_t *storage = (int32_t *)(mfself->storage->data + mfself->position);
      size_t nByteRemaining = (mfself->position + nByte <= mfself->size ? nByte : mfself->size-mfself->position);
      nread = nByteRemaining/8;
      size_t i;
      for(i = 0; i < nread; i++)
        data[i] = storage[2*i + big_endian];
      mfself->position += nread*8;
    }
  }
  else
  {
    size_t i;
    for(i = 0; i < n; i++)
    {
      size_t nByteRead = 0;
      char spaceChar = 0;
      char *spacePtr = THMemoryFile_strnextspace(mfself->storage->data+mfself->position, &spaceChar);
      int nByteRead_; int ret = sscanf(mfself->storage->data+mfself->position, "%ld%n", &data[i], &nByteRead_); nByteRead = nByteRead_; if(ret <= 0) break; else nread++;
      if(ret == EOF)
      {
        while(mfself->storage->data[mfself->position])
          mfself->position++;
      }
      else
        mfself->position += nByteRead;
      if(spacePtr)
        *spacePtr = spaceChar;
    }
    if(mfself->file.isAutoSpacing && (n > 0))
    {
      if( (mfself->position < mfself->size) && (mfself->storage->data[mfself->position] == '\n') )
        mfself->position++;
    }
  }

  if(nread != n)
  {
    mfself->file.hasError = 1; /* shouldn't we put hasError to 0 all the time ? */
    if(!mfself->file.isQuiet)
      THError("read error: read %d blocks instead of %d", nread, n);
  }

  return nread;
}

static size_t THMemoryFile_writeLong(THFile *self, long *data, size_t n)
{
  THMemoryFile *mfself = (THMemoryFile*)self;

  THArgCheck(mfself->storage != NULL, 1, "attempt to use a closed file");
  THArgCheck(mfself->file.isWritable, 1, "attempt to write in a read-only file");

  if (n == 0)
    return 0;

  if(mfself->file.isBinary)
  {
    if(mfself->longSize == 0 || mfself->longSize == sizeof(long))
    {
      size_t nByte = sizeof(long)*n;
      THMemoryFile_grow(mfself, mfself->position+nByte);
      memmove(mfself->storage->data+mfself->position, data, nByte);
      mfself->position += nByte;
    } else if(mfself->longSize == 4)
    {
      size_t nByte = 4*n;
      THMemoryFile_grow(mfself, mfself->position+nByte);
      int32_t *storage = (int32_t *)(mfself->storage->data + mfself->position);
      size_t i;
      for(i = 0; i < n; i++)
        storage[i] = data[i];
      mfself->position += nByte;
    }
    else /* if(mfself->longSize == 8) */
    {
      int big_endian = !THDiskFile_isLittleEndianCPU();
      size_t nByte = 8*n;
      THMemoryFile_grow(mfself, mfself->position+nByte);
      int32_t *storage = (int32_t *)(mfself->storage->data + mfself->position);
      size_t i;
      for(i = 0; i < n; i++)
      {
        storage[2*i + !big_endian] = 0;
        storage[2*i + big_endian] = data[i];
      }
      mfself->position += nByte;
    }
    if(mfself->position > mfself->size)
    {
      mfself->size = mfself->position;
      mfself->storage->data[mfself->size] = '\0';
    }
  }
  else
  {
    size_t i;
    for(i = 0; i < n; i++)
    {
      ssize_t nByteWritten;
      while (1)
      {
        nByteWritten = snprintf(mfself->storage->data+mfself->position, mfself->storage->size-mfself->position, "%ld", data[i]);
        if( (nByteWritten > -1) && (nByteWritten < mfself->storage->size-mfself->position) )
        {
          mfself->position += nByteWritten;
          break;
        }
        THMemoryFile_grow(mfself, mfself->storage->size + (mfself->storage->size/2) + 2);
      }
      if(mfself->file.isAutoSpacing)
      {
        if(i < n-1)
        {
          THMemoryFile_grow(mfself, mfself->position+1);
          sprintf(mfself->storage->data+mfself->position, " ");
          mfself->position++;
        }
        if(i == n-1)
        {
          THMemoryFile_grow(mfself, mfself->position+1);
          sprintf(mfself->storage->data+mfself->position, "\n");
          mfself->position++;
        }
      }
    }
    if(mfself->position > mfself->size)
    {
      mfself->size = mfself->position;
      mfself->storage->data[mfself->size] = '\0';
    }
  }

  return n;
}

static char* THMemoryFile_cloneString(const char *str, ptrdiff_t size)
{
  char *cstr = THAlloc(size);
  memcpy(cstr, str, size);
  return cstr;
}

static size_t THMemoryFile_readString(THFile *self, const char *format, char **str_)
{
  THMemoryFile *mfself = (THMemoryFile*)self;

  THArgCheck(mfself->storage != NULL, 1, "attempt to use a closed file");
  THArgCheck(mfself->file.isReadable, 1, "attempt to read in a write-only file");
  THArgCheck((strlen(format) >= 2 ? (format[0] == '*') && (format[1] == 'a' || format[1] == 'l') : 0), 2, "format must be '*a' or '*l'");

  if(mfself->position == mfself->size) /* eof ? */
  {
    mfself->file.hasError = 1;
    if(!mfself->file.isQuiet)
      THError("read error: read 0 blocks instead of 1");

    *str_ = NULL;
    return 0;
  }

  if(format[1] == 'a')
  {
    size_t str_size = mfself->size-mfself->position;

    *str_ = THMemoryFile_cloneString(mfself->storage->data+mfself->position, str_size);
    mfself->position = mfself->size;

    return str_size;
  }
  else
  {
    char *p = mfself->storage->data+mfself->position;
    int eolFound = 0;
    size_t posEol;
    size_t i;
    for(i = 0; i < mfself->size-mfself->position; i++)
    {
      if(p[i] == '\n')
      {
        posEol = i;
        eolFound = 1;
        break;
      }
    }

    if(eolFound)
    {
      *str_ = THMemoryFile_cloneString(mfself->storage->data+mfself->position, posEol);
      mfself->position += posEol+1;
      return posEol;
    }
    else /* well, we read all! */
    {
      size_t str_size = mfself->size-mfself->position;

      *str_ = THMemoryFile_cloneString(mfself->storage->data+mfself->position, str_size);
      mfself->position = mfself->size;

      return str_size;
    }
  }

  *str_ = NULL;
  return 0;
}

static size_t THMemoryFile_writeString(THFile *self, const char *str, size_t size)
{
  THMemoryFile *mfself = (THMemoryFile*)self;

  THArgCheck(mfself->storage != NULL, 1, "attempt to use a closed file");
  THArgCheck(mfself->file.isWritable, 1, "attempt to write in a read-only file");

  THMemoryFile_grow(mfself, mfself->position+size);
  memmove(mfself->storage->data+mfself->position, str, size);
  mfself->position += size;
  if(mfself->position > mfself->size)
  {
    mfself->size = mfself->position;
    mfself->storage->data[mfself->size] = '\0';
  }

  return size;
}

THFile *THMemoryFile_newWithStorage(THCharStorage *storage, const char *mode)
{
  static struct THFileVTable vtable = {
    THMemoryFile_isOpened,

    THMemoryFile_readByte,
    THMemoryFile_readChar,
    THMemoryFile_readShort,
    THMemoryFile_readInt,
    THMemoryFile_readLong,
    THMemoryFile_readFloat,
    THMemoryFile_readDouble,
    THMemoryFile_readHalf,
    THMemoryFile_readString,

    THMemoryFile_writeByte,
    THMemoryFile_writeChar,
    THMemoryFile_writeShort,
    THMemoryFile_writeInt,
    THMemoryFile_writeLong,
    THMemoryFile_writeFloat,
    THMemoryFile_writeDouble,
    THMemoryFile_writeHalf,
    THMemoryFile_writeString,

    THMemoryFile_synchronize,
    THMemoryFile_seek,
    THMemoryFile_seekEnd,
    THMemoryFile_position,
    THMemoryFile_close,
    THMemoryFile_free
  };

  THMemoryFile *mfself;
  int isReadable;
  int isWritable;

  if(storage)
  {
    THArgCheck(storage->data[storage->size-1] == '\0', 1, "provided CharStorage must be terminated by 0");
    THArgCheck(THMemoryFile_mode(mode, &isReadable, &isWritable), 2, "file mode should be 'r','w' or 'rw'");
    THCharStorage_retain(storage);
  }
  else
  {
    THArgCheck(THMemoryFile_mode(mode, &isReadable, &isWritable), 2, "file mode should be 'r','w' or 'rw'");
    storage = THCharStorage_newWithSize(1);
    storage->data[0] = '\0';
  }

  mfself = THAlloc(sizeof(THMemoryFile));

  mfself->storage = storage;
  mfself->size = (storage ? storage->size-1 : 0);
  mfself->position = 0;
  mfself->longSize = 0;

  mfself->file.vtable = &vtable;
  mfself->file.isQuiet = 0;
  mfself->file.isReadable = isReadable;
  mfself->file.isWritable = isWritable;
  mfself->file.isBinary = 0;
  mfself->file.isAutoSpacing = 1;
  mfself->file.hasError = 0;

  return (THFile*)mfself;
}

THFile *THMemoryFile_new(const char *mode)
{
  return THMemoryFile_newWithStorage(NULL, mode);
}
