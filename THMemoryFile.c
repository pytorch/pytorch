#include "THMemoryFile.h"
#include "THFilePrivate.h"

typedef struct THMemoryFile__
{
    THFile file;
    THCharStorage *storage;
    long size;
    long position;

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

static void THMemoryFile_grow(THMemoryFile *self, long size)
{
  long missingSpace;

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
  static long THMemoryFile_read##TYPEC(THFile *self, TYPE *data, long n) \
  {                                                                     \
    THMemoryFile *mfself = (THMemoryFile*)self;                         \
    long nread = 0L;                                                    \
                                                                        \
    THArgCheck(mfself->storage != NULL, 1, "attempt to use a closed file");     \
    THArgCheck(mfself->file.isReadable, 1, "attempt to read in a write-only file"); \
                                                                        \
    if(mfself->file.isBinary)                                           \
    {                                                                   \
      long nByte = sizeof(TYPE)*n;                                      \
      long nByteRemaining = (mfself->position + nByte <= mfself->size ? nByte : mfself->size-mfself->position); \
      nread = nByteRemaining/sizeof(TYPE);                              \
      memmove(data, mfself->storage->data+mfself->position, nread*sizeof(TYPE)); \
      mfself->position += nread*sizeof(TYPE);                           \
    }                                                                   \
    else                                                                \
    {                                                                   \
      long i;                                                           \
      for(i = 0; i < n; i++)                                            \
      {                                                                 \
        long nByteRead = 0;                                             \
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
  static long THMemoryFile_write##TYPEC(THFile *self, TYPE *data, long n) \
  {                                                                     \
    THMemoryFile *mfself = (THMemoryFile*)self;                         \
    long nread = 0L;                                                    \
                                                                        \
    THArgCheck(mfself->storage != NULL, 1, "attempt to use a closed file");     \
    THArgCheck(mfself->file.isWritable, 1, "attempt to write in a read-only file"); \
                                                                        \
    if(mfself->file.isBinary)                                           \
    {                                                                   \
      long nByte = sizeof(TYPE)*n;                                      \
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
      long i;                                                           \
      for(i = 0; i < n; i++)                                            \
      {                                                                 \
        long nByteWritten;                                              \
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

static void THMemoryFile_seek(THFile *self, long position)
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
      THError("unable to seek at position %d", position);
  }
}

static void THMemoryFile_seekEnd(THFile *self)
{
  THMemoryFile *mfself = (THMemoryFile*)self;
  THArgCheck(mfself->storage != NULL, 1, "attempt to use a closed file");

  mfself->position = mfself->size;
}

static long THMemoryFile_position(THFile *self)
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
                   long ret = (mfself->position + n <= mfself->size ? n : mfself->size-mfself->position);  \
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
                   long ret = (mfself->position + n <= mfself->size ? n : mfself->size-mfself->position);  \
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

READ_WRITE_METHODS(long, Long,
                   int nByteRead_; int ret = sscanf(mfself->storage->data+mfself->position, "%ld%n", &data[i], &nByteRead_); nByteRead = nByteRead_; if(ret <= 0) break; else nread++,
                   nByteWritten = snprintf(mfself->storage->data+mfself->position, mfself->storage->size-mfself->position, "%ld", data[i]),
                   1)

READ_WRITE_METHODS(float, Float,
                   int nByteRead_; int ret = sscanf(mfself->storage->data+mfself->position, "%g%n", &data[i], &nByteRead_); nByteRead = nByteRead_; if(ret <= 0) break; else nread++,
                   nByteWritten = snprintf(mfself->storage->data+mfself->position, mfself->storage->size-mfself->position, "%g", data[i]),
                   1)

READ_WRITE_METHODS(double, Double,
                   int nByteRead_; int ret = sscanf(mfself->storage->data+mfself->position, "%lg%n", &data[i], &nByteRead_); nByteRead = nByteRead_; if(ret <= 0) break; else nread++,
                   nByteWritten = snprintf(mfself->storage->data+mfself->position, mfself->storage->size-mfself->position, "%lg", data[i]),
                   1)

static char* THMemoryFile_cloneString(const char *str, long size)
{
  char *cstr = THAlloc(size);
  memcpy(cstr, str, size);
  return cstr;
}

static long THMemoryFile_readString(THFile *self, const char *format, char **str_)
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
    long str_size = mfself->size-mfself->position;

    *str_ = THMemoryFile_cloneString(mfself->storage->data+mfself->position, str_size);
    mfself->position = mfself->size;

    return str_size;
  }
  else
  {
    char *p = mfself->storage->data+mfself->position;
    long posEol = -1;
    long i;
    for(i = 0L; i < mfself->size-mfself->position; i++)
    {
      if(p[i] == '\n')
      {
        posEol = i;
        break;
      }
    }

    if(posEol >= 0)
    {
      *str_ = THMemoryFile_cloneString(mfself->storage->data+mfself->position, posEol);
      mfself->position += posEol+1;
      return posEol;
    }
    else /* well, we read all! */
    {
      long str_size = mfself->size-mfself->position;

      *str_ = THMemoryFile_cloneString(mfself->storage->data+mfself->position, str_size);
      mfself->position = mfself->size;

      return str_size;
    }
  }

  *str_ = NULL;
  return 0;
}

static long THMemoryFile_writeString(THFile *self, const char *str, long size)
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
    THMemoryFile_readString,

    THMemoryFile_writeByte,
    THMemoryFile_writeChar,
    THMemoryFile_writeShort,
    THMemoryFile_writeInt,
    THMemoryFile_writeLong,
    THMemoryFile_writeFloat,
    THMemoryFile_writeDouble,
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
