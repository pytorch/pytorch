#include "THAllocator.h"

/* stuff for mapped files */
#ifdef _WIN32
#include <windows.h>
#endif

#if HAVE_MMAP
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif
/* end of stuff for mapped files */

static void *THDefaultAllocator_alloc(void* ctx, long size) {
  return THAlloc(size);
}

static void *THDefaultAllocator_realloc(void* ctx, void* ptr, long size) {
  return THRealloc(ptr, size);
}

static void THDefaultAllocator_free(void* ctx, void* ptr) {
  THFree(ptr);
}

THAllocator THDefaultAllocator = {
  &THDefaultAllocator_alloc,
  &THDefaultAllocator_realloc,
  &THDefaultAllocator_free
};

#if defined(_WIN32) || defined(HAVE_MMAP)

struct THMapAllocatorContext_ {
  char *filename; /* file name */
  int shared; /* is shared or not */
  long size; /* mapped size */
};

THMapAllocatorContext *THMapAllocatorContext_new(const char *filename, int shared)
{
  THMapAllocatorContext *ctx = THAlloc(sizeof(THMapAllocatorContext));

  ctx->filename = THAlloc(strlen(filename)+1);
  strcpy(ctx->filename, filename);
  ctx->shared = shared;
  ctx->size = 0;

  return ctx;
}

long THMapAllocatorContext_size(THMapAllocatorContext *ctx)
{
  return ctx->size;
}

void THMapAllocatorContext_free(THMapAllocatorContext *ctx)
{
  THFree(ctx->filename);
  THFree(ctx);
}

static void *THMapAllocator_alloc(void* ctx_, long size)
{
  THMapAllocatorContext *ctx = ctx_;
  void *data = NULL;

#ifdef _WIN32
  {
    HANDLE hfile;
    HANDLE hmfile;
    DWORD size_hi, size_lo;
    size_t hfilesz;

    /* open file */
    /* FILE_FLAG_RANDOM_ACCESS ? */
    if(ctx->shared)
    {
      hfile = CreateFileA(ctx->filename, GENERIC_READ|GENERIC_WRITE, FILE_SHARE_WRITE|FILE_SHARE_READ, 0, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, 0);
      if (hfile == INVALID_HANDLE_VALUE)
        THError("could not open file <%s> in read-write mode", ctx->filename);
    }
    else
    {
      hfile = CreateFileA(ctx->filename, GENERIC_READ, FILE_SHARE_WRITE|FILE_SHARE_READ, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
      if (hfile == INVALID_HANDLE_VALUE)
        THError("could not open file <%s> in read-only mode", ctx->filename);
    }

    size_lo = GetFileSize(hfile, &size_hi);
    if(sizeof(size_t) > 4)
    {
      hfilesz = ((size_t)size_hi) << 32;
      hfilesz |= size_lo;
    }
    else
      hfilesz = (size_t)(size_lo);

    if(size > 0)
    {
      if(size > hfilesz)
      {
        if(ctx->shared)
        {
#if SIZEOF_SIZE_T > 4
          size_hi = (DWORD)((size) >> 32);
          size_lo = (DWORD)((size) & 0xFFFFFFFF);
#else
          size_hi = 0;
          size_lo = (DWORD)(size);
#endif
          if((SetFilePointer(hfile, size_lo, &size_hi, FILE_BEGIN)) == INVALID_SET_FILE_POINTER)
          {
            CloseHandle(hfile);
            THError("unable to stretch file <%s> to the right size", ctx->filename);
          }
          if(SetEndOfFile(hfile) == 0)
          {
            CloseHandle(hfile);
            THError("unable to write to file <%s>", ctx->filename);
          }
        }
        else
        {
          CloseHandle(hfile);
          THError("file <%s> size is smaller than the required mapping size <%ld>", ctx->filename, size);
        }
      }
    }
    else
      size = hfilesz;

    ctx->size = size; /* if we are here, it must be the right size */

#if SIZEOF_SIZE_T > 4
    size_hi = (DWORD)((ctx->size) >> 32);
    size_lo = (DWORD)((ctx->size) & 0xFFFFFFFF);
#else
    size_hi = 0;
    size_lo = (DWORD)(ctx->size);
#endif

    /* get map handle */
    if(ctx->shared)
    {
      if( (hmfile = CreateFileMapping(hfile, NULL, PAGE_READWRITE, size_hi, size_lo, NULL)) == NULL )
        THError("could not create a map on file <%s>", ctx->filename);
    }
    else
    {
      if( (hmfile = CreateFileMapping(hfile, NULL, PAGE_WRITECOPY, size_hi, size_lo, NULL)) == NULL )
        THError("could not create a map on file <%s>", ctx->filename);
    }

    /* map the stuff */
    if(ctx->shared)
      data = MapViewOfFile(hmfile, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    else
      data = MapViewOfFile(hmfile, FILE_MAP_COPY, 0, 0, 0);

    CloseHandle(hfile); 
    CloseHandle(hmfile); 
  }
#else /* _WIN32 */
  {
    /* open file */
    int fd;
    long fdsz;

    if(ctx->shared == TH_ALLOCATOR_MAPPED_SHARED)
    {
      if((fd = open(ctx->filename, O_RDWR | O_CREAT, (mode_t)0600)) == -1)
        THError("unable to open file <%s> in read-write mode", ctx->filename);
    }
    else if (ctx->shared == TH_ALLOCATOR_MAPPED_SHAREDMEM)
    {
#ifdef HAVE_SHM_OPEN
      if((fd = shm_open(ctx->filename, O_RDWR | O_CREAT, (mode_t)0600)) == -1)
        THError("unable to open file <%s> in read-write mode", ctx->filename);
#else
      THError("unable to open file <%s> in sharedmem mode, shm_open unavailable on this platform");
#endif
    }
    else
    {
      if((fd = open(ctx->filename, O_RDONLY)) == -1)
        THError("unable to open file <%s> in read-only mode", ctx->filename);
    }
    if((fdsz = lseek(fd, 0, SEEK_END)) == -1)
    {
      close(fd);
      THError("unable to seek at end of file <%s>", ctx->filename);
    }
    if(size > 0)
    {
      if(size > fdsz)
      {
        if(ctx->shared)
        {
          /* if it is shared mem, let's put it in correct size */
          if (ctx->shared == TH_ALLOCATOR_MAPPED_SHAREDMEM)
          {
            if(ftruncate(fd, size) == -1)
              THError("unable to resize shared memory file <%s> to the right size", ctx->filename);
          }
          if((fdsz = lseek(fd, size-1, SEEK_SET)) == -1)
          {
            close(fd);
            THError("unable to stretch file <%s> to the right size", ctx->filename);
          }
          if((write(fd, "", 1)) != 1) /* note that the string "" contains the '\0' byte ... */
          {
            close(fd);
            THError("unable to write to file <%s>", ctx->filename);
          }
        }
        else
        {
          close(fd);
          THError("file <%s> size is smaller than the required mapping size <%ld>", ctx->filename, size);
        }
      }
    }
    else
      size = fdsz;

    ctx->size = size; /* if we are here, it must be the right size */
    
    /* map it */
    if(ctx->shared)
      data = mmap(NULL, ctx->size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    else
      data = mmap(NULL, ctx->size, PROT_READ|PROT_WRITE, MAP_PRIVATE, fd, 0);

    if(close(fd) == -1)
      THError("Error closing file <%s>", ctx->filename);

    if(data == MAP_FAILED)
    {
      data = NULL; /* let's be sure it is NULL */
      THError("$ Torch: unable to mmap memory: you tried to mmap %dGB.", ctx->size/1073741824);
    }
  }
#endif

  return data;
}

static void *THMapAllocator_realloc(void* ctx, void* ptr, long size) {
  THError("cannot realloc mapped data");
  return NULL;
}

static void THMapAllocator_free(void* ctx_, void* data) {
  THMapAllocatorContext *ctx = ctx_;

#ifdef _WIN32
  if(!UnmapViewOfFile((LPINT)data))
    THError("could not unmap the shared memory file");
#else
  if (munmap(data, ctx->size))
    THError("could not unmap the shared memory file");
  if (ctx->shared == TH_ALLOCATOR_MAPPED_SHAREDMEM)
  {
#ifdef HAVE_SHM_UNLINK
    if (shm_unlink(ctx->filename) == -1)
      THError("could not unlink the shared memory file %s", ctx->filename);
#else
    THError("could not unlink the shared memory file %s, shm_unlink not available on platform", ctx->filename);
#endif
  }
#endif

  THMapAllocatorContext_free(ctx);
}

#else

THMapAllocatorContext *THMapAllocatorContext_new(const char *filename, int shared) {
  THError("file mapping not supported on your system");
  return NULL;
}

void THMapAllocatorContext_free(THMapAllocatorContext *ctx) {
  THError("file mapping not supported on your system");
}

static void *THMapAllocator_alloc(void* ctx_, long size) {
  THError("file mapping not supported on your system");
  return NULL;
}

static void *THMapAllocator_realloc(void* ctx, void* ptr, long size) {
  THError("file mapping not supported on your system");
  return NULL;
}

static void THMapAllocator_free(void* ctx, void* data) {
  THError("file mapping not supported on your system");
}

#endif

THAllocator THMapAllocator = {
  &THMapAllocator_alloc,
  &THMapAllocator_realloc,
  &THMapAllocator_free
};
