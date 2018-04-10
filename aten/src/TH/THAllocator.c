#include "THAllocator.h"
#include "THAtomic.h"

/* needed for ATOMIC_INT_LOCK_FREE */
/* cannot go in THAtomic.h because of interactions with OpenMP giving
   sorry not implemented errors */
#if defined(USE_C11_ATOMICS)
#include <stdatomic.h>
#if ATOMIC_INT_LOCK_FREE == 2
#define TH_ATOMIC_IPC_REFCOUNT 1
#endif
#elif defined(USE_MSC_ATOMICS) || defined(USE_GCC_ATOMICS)
#define TH_ATOMIC_IPC_REFCOUNT 1
#endif

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

static void *THDefaultAllocator_alloc(void* ctx, ptrdiff_t size) {
  return THAlloc(size);
}

static void *THDefaultAllocator_realloc(void* ctx, void* ptr, ptrdiff_t size) {
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
  int flags;
  ptrdiff_t size; /* mapped size */
#ifdef _WIN32
  HANDLE handle;
  HANDLE event;
  char *eventname;
#else
  int fd;
#endif
};

#define TH_ALLOC_ALIGNMENT 64

typedef struct {
  int refcount;
} THMapInfo;

char * unknown_filename = "filename not specified";
#ifdef _WIN32
char * unknown_eventname = "eventname not specified";
#endif

THMapAllocatorContext *THMapAllocatorContext_new(const char *filename, int flags)
{
  THMapAllocatorContext *ctx = THAlloc(sizeof(THMapAllocatorContext));

  if (!(flags & TH_ALLOCATOR_MAPPED_SHARED) && !(flags & TH_ALLOCATOR_MAPPED_SHAREDMEM))
    flags &= ~TH_ALLOCATOR_MAPPED_NOCREATE;
  if ((flags ^ TH_ALLOCATOR_MAPPED_EXCLUSIVE) == 0)
    THError("TH_ALLOCATOR_MAPPED_EXCLUSIVE flag requires opening the file "
        "in shared mode");

  if (filename) {
    ctx->filename = THAlloc(strlen(filename)+1);
    strcpy(ctx->filename, filename);
#ifdef _WIN32
    char *suffixname = "_event";
    size_t namelen = strlen(filename)+1+strlen(suffixname);
    ctx->eventname = THAlloc(namelen);
    strcpy(ctx->eventname, ctx->filename);
    strcat(ctx->eventname, suffixname);
#endif
  } else {
    ctx->filename = unknown_filename;
#ifdef _WIN32
    ctx->eventname = unknown_eventname;
#endif
  }

  ctx->flags = flags;
  ctx->size = 0;
#ifdef _WIN32
  ctx->handle = INVALID_HANDLE_VALUE;
#else
  ctx->fd = -1;
#endif

  return ctx;
}

THMapAllocatorContext *THMapAllocatorContext_newWithFd(const char *filename, int fd, int flags)
{
#ifdef _WIN32
  THError("THMapAllocatorContext_newWithFd is unsupported on Windows");
#else
  THMapAllocatorContext *ctx = THMapAllocatorContext_new(filename, flags);
  ctx->fd = fd;

  return ctx;
#endif
}

char * THMapAllocatorContext_filename(THMapAllocatorContext *ctx)
{
  return ctx->filename;
}

int THMapAllocatorContext_fd(THMapAllocatorContext *ctx)
{
#ifdef _WIN32
  THError("THMapAllocatorContext_fd is unsupported on Windows");
#else
  return ctx->fd;
#endif
}

ptrdiff_t THMapAllocatorContext_size(THMapAllocatorContext *ctx)
{
  return ctx->size;
}

void THMapAllocatorContext_free(THMapAllocatorContext *ctx)
{
  if (ctx->filename != unknown_filename) {
    THFree(ctx->filename);
#ifdef _WIN32
    THFree(ctx->eventname);
#endif
  }
  THFree(ctx);
}

#ifdef _WIN32
typedef struct{
  HANDLE event;
  HANDLE handle;
  HANDLE wait;
} ReleaseContext;
static VOID CALLBACK WaitForReleaseHandle(PVOID lpParam, BOOLEAN TimerOrWaitFired)
{
  if (lpParam) {
    ReleaseContext *ctx = (ReleaseContext *)lpParam;

    SetEvent(ctx->event);
    CloseHandle(ctx->event);
    CloseHandle(ctx->handle);

    UnregisterWait(ctx->wait);

    THFree(ctx);
  }
}
#endif

static void *_map_alloc(void* ctx_, ptrdiff_t size)
{
  if (size == 0)
    return NULL;

  THMapAllocatorContext *ctx = ctx_;
  void *data = NULL;

#ifdef _WIN32
  if (ctx->flags & TH_ALLOCATOR_MAPPED_SHAREDMEM)
  {
    char *filename;
    char *eventname;
    LARGE_INTEGER hfilesz;

    if (ctx->filename[0] == '/') {
      filename = ctx->filename + 1;
      eventname = ctx->eventname + 1;
    }
    else {
      filename = ctx->filename;
      eventname = ctx->eventname;
    }

    hfilesz.QuadPart = size;

    if (ctx->flags & TH_ALLOCATOR_MAPPED_EXCLUSIVE)
    {
      ctx->handle = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, hfilesz.HighPart, hfilesz.LowPart, filename);
      ctx->event = CreateEvent(NULL, FALSE, FALSE, eventname);
    }
    else if (ctx->flags & TH_ALLOCATOR_MAPPED_NOCREATE)
    {
      ctx->handle = OpenFileMapping(FILE_MAP_ALL_ACCESS, FALSE, filename);
      ctx->event = OpenEvent(EVENT_ALL_ACCESS, FALSE, eventname);
    }
    else
    {
      THError("Expected either TH_ALLOCATOR_MAPPED_EXCLUSIVE or TH_ALLOCATOR_MAPPED_NOCREATE");
    }

    if (ctx->event == NULL)
      THError("Couldn't open shared event: <%s>, error code: <%d>", eventname, GetLastError());

    if (ctx->handle == NULL)
      THError("Couldn't open shared file mapping: <%s>, error code: <%d>", filename, GetLastError());

    ctx->size = size;
    data = MapViewOfFile(ctx->handle, FILE_MAP_ALL_ACCESS, 0, 0, size);
    if (!data)
      THError("Couldn't map view of shared file <%s>, error code: <%d>", filename, GetLastError());
  }
  else
  {

    HANDLE hfile;
    HANDLE hmfile;
    LARGE_INTEGER hfilesz;

    if (ctx->flags & TH_ALLOCATOR_MAPPED_EXCLUSIVE)
      THError("exclusive file mapping is not supported on Windows");
    if (ctx->flags & TH_ALLOCATOR_MAPPED_NOCREATE)
      THError("file mapping without creation is not supported on Windows");
    if (ctx->flags & TH_ALLOCATOR_MAPPED_KEEPFD)
      THError("TH_ALLOCATOR_MAPPED_KEEPFD not supported on Windows");
    if (ctx->flags & TH_ALLOCATOR_MAPPED_FROMFD)
      THError("TH_ALLOCATOR_MAPPED_FROMFD not supported on Windows");

    /* open file */
    /* FILE_FLAG_RANDOM_ACCESS ? */
    if(ctx->flags)
    {
      hfile = CreateFileA(ctx->filename, GENERIC_READ|GENERIC_WRITE, FILE_SHARE_WRITE|FILE_SHARE_READ, 0, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, 0);
      if (hfile == INVALID_HANDLE_VALUE)
        THError("could not open file <%s> in read-write mode; error code: <%d>", ctx->filename, GetLastError());
    }
    else
    {
      hfile = CreateFileA(ctx->filename, GENERIC_READ, FILE_SHARE_WRITE|FILE_SHARE_READ, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
      if (hfile == INVALID_HANDLE_VALUE)
        THError("could not open file <%s> in read-only mode; error code: <%d>", ctx->filename, GetLastError());
    }

    if (GetFileSizeEx(hfile, &hfilesz) == 0)
      THError("could not get file size: <%s>; error code: <%d>", ctx->filename, GetLastError());

    if(size > 0)
    {
      if(size > hfilesz.QuadPart)
      {
        if(ctx->flags)
        {
          hfilesz.QuadPart = size;
          if(SetFilePointerEx(hfile, hfilesz, NULL, FILE_BEGIN) == 0)
          {
            CloseHandle(hfile);
            THError("unable to stretch file <%s> to the right size; error code: <%d>", ctx->filename, GetLastError());
          }
          if(SetEndOfFile(hfile) == 0)
          {
            CloseHandle(hfile);
            THError("unable to write to file <%s>; error code: <%d>", ctx->filename, GetLastError());
          }
        }
        else
        {
          CloseHandle(hfile);
          THError("file <%s> size is smaller than the required mapping size <%ld>; error code: <%d>", ctx->filename, size, GetLastError());
        }
      }
    }
    else
      size = hfilesz.QuadPart;

    ctx->size = size; /* if we are here, it must be the right size */

    hfilesz.QuadPart = ctx->size;

    /* get map handle */
    if(ctx->flags)
    {
      if( (hmfile = CreateFileMapping(hfile, NULL, PAGE_READWRITE, hfilesz.HighPart, hfilesz.LowPart, NULL)) == NULL )
        THError("could not create a map on file <%s>; error code: <%d>", ctx->filename, GetLastError());
    }
    else
    {
      if( (hmfile = CreateFileMapping(hfile, NULL, PAGE_WRITECOPY, hfilesz.HighPart, hfilesz.LowPart, NULL)) == NULL )
        THError("could not create a map on file <%s>; error code: <%d>", ctx->filename, GetLastError());
    }

    /* map the stuff */
    if(ctx->flags)
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
    int flags;
    struct stat file_stat;

    if (ctx->flags & (TH_ALLOCATOR_MAPPED_SHARED | TH_ALLOCATOR_MAPPED_SHAREDMEM))
      flags = O_RDWR | O_CREAT;
    else
      flags = O_RDONLY;

    if (ctx->flags & TH_ALLOCATOR_MAPPED_EXCLUSIVE)
      flags |= O_EXCL;
    if (ctx->flags & TH_ALLOCATOR_MAPPED_NOCREATE)
      flags &= ~O_CREAT;

    if (!(ctx->flags & TH_ALLOCATOR_MAPPED_FROMFD)) {
      if(ctx->flags & TH_ALLOCATOR_MAPPED_SHARED)
      {
        if((fd = open(ctx->filename, flags, (mode_t)0600)) == -1)
          THError("unable to open file <%s> in read-write mode", ctx->filename);
      }
      else if (ctx->flags & TH_ALLOCATOR_MAPPED_SHAREDMEM)
      {
#ifdef HAVE_SHM_OPEN
        if((fd = shm_open(ctx->filename, flags, (mode_t)0600)) == -1)
          THError("unable to open shared memory object <%s> in read-write mode", ctx->filename);
#else
        THError("unable to open file <%s> in sharedmem mode, shm_open unavailable on this platform", ctx->filename);
#endif
      }
      else
      {
        if((fd = open(ctx->filename, O_RDONLY)) == -1)
          THError("unable to open file <%s> in read-only mode", ctx->filename);
      }
    } else {
      fd = ctx->fd;
    }

    if(fstat(fd, &file_stat) == -1)
    {
      if (!(ctx->flags & TH_ALLOCATOR_MAPPED_FROMFD))
        close(fd);
      THError("unable to stat the file <%s>", ctx->filename);
    }

    if(size > 0)
    {
      if(size > file_stat.st_size)
      {
        if(ctx->flags)
        {
          if(ftruncate(fd, size) == -1)
            THError("unable to resize file <%s> to the right size", ctx->filename);
          if(fstat(fd, &file_stat) == -1 || file_stat.st_size < size)
          {
            close(fd);
            THError("unable to stretch file <%s> to the right size", ctx->filename);
          }
/* on macOS write returns with errno 45 (Opperation not supported) when used
 * with a file descriptor obtained via shm_open
 */
#ifndef __APPLE__
          if((write(fd, "", 1)) != 1) /* note that the string "" contains the '\0' byte ... */
          {
            close(fd);
            THError("unable to write to file <%s>", ctx->filename);
          }
#endif
        }
        else
        {
          close(fd);
          THError("file <%s> size is smaller than the required mapping size <%ld>", ctx->filename, size);
        }
      }
    }
    else
      size = file_stat.st_size;

    ctx->size = size; /* if we are here, it must be the right size */

    /* map it */
    if (ctx->flags & (TH_ALLOCATOR_MAPPED_SHARED | TH_ALLOCATOR_MAPPED_SHAREDMEM))
      data = mmap(NULL, ctx->size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    else
      data = mmap(NULL, ctx->size, PROT_READ|PROT_WRITE, MAP_PRIVATE, fd, 0);

    if (ctx->flags & TH_ALLOCATOR_MAPPED_KEEPFD) {
      ctx->fd = fd;
    } else {
      if(close(fd) == -1)
        THError("Error closing file <%s>", ctx->filename);
      ctx->fd = -1;
    }

    if (ctx->flags & TH_ALLOCATOR_MAPPED_UNLINK) {
      if (ctx->flags & TH_ALLOCATOR_MAPPED_SHAREDMEM)
      {
#ifdef HAVE_SHM_UNLINK
        if (shm_unlink(ctx->filename) == -1)
          THError("could not unlink the shared memory file %s", ctx->filename);
#else
        THError("could not unlink the shared memory file %s, shm_unlink not available on platform", ctx->filename);
#endif
      }
      else
      {
        if (unlink(ctx->filename) == -1)
          THError("could not unlink file %s", ctx->filename);
      }
    }

    if(data == MAP_FAILED)
    {
      data = NULL; /* let's be sure it is NULL */
      THError("$ Torch: unable to mmap memory: you tried to mmap %dGB.", ctx->size/1073741824);
    }
  }
#endif

  return data;
}

static void * THMapAllocator_alloc(void *ctx, ptrdiff_t size) {
  return _map_alloc(ctx, size);
}

static void *THMapAllocator_realloc(void* ctx, void* ptr, ptrdiff_t size) {
  THError("cannot realloc mapped data");
  return NULL;
}

static void THMapAllocator_free(void* ctx_, void* data) {
  if (data == NULL)
    return;

  THMapAllocatorContext *ctx = ctx_;

#ifdef _WIN32
  if ((ctx->flags & TH_ALLOCATOR_MAPPED_KEEPFD) || (ctx->flags & TH_ALLOCATOR_MAPPED_SHAREDMEM))
    CloseHandle(ctx->handle);
  if(UnmapViewOfFile(data) == 0)
    THError("could not unmap the shared memory file");
#else /* _WIN32 */
  if (ctx->flags & TH_ALLOCATOR_MAPPED_KEEPFD) {
    if (close(ctx->fd) == -1)
      THError("could not close file descriptor %d", ctx->fd);
  }

  if (munmap(data, ctx->size))
    THError("could not unmap the shared memory file");

  if (!(ctx->flags & (TH_ALLOCATOR_MAPPED_FROMFD | TH_ALLOCATOR_MAPPED_UNLINK)))
  {
    if (ctx->flags & TH_ALLOCATOR_MAPPED_SHAREDMEM)
    {
#ifdef HAVE_SHM_UNLINK
      if (shm_unlink(ctx->filename) == -1)
        THError("could not unlink the shared memory file %s", ctx->filename);
#else
      THError("could not unlink the shared memory file %s, shm_unlink not available on platform", ctx->filename);
#endif
    }
  }
#endif /* _WIN32 */

  THMapAllocatorContext_free(ctx);
}

#else

THMapAllocatorContext *THMapAllocatorContext_new(const char *filename, int flags) {
  THError("file mapping not supported on your system");
  return NULL;
}

void THMapAllocatorContext_free(THMapAllocatorContext *ctx) {
  THError("file mapping not supported on your system");
}

static void *THMapAllocator_alloc(void* ctx_, ptrdiff_t size) {
  THError("file mapping not supported on your system");
  return NULL;
}

static void *THMapAllocator_realloc(void* ctx, void* ptr, ptrdiff_t size) {
  THError("file mapping not supported on your system");
  return NULL;
}

static void THMapAllocator_free(void* ctx, void* data) {
  THError("file mapping not supported on your system");
}

#endif

#if (defined(_WIN32) || defined(HAVE_MMAP)) && defined(TH_ATOMIC_IPC_REFCOUNT)

static void * THRefcountedMapAllocator_alloc(void *_ctx, ptrdiff_t size) {
  THMapAllocatorContext *ctx = _ctx;

  if (ctx->flags & TH_ALLOCATOR_MAPPED_FROMFD)
    THError("THRefcountedMapAllocator doesn't support TH_ALLOCATOR_MAPPED_FROMFD flag");
  if (ctx->flags & TH_ALLOCATOR_MAPPED_KEEPFD)
    THError("THRefcountedMapAllocator doesn't support TH_ALLOCATOR_MAPPED_KEEPFD flag");
  if (ctx->flags & TH_ALLOCATOR_MAPPED_UNLINK)
    THError("THRefcountedMapAllocator doesn't support TH_ALLOCATOR_MAPPED_UNLINK flag");
  if (!(ctx->flags & TH_ALLOCATOR_MAPPED_SHAREDMEM))
    THError("THRefcountedMapAllocator requires TH_ALLOCATOR_MAPPED_SHAREDMEM flag");

  size = size + TH_ALLOC_ALIGNMENT;
  void *ptr = _map_alloc(ctx, size);
  char *data = ((char*)ptr) + TH_ALLOC_ALIGNMENT;
  THMapInfo *map_info = (THMapInfo*)ptr;

#ifdef _WIN32
  ReleaseContext* r_ctx = (ReleaseContext *) THAlloc(sizeof(ReleaseContext));
  r_ctx->handle = ctx->handle;
  r_ctx->event = ctx->event;
  r_ctx->wait = NULL;
  BOOL can_wait = RegisterWaitForSingleObject(&r_ctx->wait, ctx->event, WaitForReleaseHandle, (PVOID)r_ctx, INFINITE, WT_EXECUTEONLYONCE);
  if (!can_wait) {
    THError("Couldn't register wait on event, error code: <%d>", GetLastError());
  }
#endif

  if (ctx->flags & TH_ALLOCATOR_MAPPED_EXCLUSIVE)
    map_info->refcount = 1;
  else
    THAtomicIncrementRef(&map_info->refcount);

  return (void*)data;
}

static void *THRefcountedMapAllocator_realloc(void* ctx, void* ptr, ptrdiff_t size) {
  THError("cannot realloc mapped data");
  return NULL;
}

static void THRefcountedMapAllocator_free(void* ctx_, void* data) {
  THMapAllocatorContext *ctx = ctx_;

#ifdef _WIN32
  THMapInfo *info = (THMapInfo*)(((char*)data) - TH_ALLOC_ALIGNMENT);
  if (THAtomicDecrementRef(&info->refcount)) {
    SetEvent(ctx->event); 
  }
  if(UnmapViewOfFile(((char*)data) - TH_ALLOC_ALIGNMENT) == 0)
    THError("could not unmap the shared memory file");
#else /* _WIN32 */

  THMapInfo *info = (THMapInfo*)(((char*)data) - TH_ALLOC_ALIGNMENT);
  if (THAtomicDecrementRef(&info->refcount)) {
#ifdef HAVE_SHM_UNLINK
    if (shm_unlink(ctx->filename) == -1)
      THError("could not unlink the shared memory file %s", ctx->filename);
#else
    THError("could not unlink the shared memory file %s, shm_unlink not available on platform", ctx->filename);
#endif /* HAVE_SHM_UNLINK */
  }
  if (munmap(info, ctx->size))
    THError("could not unmap the shared memory file %s", ctx->filename);
#endif /* _WIN32 */

  THMapAllocatorContext_free(ctx);
}

void THRefcountedMapAllocator_incref(THMapAllocatorContext *ctx, void *data)
{
  THMapInfo *map_info = (THMapInfo*)(((char*)data) - TH_ALLOC_ALIGNMENT);
  THAtomicIncrementRef(&map_info->refcount);
}

int THRefcountedMapAllocator_decref(THMapAllocatorContext *ctx, void *data)
{
  THMapInfo *map_info = (THMapInfo*)(((char*)data) - TH_ALLOC_ALIGNMENT);
  return THAtomicDecrementRef(&map_info->refcount);
}

#else

static void * THRefcountedMapAllocator_alloc(void *ctx, ptrdiff_t size) {
  THError("refcounted file mapping not supported on your system");
  return NULL;
}

static void *THRefcountedMapAllocator_realloc(void* ctx, void* ptr, ptrdiff_t size) {
  THError("refcounted file mapping not supported on your system");
  return NULL;
}

static void THRefcountedMapAllocator_free(void* ctx_, void* data) {
  THError("refcounted file mapping not supported on your system");
}

void THRefcountedMapAllocator_incref(THMapAllocatorContext *ctx, void *data)
{
  THError("refcounted file mapping not supported on your system");
}

int THRefcountedMapAllocator_decref(THMapAllocatorContext *ctx, void *data)
{
  THError("refcounted file mapping not supported on your system");
  return 0;
}

#endif

THAllocator THMapAllocator = {
  &THMapAllocator_alloc,
  &THMapAllocator_realloc,
  &THMapAllocator_free
};

THAllocator THRefcountedMapAllocator = {
  &THRefcountedMapAllocator_alloc,
  &THRefcountedMapAllocator_realloc,
  &THRefcountedMapAllocator_free
};
