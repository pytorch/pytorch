#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THStorage.c"
#else

real* THStorage_(data)(const THStorage *self)
{
  return self->data;
}

long THStorage_(size)(const THStorage *self)
{
  return self->size;
}

THStorage* THStorage_(new)(void)
{
  return THStorage_(newWithSize)(0);
}

THStorage* THStorage_(newWithSize)(long size)
{
  THStorage *storage = THAlloc(sizeof(THStorage));
  storage->data = THAlloc(sizeof(real)*size);
  storage->size = size;
  storage->refcount = 1;
  storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM;
  return storage;
}

THStorage* THStorage_(newWithSize1)(real data0)
{
  THStorage *self = THStorage_(newWithSize)(1);
  self->data[0] = data0;
  return self;
}

THStorage* THStorage_(newWithSize2)(real data0, real data1)
{
  THStorage *self = THStorage_(newWithSize)(2);
  self->data[0] = data0;
  self->data[1] = data1;
  return self;
}

THStorage* THStorage_(newWithSize3)(real data0, real data1, real data2)
{
  THStorage *self = THStorage_(newWithSize)(3);
  self->data[0] = data0;
  self->data[1] = data1;
  self->data[2] = data2;
  return self;
}

THStorage* THStorage_(newWithSize4)(real data0, real data1, real data2, real data3)
{
  THStorage *self = THStorage_(newWithSize)(4);
  self->data[0] = data0;
  self->data[1] = data1;
  self->data[2] = data2;
  self->data[3] = data3;
  return self;
}

#if defined(_WIN32) || defined(HAVE_MMAP)

THStorage* THStorage_(newWithMapping)(const char *fileName, int isShared)
{
  THStorage *storage = THAlloc(sizeof(THStorage));
  long size;

  /* check size */
  FILE *f = fopen(fileName, "rb");
  if(f == NULL)
    THError("unable to open file <%s> for mapping (read-only mode)", fileName);
  fseek(f, 0, SEEK_END);
  size = ftell(f);
  fclose(f);
  size /= sizeof(real);

#ifdef _WIN32
  {
    HANDLE hfile;
    HANDLE hmfile;
    DWORD size_hi, size_lo;

    /* open file */
    if(isShared)
    {
      hfile = CreateFileA(fileName, GENERIC_READ|GENERIC_WRITE, FILE_SHARE_WRITE|FILE_SHARE_READ, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
      if (hfile == INVALID_HANDLE_VALUE)
        THError("could not open file <%s> in read-write mode", fileName);
    }
    else
    {
      hfile = CreateFileA(fileName, GENERIC_READ, FILE_SHARE_WRITE|FILE_SHARE_READ, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
      if (hfile == INVALID_HANDLE_VALUE)
        THError("could not open file <%s> in read-only mode", fileName);
    }

#if SIZEOF_SIZE_T > 4
    size_hi = (DWORD)((size*sizeof(real)) >> 32);
    size_lo = (DWORD)((size*sizeof(real)) & 0xFFFFFFFF);
#else
    size_hi = 0;
    size_lo = (DWORD)(size*sizeof(real));
#endif

    /* get map handle */
    if(isShared)
    {
      if( (hmfile = CreateFileMapping(hfile, NULL, PAGE_READWRITE, size_hi, size_lo, NULL)) == NULL )
        THError("could not create a map on file <%s>", fileName);
    }
    else
    {
      if( (hmfile = CreateFileMapping(hfile, NULL, PAGE_WRITECOPY, size_hi, size_lo, NULL)) == NULL )
        THError("could not create a map on file <%s>", fileName);
    }

    /* map the stuff */
    storage = THStorage_(new)();
    if(isShared)
      storage->data = MapViewOfFile(hmfile, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    else
      storage->data = MapViewOfFile(hmfile, FILE_MAP_COPY, 0, 0, 0);
      
    storage->size = size;
    if(storage->data == NULL)
    {
      THStorage_(free)(storage);
      THError("memory map failed on file <%s>", fileName);
    }
    CloseHandle(hfile); 
    CloseHandle(hmfile); 
  }
#else
  {
    /* open file */
    int fd;
    if(isShared)
    {
      fd = open(fileName, O_RDWR);
      if(fd == -1)
        THError("unable to open file <%s> in read-write mode", fileName);
    }
    else
    {
      fd = open(fileName, O_RDONLY);
      if(fd == -1)
        THError("unable to open file <%s> in read-only mode", fileName);
    }
    
    /* map it */
    storage = THStorage_(new)();
    if(isShared)
      storage->data = mmap(NULL, size*sizeof(real), PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    else
      storage->data = mmap(NULL, size*sizeof(real), PROT_READ|PROT_WRITE, MAP_PRIVATE, fd, 0);

    storage->size = size;
    if(storage->data == MAP_FAILED)
    {
      storage->data = NULL; /* let's be sure it is NULL before calling free() */
      THStorage_(free)(storage);
      THError("memory map failed on file <%s>", fileName);
    }
    close (fd);
  }
#endif

  storage->refcount = 1;
  storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_MAPPED | TH_STORAGE_FREEMEM;;
  return storage;
}

#else

THStorage* THStorage_(newWithMapping)(const char *fileName, int isShared)
{
  THError("Mapped file Storages are not supported on your system");
}

#endif

void THStorage_(setFlag)(THStorage *storage, const char flag)
{
  storage->flag |= flag;
}

void THStorage_(clearFlag)(THStorage *storage, const char flag)
{
  storage->flag &= ~flag;
}

void THStorage_(retain)(THStorage *storage)
{
  if(storage && (storage->flag & TH_STORAGE_REFCOUNTED))
    ++storage->refcount;
}

void THStorage_(free)(THStorage *storage)
{
  if(!storage)
    return;

  if((storage->flag & TH_STORAGE_REFCOUNTED) && (storage->refcount > 0))
  {
    if(--storage->refcount == 0)
    {
      if(storage->flag & TH_STORAGE_FREEMEM)
      {
#if defined(_WIN32) || defined(HAVE_MMAP)
        if(storage->flag & TH_STORAGE_MAPPED)
        {
#ifdef _WIN32
          if(!UnmapViewOfFile((LPINT)storage->data))
#else
            if (munmap(storage->data, storage->size*sizeof(real)))
#endif
              THError("could not unmap the shared memory file");
        }
        else
#endif
          THFree(storage->data);
      }
      THFree(storage);
    }
  }
}

THStorage* THStorage_(newWithData)(real *data, long size)
{
  THStorage *storage = THAlloc(sizeof(THStorage));
  storage->data = data;
  storage->size = size;
  storage->refcount = 1;
  storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM;
  return storage;
}

void THStorage_(resize)(THStorage *storage, long size)
{
  if(storage->flag & TH_STORAGE_RESIZABLE)
  {
    storage->data = THRealloc(storage->data, sizeof(real)*size);
    storage->size = size;
  }
}

void THStorage_(fill)(THStorage *storage, real value)
{
  long i;
  for(i = 0; i < storage->size; i++)
    storage->data[i] = value;
}

void THStorage_(set)(THStorage *self, long idx, real value)
{
  THArgCheck((idx >= 0) && (idx < self->size), 2, "out of bounds");
  self->data[idx] = value;
}

real THStorage_(get)(const THStorage *self, long idx)
{
  THArgCheck((idx >= 0) && (idx < self->size), 2, "out of bounds");
  return self->data[idx];
}

#endif
