#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/serialization.cpp"
#else

#define SYSCHECK(call) { ssize_t __result = call; if (__result < 0) throw std::system_error(__result, std::system_category()); }

void THPTensor_(writeMetadataRaw)(THTensor *self, int fd)
{
  SYSCHECK(write(fd, &self->nDimension, sizeof(long)));
  SYSCHECK(write(fd, self->size, sizeof(long) * self->nDimension));
  SYSCHECK(write(fd, self->stride, sizeof(long) * self->nDimension));
  SYSCHECK(write(fd, &self->storageOffset, sizeof(long)));
}

THTensor * THPTensor_(newWithMetadataFileRaw)(int fd, THStorage *storage)
{
  THTensorPtr tensor = THTensor_(new)(LIBRARY_STATE_NOARGS);
  SYSCHECK(read(fd, &tensor->nDimension, sizeof(long)));
  tensor->size = (long*)THAlloc(tensor->nDimension * sizeof(long));
  tensor->stride = (long*)THAlloc(tensor->nDimension * sizeof(long));
  SYSCHECK(read(fd, tensor->size, sizeof(long) * tensor->nDimension));
  SYSCHECK(read(fd, tensor->stride, sizeof(long) * tensor->nDimension));
  SYSCHECK(read(fd, &tensor->storageOffset, sizeof(long)));
  if (storage)
    THStorage_(retain)(LIBRARY_STATE storage);
  tensor->storage = storage;
  return tensor.release();
}

void THPStorage_(writeFileRaw)(THStorage *self, int fd)
{
  real *data;
#ifndef THC_GENERIC_FILE
  data = self->data;
#else
  std::unique_ptr<char[]> cpu_data(new char[self->size * sizeof(real)]);
  data = (real*)cpu_data.get();
  THCudaCheck(cudaMemcpy(data, self->data, self->size * sizeof(real), cudaMemcpyDeviceToHost));
#endif
  SYSCHECK(write(fd, &self->size, sizeof(long)));
  SYSCHECK(write(fd, data, sizeof(real) * self->size));
}

THStorage * THPStorage_(readFileRaw)(int fd)
{
  real *data;
  long size;
  SYSCHECK(read(fd, &size, sizeof(long)));
  THStoragePtr storage = THStorage_(newWithSize)(LIBRARY_STATE size);

#ifndef THC_GENERIC_FILE
  data = storage->data;
#else
  std::unique_ptr<char[]> cpu_data(new char[size * sizeof(real)]);
  data = (real*)cpu_data.get();
#endif

  SYSCHECK(read(fd, data, sizeof(real) * storage->size));

#ifdef THC_GENERIC_FILE
  THCudaCheck(cudaMemcpy(storage->data, data, size * sizeof(real), cudaMemcpyHostToDevice));
#endif
  return storage.release();
}

#undef SYSCHECK

#endif
