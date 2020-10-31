#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCStorage.cpp"
#else

#include <c10/util/intrusive_ptr.h>
#include <c10/util/typeid.h>

#ifdef __HIP_PLATFORM_HCC__
#include <hip/hip_version.h>
#endif

scalar_t* THCStorage_(data)(THCState *state, const THCStorage *self)
{
  return self->data<scalar_t>();
}

int THCStorage_(elementSize)(THCState *state)
{
  return sizeof(scalar_t);
}

void THCStorage_(set)(THCState *state, THCStorage *self, ptrdiff_t index, scalar_t value)
{
  THArgCheck(
      (index >= 0) && (index < (self->nbytes() / sizeof(scalar_t))),
      2,
      "index out of bounds");
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
#if HIP_VERSION >= 301
  THCudaCheck(hipMemcpyWithStream(THCStorage_(data)(state, self) + index, &value, sizeof(scalar_t),
                                  cudaMemcpyHostToDevice,
                                  stream));
#else
  THCudaCheck(cudaMemcpyAsync(THCStorage_(data)(state, self) + index, &value, sizeof(scalar_t),
                              cudaMemcpyHostToDevice,
                              stream));
  THCudaCheck(cudaStreamSynchronize(stream));
#endif
}

scalar_t THCStorage_(get)(THCState *state, const THCStorage *self, ptrdiff_t index)
{
  THArgCheck(
      (index >= 0) && (index < (self->nbytes() / sizeof(scalar_t))),
      2,
      "index out of bounds");
  scalar_t value;
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
#if HIP_VERSION >= 301
  THCudaCheck(hipMemcpyWithStream(&value, THCStorage_(data)(state, self) + index, sizeof(scalar_t),
                                  cudaMemcpyDeviceToHost, stream));
#else
  THCudaCheck(cudaMemcpyAsync(&value, THCStorage_(data)(state, self) + index, sizeof(scalar_t),
                              cudaMemcpyDeviceToHost, stream));
  THCudaCheck(cudaStreamSynchronize(stream));
#endif
  return value;
}

THCStorage* THCStorage_(new)(THCState *state)
{
  THStorage* storage = c10::make_intrusive<at::StorageImpl>(
                           c10::StorageImpl::use_byte_size_t(),
                           0,
                           c10::cuda::CUDACachingAllocator::get(),
                           true)
                           .release();
  return storage;
}

THCStorage* THCStorage_(newWithSize)(THCState *state, ptrdiff_t size)
{
  THStorage* storage = c10::make_intrusive<at::StorageImpl>(
                           c10::StorageImpl::use_byte_size_t(),
                           size * sizeof(scalar_t),
                           c10::cuda::CUDACachingAllocator::get(),
                           true)
                           .release();
  return storage;
}

THCStorage* THCStorage_(newWithAllocator)(THCState *state, ptrdiff_t size,
                                          at::Allocator* allocator)
{
  THStorage* storage = c10::make_intrusive<at::StorageImpl>(
                           c10::StorageImpl::use_byte_size_t(),
                           size * sizeof(scalar_t),
                           allocator,
                           true)
                           .release();
  return storage;
}

THCStorage* THCStorage_(newWithSize1)(THCState *state, scalar_t data0)
{
  THCStorage *self = THCStorage_(newWithSize)(state, 1);
  THCStorage_(set)(state, self, 0, data0);
  return self;
}

THCStorage* THCStorage_(newWithMapping)(THCState *state, const char *fileName, ptrdiff_t size, int isShared)
{
  THError("not available yet for THCStorage");
  return NULL;
}

THCStorage* THCStorage_(newWithDataAndAllocator)(
    THCState* state,
    at::DataPtr&& data,
    ptrdiff_t size,
    at::Allocator* allocator) {
  THStorage* storage = c10::make_intrusive<at::StorageImpl>(
                           c10::StorageImpl::use_byte_size_t(),
                           size * sizeof(scalar_t),
                           std::move(data),
                           allocator,
                           allocator != nullptr)
                           .release();
  return storage;
}

void THCStorage_(retain)(THCState *state, THCStorage *self)
{
  THStorage_retain(self);
}

void THCStorage_(free)(THCState *state, THCStorage *self)
{
  THStorage_free(self);
}
#endif
