#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCStorage.cpp"
#else

#include <c10/util/intrusive_ptr.h>
#include <c10/util/typeid.h>

scalar_t* THCStorage_(data)(THCState *state, const THCStorage *self)
{
  return self->data<scalar_t>();
}

ptrdiff_t THCStorage_(size)(THCState *state, const THCStorage *self)
{
  return THStorage_size(self);
}

int THCStorage_(elementSize)(THCState *state)
{
  return sizeof(scalar_t);
}

void THCStorage_(set)(THCState *state, THCStorage *self, ptrdiff_t index, scalar_t value)
{
  THArgCheck((index >= 0) && (index < self->numel()), 2, "index out of bounds");
  cudaStream_t stream = THCState_getCurrentStream(state);
  THCudaCheck(cudaMemcpyAsync(THCStorage_(data)(state, self) + index, &value, sizeof(scalar_t),
                              cudaMemcpyHostToDevice,
                              stream));
  THCudaCheck(cudaStreamSynchronize(stream));
}

scalar_t THCStorage_(get)(THCState *state, const THCStorage *self, ptrdiff_t index)
{
  THArgCheck((index >= 0) && (index < self->numel()), 2, "index out of bounds");
  scalar_t value;
  cudaStream_t stream = THCState_getCurrentStream(state);
  THCudaCheck(cudaMemcpyAsync(&value, THCStorage_(data)(state, self) + index, sizeof(scalar_t),
                              cudaMemcpyDeviceToHost, stream));
  THCudaCheck(cudaStreamSynchronize(stream));
  return value;
}

THCStorage* THCStorage_(new)(THCState *state)
{
  THStorage* storage = c10::make_intrusive<at::StorageImpl>(
      caffe2::TypeMeta::Make<scalar_t>(),
      0,
      state->cudaDeviceAllocator,
      true).release();
  return storage;
}

THCStorage* THCStorage_(newWithSize)(THCState *state, ptrdiff_t size)
{
  THStorage* storage = c10::make_intrusive<at::StorageImpl>(
      caffe2::TypeMeta::Make<scalar_t>(),
      size,
      state->cudaDeviceAllocator,
      true).release();
  return storage;
}

THCStorage* THCStorage_(newWithAllocator)(THCState *state, ptrdiff_t size,
                                          at::Allocator* allocator)
{
  THStorage* storage = c10::make_intrusive<at::StorageImpl>(
      caffe2::TypeMeta::Make<scalar_t>(),
      size,
      allocator,
      true).release();
  return storage;
}

THCStorage* THCStorage_(newWithSize1)(THCState *state, scalar_t data0)
{
  THCStorage *self = THCStorage_(newWithSize)(state, 1);
  THCStorage_(set)(state, self, 0, data0);
  return self;
}

THCStorage* THCStorage_(newWithSize2)(THCState *state, scalar_t data0, scalar_t data1)
{
  THCStorage *self = THCStorage_(newWithSize)(state, 2);
  THCStorage_(set)(state, self, 0, data0);
  THCStorage_(set)(state, self, 1, data1);
  return self;
}

THCStorage* THCStorage_(newWithSize3)(THCState *state, scalar_t data0, scalar_t data1, scalar_t data2)
{
  THCStorage *self = THCStorage_(newWithSize)(state, 3);
  THCStorage_(set)(state, self, 0, data0);
  THCStorage_(set)(state, self, 1, data1);
  THCStorage_(set)(state, self, 2, data2);
  return self;
}

THCStorage* THCStorage_(newWithSize4)(THCState *state, scalar_t data0, scalar_t data1, scalar_t data2, scalar_t data3)
{
  THCStorage *self = THCStorage_(newWithSize)(state, 4);
  THCStorage_(set)(state, self, 0, data0);
  THCStorage_(set)(state, self, 1, data1);
  THCStorage_(set)(state, self, 2, data2);
  THCStorage_(set)(state, self, 3, data3);
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
      caffe2::TypeMeta::Make<scalar_t>(),
      size,
      std::move(data),
      allocator,
      allocator != nullptr).release();
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
