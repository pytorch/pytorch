#pragma once

#include <cstddef>
#include <memory>
#include <functional>
#include <cstdlib>
#include <utility>
#include <cstring>
#include <algorithm>
#include "ATen/ScalarType.h"
#include "ATen/Retainable.h"

namespace at {

/**
 * Every Tensor is backed by a storage; multiple tensors may share the same storage
 * (which is why we need an indirection.)  We have a base class for storage so
 * that we can compute the data pointer without a virtual dispatch.
 *
 * Storage is NOT part of the public API.  Don't add it to methods in Tensor.
 */
class CUDAStorageImpl {
protected:
  void *data_;
  int64_t size_;

  // The scalar type of this storage.  We need this in case we need to do placement-new/placement-delete
  // after allocation
  ScalarType scalarType_;

  char flag_;
  THCDeviceAllocator *allocator_;
  void *allocatorContext_;
  CUDAStorageImpl *view_;
  int64_t device_;

  CUDAStorageImpl(CUDAStorageImpl&&) = default;
  ~CUDAStorageImpl() = default;
  CUDAStorageImpl& operator=(CUDAStorageImpl&&) = default;

public:
  CUDAStorageImpl(void* data, int64_t size, char flag, THCDeviceAllocator *allocator, void *allocatorContext, int64_t device)
    : data_(data), size_(size), flag_(flag), allocator_(allocator), allocatorContext_(allocatorContext), device_(device), refcount(1) {}

  CUDAStorageImpl(const CUDAStorageImpl&) = delete;
  CUDAStorageImpl& operator=(const CUDAStorageImpl&) = delete;

  const void *data_ptr() const {
    return data_;
  }

  void *data_ptr() {
    return data_;
  }

  template <typename T>
  T * data() const {
    return static_cast<T*>(this->data_);  
  }

  template <typename T>
  T * data() {
    return static_cast<T*>(this->data_ptr());  
  }

  int64_t size() const {
    return size_;
  }

  char & flag() {
    return flag_;  
  }

  void setFlag(char _flag) {
    flag_ = _flag;
  }

  THCDeviceAllocator *allocator() {
    return allocator_;  
  }

  void *allocatorContext() {
    return allocatorContext_;
  }

  void setAllocator(THCDeviceAllocator * _allocator) {
    allocator_ = _allocator;
  }
  void setAllocatorContext(void * _allocatorContext) {
    allocatorContext_ = _allocatorContext;
  }

  CUDAStorageImpl *view() {
    return view_;
  }

  CUDAStorageImpl *view() const {
    return view_;
  }

  void setView(CUDAStorageImpl * _view) {
    view_ = _view;
  }

  int64_t device() const {
    return device_;
  }

  void resize(THCState *state, int64_t size, int64_t realsize) {
    AT_ASSERTM(size >= 0, "invalid size");
    AT_ASSERT(this->allocator() != NULL);
    int device;
    THCudaCheck(cudaGetDevice(&device));

    if(!(this->flag() & TH_STORAGE_RESIZABLE))
      AT_ERROR("Trying to resize storage that is not resizable");

    if (this->allocator()->realloc) {
      cudaError_t err = (*this->allocator()->realloc)(
        this->allocatorContext(),
        (void**)&(this->data_),
        this->size() * realsize,
        size * realsize, THCState_getCurrentStream(state));
      if (err != cudaSuccess) {
        THCudaCheck(err);
      }
      this->size_ = size;
      this->device_ = device;
      return;
    }

    if(size == 0)
    {
      if(this->flag() & TH_STORAGE_FREEMEM) {
        THCudaCheck(
          (*this->allocator()->free)(this->allocatorContext(), this->data_ptr()));
      }
      this->data_ = NULL;
      this->size_ = 0;
      this->device_ = device;
    }
    else
    {
      real *data = NULL;
      cudaError_t err =
        (*this->allocator()->malloc)(this->allocatorContext(),
                                   (void**)&(data),
                                   size * sizeof(real),
                                   THCState_getCurrentStream(state));
      THCudaCheck(err);

      if (this->data_ptr()) {
        // Enable p2p access when the memcpy is across devices
        THCState_getPeerToPeerAccess(state, device, this->device());

        THCudaCheck(cudaMemcpyAsync(data,
                                    this->data_ptr(),
                                    THMin(this->size(), size) * realsize,
                                    cudaMemcpyDeviceToDevice,
                                    THCState_getCurrentStream(state)));
        if(this->flag() & TH_STORAGE_FREEMEM) {
          THCudaCheck(
            (*this->allocator()->free)(this->allocatorContext(), this->data_ptr()));
        }
      }

      this->data_ = data;
      this->size_ = size;
      this->device_ = device;
    }
  }

  void swap(at::CUDAStorageImpl *other) {
    #define SWAP(val) { val = storage1->val; storage1->val = storage2->val; storage2->val = val; }
    std::swap(this->data_, other->data_);
    std::swap(this->size_, other->size_);
    std::swap(this->scalarType_, other->scalarType_);
    std::swap(this->flag_, other->flag_);
    std::swap(this->allocator_, other->allocator_);
    std::swap(this->allocatorContext_, other->allocatorContext_);
    std::swap(this->view_, other->view_);
  }

  // FIXME: change this
  std::atomic<int> refcount;
};

}

