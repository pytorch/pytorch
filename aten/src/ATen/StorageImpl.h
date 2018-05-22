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
class StorageImpl {
protected:
  void *data_;
  int64_t size_;

  // The scalar type of this storage.  We need this in case we need to do placement-new/placement-delete
  // after allocation
  ScalarType scalarType_;

  char flag_;
  THAllocator *allocator_;
  void *allocatorContext_;
  StorageImpl *view_;

  StorageImpl(StorageImpl&&) = default;
  ~StorageImpl() = default;
  StorageImpl& operator=(StorageImpl&&) = default;

public:
  StorageImpl(void* data, int64_t size, char flag, THAllocator *allocator, void *allocatorContext)
    : data_(data), size_(size), flag_(flag), allocator_(allocator), allocatorContext_(allocatorContext), refcount(1) {}

  StorageImpl(const StorageImpl&) = delete;
  StorageImpl& operator=(const StorageImpl&) = delete;

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

  THAllocator *allocator() {
    return allocator_;  
  }

  void *allocatorContext() {
    return allocatorContext_;
  }

  StorageImpl *view() {
    return view_;
  }

  StorageImpl *view() const {
    return view_;
  }

  void resize(int64_t size, int64_t realsize) {
    if(this->flag_ & TH_STORAGE_RESIZABLE)
    {
      if(this->allocator()->realloc == NULL) {
        /* case when the allocator does not have a realloc defined */
        real *old_data = this->data<real>();
        ptrdiff_t old_size = this->size();
        if (size == 0) {
          this->data_ = NULL;
        } else {
          this->data_ = this->allocator()->malloc(
              this->allocatorContext(),
              realsize*size);
        }
        this->size_ = size;
        if (old_data != NULL) {
          ptrdiff_t copy_size = old_size;
          if (this->size() < copy_size) {
            copy_size = this->size();
          }
          if (copy_size > 0) {
            memcpy(this->data_ptr(), old_data, realsize*copy_size);
          }
          this->allocator()->free(this->allocatorContext(), old_data);
        }
      } else {
        this->data_ = this->allocator()->realloc(
                this->allocatorContext(),
                this->data_,
                realsize*size);
        this->size_ = size;
      }
    } else {
      AT_ERROR("Trying to resize storage that is not resizable");
    }
  }

  void swap(at::StorageImpl *other) {
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

