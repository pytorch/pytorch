#pragma once

#include <cstddef>
#include <stdint.h>

#include "ATen/ScalarType.h"
#include "ATen/core/TensorAccessor.h"

namespace at {
namespace cuda {

template <typename T>
struct RestrictPtrTraits {
  typedef T* __restrict__ PtrType;
};

template <typename T>
struct DefaultPtrTraits {
  typedef T* PtrType;
};

namespace detail {

template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits>
class CUDATensorSubAccessorBase {
public:
  typedef typename PtrTraits<T>::PtrType PtrType;

  __device__ CUDATensorSubAccessorBase(PtrType data_, const int64_t * sizes_, const int64_t * strides_)
  : data_(data_), sizes_(sizes_), strides_(strides_) {}
  __device__ int64_t stride(int64_t i) { return strides_[i]; }
  __device__ int64_t size(int64_t i) { return sizes_[i]; }
protected:
  PtrType data_;
  const int64_t* sizes_;
  const int64_t* strides_;
};

template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits>
class CUDATensorSubAccessor : public CUDATensorSubAccessorBase<T,N,PtrTraits> {
public:
  typedef typename PtrTraits<T>::PtrType PtrType;
  __device__ CUDATensorSubAccessor(PtrType data_, const int64_t * sizes_, const int64_t * strides_)
  : CUDATensorSubAccessorBase<T,N>(data_ ,sizes_, strides_) {}

  __device__ CUDATensorSubAccessor<T,N-1> operator[](int64_t i) {
    return CUDATensorSubAccessor<T,N-1>(this->data_ + this->strides_[0]*i,this->sizes_+1,this->strides_+1);
  }
};

template<typename T, template <typename U> class PtrTraits>
class CUDATensorSubAccessor<T,1,PtrTraits> : public CUDATensorSubAccessorBase<T,1,PtrTraits> {
public:
  typedef typename PtrTraits<T>::PtrType PtrType;
  __device__ CUDATensorSubAccessor(PtrType data_, const int64_t * sizes_, const   int64_t * strides_)
  : CUDATensorSubAccessorBase<T,1>(data_,sizes_,strides_) {}
  __device__ T & operator[](int64_t i) {
    return this->data_[this->strides_[0]*i];
  }
};

template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits>
class CUDATensorAccessorBase {
public:
  typedef typename PtrTraits<T>::PtrType PtrType;
  CUDATensorAccessorBase(TensorAccessor<T, N>& ta)
  {
    data_ = static_cast<PtrType>(ta.data());
    IntList sizes = ta.sizes();
    IntList strides = ta.strides();
    for (size_t i = 0; i < N; i++) {
      sizes_[i] = sizes[i];
      strides_[i] = strides[i];
    }
  }
  __host__ __device__ int64_t stride(int64_t i) { return strides_[i]; }
  __host__ __device__ int64_t size(int64_t i) { return sizes_[i]; }
protected:
  PtrType data_;
  int64_t sizes_[N];
  int64_t strides_[N];
};

} // namespace detail

template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits>
class CUDATensorAccessor : public detail::CUDATensorAccessorBase<T,N,PtrTraits> {
public:
  typedef typename PtrTraits<T>::PtrType PtrType;

  CUDATensorAccessor(TensorAccessor<T,N> ta)
    : detail::CUDATensorAccessorBase<T,N>(ta) {};

  __device__ detail::CUDATensorSubAccessor<T,N-1> operator[](int64_t i) {
    int64_t* new_sizes = this->sizes_+1;
    int64_t* new_strides = this->strides_+1;
    return detail::CUDATensorSubAccessor<T,N-1>(this->data_ + this->strides_[0]*i, new_sizes, new_strides);
  }
};

template<typename T, template <typename U> class PtrTraits>
class CUDATensorAccessor<T,1,PtrTraits> : public detail::CUDATensorAccessorBase<T,1,PtrTraits> {
public:
  typedef typename PtrTraits<T>::PtrType PtrType;
  CUDATensorAccessor(TensorAccessor<T,1> ta)
    : detail::CUDATensorAccessorBase<T,1>(ta) {}

  __device__ T & operator[](int64_t i) {
    return this->data_[this->strides_[0]*i];
  }
};

} } // at::cuda

