#pragma once

#include <cstddef>
#include <stdint.h>

namespace at {

template <typename T>
struct DefaultPtrTraits {
  typedef T* PtrType;
};

#ifdef __CUDACC__
template <typename T>
struct RestrictPtrTraits {
  typedef T* __restrict__ PtrType;
};
#endif

#ifndef AT_HOSTDEVICE
#ifdef __CUDACC__
#define AT_HOSTDEVICE __host__ __device__
#else
#define AT_HOSTDEVICE
#endif
#endif

template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits>
class TensorAccessorBase {
public:
  typedef typename PtrTraits<T>::PtrType PtrType;

  AT_HOSTDEVICE TensorAccessorBase(PtrType data_, const int64_t * sizes_, const int64_t * strides_)
  : data_(data_), sizes_(sizes_), strides_(strides_) {}
  AT_HOSTDEVICE IntList sizes() {
    return IntList(sizes_,N);
  }
  AT_HOSTDEVICE IntList strides() {
    return IntList(strides_,N);
  }
  AT_HOSTDEVICE int64_t stride(int64_t i) { return strides_[i]; }
  AT_HOSTDEVICE int64_t size(int64_t i) { return sizes_[i]; }
protected:
  PtrType data_;
  const int64_t* sizes_;
  const int64_t* strides_;
};

template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits>
class TensorAccessor : public TensorAccessorBase<T,N,PtrTraits> {
public:
  typedef typename PtrTraits<T>::PtrType PtrType;

  AT_HOSTDEVICE TensorAccessor(PtrType data_, const int64_t * sizes_, const int64_t * strides_)
  : TensorAccessorBase<T,N>(data_,sizes_,strides_) {}

  AT_HOSTDEVICE TensorAccessor<T,N-1> operator[](int64_t i) {
    return TensorAccessor<T,N-1>(this->data_ + this->strides_[0]*i,this->sizes_+1,this->strides_+1);
  }
};

template<typename T, template <typename U> class PtrTraits>
class TensorAccessor<T,1,PtrTraits> : public TensorAccessorBase<T,1,PtrTraits> {
public:
  typedef typename PtrTraits<T>::PtrType PtrType;

  AT_HOSTDEVICE TensorAccessor(PtrType data_, const int64_t * sizes_, const   int64_t * strides_)
  : TensorAccessorBase<T,1,PtrTraits>(data_,sizes_,strides_) {}
  AT_HOSTDEVICE T & operator[](int64_t i) {
    return this->data_[this->strides_[0]*i];
  }
};

template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits>
class PackedTensorAccessorBase {
public:
  typedef typename PtrTraits<T>::PtrType PtrType;
  AT_HOSTDEVICE PackedTensorAccessorBase(PtrType data_, const int64_t * sizes_, const   int64_t * strides_)
  : data_(data_)
  {
    std::copy(sizes_, sizes_ + N, std::begin(this->sizes_));
    std::copy(strides_, strides_ + N, std::begin(this->strides_));
  }
  AT_HOSTDEVICE int64_t stride(int64_t i) { return strides_[i]; }
  AT_HOSTDEVICE int64_t size(int64_t i) { return sizes_[i]; }
protected:
  PtrType data_;
  int64_t sizes_[N];
  int64_t strides_[N];
};

template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits>
class PackedTensorAccessor : public PackedTensorAccessorBase<T,N,PtrTraits> {
public:
  typedef typename PtrTraits<T>::PtrType PtrType;

  AT_HOSTDEVICE PackedTensorAccessor(PtrType data_, const int64_t * sizes_, const   int64_t * strides_)
  : PackedTensorAccessorBase<T,N,PtrTraits>(data_, sizes_, strides_) {};

  AT_HOSTDEVICE TensorAccessor<T,N-1> operator[](int64_t i) {
    int64_t* new_sizes = this->sizes_+1;
    int64_t* new_strides = this->strides_+1;
    return TensorAccessor<T,N-1>(this->data_ + this->strides_[0]*i, new_sizes, new_strides);
  }
};

template<typename T, template <typename U> class PtrTraits>
class PackedTensorAccessor<T,1,PtrTraits> : public PackedTensorAccessorBase<T,1,PtrTraits> {
public:
  typedef typename PtrTraits<T>::PtrType PtrType;
  AT_HOSTDEVICE PackedTensorAccessor(PtrType data_, const int64_t * sizes_, const   int64_t * strides_)
  : PackedTensorAccessorBase<T,1,PtrTraits>(data_, sizes_, strides_) {};

  AT_HOSTDEVICE T & operator[](int64_t i) {
    return this->data_[this->strides_[0]*i];
  }
};

}

#undef AT_HOSTDEVICE
