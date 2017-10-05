#pragma once

#include <cstddef>
#include <stdint.h>

#include "ATen/ScalarType.h"

namespace at {


template<typename T, size_t N>
class TensorAccessorBase {
public:
  TensorAccessorBase(T * data_, const int64_t * sizes_, const int64_t * strides_)
  : data_(data_), sizes_(sizes_), strides_(strides_) {}
  IntList sizes() {
    return IntList(sizes_,N);
  }
  IntList strides() {
    return IntList(strides_,N);
  }
  int64_t stride(int64_t i) { return strides()[i]; }
  int64_t size(int64_t i) { return sizes()[i]; }
protected:
  T * data_;
  const int64_t* sizes_;
  const int64_t* strides_;
};

template<typename T, size_t N>
class TensorAccessor : public TensorAccessorBase<T,N> {
public:
  TensorAccessor(T * data_, const int64_t * sizes_, const int64_t * strides_)
  : TensorAccessorBase<T,N>(data_,sizes_,strides_) {}

  TensorAccessor<T,N-1> operator[](int64_t i) {
    return TensorAccessor<T,N-1>(this->data_ + this->strides_[0]*i,this->sizes_+1,this->strides_+1);
  }
};

template<typename T>
class TensorAccessor<T,1> : public TensorAccessorBase<T,1> {
public:
  TensorAccessor(T * data_, const int64_t * sizes_, const   int64_t * strides_)
  : TensorAccessorBase<T,1>(data_,sizes_,strides_) {}
  T & operator[](int64_t i) {
    return this->data_[this->strides_[0]*i];
  }
};

}
