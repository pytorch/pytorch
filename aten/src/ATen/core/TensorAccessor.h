#pragma once

#include <c10/macros/Macros.h>
#include <stdint.h>
#include <cstddef>

namespace at {

// The PtrTraits argument to the TensorAccessor/PackedTensorAccessor
// is used to enable the __restrict__ keyword/modifier for the data
// passed to cuda.
template <typename T>
struct DefaultPtrTraits {
  typedef T* PtrType;
};

#if defined(__CUDACC__) || defined(__HIPCC__)
template <typename T>
struct RestrictPtrTraits {
  typedef T* __restrict__ PtrType;
};
#endif

// TensorAccessorBase and TensorAccessor are used for both CPU and CUDA tensors.
// For CUDA tensors it is used in device code (only). This means that we restrict ourselves
// to functions and types available there (e.g. IntList isn't).

// The PtrTraits argument is only relevant to cuda to support `__restrict__` pointers.
template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits>
class TensorAccessorBase {
public:
  typedef typename PtrTraits<T>::PtrType PtrType;

  C10_HOST_DEVICE TensorAccessorBase(
      PtrType data_,
      const int64_t* sizes_,
      const int64_t* strides_)
      : data_(data_), sizes_(sizes_), strides_(strides_) {}
  C10_HOST IntList sizes() const {
    return IntList(sizes_,N);
  }
  C10_HOST IntList strides() const {
    return IntList(strides_,N);
  }
  C10_HOST_DEVICE int64_t stride(int64_t i) const {
    return strides_[i];
  }
  C10_HOST_DEVICE int64_t size(int64_t i) const {
    return sizes_[i];
  }
  C10_HOST_DEVICE T* data() {
    return data_;
  }
  C10_HOST_DEVICE const T* data() const {
    return data_;
  }

protected:
  PtrType data_;
  const int64_t* sizes_;
  const int64_t* strides_;
};

// The `TensorAccessor` is typically instantiated for CPU `Tensor`s using
// `Tensor.accessor<T, N>()`.
// For CUDA `Tensor`s, `PackedTensorAccessor` is used on the host and only
// indexing on the device uses `TensorAccessor`s.
template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits>
class TensorAccessor : public TensorAccessorBase<T,N,PtrTraits> {
public:
  typedef typename PtrTraits<T>::PtrType PtrType;

  C10_HOST_DEVICE TensorAccessor(
      PtrType data_,
      const int64_t* sizes_,
      const int64_t* strides_)
      : TensorAccessorBase<T, N>(data_, sizes_, strides_) {}

  C10_HOST_DEVICE TensorAccessor<T, N - 1> operator[](int64_t i) {
    return TensorAccessor<T,N-1>(this->data_ + this->strides_[0]*i,this->sizes_+1,this->strides_+1);
  }

  C10_HOST_DEVICE const TensorAccessor<T, N - 1> operator[](int64_t i) const {
    return TensorAccessor<T,N-1>(this->data_ + this->strides_[0]*i,this->sizes_+1,this->strides_+1);
  }
};

template<typename T, template <typename U> class PtrTraits>
class TensorAccessor<T,1,PtrTraits> : public TensorAccessorBase<T,1,PtrTraits> {
public:
  typedef typename PtrTraits<T>::PtrType PtrType;

  C10_HOST_DEVICE TensorAccessor(
      PtrType data_,
      const int64_t* sizes_,
      const int64_t* strides_)
      : TensorAccessorBase<T, 1, PtrTraits>(data_, sizes_, strides_) {}
  C10_HOST_DEVICE T& operator[](int64_t i) {
    return this->data_[this->strides_[0]*i];
  }
};


// PackedTensorAccessorBase and PackedTensorAccessor are used on for CUDA `Tensor`s on the host
// and as
// In contrast to `TensorAccessor`s, they copy the strides and sizes on instantiation (on the host)
// in order to transfer them on the device when calling kernels.
// On the device, indexing of multidimensional tensors gives to `TensorAccessor`s.
// Use RestrictPtrTraits as PtrTraits if you want the tensor's data pointer to be marked as __restrict__.
// Instantiation from data, sizes, strides is only needed on the host and std::copy isn't available
// on the device, so those functions are host only.
template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits>
class PackedTensorAccessorBase {
public:
  typedef typename PtrTraits<T>::PtrType PtrType;
  C10_HOST PackedTensorAccessorBase(
      PtrType data_,
      const int64_t* sizes_,
      const int64_t* strides_)
      : data_(data_) {
    std::copy(sizes_, sizes_ + N, std::begin(this->sizes_));
    std::copy(strides_, strides_ + N, std::begin(this->strides_));
  }
  C10_HOST_DEVICE int64_t stride(int64_t i) const {
    return strides_[i];
  }
  C10_HOST_DEVICE int64_t size(int64_t i) const {
    return sizes_[i];
  }

protected:
  PtrType data_;
  int64_t sizes_[N];
  int64_t strides_[N];
};

template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits>
class PackedTensorAccessor : public PackedTensorAccessorBase<T,N,PtrTraits> {
public:
  typedef typename PtrTraits<T>::PtrType PtrType;

  C10_HOST PackedTensorAccessor(
      PtrType data_,
      const int64_t* sizes_,
      const int64_t* strides_)
      : PackedTensorAccessorBase<T, N, PtrTraits>(data_, sizes_, strides_){};

  C10_DEVICE TensorAccessor<T, N - 1> operator[](int64_t i) {
    int64_t* new_sizes = this->sizes_+1;
    int64_t* new_strides = this->strides_+1;
    return TensorAccessor<T,N-1>(this->data_ + this->strides_[0]*i, new_sizes, new_strides);
  }

  C10_DEVICE const TensorAccessor<T, N - 1> operator[](int64_t i) const {
    int64_t* new_sizes = this->sizes_+1;
    int64_t* new_strides = this->strides_+1;
    return TensorAccessor<T,N-1>(this->data_ + this->strides_[0]*i, new_sizes, new_strides);
  }
};

template<typename T, template <typename U> class PtrTraits>
class PackedTensorAccessor<T,1,PtrTraits> : public PackedTensorAccessorBase<T,1,PtrTraits> {
public:
  typedef typename PtrTraits<T>::PtrType PtrType;
  C10_HOST PackedTensorAccessor(
      PtrType data_,
      const int64_t* sizes_,
      const int64_t* strides_)
      : PackedTensorAccessorBase<T, 1, PtrTraits>(data_, sizes_, strides_){};

  C10_DEVICE T& operator[](int64_t i) {
    return this->data_[this->strides_[0]*i];
  }
  C10_DEVICE const T& operator[](int64_t i) const {
    return this->data_[this->strides_[0]*i];
  }
};

}
