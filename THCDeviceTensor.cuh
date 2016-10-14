#ifndef THC_DEVICE_TENSOR_INC
#define THC_DEVICE_TENSOR_INC

#include <cuda.h>
#include <cuda_runtime.h>

// A CUDA 6.5 compatible version of static_assert. Remove once on CUDA 7.0.
template <bool>
struct THCStaticAssert;

template <>
struct THCStaticAssert<true> {
};

#define thc_static_assert(expr) (THCStaticAssert<(expr) != 0>())

/// Our tensor type
template <typename T,
          int Dim,
          typename IndexT,
          template <typename U> class PtrTraits>
class THCDeviceTensor;

/// Type of a subspace of a tensor
namespace detail {
template <typename TensorType,
          int SubDim,
          template <typename U> class PtrTraits>
class THCDeviceSubTensor;
}

template <typename T>
struct RestrictPtrTraits {
  typedef T* __restrict__ PtrType;
};

template <typename T>
struct DefaultPtrTraits {
  typedef T* PtrType;
};

/**
   Templated multi-dimensional array that supports strided access of
   elements. Main access is through `operator[]`; e.g.,
   `tensor[x][y][z]`.

- `T` is the contained type (e.g., `float`)
- `Dim` is the tensor rank
- `IndexT` is the integer type used for size/stride arrays, and for
- all indexing math. Default is `int`, but for large tensors, `long`
- can be used instead.
- `PtrTraits` are traits applied to our data pointer (T*). By default,
- this is just T*, but RestrictPtrTraits can be used to apply T*
- __restrict__ for alias-free analysis.
*/
template <typename T,
          int Dim,
          typename IndexT = int,
          template <typename U> class PtrTraits = DefaultPtrTraits>
class THCDeviceTensor {
 public:
  enum { NumDim = Dim };
  typedef T DataType;
  typedef IndexT IndexType;
  typedef typename PtrTraits<T>::PtrType DataPtrType;
  typedef THCDeviceTensor<T, Dim, IndexT, PtrTraits> TensorType;

  /// Default constructor
  __host__ __device__ THCDeviceTensor();

  /// Constructor that calculates strides with no padding
  __host__ __device__ THCDeviceTensor(DataPtrType data,
#ifdef _MSC_VER
                                      const IndexT (&sizes)[Dim]);
#else
                                      const IndexT sizes[Dim]);
#endif

  /// Constructor that takes arbitrary size/stride arrays
  __host__ __device__ THCDeviceTensor(DataPtrType data,
#ifdef _MSC_VER
                                      const IndexT (&sizes)[Dim],
                                      const IndexT (&strides)[Dim]);
#else
                                      const IndexT sizes[Dim],
                                      const IndexT strides[Dim]);
#endif

  /// Returns true if the two tensors are of the same dimensionality,
  /// size and stride.
  template <int OtherDim>
  __host__ __device__ bool
  isSameSizeAndStride(
    const THCDeviceTensor<T, OtherDim, IndexT, PtrTraits>& rhs) const;

  /// Cast to a tensor of a different type of the same size and stride
  template <typename U>
  __host__ __device__ THCDeviceTensor<U, Dim, IndexT, PtrTraits> cast();

  /// Const version of `cast`
  template <typename U>
  __host__ __device__
  const THCDeviceTensor<U, Dim, IndexT, PtrTraits> cast() const;

  /// Returns a raw pointer to the start of our data.
  __host__ __device__ __forceinline__ DataPtrType data() {
    return data_;
  }

  /// Returns a raw pointer to the start of our data (const).
  __host__ __device__ __forceinline__
  const DataPtrType data() const {
    return data_;
  }

  /// Cast to a different datatype
  template <typename U>
  __host__ __device__ __forceinline__
  typename PtrTraits<U>::PtrType dataAs() {
    return reinterpret_cast<typename PtrTraits<U>::PtrType>(data_);
  }

  /// Cast to a different datatype
  template <typename U>
  __host__ __device__ __forceinline__
  const typename PtrTraits<const U>::PtrType dataAs() const {
    return reinterpret_cast<typename PtrTraits<const U>::PtrType>(data_);
  }

  /// Returns a read/write view of a portion of our tensor.
  __host__ __device__ __forceinline__
  detail::THCDeviceSubTensor<TensorType, Dim - 1, PtrTraits>
    operator[](IndexT);

  /// Returns a read/write view of a portion of our tensor (const).
  __host__ __device__ __forceinline__
  const detail::THCDeviceSubTensor<TensorType, Dim - 1, PtrTraits>
    operator[](IndexT) const;

  /// Returns the size of a given dimension, `[0, Dim - 1]`. No bounds
  /// checking.
  __host__ __device__ __forceinline__ int getSize(int i) const {
    return size_[i];
  }

  /// Returns the stride of a given dimension, `[0, Dim - 1]`. No bounds
  /// checking.
  __host__ __device__ __forceinline__ int getStride(int i) const {
    return stride_[i];
  }

  /// Returns the total number of elements contained within our data
  /// (product of `getSize(i)`)
  __host__ __device__ ptrdiff_t numElements() const;

  /// Returns the size array.
  __host__ __device__ __forceinline__ const IndexT* sizes() const {
    return size_;
  }

  /// Returns the stride array.
  __host__ __device__ __forceinline__ const IndexT* strides() const {
    return stride_;
  }

  /// Returns true if there is no padding within the tensor and no
  /// re-ordering of the dimensions.
  /// ~~~
  /// (stride(i) == size(i + 1) * stride(i + 1)) && stride(dim - 1) == 0
  /// ~~~
  __host__ __device__ bool isContiguous() const;

  /// Returns whether a given dimension has only increasing stride
  /// from the previous dimension. A tensor that was permuted by
  /// exchanging size and stride only will fail this check.
  /// If `i == 0` just check `size > 0`. Returns `false` if `stride` is `<= 0`.
  __host__ __device__ bool isConsistentlySized(int i) const;

  // Returns whether at each dimension `stride <= size`.
  // If this is not the case then iterating once over the size space will
  // touch the same memory locations multiple times.
  __host__ __device__ bool isConsistentlySized() const;

  /// Returns true if the given dimension index has no padding
  __host__ __device__ bool isContiguousDim(int i) const;

  /// Returns a tensor of the same dimension after transposing the two
  /// dimensions given. Does not actually move elements; transposition
  /// is made by permuting the size/stride arrays.
  /// If the dimensions are not valid, asserts.
  __host__ __device__ THCDeviceTensor<T, Dim, IndexT, PtrTraits>
  transpose(int dim1, int dim2) const;

  /// Upcast a tensor of dimension `D` to some tensor of dimension
  /// D' > D by padding the leading dimensions by 1
  /// e.g., upcasting a 2-d tensor `[2][3]` to a 4-d tensor `[1][1][2][3]`
  template <int NewDim>
  __host__ __device__ THCDeviceTensor<T, NewDim, IndexT, PtrTraits>
  upcastOuter();

  /// Upcast a tensor of dimension `D` to some tensor of dimension
  /// D' > D by padding the lowest/most varying dimensions by 1
  /// e.g., upcasting a 2-d tensor `[2][3]` to a 4-d tensor `[2][3][1][1]`
  template <int NewDim>
  __host__ __device__ THCDeviceTensor<T, NewDim, IndexT, PtrTraits>
  upcastInner();

  /// Downcast a tensor of dimension `D` to some tensor of dimension
  /// D' < D by collapsing the leading dimensions. asserts if there is
  /// padding on the leading dimensions.
  template <int NewDim>
  __host__ __device__
  THCDeviceTensor<T, NewDim, IndexT, PtrTraits> downcastOuter();

  /// Downcast a tensor of dimension `D` to some tensor of dimension
  /// D' < D by collapsing the leading dimensions. asserts if there is
  /// padding on the leading dimensions.
  template <int NewDim>
  __host__ __device__
  THCDeviceTensor<T, NewDim, IndexT, PtrTraits> downcastInner();

  /// Returns a tensor that is a view of the `SubDim`-dimensional slice
  /// of this tensor, starting at `at`.
  template <int SubDim>
  __host__ __device__ THCDeviceTensor<T, SubDim, IndexT, PtrTraits>
  view(DataPtrType at);

  /// Returns a tensor that is a view of the `SubDim`-dimensional slice
  /// of this tensor, starting where our data begins
  template <int SubDim>
  __host__ __device__ THCDeviceTensor<T, SubDim, IndexT, PtrTraits>
  view();

  /// Zeroes out the tensor asynchronously. Asserts if the contents
  /// in question are not contiguous.
  void zero(cudaStream_t stream = 0);

 private:
  /// Raw pointer to where the tensor data begins
  DataPtrType data_;

  /// Array of strides (in sizeof(T) terms) per each dimension
  IndexT stride_[Dim];

  /// Size per each dimension
  IndexT size_[Dim];
};

namespace detail {

/// Specialization for a view of a single value (0-dimensional)
template <typename TensorType, template <typename U> class PtrTraits>
class THCDeviceSubTensor<TensorType, 0, PtrTraits> {
 public:
  __host__ __device__ THCDeviceSubTensor<TensorType, 0, PtrTraits>
  operator=(typename TensorType::DataType val) {
    *data_ = val;
    return *this;
  }

  // operator T&
  __host__ __device__ operator typename TensorType::DataType&() {
    return *data_;
  }

  // const operator T& returning const T&
  __host__ __device__ operator const typename TensorType::DataType&() const {
    return *data_;
  }

  // operator& returning T*
  __host__ __device__ typename TensorType::DataType* operator&() {
    return data_;
  }

  // const operator& returning const T*
  __host__ __device__ const typename TensorType::DataType* operator&() const {
    return data_;
  }

  /// Returns a raw accessor to our slice.
  __host__ __device__ __forceinline__ typename TensorType::DataPtrType data() {
    return data_;
  }

  /// Returns a raw accessor to our slice (const).
  __host__ __device__ __forceinline__
  const typename TensorType::DataPtrType data() const {
    return data_;
  }

  /// Cast to a different datatype.
  template <typename T>
  __host__ __device__ T& as() {
    return *dataAs<T>();
  }

  /// Cast to a different datatype (const).
  template <typename T>
  __host__ __device__ const T& as() const {
    return *dataAs<T>();
  }

  /// Cast to a different datatype
  template <typename T>
  __host__ __device__ __forceinline__
  typename PtrTraits<T>::PtrType dataAs() {
    return reinterpret_cast<typename PtrTraits<T>::PtrType>(data_);
  }

  /// Cast to a different datatype (const)
  template <typename T>
  __host__ __device__ __forceinline__
  typename PtrTraits<const T>::PtrType dataAs() const {
    return reinterpret_cast<typename PtrTraits<const T>::PtrType>(data_);
  }

  /// Use the texture cache for reads
  __device__ __forceinline__ typename TensorType::DataType ldg() const {
#if __CUDA_ARCH__ >= 350
    return __ldg(data_);
#else
    return *data_;
#endif
  }

  /// Use the texture cache for reads; cast as a particular type
  template <typename T>
  __device__ __forceinline__ T ldgAs() const {
#if __CUDA_ARCH__ >= 350
    return __ldg(dataAs<T>());
#else
    return as<T>();
#endif
  }

  private:
  /// One dimension greater can create us
  friend class THCDeviceSubTensor<TensorType, 1, PtrTraits>;

  /// Our parent tensor can create us
  friend class THCDeviceTensor<typename TensorType::DataType,
                               1,
                               typename TensorType::IndexType,
                               PtrTraits>;

  __host__ __device__ __forceinline__ THCDeviceSubTensor(
    TensorType& t,
    typename TensorType::DataPtrType data)
      : tensor_(t),
        data_(data) {
  }

  /// The tensor we're referencing
  TensorType& tensor_;

  /// Where our value is located
  typename TensorType::DataPtrType const data_;
};

/// A `SubDim`-rank slice of a parent THCDeviceTensor
template <typename TensorType,
          int SubDim,
          template <typename U> class PtrTraits>
class THCDeviceSubTensor {
 public:
  /// Returns a view of the data located at our offset (the dimension
  /// `SubDim` - 1 tensor).
  __host__ __device__ __forceinline__
  THCDeviceSubTensor<TensorType, SubDim - 1, PtrTraits>
    operator[](typename TensorType::IndexType index) {
    return THCDeviceSubTensor<TensorType, SubDim - 1, PtrTraits>(
      tensor_,
      data_ + index * tensor_.getStride(TensorType::NumDim - SubDim));
  }

  /// Returns a view of the data located at our offset (the dimension
  /// `SubDim` - 1 tensor) (const).
  __host__ __device__ __forceinline__
  const THCDeviceSubTensor<TensorType, SubDim - 1, PtrTraits>
    operator[](typename TensorType::IndexType index) const {
    return THCDeviceSubTensor<TensorType, SubDim - 1, PtrTraits>(
      tensor_,
      data_ + index * tensor_.getStride(TensorType::NumDim - SubDim));
  }

  // operator& returning T*
  __host__ __device__ typename TensorType::DataType* operator&() {
    return data_;
  }

  // const operator& returning const T*
  __host__ __device__ const typename TensorType::DataType* operator&() const {
    return data_;
  }

  /// Returns a raw accessor to our slice.
  __host__ __device__ __forceinline__ typename TensorType::DataPtrType data() {
    return data_;
  }

  /// Returns a raw accessor to our slice (const).
  __host__ __device__ __forceinline__
  const typename TensorType::DataPtrType data() const {
    return data_;
  }

  /// Cast to a different datatype.
  template <typename T>
  __host__ __device__ T& as() {
    return *dataAs<T>();
  }

  /// Cast to a different datatype (const).
  template <typename T>
  __host__ __device__ const T& as() const {
    return *dataAs<T>();
  }

  /// Cast to a different datatype
  template <typename T>
  __host__ __device__ __forceinline__
  typename PtrTraits<T>::PtrType dataAs() {
    return reinterpret_cast<typename PtrTraits<T>::PtrType>(data_);
  }

  /// Cast to a different datatype (const)
  template <typename T>
  __host__ __device__ __forceinline__
  typename PtrTraits<const T>::PtrType dataAs() const {
    return reinterpret_cast<typename PtrTraits<const T>::PtrType>(data_);
  }

  /// Use the texture cache for reads
  __device__ __forceinline__ typename TensorType::DataType ldg() const {
#if __CUDA_ARCH__ >= 350
    return __ldg(data_);
#else
    return *data_;
#endif
  }

  /// Use the texture cache for reads; cast as a particular type
  template <typename T>
  __device__ __forceinline__ T ldgAs() const {
#if __CUDA_ARCH__ >= 350
    return __ldg(dataAs<T>());
#else
    return as<T>();
#endif
  }

  /// Returns a tensor that is a view of the SubDim-dimensional slice
  /// of this tensor, starting where our data begins
  THCDeviceTensor<typename TensorType::DataType,
               SubDim,
               typename TensorType::IndexType,
               PtrTraits> view() {
    return tensor_.template view<SubDim>(data_);
  }

 private:
  /// One dimension greater can create us
  friend class THCDeviceSubTensor<TensorType, SubDim + 1, PtrTraits>;

  /// Our parent tensor can create us
  friend class
  THCDeviceTensor<typename TensorType::DataType,
               TensorType::NumDim,
               typename TensorType::IndexType,
               PtrTraits>;

  __host__ __device__ __forceinline__ THCDeviceSubTensor(
    TensorType& t,
    typename TensorType::DataPtrType data)
      : tensor_(t),
        data_(data) {
  }

  /// The tensor we're referencing
  TensorType& tensor_;

  /// The start of our sub-region
  typename TensorType::DataPtrType const data_;
};

} // namespace detail

template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
__host__ __device__ __forceinline__
detail::THCDeviceSubTensor<THCDeviceTensor<T, Dim, IndexT, PtrTraits>,
                        Dim - 1, PtrTraits>
THCDeviceTensor<T, Dim, IndexT, PtrTraits>::operator[](IndexT index) {
  return detail::THCDeviceSubTensor<TensorType, Dim - 1, PtrTraits>(
    detail::THCDeviceSubTensor<TensorType, Dim, PtrTraits>(
      *this, data_)[index]);
}

template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
__host__ __device__ __forceinline__
const detail::THCDeviceSubTensor<THCDeviceTensor<T, Dim, IndexT, PtrTraits>,
                              Dim - 1, PtrTraits>
THCDeviceTensor<T, Dim, IndexT, PtrTraits>::operator[](IndexT index) const {
  return detail::THCDeviceSubTensor<TensorType, Dim - 1, PtrTraits>(
    detail::THCDeviceSubTensor<TensorType, Dim, PtrTraits>(
      const_cast<TensorType&>(*this), data_)[index]);
}

#include "THCDeviceTensor-inl.cuh"

#endif // THC_DEVICE_TENSOR_INC
