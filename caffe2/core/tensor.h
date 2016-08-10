#ifndef CAFFE2_CORE_TENSOR_H_
#define CAFFE2_CORE_TENSOR_H_

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <typeinfo>
#include <type_traits>
#include <vector>

#include "caffe2/core/common.h"
#include "caffe2/core/flags.h"
#include "caffe2/core/context.h"
#include "caffe2/core/typeid.h"
#include "caffe2/core/logging.h"

// A global boolean variable to control whether we free memory when a Tensor
// is shrinked to a smaller size. As a result, a Tensor is always going to
// keep the memory allocated for its maximum capacity reshaped to so far.
// This is disabled by default unless explicitly enabled by the commandline
// argument.
CAFFE2_DECLARE_bool(caffe2_keep_on_shrink);

namespace caffe2 {

// Data type for Tensor Index. We use size_t to be safe here as well as for
// large matrices that are common in sparse math.
typedef int64_t TIndex;

/**
 * A utility function to convert vector<int> to vector<TIndex>.
 */
inline vector<TIndex> ToVectorTIndex(const std::vector<int>& src) {
  return vector<TIndex>(src.begin(), src.end());
}

/**
 * @brief Tensor is the basic class in Caffe2 that stores a contiguous memory
 * with its shape information.
 *
 * The Tensor class is essentially a wrapper around a device-specific memory
 * (the device is specified by the Context template argument), and deals with
 * the allocation and de-allocation of such memory. We make a simplified
 * assumption that the memory is always contiguous.
 */
template <class Context>
class Tensor {
 public:
  /**
   * Initializes an empty tensor.
   */
  Tensor() {}

  /**
   * @brief Creates a tensor of the given dimension.
   *
   * Note that the actual data allocation is not going to be carried out until
   * the first time mutable_data() is called.
   */
  explicit Tensor(const vector<TIndex>& dims) { Resize(dims); }
  explicit Tensor(const vector<int>& dims) { Resize(dims); }

  /**
   * @brief Creates a tensor from a source tensor, copying over the content.
   *
   * Note that the source tensor can be from a different device context. The
   * second argument provides a device context object (either Context or
   * SrcContext) that will be responsible for copying the underlying data.
   * If you do not wish to pass in a Context object, an equivalent constructor
   * function exists that will create an implicit context object for copy, but
   * be noted that this will cause a potential performance hit.
   */
  template <class SrcContext, class ContextForCopy>
  Tensor(const Tensor<SrcContext>& src, ContextForCopy* context) {
    CopyFrom(src, context);
  }

  /**
   * @brief Creates a tensor from a source tensor, copying over the content.
   *
   * Note that this may have a potential performance hit, since a temporary
   * context object will be created for the memory copy. Prefer explicitly
   * providing a context for copy if you can.
   */
  template <class SrcContext>
  Tensor(const Tensor<SrcContext>& src) {
    CopyFrom(src);
  }

  /**
   * @brief Creates a tensor, and fills its contents with the given values.
   */
  template <typename T>
  Tensor(const vector<TIndex>& dims, const vector<T>& values, Context* context)
      : meta_(TypeMeta::Make<T>()) {
    Resize(dims);
    CHECK_EQ(values.size(), size_);
    T* data = mutable_data<T>();
    for (TIndex i = 0; i < size_; ++i) {
      data[i] = values[i];
    }
  }

  /**
   * @brief Creates a scalar tensor, and fills its content with the given value.
   */
  template <typename T,
            typename = typename std::enable_if<std::is_scalar<T>::value>::type>
  Tensor(const T& value, Context* context) {
    Resize(vector<TIndex>{});
    T* data = mutable_data<T>();
    for (TIndex i = 0; i < size_; ++i) {
      data[i] = value;
    }
  }

  /**
   * @brief Copies the data from a source tensor, with a contex provided to
   * carry out the underlying memcpy operation.
   */
  template <class SrcContext, class ContextForCopy>
  void CopyFrom(const Tensor<SrcContext>& src, ContextForCopy* context) {
    if ((void*)&src == (void*)this) {
      return;
    }
    meta_ = src.meta();
    Resize(src.dims());
    if (size() > 0) {
      if (meta_.copy()) {
        meta_.copy()(src.raw_data(), raw_mutable_data(), size());
      } else {
        context->template CopyBytes<SrcContext, Context>(
            nbytes(), src.raw_data(), raw_mutable_data());
      }
    }
  }

  /**
   * @brief Copies the data from a source tensor.
   *
   * Note that this may have a potential performance hit, since a temporary
   * context object will be created for the memory copy. Prefer explicitly
   * providing a context for copy if you can.
   */
  template <class SrcContext>
  inline void CopyFrom(const Tensor<SrcContext>& src) {
    SrcContext tmp_context;
    CopyFrom(src, &tmp_context);
  }

  virtual ~Tensor() {}

  /**
   * @brief Extends the outer-most dimension of this tensor by num elements,
   * preserving the existing data.
   *
   * The underlying data may be reallocated in order to accommodate the new
   * elements, in which case this tensors' capacity is grown at a factor of
   * growthPct. This ensures that Extend runs on an amortized O(1) time
   * complexity.
   */
  template <class ContextForCopy>
  void Extend(TIndex num, int growthPct, ContextForCopy* context) {
    CHECK_GE(dims_.size(), 1);
    auto oldSize = size_;
    auto newDims = dims_;
    newDims[0] += num;
    if (!data_) {
      Resize(newDims);
      return;
    }
    auto newSize = std::accumulate(
        newDims.begin(), newDims.end(), 1, std::multiplies<TIndex>());
    if (newSize * meta_.itemsize() > capacity_) {
      auto newCapacity = dims_;
      newCapacity[0] = std::max(newDims[0], dims_[0] * (growthPct + 100) / 100);
      auto oldData = std::move(data_);
      Resize(newCapacity);
      auto* newData = raw_mutable_data(meta_);
      context->template CopyItems<ContextForCopy, ContextForCopy>(
          meta_, oldSize, oldData.get(), newData);
    }
    dims_ = newDims;
    size_ = newSize;
  }

  /**
   * @brief Resizes a tensor.
   *
   * Resize takes in a vector of ints specifying the dimensions of the tensor.
   * You can pass in an empty vector to specify that it is a scalar (i.e.
   * containing one single item).
   *
   * The underlying storage may be deleted after calling Resize: if the new
   * shape leads to a different number of items in the tensor, the old memory
   * is deleted and new memory will be allocated next time you call
   * mutable_data(). However, if the shape is different but the total number of
   * items is the same, the underlying storage is kept.
   */
  template <typename... Ts>
  void Resize(Ts... dim_source) {
    bool size_changed = SetDims(dim_source...);
    // If needed, we will free the data. the next mutable_data() call
    // will create the data storage.
    if (size_changed && (capacity_ < size_ * meta_.itemsize() ||
                         !FLAGS_caffe2_keep_on_shrink)) {
      data_.reset();
      capacity_ = 0;
    }
  }

  /**
   * Resize the tensor like the source tensor. Note that this is just a
   * sugar wrapper that essentially calls Resize(src_tensor.dims()).
   */
  template <class OtherContext>
  inline void ResizeLike(const Tensor<OtherContext>& src_tensor) {
    // Note: need casting for different context types.
    if (static_cast<void*>(this) != static_cast<const void*>(&src_tensor)) {
      Resize(src_tensor.dims());
    }
  }

  /**
   * Resizes the tensor without touching underlying storage.
   * This requires the total size of the tensor to remains constant.
   */
  inline void Reshape(const vector<TIndex>& dims) {
    TIndex new_size = 1;
    for (auto d : dims) {
      CHECK_GE(d, 0);
      new_size *= d;
    }
    CAFFE_ENFORCE(
        new_size == size_,
        "New size and old size are not equal. You cannot use Reshape, "
        "but should use Resize."
        // TODO(jiayq): remove the following warning after pending diffs
        // stabilize.
        " The old caffe2 mixes Reshape and Resize but this behavior has "
        "been changed. If you find this error, most likely you will need "
        "to change corresponding code from Reshape to Resize.");
    dims_ = dims;
  }

  inline void Reshape(const vector<int>& dims) {
    Reshape(ToVectorTIndex(dims));
  }

  /**
   * A utility function to print the debug string for the tensor. Note that this
   * is very slow since it involves quite some string operations, so do not use
   * it in your performance-critical code.
   */
  string DebugString() const {
    std::stringstream ss;
    ss << "A Tensor of item size " << itemsize() << " and type "
       << meta_.name() << " and dimension (";
    for (int d : dims_) {
      ss << d << ",";
    }
    ss << ").";
    return ss.str();
  }

  /**
   * @brief Shares the data with another tensor.
   *
   * To share data between two tensors, the sizes of the two tensors must be
   * equal already. The reason we do not implicitly do a Resize to make the two
   * tensors have the same shape is that, we want to allow tensors of different
   * shapes but the same number of items to still be able to share data. This
   * allows one to e.g. have a n-dimensional Tensor and a flattened version
   * sharing the same underlying storage.
   *
   * The source tensor should already have its data allocated.
   */
  void ShareData(const Tensor& src) {
    meta_ = src.meta();
    CHECK_EQ(src.size_, size_)
        << "Size mismatch - did you call reshape before sharing the data?";
    // It is possible that the source tensor hasn't called mutable_data() yet,
    // in which case ShareData() doesn't make much sense since we don't really
    // know what to share yet.
    CHECK(src.data_.get()) << "Source tensor has no content yet.";
    // Finally, do sharing.
    data_ = src.data_;
    capacity_ = src.capacity_;
  }

  /**
   * @brief Shares the data with an externally managed pointer.
   *
   * This is similar to SharData() but the tensor does not take over ownership
   * of the pointer, so the caller can explicitly manage the memory storage.
   * One needs to make sure that the external memory is deallocated only after
   * the tensor finishes using it.
   */
  template <typename T>
  void ShareExternalPointer(T* src, size_t capacity = 0) {
    meta_ = TypeMeta::Make<T>();
    CHECK(size_ > 0)
        << "To share data with a raw pointer, you need to set shape first.";
    data_.reset(src, [](void*)->void {});
    // Sets capacity. If not specified, we will implicitly assume that
    // the capacity is the current size.
    if (capacity) {
      capacity_ = capacity;
    } else {
      capacity_ = nbytes();
    }
  }

  /**
   * Returns a const raw void* pointer of the underlying storage. mutable_data()
   * or raw_mutable_data() must have been called prior to this function call.
   */
  inline const void* raw_data() const {
    CAFFE_ENFORCE(data_.get() || size_ == 0);
    return data_.get();
  }

  /**
   * Returns a typed pointer of the underlying storage. mutable_data() or
   * raw_mutable_data() must have been called prior to this function call, and
   * the data type must be of the correct type. If you want to get a void*
   * pointer instead, use raw_data().
   */
  template <typename T>
  inline const T* data() const {
    CAFFE_ENFORCE(
        data_.get() || size_ == 0,
        "The tensor is uninitialized. You probably need to call ",
        "Resize() and mutable_data() first.");
    CAFFE_ENFORCE(
        IsType<T>(),
        "Tensor type mistmatch, caller expects elements to be ",
        TypeMeta::Name<T>(),
        " while tensor contains ",
        meta_.name());
    return static_cast<T*>(data_.get());
  }

  /**
   * Returns a mutable raw pointer of the underlying storage. Since we will need
   * to know the type of the data for allocation, a TypeMeta object is passed in
   * to specify the necessary information. This is conceptually equivalent of
   * calling mutable_data<T>() where the TypeMeta parameter meta is derived from
   * the type T. This function differs from mutable_data<T>() in the sense that
   * the type T can be specified during runtime via the TypeMeta object.
   *
   * If the existing data does not match the desired type, it will be deleted
   * and a new storage will be created.
   */
  inline void* raw_mutable_data(const TypeMeta& meta) {
    // For 0-size tensors it's fine to return any pointer (including nullptr)
    if (meta_ == meta && (data_.get() || size_ == 0)) {
      return data_.get();
    } else {
      meta_ = meta;
      CHECK_GE(size_, 0)
          << "Tensor is not initialized. You probably need to call Resize() "
          << "before calling mutable_data()";
      if (size_ == 0) {
        return data_.get();
      }
      if (meta.ctor()) {
        // For types that need placement new, we will call it, as well as
        // making sure that when the data is freed, it calls the right
        // destruction procedure.
        auto size = size_;
        auto dtor = meta_.dtor();
        data_.reset(
            static_cast<void*>(Context::New(size_ * meta_.itemsize())),
            [size, dtor](void* ptr) -> void {
                dtor(ptr, size);
                Context::Delete(ptr);
            });
        meta_.ctor()(data_.get(), size_);
      } else {
        // For fundamental type, new and delete is easier.
        data_.reset(static_cast<void*>(Context::New(size_ * meta_.itemsize())),
                    Context::Delete);
      }
      capacity_ = size_ * meta_.itemsize();
      return data_.get();
    }
  }

  /**
   * Returns a mutable raw pointer of the underlying storage. This can only be
   * used when you know for sure that the underlying storage of the tensor is
   * already created via an earlier raw_mutable_data(meta) call or a
   * mutable_data<T>() call.
   *
   * If the existing data does not match the desired type, it will be deleted
   * and a new storage will be created.
   */
  inline void* raw_mutable_data() {
    CHECK_NE(meta_.id(), 0)
        << "Calling raw_mutable_data() without meta, but the current meta is "
           "of unknown type.";
    return raw_mutable_data(meta_);
  }

  /**
   * Returns a typed pointer of the underlying storage.
   *
   * If the existing data does not match the desired type, it will be deleted
   * and a new storage will be created.
   */
  template <typename T>
  inline T* mutable_data() {
    if ((size_ == 0 || data_.get()) && IsType<T>()) {
      return static_cast<T*>(data_.get());
    }
    return static_cast<T*>(raw_mutable_data(TypeMeta::Make<T>()));
  }


  /**
   * Returns the number of dimensions of the data.
   */
  inline int ndim() const { return dims_.size(); }
  /**
   * Returns the size (i.e. the number of items) of the tensor.
   */
  inline TIndex size() const { return size_; }
  /**
   * Return the number of bytes each item takes in the tensor.
   */
  inline size_t itemsize() const { return meta_.itemsize(); }
  /**
   * Returns the total number of bytes of the storage.
   *
   * This is equivalent to calling size() * itemsize().
   */
  inline size_t nbytes() const { return size_ * meta_.itemsize(); }
  /**
   * Returns the dimensions of the tensor as a vector.
   */
  inline const vector<TIndex>& dims() const { return dims_; }
  /**
   * Return product of all dimensions starting from K
   */
  inline TIndex size_from_dim(int k) const {
    TIndex r = 1;
    for (int i = k; i < dims_.size(); ++i) {
      r *= dims_[i];
    }
    return r;
  }

  // Product of all dims up to
  inline TIndex size_to_dim(int k) const {
    CHECK(k < dims_.size());
    TIndex r = 1;
    for (int i = 0; i < k; ++i) {
      r *= dims_[i];
    }
    return r;
  }

  /**
  * Returns the 'canonical' version of a (usually)  user-specified axis,
  * allowing for negative indexing (e.g., -1 for the last axis).
  *
  * @param axis_index the axis index.
  *        If 0 <= index < ndim(), return index.
  *        If -ndim <= index <= -1, return (ndim() - (-index)),
  *        e.g., the last axis index (ndim() - 1) if index == -1,
  *        the second to last if index == -2, etc.
  *        Dies on out of range index.
  */
  inline int canonical_axis_index(int axis_index) const {
    CHECK_GE(axis_index, -ndim());
    CHECK_LT(axis_index, ndim());
    if (axis_index < 0) {
      return axis_index + ndim();
    }
    return axis_index;
  }
  /**
   * Checks if the tensor content is of the given data type.
   */
  template <typename T>
  inline bool IsType() const { return meta_.Match<T>(); }
  /**
   * Returns the TypeMeta object associated with the current data type.
   */
  inline const TypeMeta& meta() const { return meta_; }

  /**
   * Returns the i-th dimension of the tensor in int.
   *
   * This function returns an int value instead of TIndex, which depending on
   * the typedef could be int64. If you want int64 dim values, make sure you
   * call dim() instead.
   */
  inline int dim32(const int i) const {
    DCHECK_LT(i, dims_.size()) << "Exceeding ndim limit " << dims_.size();
    DCHECK_GE(i, 0) << "Cannot have negative index";
    CHECK_LT(dims_[i], std::numeric_limits<int>::max());
    return static_cast<int>(dims_[i]);
  }

  /**
   * Returns the i-th dimension of the tensor. Note that the passed in index
   * must be between 0 (inclusive) and the number of dimensions, otherwise
   * this function will produce a fatal message.
   */
  inline TIndex dim(const int i) const {
    DCHECK_LT(i, dims_.size()) << "Exceeding ndim limit " << dims_.size();
    DCHECK_GE(i, 0) << "Cannot have negative index";
    return dims_[i];
  }

 protected:
  vector<TIndex> dims_;
  TIndex size_ = -1;
  TypeMeta meta_;
  std::shared_ptr<void> data_;
  size_t capacity_ = 0;
  // In case of chunk load we store how much data was already loaded

 private:
  bool SetDims(const vector<TIndex>& src) {
    auto old_size = size_;
    dims_.resize(src.size());
    size_ = 1;
    for (int i = 0; i < src.size(); ++i) {
      size_ *= src[i];
      dims_[i] = src[i];
    }
    return size_ != old_size;
  }

  bool SetDims(const vector<long int>& src) {
    auto old_size = size_;
    dims_.resize(src.size());
    size_ = 1;
    for (int i = 0; i < src.size(); ++i) {
      size_ *= src[i];
      dims_[i] = src[i];
    }
    return size_ != old_size;
  }

  bool SetDims(const vector<int>& src) {
    auto old_size = size_;
    dims_.resize(src.size());
    size_ = 1;
    for (int i = 0; i < src.size(); ++i) {
      size_ *= src[i];
      dims_[i] = src[i];
    }
    return size_ != old_size;
  }

  // TODO(jiayq): maybe rewrite the following functions with initializer list.
  // NVCC does not play well with initializer lists last time, but worth
  // another shot.
  bool SetDims(const TIndex d0) {
    auto old_size = size_;
    dims_.resize(1);
    dims_[0] = d0;
    size_ = d0;
    return size_ != old_size;
  }

  bool SetDims(const TIndex d0, const TIndex d1) {
    auto old_size = size_;
    dims_.resize(2);
    dims_[0] = d0;
    dims_[1] = d1;
    size_ = d0 * d1;
    return size_ != old_size;
  }

  bool SetDims(const TIndex d0, const TIndex d1, const TIndex d2) {
    auto old_size = size_;
    dims_.resize(3);
    dims_[0] = d0;
    dims_[1] = d1;
    dims_[2] = d2;
    size_ = d0 * d1 * d2;
    return size_ != old_size;
  }

  bool
  SetDims(const TIndex d0, const TIndex d1, const TIndex d2, const TIndex d3) {
    auto old_size = size_;
    dims_.resize(4);
    dims_[0] = d0;
    dims_[1] = d1;
    dims_[2] = d2;
    dims_[3] = d3;
    size_ = d0 * d1 * d2 * d3;
    return size_ != old_size;
  }

  // Note(jiayq): possibly a rule-of-three violation, but we explicitly
  // discourage the use of = for Tensors.
  Tensor& operator=(const Tensor& src) = delete;
};

// For simplicity, we will typedef Tensor<CPUContext> to TensorCPU.
typedef Tensor<CPUContext> TensorCPU;

}  // namespace caffe2
#endif  // CAFFE2_CORE_TENSOR_H_
