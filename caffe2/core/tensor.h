#ifndef CAFFE2_CORE_TENSOR_H_
#define CAFFE2_CORE_TENSOR_H_

#include <cstddef>
#include <sstream>
#include <typeinfo>
#include <type_traits>
#include <vector>

#include "caffe2/core/common.h"
#include "caffe2/core/context.h"
#include "caffe2/core/typeid.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/core/logging.h"

namespace caffe2 {

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
  Tensor() : size_(0), data_(nullptr) {}

  /**
   * @brief Creates a tensor of the given dimension.
   *
   * Note that the actual data allocation is not going to be carried out until
   * the first time mutable_data() is called.
   */
  explicit Tensor(const vector<int>& dims) { Reshape(dims); }

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
  Tensor(const Tensor<SrcContext>& src, ContextForCopy* context)
      : meta_(src.meta()) {
    Reshape(src.dims());
    context->template Memcpy<SrcContext, Context>(
        nbytes(), src.raw_data(), raw_mutable_data());
  }

  /**
   * @brief Creates a tensor from a source tensor, copying over the content.
   *
   * Note that this may have a potential performance hit, since a temporary
   * context object will be created for the memory copy. Prefer explicitly
   * providing a context for copy if you can.
   */
  template <class SrcContext>
  Tensor(const Tensor<SrcContext>& src)
      : meta_(src.meta()) {
    Reshape(src.dims());
    SrcContext tmp_context;
    tmp_context.template Memcpy<SrcContext, Context>(
        nbytes(), src.raw_data(), raw_mutable_data());
  }

  /**
   * @brief Creates a tensor, and fills its contents with the given values.
   */
  template <typename T>
  Tensor(const vector<int>& dims, const vector<T>& values, Context* context)
      : meta_(TypeMeta::Make<T>()) {
    Reshape(dims);
    CAFFE_CHECK_EQ(values.size(), size_);
    context->template Copy<T, CPUContext, Context>(
        values.size(), values.data(), mutable_data<T>());
  }

  /**
   * @brief Creates a scalar tensor, and fills its content with the given value.
   */
  template <typename T,
            typename = typename std::enable_if<std::is_scalar<T>::value>::type>
  Tensor(const T& value, Context* context) {
    Reshape(vector<int>{});
    context->template Copy<T, CPUContext, Context>(
        1, &value, mutable_data<T>());
  }

  virtual ~Tensor() {}

  /**
   * @brief Reshapes a tensor.
   *
   * Reshape takes in a vector of ints specifying the dimensions of the tensor.
   * You can pass in an empty vector to specify that it is a scalar (i.e.
   * containing one single item).
   *
   * The underlying storage may be deleted after calling Reshape: if the new
   * shape leads to a different number of items in the tensor, the old memory
   * is deleted and new memory will be allocated next time you call
   * mutable_data(). However, if the shape is different but the total number of
   * items is the same, the underlying storage is kept.
   */
  void Reshape(const vector<int>& dims) {
    dims_ = dims;
    // Calculate the size.
    int new_size = 1;
    for (int d : dims_) {
      CAFFE_CHECK_GT(d, 0);
      new_size *= d;
    }
    // If the size changes, we will free the data. the next mutable_data() call
    // will create the data storage.
    if (data_.get() && size_ != new_size) {
      data_.reset();
    }
    size_ = new_size;
  }

  /**
   * Reshape the tensor like the source tensor. Note that this is just a
   * sugar wrapper that essentially calls Reshape(src_tensor.dims()).
   */
  template <class OtherContext>
  inline void ReshapeLike(const Tensor<OtherContext>& src_tensor) {
    Reshape(src_tensor.dims());
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
   * equal already. The reason we do not implicitly do a Reshape to make the two
   * tensors have the same shape is that, we want to allow tensors of different
   * shapes but the same number of items to still be able to share data. This
   * allows one to e.g. have a n-dimensional Tensor and a flattened version
   * sharing the same underlying storage.
   *
   * The source tensor should already have its data allocated.
   */
  void ShareData(const Tensor& src) {
    meta_ = src.meta();
    CAFFE_CHECK_EQ(src.size_, size_)
        << "Size mismatch - did you call reshape before sharing the data?";
    // It is possible that the source tensor hasn't called mutable_data() yet,
    // in which case ShareData() does make much sense since we don't really know
    // what to share yet.
    CAFFE_CHECK(src.data_.get()) << "Source tensor has no content yet.";
    // Finally, do sharing.
    data_ = src.data_;
  }

  /**
   * Returns a const raw void* pointer of the underlying storage. mutable_data()
   * or raw_mutable_data() must have been called prior to this function call.
   */
  inline const void* raw_data() const {
    CAFFE_CHECK_NOTNULL(data_.get());
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
    CAFFE_CHECK_NOTNULL(data_.get());
    CAFFE_CHECK(IsType<T>());
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
    if (!data_.get() || meta_ != meta) {
      meta_ = meta;
      CAFFE_CHECK_GT(size_, 0);
      data_.reset(static_cast<void*>(Context::New(size_ * meta_.itemsize())),
                  Context::Delete);
      return data_.get();
    } else {
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
    CAFFE_CHECK_NE(meta_.id(), 0)
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
    return static_cast<T*>(
        raw_mutable_data(TypeMeta::Make<T>()));
  }

  /**
   * Returns the number of dimensions of the data.
   */
  inline int ndim() const { return dims_.size(); }
  /**
   * Returns the size (i.e. the number of items) of the tensor.
   */
  inline int size() const { return size_; }
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
  inline const vector<int>& dims() const { return dims_; }
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
   * Returns the i-th dimension of the tensor. Note that the passed in index
   * must be between 0 (inclusive) and the number of dimensions, otherwise
   * this function will produce a fatal message.
   */
  inline int dim(const int i) const {
    CAFFE_CHECK_LT(i, dims_.size()) << "Exceeding ndim limit " << dims_.size();
    CAFFE_CHECK_GE(i, 0) << "Cannot have negative index";
    return dims_[i];
  }

 protected:
  vector<int> dims_;
  int size_;
  TypeMeta meta_;
  std::shared_ptr<void> data_;

  DISABLE_COPY_AND_ASSIGN(Tensor);
};

// For simplicity, we will typedef Tensor<CPUContext> to TensorCPU.
template <class Context> class Tensor;
typedef Tensor<CPUContext> TensorCPU;

}  // namespace caffe2
#endif  // CAFFE2_CORE_TENSOR_H_
