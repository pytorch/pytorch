#ifndef CAFFE2_CORE_BLOB_H_
#define CAFFE2_CORE_BLOB_H_

#include <cstddef>
#include <sstream>
#include <typeinfo>
#include <type_traits>
#include <vector>

#include "caffe2/core/common.h"
#include "caffe2/core/context.h"
#include "caffe2/core/registry.h"
#include "caffe2/core/typeid.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/core/logging.h"

namespace caffe2 {

// Blob is a general container that hosts a pointer as well as checking its
// type, and takes charge of deleting it when the blob is deallocated. A blob
// could contain ANYTHING, although the most common case is to contain a Tensor.
class Blob {
 public:
  // Initialize an empty blob.
  Blob() : meta_(), pointer_(nullptr) {}
  ~Blob() { Reset(); }
  // Checks if the content stored in the blob is of type T.
  template <class T>
  bool IsType() const { return meta_.Match<T>(); }
  // Returns the typename of the blob.
  inline const char* TypeName() const { return meta_.name(); }
  // Gets the const reference of the stored object.
  template <class T>
  const T& Get() const {
    CAFFE_CHECK(IsType<T>()) << "wrong type for the Blob instance. Expected "
                       << meta_.name() << " got "
                       << TypeMeta::Name<T>();
    return *static_cast<const T*>(pointer_);
  }
  // Gets a mutable pointer to the stored object. If the current object is not
  // of the right type, a new object is created. Note that type T should have
  // a default constructor. Otherwise, create the object yourself and use Reset.
  template <class T>
  T* GetMutable() {
    if (!IsType<T>()) {
      return Reset<T>(new T());
    }
    return static_cast<T*>(pointer_);
  }

  inline void Reset() {
    if (pointer_) {
      destroy_(pointer_);
      pointer_ = nullptr;
    }
  }

  template <class T>
  T* Reset(T* allocated) {
    if (pointer_) { destroy_(pointer_); }
    CAFFE_VLOG(1) << "Create new mutable object " << TypeMeta::Name<T>();
    meta_ = TypeMeta::Make<T>();
    pointer_ = static_cast<void*>(allocated);
    destroy_ = &Destroy<T>;
    return allocated;
  }

  // Serializes the current blob, if possible. This serialization uses
  // registration so we don't need to deal with multiple platform problems.
  string Serialize(const string& name) const;

 private:
  template <class T>
  static void Destroy(void* pointer) {
    delete static_cast<T*>(pointer);
  }
  typedef void (*DestroyCall)(void *);
  TypeMeta meta_;
  void* pointer_;
  DestroyCall destroy_;

  DISABLE_COPY_AND_ASSIGN(Blob);
};

template <class Context>
class Tensor {
 public:
  Tensor() : size_(0), data_(nullptr) {}

  // Creates a tensor. The actual data allocation is going to be carried out
  // till the first time mutable_data() is called, so there is no overhead of
  // creating multiple tensors just as placeholders (although I haven't got a
  // clear idea where such cases would happen).
  explicit Tensor(const vector<int>& dims) { Reshape(dims); }

  // Creates a tensor from a source tensor that is possibly from a different
  // device context. A device context object (either Context or SrcContext) is
  // provided for copying the underlying data.
  template <class SrcContext, class ContextForCopy>
  Tensor(const Tensor<SrcContext>& src, ContextForCopy* context)
      : meta_(src.meta()) {
    Reshape(src.dims());
    context->template Memcpy<SrcContext, Context>(
        nbytes(), src.raw_data(), raw_mutable_data());
  }

  // A fallback choice, where we will use a default context - do not use it if
  // you can explicitly provide a context for copy.
  template <class SrcContext>
  Tensor(const Tensor<SrcContext>& src)
      : meta_(src.meta()) {
    Reshape(src.dims());
    SrcContext tmp_context;
    tmp_context.template Memcpy<SrcContext, Context>(
        nbytes(), src.raw_data(), raw_mutable_data());
  }

  // Creates a tensor, and fills its contents with the given values. We need to
  // have a context passed in as the copy function is device dependent.
  template <typename T>
  Tensor(const vector<int>& dims, const vector<T>& values, Context* context)
      : meta_(TypeMeta::Make<T>()) {
    Reshape(dims);
    CAFFE_CHECK_EQ(values.size(), size_);
    context->template Copy<T, CPUContext, Context>(
        values.size(), values.data(), mutable_data<T>());
  }

  template <typename T,
            typename = typename std::enable_if<std::is_scalar<T>::value>::type>
  Tensor(const T& value, Context* context) {
    Reshape(vector<int>{});
    context->template Copy<T, CPUContext, Context>(
        1, &value, mutable_data<T>());
  }

  virtual ~Tensor() {}

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

  template <class OtherContext>
  inline void ReshapeLike(const Tensor<OtherContext>& src_tensor) {
    Reshape(src_tensor.dims());
  }

  // A utility function to print the debug string for the tensor. Note that this
  // is very slow since it involves quite some string operations, so do not use
  // it in your performance-critical code.
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

  void ShareData(const Tensor& src) {
    // To share data, the sizes must be equal.
    // The reason we do not force the ShareData to have an explicit reshape is
    // because we want to allow tensors to have different shape but still
    // maintain the same underlying data storage, as long as the contents are
    // of the same size.
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

  inline const void* raw_data() const {
    CAFFE_CHECK_NOTNULL(data_.get());
    return data_.get();
  }

  template <typename T>
  inline const T* data() const {
    CAFFE_CHECK_NOTNULL(data_.get());
    CAFFE_CHECK(IsType<T>());
    return static_cast<T*>(data_.get());
  }

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

  // If you know that the meta information of the tensor is already set, you
  // can directly call without passing in a meta object.
  inline void* raw_mutable_data() {
    CAFFE_CHECK_NE(meta_.id(), 0)
        << "Calling raw_mutable_data() without meta, but the current meta is "
           "of unknown type.";
    return raw_mutable_data(meta_);
  }

  template <typename T>
  inline T* mutable_data() {
    return static_cast<T*>(
        raw_mutable_data(TypeMeta::Make<T>()));
  }

  inline int ndim() const { return dims_.size(); }
  inline int size() const { return size_; }
  inline int itemsize() const { return meta_.itemsize(); }
  inline int nbytes() const { return size_ * meta_.itemsize(); }
  inline const vector<int>& dims() const { return dims_; }
  template <typename T>
  inline bool IsType() const { return meta_.Match<T>(); }
  inline const TypeMeta& meta() const { return meta_; }

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


}  // namespace caffe2
#endif  // CAFFE2_CORE_BLOB_H_
