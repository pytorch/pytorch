#ifndef CAFFE2_CORE_BLOB_H_
#define CAFFE2_CORE_BLOB_H_

#include <cstddef>
#include <vector>

#include "caffe2/core/common.h"
#include "caffe2/core/context.h"
#include "caffe2/core/typeid.h"
#include "caffe2/proto/caffe2.pb.h"
#include "glog/logging.h"

namespace caffe2 {

namespace internal {
// Destroy is a templated function that allows us to memorize the type of the
// pointer we are storing in a void*.
template <class T>
void Destroy(void* pointer) {
  delete static_cast<T*>(pointer);
}
}  // namespace internal

// Blob is a general container that hosts a pointer as well as checking its
// type, and takes charge of deleting it when the blob is deallocated. A blob
// could contain ANYTHING, although the most common case is to contain a Tensor.
class Blob {
 public:
  typedef void (*DestroyCall)(void *);

  Blob() : id_(internal::gUnknownType), pointer_(nullptr) {}

  ~Blob() { Reset(); }

  template <class T>
  inline bool IsType() const { return internal::IsTypeId<T>(id_); }
  inline string TypeName() const { return internal::TypeName(id_); }
  template <class T>
  const T& Get() const {
    CHECK(IsType<T>()) << "wrong type for the Blob instance. Expected "
                       << internal::TypeName<T>() << " got "
                       << internal::TypeName(id_);
    return *static_cast<const T*>(pointer_);
  }

  template <class T>
  T* GetMutable() {
    if (!IsType<T>()) {
      VLOG(1) << "Create new mutable object " << internal::TypeName<T>();
      if (pointer_) destroy_(pointer_);
      // If we are not of the right type, create a new instance.
      pointer_ = static_cast<void*>(new T());
      destroy_ = &internal::Destroy<T>;
    }
    id_ = internal::GetTypeId<T>();
    return static_cast<T*>(pointer_);
  }

  inline void Reset() {
    if (pointer_) {
      destroy_(pointer_);
      pointer_ = nullptr;
    }
  }

 private:
  internal::TypeId id_;
  void* pointer_;
  DestroyCall destroy_;

  DISABLE_COPY_AND_ASSIGN(Blob);
};


template <typename dtype, class Context>
class Tensor {
 public:
  Tensor() : ndim_(0), size_(0), data_(nullptr),
             own_data_(true), data_source_(nullptr) {}

  // Creates a tensor. The actual data allocation is going to be carried out
  // till the first time mutable_data() is called, so there is no overhead of
  // creating multiple tensors just as placeholders (although I haven't got a
  // clear idea where such cases would happen).
  explicit Tensor(const vector<int>& dims)
      : data_(nullptr), own_data_(true), data_source_(nullptr) {
    Reshape(dims);
  }

  template <class SrcContext>
  Tensor(const Tensor<dtype, SrcContext>& src, Context* context)
      : data_(nullptr), own_data_(true), data_source_(nullptr) {
    Reshape(src.dims());
    context->template Copy<dtype, Context, SrcContext>(
        mutable_data(), src.data(), src.size());
  }

  // Creates a tensor, and fills its contents with the given values. We need to
  // have a context passed in as the copy function is device dependent.
  Tensor(const vector<int>& dims, vector<dtype> values, Context* context)
      : data_(nullptr), own_data_(true), data_source_(nullptr) {
    Reshape(dims);
    CHECK_EQ(values.size(), size_);
    context->template Copy<dtype, Context, CPUContext>(
        mutable_data(), values.data(), values.size());
  }

  // Special case of above: create a tensor of shape 1, and the given value.
  Tensor(const dtype& value, Context* context)
      : data_(nullptr), own_data_(true), data_source_(nullptr) {
    Reshape(std::vector<int>(1, 1));
    context->template Copy<dtype, Context, CPUContext>(
      mutable_data(), &value, 1);
  }

  virtual ~Tensor() {
    Free();
  }

  void Reshape(const vector<int>& dims) {
    CHECK_GT(dims.size(), 0);
    dims_ = dims;
    ndim_ = dims_.size();
    // Calculate the size.
    int new_size = 1;
    for (int d : dims_) {
      CHECK_GT(d, 0);
      new_size *= d;
    }
    // If the size changes, we will call Free(). The next data() call will
    // re-allocate the memory.
    if (data_ && size_ != new_size) {
      Free();
    }
    size_ = new_size;
  }

  template <typename other_type, class OtherContext>
  inline void ReshapeLike(const Tensor<other_type, OtherContext>& src_tensor) {
    Reshape(src_tensor.dims());
  }

  void ShareData(const Tensor& src) {
    // To share data, the sizes must be equal.
    CHECK_EQ(src.size_, size_)
        << "Size mismatch - did you call reshape before sharing the data?";
    if (data_) Free();
    own_data_ = false;
    data_source_ = &src;
  }

  inline int ndim() const { return ndim_; }
  inline int size() const { return size_; }
  inline const vector<int>& dims() const { return dims_; }
  inline int dim(const int i) const {
    CHECK_LT(i, ndim_) << "Exceeding ndim limit " << ndim_;
    CHECK_GE(i, 0) << "Cannot have negative index";
    return dims_[i];
  }

  const dtype* data() const {
    if (own_data_) {
      CHECK_NOTNULL(data_);
      return data_;
    } else {
      CHECK_NOTNULL(data_source_);
      CHECK_EQ(data_source_->size_, size_) << "Source data size has changed.";
      CHECK_NOTNULL(data_source_->data());
      return data_source_->data();
    }
  }

  dtype* mutable_data() {
    CHECK(own_data_) << "Cannot call mutable_data() from a shared tensor.";
    CHECK_GT(size_, 0) << "Cannot call mutable_data on a size 0 tensor.";
    if (!data_) Allocate();
    CHECK_NOTNULL(data_);
    return data_;
  }

  void Allocate() {
    CHECK(data_ == nullptr);
    CHECK_GT(size_, 0);
    data_ = static_cast<dtype*>(Context::New(size_ * sizeof(dtype)));
  }

  void Free() {
    if (own_data_) {
      if (data_) {
        Context::Delete(data_);
      }
    }
    own_data_ = true;
    data_ = nullptr;
  }

 protected:
  int ndim_;
  vector<int> dims_;
  int size_;
  dtype* data_;
  bool own_data_;
  const Tensor* data_source_;

  DISABLE_COPY_AND_ASSIGN(Tensor);
};

}  // namespace caffe2
#endif  // CAFFE2_CORE_BLOB_H_
