#ifndef CAFFE2_CORE_BLOB_H_
#define CAFFE2_CORE_BLOB_H_

#include <cstddef>
#include <vector>

#include "caffe2/core/common.h"
#include "caffe2/core/context.h"
#include "caffe2/core/registry.h"
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

  // Serializes the current blob, if possible. This serialization uses
  // registration so we don't need to deal with multiple platform problems.
  inline string Serialize(const string& name) const;

 private:
  internal::TypeId id_;
  void* pointer_;
  DestroyCall destroy_;

  DISABLE_COPY_AND_ASSIGN(Blob);
};

// BlobSerializerBase is a class that serializes a blob to a string. This class
// exists purely for the purpose of registering type-specific serialization
// code.
class BlobSerializerBase {
 public:
  virtual string Serialize(const Blob& blob, const string& name) = 0;
};

// THe Blob serialization registry and serializer creator functions.
DECLARE_TYPED_REGISTRY(BlobSerializerRegistry, internal::TypeId,
                       BlobSerializerBase);
#define REGISTER_BLOB_SERIALIZER(name, id, ...) \
  REGISTER_TYPED_CLASS(BlobSerializerRegistry, name, id, __VA_ARGS__)
// Creates an operator with the given operator definition.
inline BlobSerializerBase* CreateSerializer(internal::TypeId id) {
  return BlobSerializerRegistry()->Create(id);
}

// The blob serialization member function implementation.
inline string Blob::Serialize(const string& name) const {
  std::unique_ptr<BlobSerializerBase> serializer(CreateSerializer(id_));
  return serializer->Serialize(*this, name);
}


template <typename dtype, class Context>
class Tensor {
 public:
  Tensor() : ndim_(0), size_(0), data_(nullptr) {}

  // Creates a tensor. The actual data allocation is going to be carried out
  // till the first time mutable_data() is called, so there is no overhead of
  // creating multiple tensors just as placeholders (although I haven't got a
  // clear idea where such cases would happen).
  explicit Tensor(const vector<int>& dims)
      : data_(nullptr) {
    Reshape(dims);
  }

  template <class SrcContext, class ContextForCopy>
  Tensor(const Tensor<dtype, SrcContext>& src, ContextForCopy* context)
      : data_(nullptr) {
    Reshape(src.dims());
    context->template Copy<dtype, Context, SrcContext>(
        mutable_data(), src.data(), src.size());
  }

  // Creates a tensor, and fills its contents with the given values. We need to
  // have a context passed in as the copy function is device dependent.
  Tensor(const vector<int>& dims, const vector<dtype>& values, Context* context)
      : data_(nullptr) {
    Reshape(dims);
    CHECK_EQ(values.size(), size_);
    context->template Copy<dtype, Context, CPUContext>(
        mutable_data(), values.data(), values.size());
  }

  // Special case of above: create a tensor of shape 1, and the given value.
  Tensor(const dtype& value, Context* context)
      : data_(nullptr) {
    Reshape(std::vector<int>());
    context->template Copy<dtype, Context, CPUContext>(
        mutable_data(), &value, 1);
  }

  virtual ~Tensor() {}

  void Reshape(const vector<int>& dims) {
    dims_ = dims;
    ndim_ = dims_.size();
    // Calculate the size.
    int new_size = 1;
    for (int d : dims_) {
      CHECK_GT(d, 0);
      new_size *= d;
    }
    // If the size changes, we will free the data. the next mutable_data() call
    // will create the data storage.
    if (data_.get() && size_ != new_size) {
      data_.reset();
    }
    size_ = new_size;
  }

  template <typename other_type, class OtherContext>
  inline void ReshapeLike(const Tensor<other_type, OtherContext>& src_tensor) {
    Reshape(src_tensor.dims());
  }

  void ShareData(const Tensor& src) {
    // To share data, the sizes must be equal.
    // The reason we do not force the ShareData to have an explicit reshape is
    // because we want to allow tensors to have different shape but still
    // maintain the same underlying data storage, as long as the contents are
    // of the same size.
    CHECK_EQ(src.size_, size_)
        << "Size mismatch - did you call reshape before sharing the data?";
    // It is possible that the source tensor hasn't called mutable_data() yet,
    // in which case ShareData() does make much sense since we don't really know
    // what to share yet.
    CHECK(src.data_.get())
        << "Source tensor has no content yet.";
    // Finally, do sharing.
    data_ = src.data_;
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
    CHECK_NOTNULL(data_.get());
    return data_.get();
  }

  dtype* mutable_data() {
    if (!data_.get()) Allocate();
    return data_.get();
  }

  void Allocate() {
    CHECK_GT(size_, 0);
    data_.reset(static_cast<dtype*>(Context::New(size_ * sizeof(dtype))),
                Context::Delete);
  }

 protected:
  int ndim_;
  vector<int> dims_;
  int size_;
  std::shared_ptr<dtype> data_;
  DISABLE_COPY_AND_ASSIGN(Tensor);
};

}  // namespace caffe2
#endif  // CAFFE2_CORE_BLOB_H_
