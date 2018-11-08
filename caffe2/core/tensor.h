#ifndef CAFFE2_CORE_TENSOR_H_
#define CAFFE2_CORE_TENSOR_H_

#include "caffe2/core/storage.h"
#include "caffe2/core/tensor_impl.h"

#include <ATen/core/UndefinedTensorImpl.h>
#include <ATen/core/intrusive_ptr.h>
#include "ATen/core/TensorOptions.h"

namespace caffe2 {

using at::UndefinedTensorImpl;

/**
 * @brief Tensor class holds a shared pointer to the implementation TensorImpl,
 * redirects API calls to TensorImpl;
 * Copying of Tensor results in sharing the same underlying implementation
 * object
 *
 * NB: See TensorImpl for documentation on these methods.
 */
class CAFFE2_API Tensor final {
 protected:
  using TensorImplPtr = c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>;
  TensorImplPtr impl_;

 public:
  Tensor() : impl_() {}

  operator bool() const {
    return impl_.defined();
  }

  TensorImpl* unsafeGetTensorImpl() const {
    return impl_.get();
  }

  /**
   * @brief Creates a tensor of the given device type.
   *
   * Note that the actual data allocation is not going to be carried out until
   * you resize the tensor and then call mutable_data().
   */
  explicit Tensor(at::Device device)
    : impl_(c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(
        Storage(device),
        at::detail::computeTensorTypeId(at::device(device).layout(at::kStrided)),
        /*is_variable=*/ false
      )) {
  }

  /**
   * @brief Creates a tensor of the given dimension.
   *
   * Note that the actual data allocation is not going to be carried out until
   * the first time mutable_data() is called.
   */
  explicit Tensor(at::IntList dims, DeviceType type) : Tensor(type) {
    // TODO: here, we create a Storage
    // and immediately discard it in Resize() since
    // reset_tensor will be true and FreeMemory will be called,
    // we might want to avoid creating Storage twice?
    Resize(dims);
  }

  explicit Tensor(const vector<int>& dims, DeviceType type)
      : Tensor(type) {
    Resize(dims);
  }

  /**
   * @brief: Create a Tensor of at::DeviceType `type` and initialize it with
   * src Tensor
   */
  Tensor(const Tensor& src, DeviceType type)
      : Tensor(type) {
    CopyFrom(src);
  }

  Tensor Clone() const {
    Tensor x(GetDevice());
    x.CopyFrom(*this);
    return x;
  }

  DeviceType GetDeviceType() const {
    return impl_->device_type();
  }

  at::Device GetDevice() const {
    return impl_.get()->GetDevice();
  }

  void CopyFrom(const Tensor& src, BaseContext* context = nullptr) const {
    impl_.get()->CopyFrom(*src.impl_.get(), context);
  }

  /**
   * @brief Extend the outer-most dimension of this tensor
   *        to dimension of `num`.
   */
  void ExtendTo(int64_t num, float growthPct, BaseContext* context) const {
    CAFFE_ENFORCE_GE_WITH_CALLER(impl_->dim(), 1);
    CAFFE_ENFORCE_GE_WITH_CALLER(growthPct, 0);
    CAFFE_ENFORCE(context != nullptr, "Context must be provided.");
    Extend(num - impl_->size(0), growthPct, context);
  }

  void Extend(int64_t num, float growthPct, BaseContext* context) const {
    impl_.get()->Extend(num, growthPct, context);
  }

  /**
   * @brief Shrinks the outer-most dimension to given size, keeping the data.
   *
   * This method guarantees that no re-allocations are carried out, which means
   * that the extra capacity after the end of the shrunk tensor is maintained.
   * Notably, this function does NOT respect caffe2_keep_on_shrink.
   */
  void ShrinkTo(int64_t outer_dim) const {
    CAFFE_ENFORCE_WITH_CALLER(
        impl_->is_contiguous(),
        "Right now ShrinkTo is only supported on contiguous Tensor.");
    CAFFE_ENFORCE_WITH_CALLER(impl_->dim() >= 1, "Tensor must be at least 1D");
    CAFFE_ENFORCE_WITH_CALLER(
        outer_dim <= impl_->size(0),
        "New outer dimension must be smaller than current.");
    CAFFE_ENFORCE(
        impl_->storage().unique(),
        "Can't call ShrinkTo on shared storage, please call Resize instead.");
    impl_.get()->set_size(0, outer_dim);
  }

  template <class T>
  void ReserveSpace(const T& outer_dim) const {
    impl_.get()->ReserveSpace(outer_dim);
  }

  template <typename... Ts>
  void Resize(Ts... dim_source) const {
    impl_.get()->Resize(dim_source...);
  }

  /**
   * Resize the tensor like the source tensor. Note that this is just a
   * sugar wrapper that essentially calls Resize(src_tensor.dims()).
   * This method respects caffe2_keep_on_shrink.
   */
  inline void ResizeLike(const Tensor& src_tensor) const {
    CAFFE_ENFORCE_WITH_CALLER(
        src_tensor.is_contiguous(),
        "Right now ResizeLike is only supported for contiguous Tensor.");
    if (impl_ != src_tensor.impl_) {
      impl_.get()->Resize(src_tensor.sizes());
    }
  }

  inline void Reshape(const vector<int64_t>& dims) const {
    impl_.get()->Reshape(dims);
  }

  inline void Reshape(const vector<int>& dims) const {
    impl_.get()->Reshape(ToVectorint64_t(dims));
  }

  inline void FreeMemory() const {
    impl_.get()->FreeMemory();
  }

  /**
   * A utility function to print the debug string for the tensor. Note that this
   * is very slow since it involves quite some string operations, so do not use
   * it in your performance-critical code.
   */
  string DebugString() const {
    std::stringstream ss;
    ss << "A Tensor of item size " << impl_->storage().itemsize() << " and type "
       << impl_->dtype().name() << " and dimension (";
    for (int d : impl_->sizes()) {
      ss << d << ",";
    }
    ss << ").";
    return ss.str();
  }

  // NB: a.swap(b) is not equivalent to std::swap(a, b);
  // swap method swaps the CONTENTS of the tensors, while std::swap
  // swaps the POINTERS.
  void swap(const Tensor& other) const noexcept {
    // NB: use get() to get a non-const pointer!
    std::swap(*impl_.get(), *other.impl_.get());
  }

  void ShareData(const Tensor& src) const {
    impl_.get()->ShareData(*src.impl_.get());
  }

  /**
   * @brief Shares the data with an externally managed pointer.
   *
   * This is similar to ShareData() but the source is a pointer with an advanced
   * deleter option. In default, no deletion takes place, and one needs to make
   * sure that the external memory is deallocated only after the tensor finishes
   * using it. If a Deleter object is passed in, when this tensor is reallocated
   * or freed, the deleter function is going to be called.
   */
  template <typename T>
  void ShareExternalPointer(
      T* src,
      size_t capacity = 0,
      MemoryDeleter d = nullptr) const {
    ShareExternalPointer((void*)src, caffe2::TypeMeta::Make<T>(), capacity, d);
  }

  template <typename T>
  void ShareExternalPointer(at::DataPtr&& data_ptr, size_t capacity = 0) const {
    ShareExternalPointer(std::move(data_ptr), caffe2::TypeMeta::Make<T>(), capacity);
  }

  void ShareExternalPointer(
      void* src,
      const TypeMeta& data_type,
      size_t capacity = 0,
      MemoryDeleter d = nullptr) const {
    CAFFE_ENFORCE_WITH_CALLER(
        impl_->is_contiguous(),
        "Right now ShareExternalPointer is only supported for contiguous Tensor.");
    CAFFE_ENFORCE_WITH_CALLER(
        data_type.id() != caffe2::TypeIdentifier::uninitialized(),
        "To share with a raw external pointer you need to pass in an "
        "initialized data_type(TypeMeta).");
    impl_.get()->ShareExternalPointer(
        at::DataPtr(src, src, d, impl_->device_type()), data_type, capacity);
  }

  void ShareExternalPointer(
      at::DataPtr&& data_ptr,
      const TypeMeta& data_type,
      size_t capacity) {
    impl_.get()->ShareExternalPointer(std::move(data_ptr), data_type, capacity);
  }

  /**
   * Returns a const raw void* pointer of the underlying storage. mutable_data()
   * or raw_mutable_data() must have been called prior to this function call.
   */
  inline const void* raw_data() const {
    return impl_->data();
  }

  template <typename T>
  inline const T* data() const {
    return impl_.get()->data<T>();
  }

  inline void* raw_mutable_data(const TypeMeta& meta) const {
    return impl_.get()->raw_mutable_data(meta);
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
  inline void* raw_mutable_data() const {
    const auto& data_type = impl_->dtype();
    CAFFE_ENFORCE_WITH_CALLER(
        data_type.id() != caffe2::TypeIdentifier::uninitialized(),
        "Calling raw_mutable_data() without meta, but the current meta is "
        "of unknown type.");
    return raw_mutable_data(data_type);
  }

  template <typename T>
  inline T* mutable_data() const {
    return impl_.get()->mutable_data<T>();
  }

  /**
   * Returns the number of dimensions of the data.
   */
  inline int dim() const {
    return impl_->dim();
  }

  /**
   * (To be deprecated) Returns the number of dimensions of the data.
   */
  inline int ndim() const {
    return impl_->dim();
  }

  /**
   * (To be deprecated) Returns the size (i.e. the number of items) of the
   * tensor.
   */
  inline int64_t size() const {
    return impl_->numel();
  }

  /**
   * Returns the number of items of the tensor.
   */
  inline int64_t numel() const {
    return impl_->numel();
  }

  /**
   * Return the number of bytes each item takes in the tensor.
   */
  inline size_t itemsize() const {
    return impl_->storage().itemsize();
  }

  /**
   * Returns the total number of bytes of the storage.
   *
   * This is equivalent to calling size() * itemsize().
   */
  inline size_t nbytes() const {
    return impl_->numel() * itemsize();
  }

  inline at::IntList sizes() const {
    return impl_.get()->sizes();
  }

  // To be deprecated
  inline at::IntList dims() const {
    return impl_.get()->sizes();
  }

  inline int64_t size_from_dim(int k) const {
    return size_from_dim_(k, impl_->sizes());
  }

  inline int64_t size_to_dim(int k) const {
    return size_to_dim_(k, impl_->sizes());
  }

  inline int64_t size_between_dim(int k, int l) const {
    return size_between_dim_(k, l, impl_->sizes());
  }

  /**
   * Returns the 'canonical' version of a (usually)  user-specified axis,
   * allowing for negative indexing (e.g., -1 for the last axis).
   *
   * @param axis_index the axis index.
   *        If 0 <= index < dim(), return index.
   *        If -ndim <= index <= -1, return (dim() - (-index)),
   *        e.g., the last axis index (dim() - 1) if index == -1,
   *        the second to last if index == -2, etc.
   *        Dies on out of range index.
   */
  inline int canonical_axis_index(int axis_index) const {
    return canonical_axis_index_(axis_index, impl_->dim());
  }

  inline int64_t stride(int64_t dim) const {
    return impl_.get()->stride(dim);
  }

  inline at::IntList strides() {
    return impl_.get()->strides();
  }

  inline bool is_contiguous() const {
    return impl_.get()->is_contiguous();
  }

  /**
   * Checks if the tensor content is of the given data type.
   */
  template <typename T>
  inline bool IsType() const {
    return impl_->storage().IsType<T>();
  }

  /**
   * Returns the TypeMeta object associated with the current data type.
   */
  inline const TypeMeta& dtype() const {
    return impl_->dtype();
  }

  /**
   * (To be deprecated) Returns the TypeMeta object associated with the current
   * data type.
   */
  inline const TypeMeta& meta() const {
    return impl_->dtype();
  }

  /**
   * Returns the i-th dimension of the tensor in int.
   *
   * This function returns an int value instead of int64_t, which depending on
   * the typedef could be int64. If you want int64 dim values, make sure you
   * call dim() instead.
   */
  inline int dim32(const int i) const {
#ifndef NDEBUG
    CAFFE_ENFORCE_LT_WITH_CALLER(i, static_cast<int>(impl_->dim()), "Exceeding ndim limit");
    CAFFE_ENFORCE_GE_WITH_CALLER(i, 0, "Cannot have negative dimension index");
#endif
    auto s = impl_->size(i);
    CAFFE_ENFORCE_LT_WITH_CALLER(s, std::numeric_limits<int>::max());
    return static_cast<int>(s);
  }

  inline int64_t size(const int i) const {
    return impl_->size(i);
  }

  // To be deprecated
  inline int64_t dim(const int i) const {
    return impl_->size(i);
  }

  const Storage& storage() {
    return impl_->storage();
  }

  const Storage& storage() const {
    return impl_->storage();
  }

  bool storage_initialized() const {
    return impl_->storage_initialized();
  }

  bool dtype_initialized() const {
    return impl_->dtype_initialized();
  }
};

CAFFE2_API void ReinitializeTensor(Tensor* t, at::IntList dims, at::TensorOptions options);

CAFFE2_API void ReinitializeAndCopyFrom(
    Tensor* t,
    at::TensorOptions options,
    const Tensor& src,
    BaseContext* context = nullptr);

CAFFE_DECLARE_PREALLOCATED_KNOWN_TYPE(12, Tensor)

using TensorCPU = Tensor;

constexpr int k_limit_default_ = 1000;

// TODO: the following logic can be merged into regular Tensor class methods
// after MKLMemory starts to implement Tensor interface

// Type call registry
typedef TypeMeta (*TypeCall)(const void*);
TypeCall GetTypeCallFunction(TypeIdentifier id);
void RegisterTypeCallFunction(TypeIdentifier id, TypeCall c);

// Shape call registry
typedef vector<int64_t> (*TensorInfoCall)(
    const void*,
    size_t* capacity,
    DeviceOption* device);
TensorInfoCall GetTensorInfoFunction(TypeIdentifier id);
void RegisterTensorInfoFunction(TypeIdentifier id, TensorInfoCall c);

// resize helper function
void TensorVectorResize(
    std::vector<Tensor>& tensors,
    int size,
    DeviceType type);

// Tensor factory function
CAFFE2_API Tensor empty(at::IntList dims, at::TensorOptions options);

/**
 * @brief Creates a CPU tensor, and fills its contents with the given values.
 * Values are copied in
 */
// TODO: can be unified with at::from_blob when Tensor is merged and string
// types are supported
template <typename T>
Tensor TensorCPUFromValues(at::IntList dims, at::ArrayRef<T> values) {
  Tensor r = empty(dims, at::device(CPU).dtype<T>());
  CAFFE_ENFORCE_EQ(values.size(), r.size());
  CPUContext context;
  context.CopyItemsFromCPU(
      r.dtype(), values.size(), values.data(), r.mutable_data<T>());
  return r;
}

class CAFFE2_API TensorPrinter {
 public:
  explicit TensorPrinter(
      const std::string& tensor_name = "",
      const std::string& file_name = "",
      int limit = k_limit_default_);
  ~TensorPrinter();

  template <class T>
  void Print(const Tensor& tensor);

  void PrintMeta(const Tensor& tensor);

  string MetaStr(const Tensor& tensor);

 private:
  bool to_file_;
  int limit_;
  std::unique_ptr<std::ofstream> log_file_;
  std::string tensor_name_;
};

template <class T>
void TensorPrinter::Print(const Tensor& tensor) {
  std::stringstream values_stream;
  // One most likely doesn't want to print int64-number of items for visual
  // inspection, so we cast down to int here.
  int total_count = static_cast<int>(std::min(tensor.numel(), int64_t(limit_)));
  const T* tensor_data = tensor.template data<T>();
  for (int i = 0; i < total_count - 1; ++i) {
    values_stream << tensor_data[i] << ",";
  }
  // We do not add a comma after the last item.
  values_stream << tensor_data[total_count - 1];
  if (to_file_) {
    (*log_file_) << MetaStr(tensor) << values_stream.str() << std::endl;
  } else {
    // Log to console.
    LOG(INFO) << MetaStr(tensor) << values_stream.str();
  }
}

} // namespace caffe2
#endif // CAFFE2_CORE_TENSOR_H_
