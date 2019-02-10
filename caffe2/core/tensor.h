#ifndef CAFFE2_CORE_TENSOR_H_
#define CAFFE2_CORE_TENSOR_H_

#include "caffe2/core/storage.h"
#include "caffe2/core/tensor_impl.h"

#include <ATen/core/UndefinedTensorImpl.h>
#include <c10/util/intrusive_ptr.h>
#include "ATen/core/Tensor.h"
#include <c10/core/TensorOptions.h>
#include <c10/core/Tensor.h>

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
 private:
  enum Unsafe { IDoWantAliasing };
  Tensor(const Tensor& other, Unsafe _) : impl_(other.getIntrusivePtr()) {}

 protected:
  using TensorImplPtr = c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>;
  TensorImplPtr impl_;

  void enforce_invariants();

 public:
  Tensor() : impl_() {}

  // caffe2::Tensor is explicitly marked as moveable-only because before
  // the refactoring the class used to be a value type and a lot of user code
  // is written this way. With PyTorch unification, caffe2::Tensor actually
  // has semantics of a shared_ptr now (via intrusive_ptr). However, to prevent
  // accidental mistakes when changing legacy code we keep caffe2::Tensor
  // to have movable semantics.
  //
  // If you need to get a pointer to the same Tensor instance (not to be
  // confused with shared storage), `UnsafeSharedInstance` can be used. It has
  // the same behavior as `at::Tensor a = b`.
  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;
  Tensor(Tensor&&) = default;
  Tensor& operator=(Tensor&&) = default;

  operator bool() const {
    return impl_.defined();
  }

  TensorImpl* unsafeGetTensorImpl() const {
    return impl_.get();
  }

  Tensor UnsafeSharedInstance() const {
    return Tensor(*this, IDoWantAliasing);
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
        c10::computeTensorTypeId(at::device(device).layout(at::kStrided)),
        /*is_variable=*/ false
      )) {
  }

  /**
   * @brief Creates a tensor of the given dimension.
   *
   * Note that the actual data allocation is not going to be carried out until
   * the first time mutable_data() is called.
   */
  explicit Tensor(at::IntArrayRef dims, DeviceType type) : Tensor(type) {
    // TODO: here, we create a Storage
    // and immediately discard it in Resize() since
    // reset_tensor will be true and FreeMemory will be called,
    // we might want to avoid creating Storage twice?
    Resize(dims);
  }

  // we want to preserve index information
  explicit Tensor(at::IntArrayRef dims, at::Device device): Tensor(device) {
    Resize(dims);
  }

  // TODO: remove?
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

  /**
   * @brief Mutual conversion with at::Tensor
   *
   * The tensor will share the same instance (data, strides, sizes, etc) but
   * a different subset of APIs would be available
   */
  explicit Tensor(const at::Tensor& tensor)
      : impl_(std::move(tensor.getIntrusivePtr())) {
    enforce_invariants();
  }

  explicit operator at::Tensor() const& {
    return at::Tensor::wrap_tensor_impl(impl_);
  }

  explicit operator at::Tensor() && {
    return at::Tensor::wrap_tensor_impl(std::move(impl_));
  }

  /**
   * @brief Mutual conversion with C10Tensor
   *
   * The tensor will share the same instance (data, strides, sizes, etc) but
   * a different subset of APIs would be available
   */
  explicit Tensor(C10Tensor tensor) : impl_(std::move(tensor).impl()) {
    enforce_invariants();
  }

  explicit operator C10Tensor() const & {
    return C10Tensor(impl_);
  }

  explicit operator C10Tensor() && {
    return C10Tensor(std::move(impl_));
  }

  bool is_same(const Tensor& other) const noexcept {
    return impl_ == other.impl_;
  }

  Tensor Clone() const {
    Tensor x(GetDevice());
    x.CopyFrom(*this);
    return x;
  }

  /**
   * Clone self as a Tensor that share the same Storage,
   * that is, both Tensors are views on the same Storage.
   * If we change the sizes or strides of one Tensor, it
   * does not affect the other Tensor that it shares Storage
   * with.
   * A similar yet different usage is `Tensor x = y;`, this
   * will make x and y pointing to the same Tensor and resizing
   * one of them will resize the other as well.
   *
   * TODO: Deduplicate this with THTensor_(newWithTensor)
   * (exposed in ATen as at::alias but not otherwise available)
   */
  Tensor Alias() const {
    Tensor x(sizes(), GetDevice());
    if (!dtype_initialized()) {
      C10_LOG_EVERY_MS(WARNING, 1000) <<
                   "Cloning a tensor that don't have a data type (did you call mutable_data<T> on the tensor?)";
    }
    AT_ASSERTM(
        storage_initialized(),
        "Cloning a tensor that has no content and has size > 0");
    // set_storage already sets data_type_ of TensorImpl
    x.impl_->set_storage(storage());
    x.impl_->set_storage_offset(impl_->storage_offset());
    x.impl_->set_sizes_and_strides(sizes(), strides());
    return x;
  }

  DeviceType GetDeviceType() const {
    return impl_->device_type();
  }

  at::Device GetDevice() const {
    return impl_.get()->GetDevice();
  }

  /**
   * @brief Copies the data from a source tensor, with a contex provided to
   * carry out the underlying memcpy operation.  This method respects
   * caffe2_keep_on_shrink.
   *
   * After CopyFrom, this function guarantees that the destination tensor will
   * have the same initialization state and dtype as src.  This function
   * preserves the DeviceType of the source tensor (so, e.g., if you allocate
   * a tensor on CPU and then CopyFrom a CUDA tensor, that will to a
   * CUDA-to-CPU transfer).
   *
   * 'async' parameter triggers async copy for CUDA tensors
   */
  void CopyFrom(const Tensor& src, bool async = false) {
    AT_ASSERT(!impl_->is_variable());  // TODO: remove this when Variable and Tensor are merged
    AT_ASSERTM(
        src.impl_->is_contiguous(),
        "Right now only copy of contiguous source Tensor is supported.");
    AT_ASSERTM(
        src.impl_->storage_initialized(),
        "Cannot copy from an uninitialized Tensor");

    if (src.impl_.get() == impl_.get()) {
      return;
    }

    // Test if we need to allocate a new storage
    // Uninitialized storages are guaranteed to be uniquely owned,
    // so we don't need to swap in dst case.
    // If the dtype changed, we need to reallocate storage.
    if (impl_->dtype() != src.impl_->dtype()) {
      // NB: copy preserves device_type
      // This storage will get initialized by the mutable_data call below.
      impl_->set_storage(at::Storage(impl_->device_type(), src.impl_->dtype()));
    }
    impl_->Resize(src.impl_->sizes());

    if (impl_->numel() > 0) {
      if (impl_->dtype().copy()) {
        AT_ASSERTM(
            impl_->device_type() == ::at::DeviceType::CPU,
            "In CopyFrom source and dest tensors must both be CPU for "
            "non-POD copy, but dest tensor was ",
            impl_->device_type());
        AT_ASSERTM(
            src.impl_->device_type() == ::at::DeviceType::CPU,
            "In CopyFrom source and dest tensors must both be CPU for "
            "non-POD copy, but src tensor was ",
            src.impl_->device_type());
        impl_->dtype().copy()(src.impl_->data(), impl_->raw_mutable_data(impl_->dtype()), impl_->numel());
      } else {
        // The following copy uses the current (thread local) stream for copying
        // and also takes the GPU id from the device() field passed in.
        //
        // TODO: Potentially more enforcements are necessary to avoid accidental
        // switch to sync copy if the currently set device is wrong.
        //
        // Specifically, we might need to switch to a different context device
        // here explicitly to avoid relying on user synchronizing things
        // properly.
        //
        // note: raw_mutable_data initializes device here
        void* new_data = impl_->raw_mutable_data(impl_->dtype());
        at::CopyBytes(
            impl_->numel() * impl_->itemsize(),
            src.impl_->data(),
            src.impl_->device(),
            new_data,
            impl_->device(),
            async);
      }
    }
  }

  /**
   * @brief Extend the outer-most dimension of this tensor
   *        to dimension of `num`.
   */
  void ExtendTo(int64_t num, float growthPct) const {
    CAFFE_ENFORCE_GE_WITH_CALLER(impl_->dim(), 1);
    CAFFE_ENFORCE_GE_WITH_CALLER(growthPct, 0);
    Extend(num - impl_->size(0), growthPct);
  }

  void Extend(int64_t num, float growthPct) const {
    impl_.get()->Extend(num, growthPct);
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

  // To be deprecated
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

  const c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>& getIntrusivePtr()
      const {
    return impl_;
  }

  bool defined() const {
    return impl_;
  }

  /**
   * Returns a raw void* pointer of the underlying storage. mutable_data()
   * or raw_mutable_data() must have been called prior to this function call.
   */
  inline void* raw_data() const {
    return impl_->data();
  }

  template <typename T>
  inline T* data() const {
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

  inline at::IntArrayRef sizes() const {
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

  inline at::IntArrayRef strides() const {
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

/**
 * Reinitialize a Tensor to given dims and options if necessary, note that
 * this will not do anything if the
 * Tensor already has correct size and data type
 */
CAFFE2_API void ReinitializeTensor(Tensor* t, at::IntArrayRef dims, at::TensorOptions options);

CAFFE2_API void ReinitializeAndCopyFrom(
    Tensor* t,
    at::TensorOptions options,
    const Tensor& src,
    bool async = false);

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
CAFFE2_API Tensor empty(at::IntArrayRef dims, at::TensorOptions options);

/**
 * @brief Creates a CPU tensor, and fills its contents with the given values.
 * Values are copied in
 */
// TODO: can be unified with at::from_blob when Tensor is merged and string
// types are supported
template <typename T>
Tensor TensorCPUFromValues(at::IntArrayRef dims, at::ArrayRef<T> values) {
  Tensor r = empty(dims, at::device(CPU).dtype<T>());
  CAFFE_ENFORCE_EQ(values.size(), r.numel());
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
  if (total_count) {
    // We do not add a comma after the last item.
    values_stream << tensor_data[total_count - 1];
  }

  if (to_file_) {
    (*log_file_) << MetaStr(tensor) << values_stream.str() << std::endl;
  } else {
    // Log to console.
    LOG(INFO) << MetaStr(tensor) << values_stream.str();
  }
}

} // namespace caffe2
#endif // CAFFE2_CORE_TENSOR_H_
