#pragma once

#include "ATen/core/Device.h"
#include "ATen/core/Layout.h"
#include "ATen/core/Scalar.h"
#include "ATen/core/ScalarType.h"
#include "ATen/core/SparseTensorRef.h"
#include "ATen/core/Storage.h"
#include "ATen/core/TensorAccessor.h"
#include "ATen/core/TensorImpl.h"
#include "ATen/core/optional.h"
#include "ATen/core/UndefinedTensorImpl.h"
#include "ATen/core/Error.h"

namespace at {
struct Generator;
struct Type;
class Tensor;
struct TensorOptions;
} // namespace at

namespace at {
// Tensor is a "generic" object holding a pointer to the underlying TensorImpl object, which
// has an embedded reference count. In this way, Tensor is similar to boost::intrusive_ptr.
//
// For example:
//
// void func(Tensor a) {
//   Tensor b = a;
//   ...
// }
//
// In this example, when we say Tensor b = a, we are creating a new object that points to the
// same underlying TensorImpl, and bumps its reference count. When b goes out of scope, the
// destructor decrements the reference count by calling release() on the TensorImpl it points to.
// The existing constructors, operator overloads, etc. take care to implement the correct semantics.
//
// Note that Tensor can also be NULL, i.e. it is not associated with any underlying TensorImpl, and
// special care must be taken to handle this.
class AT_API Tensor {
public:
  Tensor(){};
  Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl> tensor_impl)
      : impl_(std::move(tensor_impl)) {
    if (impl_.get() == nullptr) {
      throw std::runtime_error("TensorBaseImpl with nullptr not supported");
    }
  }

  Tensor(const Tensor&) = default;
  Tensor(Tensor&&) = default;

  int64_t dim() const {
    return impl_->dim();
  }

  TensorImpl * unsafeGetTensorImpl() const {
    return impl_.get();
  }
  TensorImpl * unsafeReleaseTensorImpl() {
    return impl_.release();
  }
  const c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>& getIntrusivePtr() const {
    return impl_;
  }

  bool defined() const {
    return impl_;
  }

  void reset() {
    impl_.reset();
  }

  // The following overloads are very intruiging.  Consider the following
  // program:
  //
  //    x[1] = 3;
  //
  // We would expect that the first entry of x is written to 3.  But how can we
  // actually achieve this?  x[1] evaluates to a tensor...
  //
  // The answer is, using a ref-qualifier.  x[1] is an rvalue, which cannot be
  // (profitably) assigned to in the traditional sense, so we overload
  // assignment to mean, "Actually, copy 3 into the tensor data."  This is done
  // with an rvalue-reference ref-qualified overload (the methods with && at the
  // end of their type.)
  //
  // There's one more fly in the ointment: We also want
  //
  //    Tensor x = y;
  //
  // to work, and we want it NOT to copy.  So we need a traditional operator=
  // overload.  But we MUST specify a mutable lvalue ref-qualifier, to
  // disambiguate the traditional overload from the rvalue-reference
  // ref-qualified overload.  Otherwise, it will be ambiguous, because
  // a non ref-qualified method is eligible for all situations.

  // Unfortunately, we have to write these constructors out manually
  // to work around an MSVC bug:
  //    error C2580: 'at::Tensor &at::Tensor::operator =(const at::Tensor &) &':
  //    multiple versions of a defaulted special member functions are not allowed
  // Tensor& operator=(const Tensor&) & = default;
  // Tensor& operator=(Tensor&&) & = default;
  Tensor& operator=(const Tensor& x) & {
    impl_ = x.impl_;
    return *this;
  }
  Tensor& operator=(Tensor&& x) & {
    impl_ = std::move(x.impl_);
    return *this;
  }

  Tensor& operator=(Scalar v) &&;
  Tensor& operator=(const Tensor&) &&;
  Tensor& operator=(Tensor&&) &&;

  bool is_same(const Tensor& other) const noexcept {
    return impl_ == other.impl_;
  }
  size_t use_count() const noexcept {
    return impl_.use_count();
  }
  size_t weak_use_count() const noexcept {
    return impl_.weak_use_count();
  }

  const char * toString() const;

  IntList sizes() const {
    return impl_->sizes();
  }
  IntList strides() const {
    return impl_->strides();
  }
  int64_t ndimension() const {
    return dim();
  }
  Type & type() const {
    return impl_->type();
  }
  TensorTypeId type_id() const {
    return impl_->type_id();
  }
  ScalarType scalar_type() const {
    return dataTypeToScalarType(impl_->dtype().id());
  }
  const Storage& storage() const {
    return impl_->storage();
  }
  Tensor toType(const Type & t, bool non_blocking=false) const;
  Tensor & copy_(const Tensor & src, bool non_blocking=false);
  Tensor toType(ScalarType t) const;
  Tensor toBackend(Backend b) const;

  /// Returns true if the `Tensor` is actually a `torch::autograd::Variable`.
  /// Defined in Type.h because of include order issues.
  bool is_variable() const noexcept;

  /// Returns a `Tensor`'s layout. Defined in Type.h
  Layout layout() const noexcept;

  /// Returns a `Tensor`'s dtype (`ScalarType`). Defined in Type.h
  ScalarType dtype() const noexcept;

  /// Returns a `Tensor`'s device.
  Device device() const;

  /// Returns the `TensorOptions` corresponding to this `Tensor`. Defined in
  /// TensorOptions.h.
  TensorOptions options() const;

  template<typename T>
  T * data() const {
    return impl_->data<T>();
  }

  // Purposely not defined here to avoid inlining
  void print() const;

  //toLongData(), toFloatData() etc.
  #define TO_TYPE_DATA(T,name,_) \
  T * to##name##Data() const;
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF(TO_TYPE_DATA)
  #undef TO_TYPE_DATA

  #define TO_C_TYPE(T,name,_) \
  T toC##name () const;
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF(TO_C_TYPE)
  #undef TO_C_TYPE

  // Return a `TensorAccessor` for CPU `Tensor`s. You have to specify scalar type and
  // dimension.
  template<typename T, size_t N>
  TensorAccessor<T,N> accessor() const& {
    static_assert(N > 0, "accessor is used for indexing tensor, for scalars use *data<T>()");
    AT_CHECK(dim() == N, "expected ", N, " dims but tensor has ", dim());
    return TensorAccessor<T,N>(data<T>(),sizes().data(),strides().data());
  }
  template<typename T, size_t N>
  TensorAccessor<T,N> accessor() && = delete;

  // Return a `PackedTensorAccessor` for CUDA `Tensor`s. You have to specify scalar type and
  // dimension. You can optionally specify RestrictPtrTraits as a template parameter to
  // cast the data pointer to a __restrict__ pointer.
  // In order to use this, your CUDA kernel has to take a corresponding PackedTensorAccessor
  // as an argument.
  template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits>
    PackedTensorAccessor<T,N,PtrTraits> packed_accessor() const& {
    static_assert(N > 0, "accessor is used for indexing tensor, for scalars use *data<T>()");
    AT_CHECK(dim() == N, "expected ", N, " dims but tensor has ", dim());
    return PackedTensorAccessor<T,N,PtrTraits>(static_cast<typename PtrTraits<T>::PtrType>(data<T>()),sizes().data(),strides().data());
  }
  template<typename T, size_t N,  template <typename U> class PtrTraits = DefaultPtrTraits>
  PackedTensorAccessor<T,N> packed_accessor() && = delete;

  Tensor operator-() const;
  Tensor& operator+=(const Tensor & other);
  Tensor& operator+=(Scalar other);
  Tensor& operator-=(const Tensor & other);
  Tensor& operator-=(Scalar other);
  Tensor& operator*=(const Tensor & other);
  Tensor& operator*=(Scalar other);
  Tensor& operator/=(const Tensor & other);
  Tensor& operator/=(Scalar other);
  Tensor operator[](Scalar index) const;
  Tensor operator[](Tensor index) const;
  Tensor operator[](int64_t index) const;

  Tensor cpu() const;
  Tensor cuda() const;

  // ~~~~~ Autograd API ~~~~~

  Tensor& set_requires_grad(bool requires_grad) {
    impl_->set_requires_grad(requires_grad);
    return *this;
  }
  bool requires_grad() const {
    return impl_->requires_grad();
  }

  Tensor& grad() {
    return impl_->grad();
  }
  const Tensor& grad() const {
    return impl_->grad();
  }

  void set_data(Tensor new_data);

  /// Computes the gradient of current tensor w.r.t. graph leaves.
  void backward(
      at::optional<Tensor> gradient = at::nullopt,
      bool keep_graph = false,
      bool create_graph = false);

  // STOP.  Thinking of adding a method here, which only makes use
  // of other ATen methods?  Define it in native_functions.yaml.

  //example
  //Tensor * add(Tensor & b);
  ${tensor_method_declarations}

  template <typename F, typename... Args>
  auto m(F func, Args&&... params) const -> decltype(func(*this, std::forward<Args>(params)...)) {
    return func(*this, std::forward<Args>(params)...);
  }

  friend struct WeakTensor;

protected:
  c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl> impl_;

public:
  explicit Tensor(Storage storage)
      : impl_(c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(std::move(storage))) {}

  /**
   * @brief Creates a tensor of the given dimension.
   *
   * Note that the actual data allocation is not going to be carried out until
   * the first time mutable_data() is called.
   */
  explicit Tensor(const std::vector<int64_t>& dims, DeviceType type)
      : Tensor(Storage(type)) {
    // TODO: here, we create a Storage
    // and immediately discard it in Resize() since
    // reset_tensor will be true and FreeMemory will be called,
    // we might want to avoid creating Storage twice?
    Resize(dims);
  }

  explicit Tensor(const std::vector<int>& dims, DeviceType type)
      : Tensor(Storage(type)) {
    Resize(dims);
  }

  /**
   * context_for_copy is required to have the same DeviceType as src
   */
  Tensor(const Tensor& src, BaseContext* context_for_copy, DeviceType type)
      : Tensor(Storage(type)) {
    CopyFrom(src, context_for_copy);
  }

  /**
   * @brief: Create a Tensor of at::DeviceType `type` and initialize it with
   * src Tensor
   */
  Tensor(const Tensor& src, DeviceType type)
      : Tensor(Storage(type)) {
    CopyFrom(src);
  }

  /**
   * @brief Creates a tensor, and fills its contents with the given values.
   * The type of tensor will be decided by the context parameter
   */
  template <typename T>
  Tensor(
      const std::vector<int64_t>& dims,
      const std::vector<T>& values,
      BaseContext* context)
      : Tensor(Storage(context->device_type(), caffe2::TypeMeta::Make<T>())) {
    Resize(dims);
    CAFFE_ENFORCE_EQ_WITH_CALLER(values.size(), size());
    context->CopyItemsFromCPU(
        storage().dtype(), size(), values.data(), mutable_data<T>());
  }

  /**
   * @brief Creates a scalar tensor, and fills its content with the given value.
   * The type of tensor will be decided by the context parameter
   */
  template <
      typename T,
      typename = typename std::enable_if<std::is_scalar<T>::value>::type>
  Tensor(const T& value, BaseContext* context)
      : Tensor(Storage(context->device_type(), caffe2::TypeMeta::Make<T>())) {
    Resize(std::vector<int64_t>{});
    context->CopyItemsFromCPU(
        storage().dtype(), size(), &value, mutable_data<T>());
  }

  Tensor Clone() const {
    Tensor x(GetDeviceType());
    x.CopyFrom(*this);
    return x;
  }

  BaseStaticContext* GetStaticContext() const {
    return impl_.get()->GetStaticContext();
  }

  std::unique_ptr<BaseContext> CreateContext() const {
    return impl_.get()->CreateContext();
  }

  DeviceType GetDeviceType() const {
    return impl_.get()->GetDeviceType();
  }

  void CopyFrom(const Tensor& src, BaseContext* context = nullptr) const {
    impl_.get()->CopyFrom(*src.impl_.get(), context);
  }

  void ExtendTo(int64_t num, float growthPct, BaseContext* context) const {
    impl_.get()->ExtendTo(num, growthPct, context);
  }

  void Extend(int64_t num, float growthPct, BaseContext* context) const {
    impl_.get()->Extend(num, growthPct, context);
  }

  void ShrinkTo(int64_t outer_dim) const {
    impl_.get()->ShrinkTo(outer_dim);
  }

  template <class T>
  void ReserveSpace(const T& outer_dim) const {
    impl_.get()->ReserveSpace(outer_dim);
  }

  template <typename... Ts>
  void Resize(Ts... dim_source) const {
    impl_.get()->Resize(dim_source...);
  }

  inline void ResizeLike(const Tensor& src_tensor) const {
    impl_.get()->ResizeLike(*src_tensor.impl_.get());
  }

  inline void Reshape(const std::vector<int64_t>& dims) const {
    impl_.get()->Reshape(dims);
  }

  inline void Reshape(const std::vector<int>& dims) const {
    impl_.get()->Reshape(dims);
  }

  inline void FreeMemory() const {
    impl_.get()->FreeMemory();
  }

  std::string DebugString() const {
    return impl_.get()->DebugString();
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

  template <typename T>
  void ShareExternalPointer(
      T* src,
      size_t capacity = 0,
      caffe2::MemoryDeleter d = nullptr) const {
    impl_.get()->ShareExternalPointer<T>(src, capacity, d);
  }

  template <typename T>
  void ShareExternalPointer(at::DataPtr&& data_ptr, size_t capacity = 0) const {
    impl_.get()->ShareExternalPointer<T>(std::move(data_ptr), capacity);
  }

  void ShareExternalPointer(
      void* src,
      const caffe2::TypeMeta& meta,
      size_t capacity = 0,
      caffe2::MemoryDeleter d = nullptr) const {
    impl_.get()->ShareExternalPointer(src, meta, capacity, d);
  }

  void ShareExternalPointer(
      at::DataPtr&& data_ptr,
      const caffe2::TypeMeta& data_type,
      size_t capacity) {
    impl_.get()->ShareExternalPointer(std::move(data_ptr), data_type, capacity);
  }

  inline const void* raw_data() const {
    return impl_.get()->raw_data();
  }

  inline void* raw_mutable_data(const caffe2::TypeMeta& meta) const {
    return impl_.get()->raw_mutable_data(meta);
  }

  inline void* raw_mutable_data() const {
    return impl_.get()->raw_mutable_data();
  }

  template <typename T>
  inline T* mutable_data() const {
    return impl_.get()->mutable_data<T>();
  }

  inline int ndim() const {
    return impl_.get()->ndim();
  }

  inline int64_t size() const {
    return impl_.get()->size();
  }

  inline size_t itemsize() const {
    return impl_.get()->itemsize();
  }

  inline size_t nbytes() const {
    return impl_.get()->nbytes();
  }

  inline size_t capacity_nbytes() const {
    return impl_.get()->capacity_nbytes();
  }

  inline const std::vector<int64_t>& dims() const {
    return impl_.get()->dims();
  }

  inline int64_t size_from_dim(int k) const {
    return impl_.get()->size_from_dim(k);
  }

  inline int64_t size_to_dim(int k) const {
    return impl_.get()->size_to_dim(k);
  }

  inline int64_t size_between_dim(int k, int l) const {
    return impl_.get()->size_between_dim(k, l);
  }

  inline int canonical_axis_index(int axis_index) const {
    return impl_.get()->canonical_axis_index(axis_index);
  }

  template <typename T>
  inline bool IsType() const {
    return impl_.get()->IsType<T>();
  }

  inline const caffe2::TypeMeta& meta() const {
    return impl_.get()->meta();
  }

  inline int dim32(const int i) const {
    return impl_.get()->dim32(i);
  }

  inline int64_t dim(const int i) const {
    return impl_.get()->dim(i);
  }

  inline void ExtractDeviceOption(caffe2::DeviceOption* device) const {
    return impl_.get()->ExtractDeviceOption(device);
  }
};

struct AT_API WeakTensor {
  WeakTensor(const Tensor& t) : weak_impl_(t.impl_) {}

  // XXX: this can return undefined tensors
  // Ideally it would be at::optional<Tensor>, but MSVC is too cool for that
  Tensor lock() const {
    return Tensor(weak_impl_.lock());
  }

  bool is_same(const WeakTensor& other) const noexcept {
    return weak_impl_ == other.weak_impl_;
  }

  size_t use_count() const noexcept {
    return weak_impl_.use_count();
  }
  size_t weak_use_count() const noexcept {
    return weak_impl_.weak_use_count();
  }

  TensorImpl* unsafeGetTensorImpl() const {
    return weak_impl_._unsafe_get_target();
  }

private:
  c10::weak_intrusive_ptr<TensorImpl, UndefinedTensorImpl> weak_impl_;
};
} // namespace at

#include "ATen/core/TensorMethods.h"
