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
struct Tensor;
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
struct AT_API Tensor {
  Tensor(){};
  Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl> tensor_impl)
      : tensor_impl_(std::move(tensor_impl)) {
    if (tensor_impl_.get() == nullptr) {
      throw std::runtime_error("TensorBaseImpl with nullptr not supported");
    }
  }

  Tensor(const Tensor&) = default;
  Tensor(Tensor&&) = default;

  int64_t dim() const {
    return tensor_impl_->dim();
  }

  TensorImpl * unsafeGetTensorImpl() const {
    return tensor_impl_.get();
  }
  TensorImpl * unsafeReleaseTensorImpl() {
    return tensor_impl_.release();
  }
  const c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>& getIntrusivePtr() const {
    return tensor_impl_;
  }

  bool defined() const {
    return tensor_impl_;
  }

  void reset() {
    tensor_impl_.reset();
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
    tensor_impl_ = x.tensor_impl_;
    return *this;
  }
  Tensor& operator=(Tensor&& x) & {
    tensor_impl_ = std::move(x.tensor_impl_);
    return *this;
  }

  Tensor& operator=(Scalar v) &&;
  Tensor& operator=(const Tensor&) &&;
  Tensor& operator=(Tensor&&) &&;

  bool is_same(const Tensor& other) const noexcept {
    return tensor_impl_ == other.tensor_impl_;
  }
  size_t use_count() const noexcept {
    return tensor_impl_.use_count();
  }
  size_t weak_use_count() const noexcept {
    return tensor_impl_.weak_use_count();
  }

  const char * toString() const;

  IntList sizes() const {
    return tensor_impl_->sizes();
  }
  IntList strides() const {
    return tensor_impl_->strides();
  }
  int64_t ndimension() const {
    return dim();
  }
  Type & type() const {
    return tensor_impl_->type();
  }
  TensorTypeId type_id() const {
    return tensor_impl_->type_id();
  }
  ScalarType scalar_type() const {
    return dataTypeToScalarType(tensor_impl_->dtype().id());
  }
  const Storage& storage() const {
    return tensor_impl_->storage();
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
  T * data() const;

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
    tensor_impl_->set_requires_grad(requires_grad);
    return *this;
  }
  bool requires_grad() const {
    return tensor_impl_->requires_grad();
  }

  Tensor& grad() {
    return tensor_impl_->grad();
  }
  const Tensor& grad() const {
    return tensor_impl_->grad();
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
  c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl> tensor_impl_;
};

struct AT_API WeakTensor {
  WeakTensor(const Tensor& t) : weak_tensor_impl_(t.tensor_impl_) {}

  // XXX: this can return undefined tensors
  // Ideally it would be at::optional<Tensor>, but MSVC is too cool for that
  Tensor lock() const {
    return Tensor(weak_tensor_impl_.lock());
  }

  bool is_same(const WeakTensor& other) const noexcept {
    return weak_tensor_impl_ == other.weak_tensor_impl_;
  }

  size_t use_count() const noexcept {
    return weak_tensor_impl_.use_count();
  }
  size_t weak_use_count() const noexcept {
    return weak_tensor_impl_.weak_use_count();
  }

  TensorImpl* unsafeGetTensorImpl() const {
    return weak_tensor_impl_._unsafe_get_target();
  }

private:
  c10::weak_intrusive_ptr<TensorImpl, UndefinedTensorImpl> weak_tensor_impl_;
};
} // namespace at

#include "ATen/core/TensorMethods.h"
