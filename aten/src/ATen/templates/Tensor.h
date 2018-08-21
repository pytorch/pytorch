#pragma once

// ${generated_comment}

#include "ATen/Device.h"
#include "ATen/Layout.h"
#include "ATen/Scalar.h"
#include "ATen/ScalarType.h"
#include "ATen/SparseTensorRef.h"
#include "ATen/Storage.h"
#include "ATen/TensorAccessor.h"
#include "ATen/TensorBase.h"
#include "ATen/TensorImpl.h"
#include "ATen/core/optional.h"

namespace at {
struct Generator;
struct Type;
struct Tensor;
struct TensorOptions;
namespace detail {
void set_data(Tensor& tensor, Tensor new_data);
} // namespace detail
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
struct Tensor : public detail::TensorBase {
  using TensorBase = detail::TensorBase;
  Tensor() : TensorBase() {}
  Tensor(TensorImpl * self, bool retain) : TensorBase(self, retain) {}
  Tensor(const TensorBase & rhs) : TensorBase(rhs) {}
  Tensor(const Tensor & rhs) = default;
  Tensor(Tensor && rhs) noexcept = default;

  // reimplemented from TensorBase so the return type is Tensor rather than TensorBase
  Tensor & operator=(Tensor && rhs) & {
    rhs.swap(*this);
    return *this;
  }
  Tensor & operator=(Tensor const & rhs) & {
      //Tensor ctor retains original rhs.pImpl
      //then rhs.pImpl is swapped with this->pImpl
      //finally Tensor dtor releases rhs.pImpl, which was originally this->pImpl
      Tensor(rhs).swap(*this);
      return *this;
  }

  inline Tensor & operator=(Tensor const & rhs) &&;
  Tensor & operator=(Scalar v) &&;
  const char * toString() const {
    return pImpl->toString();
  }
  IntList sizes() const {
    return pImpl->sizes();
  }
  IntList strides() const {
    return pImpl->strides();
  }
  int64_t ndimension() const {
    return dim();
  }
  Type & type() const {
    return pImpl->type();
  }
  std::unique_ptr<Storage> storage() const {
    return pImpl->storage();
  }
  inline Tensor toType(const Type & t, bool non_blocking=false) const;
  inline Tensor & copy_(const Tensor & src, bool non_blocking=false);
  inline Tensor toType(ScalarType t) const;
  inline Tensor toBackend(Backend b) const;

  /// New-style `to()` methods.
  /// NB: These methods are defined in TensorOptions.h.
  Tensor to(Device device, ScalarType dtype, bool non_blocking = false) const;
  Tensor to(ScalarType dtype, bool non_blocking = false) const;
  Tensor to(Device device, bool non_blocking = false) const;

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

  // non-retaining
  TensorImpl * unsafeGetTensorImpl() const {
    return pImpl;
  }

  // Purposely not defined here to avoid inlining
  void print() const;

  //toLongData(), toFloatData() etc.
  #define TO_TYPE_DATA(T,name,_) \
  T * to##name##Data() const;
  AT_FORALL_SCALAR_TYPES(TO_TYPE_DATA)
  #undef TO_TYPE_DATA

  #define TO_C_TYPE(T,name,_) \
  T toC##name () const;
  AT_FORALL_SCALAR_TYPES(TO_C_TYPE)
  #undef TO_C_TYPE

  template<typename T, size_t N>
  TensorAccessor<T,N> accessor() const& {
    static_assert(N > 0, "accessor is used for indexing tensor, for scalars use *data<T>()");
    AT_CHECK(dim() == N, "expected ", N, " dims but tensor has ", dim());
    return TensorAccessor<T,N>(data<T>(),sizes().data(),strides().data());
  }
  template<typename T, size_t N>
  TensorAccessor<T,N> accessor() && = delete;

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
    pImpl->set_requires_grad(requires_grad);
    return *this;
  }
  bool requires_grad() const {
    return pImpl->requires_grad();
  }

  Tensor& grad() {
    return pImpl->grad();
  }
  const Tensor& grad() const {
    return pImpl->grad();
  }

  Tensor detach() const {
    return pImpl->detach();
  }
  void detach_() {
    pImpl->detach_();
  }

  /// Computes the gradient of current tensor w.r.t. graph leaves.
  void backward(
      at::optional<Tensor> gradient = at::nullopt,
      bool keep_graph = false,
      bool create_graph = false);

  friend void detail::set_data(Tensor& tensor, Tensor new_data);

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
};

struct WeakTensor : public detail::WeakTensorBase {
  using WeakTensorBase = detail::WeakTensorBase;
  WeakTensor() : WeakTensorBase() {}
  WeakTensor(TensorImpl * self, bool retain) : WeakTensorBase(self, retain) {}
  WeakTensor(const WeakTensor & rhs) = default;
  WeakTensor(WeakTensor && rhs) noexcept = default;
  WeakTensor(const Tensor& t) : WeakTensorBase(t.pImpl, true) {}

  // reimplemented from TensorBase so the return type is WeakTensor rather than TensorBase
  WeakTensor & operator=(WeakTensor && rhs) & {
    rhs.swap(*this);
    return *this;
  }
  WeakTensor & operator=(WeakTensor const & rhs) & {
    //Tensor ctor retains original rhs.pImpl
    //then rhs.pImpl is swapped with this->pImpl
    //finally Tensor dtor releases rhs.pImpl, which was originally this->pImpl
    WeakTensor(rhs).swap(*this);
    return *this;
  }

  WeakTensor & operator=(const Tensor& t) {
    WeakTensor(t.pImpl, true).swap(*this);
    return *this;
  }

  // non-retaining
  TensorImpl * unsafeGetTensorImpl() const {
    return pImpl;
  }

  // XXX: this can return undefined tensors
  // Ideally it would be at::optional<Tensor>, but MSVC is too cool for that
  Tensor lock() const {
    return pImpl->weak_lock() ? Tensor(pImpl, false) : Tensor();
  }
};

namespace detail {
inline void set_data(Tensor& tensor, Tensor new_data) {
  tensor.pImpl->set_data(new_data);
}
} // namespace detail
} // namespace at
