#pragma once

#include "ATen/Generator.h"
#include "ATen/Scalar.h"
#include "ATen/ScalarType.h"
#include "ATen/SparseTensorRef.h"
#include "ATen/Storage.h"
#include "ATen/TensorAccessor.h"
#include "ATen/TensorBase.h"
#include "ATen/TensorImpl.h"
#include "ATen/Utils.h"

namespace at {
struct Type;
struct Tensor;
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

  /// Returns true if the `Tensor` is actually a `torch::autograd::Variable`,
  /// or has undefined type. Defined in Type.h because of include order issues.
  bool is_variable_or_undefined() const noexcept;

  template<typename T>
  T * data() const;

  void * unsafeGetTH(bool retain) const {
    return pImpl->unsafeGetTH(retain);
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
  TensorAccessor<T,N> accessor() {
    static_assert(N > 0, "accessor is used for indexing tensor, for scalars use *data<T>()");
    AT_CHECK(dim() == N, "expected ", N, " dims but tensor has ", dim());
    return TensorAccessor<T,N>(data<T>(),sizes().data(),strides().data());
  }

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

  // ~~~~~ Autograd API ~~~~~

  void set_requires_grad(bool requires_grad) {
    pImpl->set_requires_grad(requires_grad);
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
};

namespace detail {
inline void set_data(Tensor& tensor, Tensor new_data) {
  tensor.pImpl->set_data(new_data);
}
} // namespace detail
} // namespace at
