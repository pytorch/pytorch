#pragma once

#include <ATen/core/TensorBody.h>
#include <c10/util/Exception.h>

namespace at {
// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
class TORCH_API OptionalTensorRef {
 public:
  OptionalTensorRef() = default;

  ~OptionalTensorRef() {
    ref_.unsafeReleaseTensorImpl();
  }

  OptionalTensorRef(const TensorBase& src)
      : ref_(Tensor::unsafe_borrow_t{}, src) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(src.defined());
  }

  OptionalTensorRef(const OptionalTensorRef& rhs)
      : ref_(Tensor::unsafe_borrow_t{}, rhs.ref_) {}

  OptionalTensorRef(OptionalTensorRef&& rhs) = default;
  OptionalTensorRef& operator=(OptionalTensorRef rhs) {
    std::swap(ref_, rhs.ref_);
    return *this;
  }

  bool has_value() const {
    return ref_.defined();
  }

  const Tensor& getTensorRef() const & {
    return ref_;
  }

  const Tensor& operator*() const & {
    return ref_;
  }

  const Tensor* operator->() const & {
    return &ref_;
  }

  operator bool() const {
    return ref_.defined();
  }

 private:
  Tensor ref_;
};

// Use to convert a TensorBase (that may be undefined) to an at::Tensor
// without bumping refcount.
class TORCH_API TensorRef {
 public:
  ~TensorRef() {
    ref_.unsafeReleaseTensorImpl();
  }

  TensorRef(const TensorBase& src)
      : ref_(Tensor::unsafe_borrow_t{}, src) {}
  TensorRef(TensorRef&& other) = default;
  TensorRef(const TensorRef&) = default;
  TensorRef& operator=(const TensorRef&) = default;
  TensorRef& operator=(TensorRef&&) = default;

  const Tensor& operator*() const & {
    return ref_;
  }
 private:
  Tensor ref_;
};

template <typename T>
auto Tensor::register_hook(T&& hook) const -> Tensor::hook_return_void_t<T> {
  // Return the grad argument in case of a hook with void return type to have an
  // std::function with Tensor return type
  static_assert(std::is_same_v<decltype(hook(Tensor())), void>,
                "Expected hook to return void");
  return _register_hook([fn=std::forward<T>(hook)](const TensorBase& grad_base) {
    TensorRef grad(grad_base);
    fn(*grad);
    return Tensor();
  });
}

template <typename T>
auto Tensor::register_hook(T&& hook) const -> Tensor::hook_return_var_t<T> {
  return _register_hook([fn=std::forward<T>(hook)](const TensorBase& grad_base) {
    TensorRef grad(grad_base);
    Tensor ret = fn(*grad);
    return TensorBase(std::move(ret));
  });
}

} // namespace at
