#pragma once

// ${generated_comment}

#include <c10/core/Scalar.h>
#include <ATen/Tensor.h>
#include <c10/core/Storage.h>
#include <ATen/core/Generator.h>
#include <c10/util/Deprecated.h>
#include <ATen/DeviceGuard.h>
#include <c10/core/TensorOptions.h>
#include <ATen/core/Reduction.h>
#include <c10/util/Optional.h>
#include <ATen/TensorUtils.h>
#include <ATen/Context.h>
#include <ATen/TracerMode.h>

namespace at {

// These functions are defined in ATen/Utils.cpp.
#define TENSOR(T, S)                                                          \
  TORCH_API Tensor tensor(ArrayRef<T> values, const TensorOptions& options); \
  inline Tensor tensor(                                                       \
      std::initializer_list<T> values, const TensorOptions& options) {        \
    return at::tensor(ArrayRef<T>(values), options);                          \
  }                                                                           \
  inline Tensor tensor(T value, const TensorOptions& options) {               \
    return at::tensor(ArrayRef<T>(value), options);                           \
  }                                                                           \
  inline Tensor tensor(ArrayRef<T> values) {                                  \
    return at::tensor(std::move(values), at::dtype(k##S));                    \
  }                                                                           \
  inline Tensor tensor(std::initializer_list<T> values) {                     \
    return at::tensor(ArrayRef<T>(values));                                   \
  }                                                                           \
  inline Tensor tensor(T value) {                                             \
    return at::tensor(ArrayRef<T>(value));                                    \
  }
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TENSOR)
AT_FORALL_COMPLEX_TYPES(TENSOR)
#undef TENSOR

${function_declarations}

// Special C++ only overloads for std()-like functions (See gh-40287)
// These are needed because int -> bool conversion takes precedence over int -> IntArrayRef
// So, for example std(0) would select the std(unbiased=False) overload
TORCH_API Tensor var(const Tensor& self, int dim);
TORCH_API std::tuple<Tensor,Tensor> var_mean(const Tensor& self, int dim);
TORCH_API Tensor std(const Tensor& self, int dim);
TORCH_API std::tuple<Tensor,Tensor> std_mean(const Tensor& self, int dim);


// Special C++ only overloads for convnd functions (See gh-45667)
// These are needed because {1, 2} is ambiguous between string and IntArrayRef overloads
TORCH_API at::Tensor conv1d(
    const Tensor& input, const Tensor& weight, const Tensor& bias, IntArrayRef stride,
    std::initializer_list<int64_t> padding, IntArrayRef dilation = 1, int64_t groups = 1);
TORCH_API at::Tensor conv2d(
    const Tensor& input, const Tensor& weight, const Tensor& bias, IntArrayRef stride,
    std::initializer_list<int64_t> padding, IntArrayRef dilation = 1, int64_t groups = 1);
TORCH_API at::Tensor conv3d(
    const Tensor& input, const Tensor& weight, const Tensor& bias, IntArrayRef stride,
    std::initializer_list<int64_t> padding, IntArrayRef dilation = 1, int64_t groups = 1);

namespace detail {

TORCH_API void noopDelete(void*);

} // namespace detail

/// Provides a fluent API to construct tensors from external data.
///
/// The fluent API can be used instead of `from_blob` functions in case the
/// required set of parameters does not align with the existing overloads.
///
///     at::Tensor tensor = at::for_blob(data, sizes)
///             .strides(strides)
///             .context(context, [](void *ctx) { delete static_cast<Ctx*>(ctx); })
///             .options(...)
///             .make_tensor();
///
class TORCH_API TensorMaker {
  friend TensorMaker for_blob(void* data, IntArrayRef sizes) noexcept;

 public:
  using ContextDeleter = DeleterFnPtr;

  TensorMaker& strides(optional<IntArrayRef> value) noexcept {
    strides_ = value;

    return *this;
  }

  TensorMaker& deleter(std::function<void(void*)> value) noexcept {
    deleter_ = std::move(value);

    return *this;
  }

  TensorMaker& context(void* value, ContextDeleter deleter = nullptr) noexcept {
    ctx_ = std::unique_ptr<void, ContextDeleter>{
        value, deleter ? deleter : detail::noopDelete};

    return *this;
  }

  TensorMaker& target_device(optional<Device> value) noexcept {
    device_ = value;

    return *this;
  }

  TensorMaker& options(TensorOptions value) noexcept {
    opts_ = value;

    return *this;
  }

  Tensor make_tensor();

 private:
  explicit TensorMaker(void* data, IntArrayRef sizes) noexcept
      : data_{data}, sizes_{sizes} {}

  std::size_t computeStorageSize() const noexcept;

  DataPtr makeDataPtrFromDeleter() const;

  DataPtr makeDataPtrFromContext() noexcept;

  IntArrayRef makeTempSizes() const noexcept;

  Tensor makeEmptyTensor() const;

  void* data_;
  IntArrayRef sizes_;
  optional<IntArrayRef> strides_{};
  std::function<void(void*)> deleter_{};
  std::unique_ptr<void, ContextDeleter> ctx_{nullptr, detail::noopDelete};
  optional<Device> device_{};
  TensorOptions opts_{};
};

inline TensorMaker for_blob(void* data, IntArrayRef sizes) noexcept {
  return TensorMaker{data, sizes};
}

inline Tensor from_blob(
    void* data,
    IntArrayRef sizes,
    IntArrayRef strides,
    const std::function<void(void*)>& deleter,
    const TensorOptions& options = {},
    const c10::optional<Device> target_device = c10::nullopt) {
  return for_blob(data, sizes)
      .strides(strides)
      .deleter(deleter)
      .options(options)
      .target_device(target_device)
      .make_tensor();
}

inline Tensor from_blob(
    void* data,
    IntArrayRef sizes,
    const std::function<void(void*)>& deleter,
    const TensorOptions& options = {}) {
  return for_blob(data, sizes)
      .deleter(deleter)
      .options(options)
      .make_tensor();
}

inline Tensor from_blob(
    void* data,
    IntArrayRef sizes,
    IntArrayRef strides,
    const TensorOptions& options = {}) {
  return for_blob(data, sizes)
      .strides(strides)
      .options(options)
      .make_tensor();
}

inline Tensor from_blob(
    void* data,
    IntArrayRef sizes,
    const TensorOptions& options = {}) {
  return for_blob(data, sizes).options(options).make_tensor();
}

inline int64_t numel(const Tensor& tensor) {
  return tensor.numel();
}

inline int64_t size(const Tensor& tensor, int64_t dim) {
  return tensor.size(dim);
}

inline int64_t stride(const Tensor& tensor, int64_t dim) {
  return tensor.stride(dim);
}

inline bool is_complex(const Tensor& tensor) {
  return tensor.is_complex();
}

inline bool is_floating_point(const Tensor& tensor) {
  return tensor.is_floating_point();
}

inline bool is_signed(const Tensor& tensor) {
  return tensor.is_signed();
}

}
