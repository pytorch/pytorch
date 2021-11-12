#pragma once

// ${generated_comment}

#ifdef TORCH_ASSERT_NO_OPERATORS
#error This change adds a dependency on native_functions.yaml,            \
  meaning the file will need to be re-compiled every time an operator     \
  is changed or added. Consider if your change would be better placed in  \
  another file, or if a more specific header might achieve the same goal. \
  See NOTE: [Tensor vs. TensorBase]
#endif

${Functions_includes}

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

// Special C++ only overloads for std()-like functions (See gh-40287)
// These are needed because int -> bool conversion takes precedence over int -> IntArrayRef
// So, for example std(0) would select the std(unbiased=False) overload
TORCH_API inline Tensor var(const Tensor& self, int dim) {
  return at::var(self, IntArrayRef{dim});
}
TORCH_API inline std::tuple<Tensor, Tensor> var_mean(const Tensor& self, int dim) {
  return at::var_mean(self, IntArrayRef{dim});
}
TORCH_API inline Tensor std(const Tensor& self, int dim) {
  return at::std(self, IntArrayRef{dim});
}
TORCH_API inline std::tuple<Tensor, Tensor> std_mean(const Tensor& self, int dim) {
  return at::std_mean(self, IntArrayRef{dim});
}

namespace detail {

TORCH_API inline void noopDelete(void*) {}

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

  TensorMaker& storage_offset(optional<int64_t> value) noexcept {
    storage_offset_ = value;

    return *this;
  }

  TensorMaker& deleter(std::function<void(void*)> value) noexcept {
    deleter_ = std::move(value);

    return *this;
  }

  TensorMaker& context(void* value, ContextDeleter deleter = nullptr) noexcept {
    ctx_ = std::unique_ptr<void, ContextDeleter>{
        value, deleter != nullptr ? deleter : detail::noopDelete};

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

  void* data_;
  IntArrayRef sizes_;
  optional<IntArrayRef> strides_{};
  optional<int64_t> storage_offset_{};
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

inline bool is_inference(const Tensor& tensor) {
  return tensor.is_inference();
}

inline bool is_conj(const Tensor& tensor) {
  return tensor.is_conj();
}

inline Tensor conj(const Tensor& tensor) {
  return tensor.conj();
}

inline bool is_neg(const Tensor& tensor) {
  return tensor.is_neg();
}

}
