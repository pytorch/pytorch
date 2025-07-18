#pragma once
#include <ATen/core/Tensor.h>

namespace at {

namespace detail {

inline void noopDelete(void*) {}

} // namespace detail

/// Provides a fluent API to construct tensors from external data.
///
/// The fluent API can be used instead of `from_blob` functions in case the
/// required set of parameters does not align with the existing overloads.
///
///     at::Tensor tensor = at::for_blob(data, sizes)
///             .strides(strides)
///             .context(context, [](void *ctx) { delete static_cast<Ctx*>(ctx);
///             }) .options(...) .make_tensor();
///
class TORCH_API TensorMaker {
  friend TensorMaker for_blob(void* data, IntArrayRef sizes) noexcept;

 public:
  using ContextDeleter = DeleterFnPtr;

  TensorMaker& strides(OptionalIntArrayRef value) noexcept {
    strides_ = value;

    return *this;
  }

  TensorMaker& storage_offset(std::optional<int64_t> value) noexcept {
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

  TensorMaker& target_device(std::optional<Device> value) noexcept {
    device_ = value;

    return *this;
  }

  TensorMaker& options(TensorOptions value) noexcept {
    opts_ = value;

    return *this;
  }

  TensorMaker& resizeable_storage() noexcept {
    resizeable_ = true;

    return *this;
  }

  TensorMaker& allocator(c10::Allocator* allocator) noexcept {
    allocator_ = allocator;

    return *this;
  }

  Tensor make_tensor();

 private:
  explicit TensorMaker(void* data, IntArrayRef sizes) noexcept
      : data_{data}, sizes_{sizes} {}

  std::size_t computeStorageSize() const noexcept;

  DataPtr makeDataPtrFromDeleter() noexcept;

  DataPtr makeDataPtrFromContext() noexcept;

  IntArrayRef makeTempSizes() const noexcept;

  void* data_;
  IntArrayRef sizes_;
  OptionalIntArrayRef strides_{};
  std::optional<int64_t> storage_offset_{};
  std::function<void(void*)> deleter_{};
  std::unique_ptr<void, ContextDeleter> ctx_{nullptr, detail::noopDelete};
  std::optional<Device> device_{};
  TensorOptions opts_{};
  bool resizeable_{};
  c10::Allocator* allocator_{};
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
    const std::optional<Device> target_device = std::nullopt) {
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
    IntArrayRef strides,
    int64_t storage_offset,
    const std::function<void(void*)>& deleter,
    const TensorOptions& options = {},
    const std::optional<Device> target_device = std::nullopt) {
  return for_blob(data, sizes)
      .strides(strides)
      .storage_offset(storage_offset)
      .deleter(deleter)
      .options(options)
      .target_device(target_device)
      .make_tensor();
}

inline Tensor from_blob(
    void* data,
    IntArrayRef sizes,
    std::function<void(void*)> deleter,
    const TensorOptions& options = {},
    const std::optional<Device> target_device = std::nullopt) {
  return for_blob(data, sizes)
      .deleter(std::move(deleter))
      .options(options)
      .target_device(target_device)
      .make_tensor();
}

inline Tensor from_blob(
    void* data,
    IntArrayRef sizes,
    IntArrayRef strides,
    const TensorOptions& options = {}) {
  return for_blob(data, sizes).strides(strides).options(options).make_tensor();
}

inline Tensor from_blob(
    void* data,
    IntArrayRef sizes,
    const TensorOptions& options = {}) {
  return for_blob(data, sizes).options(options).make_tensor();
}

} // namespace at
