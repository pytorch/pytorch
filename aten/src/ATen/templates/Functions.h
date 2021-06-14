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
#include <ATen/Operators.h>

${static_dispatch_extra_headers}

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

${function_definitions}

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

  Tensor make_tensor() {
    AutoDispatchBelowADInplaceOrView guard{}; // TODO: Remove.
    tracer::impl::NoTracerDispatchMode tracer_guard{};

    check_size_nonnegative(sizes_);

    TORCH_CHECK_VALUE(
        !deleter_ || !ctx_,
        "The deleter and context arguments are mutually exclusive.");

    if (device_ == nullopt) {
      device_ = globalContext().getDeviceFromPtr(data_, opts_.device().type());
    }

    if (opts_.device().has_index()) {
      // clang-format off
      TORCH_CHECK_VALUE(
          opts_.device() == *device_,
          "Specified device ", opts_.device(), " does not match device of data ", *device_);
      // clang-format on
    }

    std::size_t size_bytes = computeStorageSize();

    DataPtr data_ptr{};
    if (deleter_) {
      data_ptr = makeDataPtrFromDeleter();
    } else {
      data_ptr = makeDataPtrFromContext();
    }

    Storage storage{Storage::use_byte_size_t{}, size_bytes, std::move(data_ptr)};

    Tensor tensor = detail::make_tensor<TensorImpl>(
        std::move(storage), opts_.computeDispatchKey(), opts_.dtype());

    if (sizes_.size() != 1 || sizes_[0] != 0) {
      TensorImpl* tensor_impl = tensor.unsafeGetTensorImpl();

      if (strides_) {
        tensor_impl->set_sizes_and_strides(sizes_, *strides_);
      } else {
        tensor_impl->set_sizes_contiguous(sizes_);
      }
    }

    return tensor;
  }

 private:
  explicit TensorMaker(void* data, IntArrayRef sizes) noexcept
      : data_{data}, sizes_{sizes} {}

  std::size_t computeStorageSize() const noexcept {
    std::size_t itemsize = opts_.dtype().itemsize();

    if (strides_) {
      return detail::computeStorageNbytes(sizes_, *strides_, itemsize);
    }

    std::size_t size = 1;
    for (std::int64_t s : sizes_) {
      size *= static_cast<std::size_t>(s);
    }
    return size * itemsize;
  }

  inline DataPtr makeDataPtrFromDeleter() const {
    return InefficientStdFunctionContext::makeDataPtr(data_, deleter_, *device_);
  }

  inline DataPtr makeDataPtrFromContext() noexcept {
    return DataPtr{data_, ctx_.release(), ctx_.get_deleter(), *device_};
  }

  IntArrayRef makeTempSizes() const noexcept {
    static std::int64_t zeros[5] = {0, 0, 0, 0, 0};
    if (opts_.has_memory_format()) {
      MemoryFormat format = *opts_.memory_format_opt();
      if (format == MemoryFormat::ChannelsLast) {
        return IntArrayRef(zeros, 4);
      }
      if (format == MemoryFormat::ChannelsLast3d) {
        return IntArrayRef(zeros, 5);
      }
    }
    return IntArrayRef(zeros, 1);
  }

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

inline bool is_inference(const Tensor& tensor) {
  return tensor.is_inference();
}

inline bool is_conj(const Tensor& tensor) {
  return tensor.is_conj();
}

inline Tensor conj(const Tensor& tensor) {
  return tensor.conj();
}

}
