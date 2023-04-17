#pragma once

#include <c10/core/Backend.h>
#include <c10/core/DefaultDtype.h>
#include <c10/core/Device.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/Layout.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <c10/core/ScalarTypeToTypeMeta.h>

#include <c10/macros/Macros.h>
#include <c10/util/C++17.h>
#include <c10/util/Optional.h>

#include <cstddef>
#include <iosfwd>
#include <utility>

namespace c10 {

DispatchKey computeDispatchKey(
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device);

inline ScalarType dtype_or_default(c10::optional<ScalarType> dtype) {
  return value_or_else(dtype, [] { return get_default_dtype_as_scalartype(); });
}

inline caffe2::TypeMeta dtype_or_default(
    c10::optional<caffe2::TypeMeta> dtype) {
  return value_or_else(dtype, [] { return get_default_dtype(); });
}

inline Layout layout_or_default(c10::optional<Layout> layout) {
  return layout.value_or(kStrided);
}

inline Device device_or_default(c10::optional<Device> device) {
  return value_or_else(device, [] { return Device(kCPU); });
}

inline bool pinned_memory_or_default(c10::optional<bool> pinned_memory) {
  return pinned_memory.value_or(false);
}

/// A class to encapsulate construction axes of an Tensor.  TensorOptions was
/// designed to support the Python style API for specifying construction options
/// on factory functions, e.g.,
///
///     torch.zeros(2, 3, dtype=torch.int32)
///
/// Because C++ doesn't natively support keyword arguments, there must be
/// another way of specifying keyword-like arguments.  TensorOptions is a
/// builder class which can be used to construct this "dictionary" of keyword
/// arguments: functions which support TensorOptions conventionally take this
/// argument optionally as their last argument.
///
/// WARNING: In PyTorch, there are `torch::` variants of factory functions,
/// e.g., torch::zeros for at::zeros.  These return Variables (while the
/// stock ATen functions return plain Tensors).  If you mix these functions
/// up, you WILL BE SAD.
///
/// Rather than use the constructor of this class directly, you should prefer to
/// use the constructor functions, and then chain setter methods on top of them.
///
///     at::device(at::kCUDA).dtype(kInt)
///     at::dtype(at::kInt)
///
/// Additionally, anywhere a TensorOptions is expected, you can directly
/// pass at::kCUDA / at::kInt, and it will implicitly convert to a
/// TensorOptions.
///
/// Here are some recommended ways to create a 2x2 tensor of zeros
/// with certain properties.  These all *implicitly* make use of
/// TensorOptions, even if they don't mention the class explicitly:
///
///     at::zeros({2,2}, at::kCUDA);
///     at::zeros({2,2}, at::kLong);
///     at::zeros({2,2}, at::device(at::kCUDA).dtype(at::kLong()));
///     at::zeros({2,2}, at::device({at::kCUDA, 1})); // place on device 1
///     at::zeros({2,2}, at::requires_grad());
///

/// NOTE [ TensorOptions Constructors ]
///
/// TensorOptions is like a dictionary with entries from the set:
/// {requires_grad, device, dtype, layout}, where each entry may be
/// unspecified (i.e., is optional). It is used to specify the properties of
/// tensors in many places both in C++ internal and API, e.g., tensor factory
/// methods like `at::empty({10}, options)`, tensor conversions like
/// `tensor.to(...)`, etc.
///
/// To provide a simple API that is consistent with Python, where one can do
/// `torch.empty(sizes, X)` with `X` being a `torch.device`, `torch.dtype`, or a
/// `torch.layout`, we want TensorOptions to be implicitly convertible from
/// `ScalarType dtype`, `Layout layout` and `Device device`. Therefore, we have
/// three implicit constructors from each of these three types.
///
/// This is sufficient for `ScalarType` and `Layout` as they are simple Enum
/// classes. However, `Device` is an ordinary class with implicit constructors
/// `Device(DeviceType, DeviceIndex = -1)` and `Device(std::string)` to be
/// consistent with Python API, where strings are treated as equivalent with a
/// `torch.device` object (e.g., "cuda:1" can be passed to everywhere a
/// `torch.device("cuda:1")` is accepted). To support the syntax
/// `at::empty({10}, {kCUDA, 1})` and `tensor.to(kCUDA)`, we need to make sure
/// that `TensorOptions` is implicitly constructible with any arguments that a
/// `Device` can constructed from. So we have,
///
///    /* implicit */ TensorOptions(T&& device) : TensorOptions() {
///      this->set_device(device);
///    }
///
///    template <typename... Args,
///             typename = std::enable_if_t<std::is_constructible<Device,
///             Args&&...>::value>>
///    /* implicit */  TensorOptions(Args&&... args)
///     : TensorOptions(Device(std::forward<Args>(args)...)) {}
///
///
/// But this will be problematic. Consider this: `TensorOptions({kCUDA, 1})`.
/// Compiler will complain about ambiguity between the copy constructor and the
/// `Device` constructor because `{kCUDA, 1}` can be converted to both a
/// `TensorOption` and a `Device`.
///
/// To get around this, we templatize the `Device` constructor. Since overload
/// resolution is done before template resolution, our problem is solved.

DispatchKey computeDispatchKey(
    optional<ScalarType> dtype,
    optional<Layout> layout,
    optional<Device> device);

struct C10_API TensorOptions {
  TensorOptions()
      : requires_grad_(false),
        pinned_memory_(false),
        has_device_(false),
        has_dtype_(false),
        has_layout_(false),
        has_requires_grad_(false),
        has_pinned_memory_(false),
        has_memory_format_(false) {}

  /// Constructs a `TensorOptions` object with the given layout.
  /* implicit */ TensorOptions(Layout layout) : TensorOptions() {
    this->set_layout(layout);
  }

  /// Constructs a `TensorOptions` object with the given device.
  /// See NOTE [ TensorOptions Constructors ] on why this is templatized.
  template <
      typename T,
      typename = std::enable_if_t<std::is_same<std::decay_t<T>, Device>::value>>
  /* implicit */ TensorOptions(T&& device) : TensorOptions() {
    this->set_device(std::forward<T>(device));
  }

  /// Constructs a `TensorOptions` object from arguments allowed in `Device`
  /// constructors.
  ///
  /// See NOTE [ TensorOptions Constructors ].
  ///
  /// NB: Ideally we only allow implicit constructors here. But there is no easy
  ///     way to detect them. So we have this one that allows explicit
  ///     constructors too.
  template <
      typename... Args,
      typename =
          std::enable_if_t<std::is_constructible<Device, Args&&...>::value>>
  /* implicit */ TensorOptions(Args&&... args)
      : TensorOptions(Device(std::forward<Args>(args)...)) {}

  /// Constructs a `TensorOptions` object with the given dtype.
  /* implicit */ TensorOptions(caffe2::TypeMeta dtype) : TensorOptions() {
    this->set_dtype(dtype);
  }

  /// legacy constructor to support ScalarType
  /* implicit */ TensorOptions(ScalarType dtype) : TensorOptions() {
    this->set_dtype(dtype);
  }

  /// Constructs a `TensorOptions` object with the given memory format.
  /* implicit */ TensorOptions(MemoryFormat memory_format) : TensorOptions() {
    set_memory_format(memory_format);
  }

  /// Return a copy of `TensorOptions` with `device` set to the given one, or
  /// cleared if `device` is `nullopt`.
  C10_NODISCARD TensorOptions
  device(c10::optional<Device> device) const noexcept {
    TensorOptions r = *this;
    r.set_device(device);
    return r;
  }

  /// Return a copy of `TensorOptions` with `device` set to the given one.
  /// (This overload ensures that variadic template c10::optional constructor
  /// for Device work correctly.)
  template <typename... Args>
  C10_NODISCARD TensorOptions device(Args&&... args) const noexcept {
    return device(
        c10::optional<Device>(c10::in_place, std::forward<Args>(args)...));
  }

  /// Return a copy of `TensorOptions`, but with device set to CUDA, and the
  /// device index set to the given one.
  ///
  /// TODO: This function encourages bad behavior (assuming CUDA is
  /// the only device that matters).  Get rid of it / rename it.
  C10_NODISCARD TensorOptions
  device_index(c10::DeviceIndex device_index) const noexcept {
    return device(Device::Type::CUDA, device_index);
  }

  /// Return a copy of `TensorOptions` with `dtype` set to the given one.
  C10_NODISCARD TensorOptions
  dtype(c10::optional<caffe2::TypeMeta> dtype) const noexcept {
    TensorOptions r = *this;
    r.set_dtype(dtype);
    return r;
  }

  // legacy function to support ScalarType
  C10_NODISCARD TensorOptions
  dtype(c10::optional<ScalarType> dtype) const noexcept {
    TensorOptions r = *this;
    r.set_dtype(dtype);
    return r;
  }

  // Since dtype is taken...
  template <typename T>
  TensorOptions& dtype() {
    dtype_ = caffe2::TypeMeta::Make<T>();
    has_dtype_ = true;
    return *this;
  }

  /// Sets the layout of the `TensorOptions`.
  C10_NODISCARD TensorOptions
  layout(c10::optional<Layout> layout) const noexcept {
    TensorOptions r = *this;
    r.set_layout(layout);
    return r;
  }

  /// Sets the `requires_grad` property of the `TensorOptions`.
  C10_NODISCARD TensorOptions
  requires_grad(c10::optional<bool> requires_grad) const noexcept {
    TensorOptions r = *this;
    r.set_requires_grad(requires_grad);
    return r;
  }

  /// Sets the `pinned_memory` property on the `TensorOptions`.
  C10_NODISCARD TensorOptions
  pinned_memory(c10::optional<bool> pinned_memory) const noexcept {
    TensorOptions r = *this;
    r.set_pinned_memory(pinned_memory);
    return r;
  }

  /// Sets the `memory_format` property on `TensorOptions`.
  C10_NODISCARD TensorOptions
  memory_format(c10::optional<MemoryFormat> memory_format) const noexcept {
    TensorOptions r = *this;
    r.set_memory_format(memory_format);
    return r;
  }

  /// Returns the device of the `TensorOptions`.
  Device device() const noexcept {
    return device_or_default(device_opt());
  }

  /// Returns whether the device is specified.
  bool has_device() const noexcept {
    return has_device_;
  }

  /// Returns the device of the `TensorOptions`, or `c10::nullopt` if
  /// device is not specified.
  c10::optional<Device> device_opt() const noexcept {
    return has_device_ ? c10::make_optional(device_) : c10::nullopt;
  }

  /// Returns the device index of the `TensorOptions`.
  int32_t device_index() const noexcept {
    return device().index();
  }

  /// Returns the dtype of the `TensorOptions`.
  caffe2::TypeMeta dtype() const noexcept {
    return dtype_or_default(dtype_opt());
  }

  /// Returns whether the dtype is specified.
  bool has_dtype() const noexcept {
    return has_dtype_;
  }

  /// Returns the dtype of the `TensorOptions`, or `c10::nullopt` if
  /// device is not specified.
  c10::optional<caffe2::TypeMeta> dtype_opt() const noexcept {
    return has_dtype_ ? c10::make_optional(dtype_) : c10::nullopt;
  }

  /// Returns the layout of the `TensorOptions`.
  Layout layout() const noexcept {
    return layout_or_default(layout_opt());
  }

  /// Returns whether the layout is specified.
  bool has_layout() const noexcept {
    return has_layout_;
  }

  /// Returns the layout of the `TensorOptions`, or `c10::nullopt` if
  /// layout is not specified.
  c10::optional<Layout> layout_opt() const noexcept {
    return has_layout_ ? c10::make_optional(layout_) : c10::nullopt;
  }

  /// Returns the `requires_grad` property of the `TensorOptions`.
  bool requires_grad() const noexcept {
    return has_requires_grad_ ? requires_grad_ : false;
  }

  /// Returns whether the `requires_grad` is specified.
  bool has_requires_grad() const noexcept {
    return has_requires_grad_;
  }

  /// Returns the `requires_grad` property of the `TensorOptions`, or
  /// `c10::nullopt` if `requires_grad` is not specified.
  c10::optional<bool> requires_grad_opt() const noexcept {
    return has_requires_grad_ ? c10::make_optional(requires_grad_)
                              : c10::nullopt;
  }

  /// Returns the `pinned_memory` property of the `TensorOptions`.
  bool pinned_memory() const noexcept {
    return pinned_memory_or_default(pinned_memory_opt());
  }

  /// Returns whether the `pinned_memory` is specified.
  bool has_pinned_memory() const noexcept {
    return has_pinned_memory_;
  }

  /// Returns if the layout is sparse
  bool is_sparse() const {
    return layout_ == c10::Layout::Sparse;
  }

  bool is_sparse_csr() const {
    return layout_ == c10::Layout::SparseCsr;
  }

  // For compatibility with legacy tensor.type() comparisons
  bool type_equal(const TensorOptions& other) const {
    return computeDispatchKey() == other.computeDispatchKey() &&
        typeMetaToScalarType(dtype_) == typeMetaToScalarType(other.dtype());
  }

  /// Returns the `pinned_memory` property of the `TensorOptions`, or
  /// `c10::nullopt` if `pinned_memory` is not specified.
  c10::optional<bool> pinned_memory_opt() const noexcept {
    return has_pinned_memory_ ? c10::make_optional(pinned_memory_)
                              : c10::nullopt;
  }

  /// Returns whether the `memory_layout` is specified
  bool has_memory_format() const noexcept {
    return has_memory_format_;
  }

  // NB: memory_format() getter is PURPOSELY not defined, as the default
  // behavior of memory_format varies from function to function.

  /// Returns the `memory_layout` property of `TensorOptions, or
  /// `c10::nullopt` if `memory_format` is not specified.
  c10::optional<MemoryFormat> memory_format_opt() const noexcept {
    return has_memory_format_ ? c10::make_optional(memory_format_)
                              : c10::nullopt;
  }

  // Resolves the ATen backend specified by the current construction axes.
  // TODO: Deprecate this
  Backend backend() const {
    return at::dispatchKeyToBackend(computeDispatchKey());
  }

  /// Return the right-biased merge of two TensorOptions.  This has the
  /// effect of overwriting settings from self with specified options
  /// of options.
  ///
  /// NB: This merging operation does NOT respect device merges.
  /// For example, if you device({kCUDA, 1}).merge_in(kCUDA)
  /// you will get kCUDA in the end!  Functions like Tensor.new_empty
  /// ensure the right device is selected anyway by way of a
  /// device guard.
  ///
  TensorOptions merge_in(TensorOptions options) const noexcept {
    TensorOptions merged = *this;
    if (options.has_device())
      merged.set_device(options.device_opt());
    if (options.has_dtype())
      merged.set_dtype(options.dtype_opt());
    if (options.has_layout())
      merged.set_layout(options.layout_opt());
    // NB: requires grad is right biased; not a logical AND/OR!
    if (options.has_requires_grad())
      merged.set_requires_grad(options.requires_grad_opt());
    if (options.has_pinned_memory())
      merged.set_pinned_memory(options.pinned_memory_opt());
    if (options.has_memory_format())
      merged.set_memory_format(options.memory_format_opt());
    return merged;
  }

  // TODO remove after TensorOptions rationalization
  TensorOptions merge_memory_format(
      c10::optional<MemoryFormat> optional_memory_format) const noexcept {
    TensorOptions merged = *this;
    if (optional_memory_format.has_value()) {
      merged.set_memory_format(*optional_memory_format);
    }
    return merged;
  }

  // INVARIANT: computeDispatchKey returns only the subset of dispatch keys for
  // which dispatchKeyToBackend is injective, if it is defined at all  (for
  // the most part, this just means that this function never returns an
  // Autograd key)
  DispatchKey computeDispatchKey() const {
    return c10::computeDispatchKey(
        optTypeMetaToScalarType(dtype_opt()), layout_opt(), device_opt());
  }

 private:
  // These methods are currently private because I'm not sure if it's wise
  // to actually publish them.  They are methods because I need them in
  // the constructor and the functional API implementation.
  //
  // If you really, really need it, you can make these public, but check if you
  // couldn't just do what you need with the functional API.  Similarly, these
  // methods are not chainable, because if you wanted chaining, you probably
  // want to use the functional API instead.  (It's probably OK to make
  // these chainable, because these functions are all explicitly annotated
  // with a ref-qualifier, the trailing &, that makes them illegal to call
  // on temporaries.)

  /// Mutably set the device of `TensorOptions`.
  void set_device(c10::optional<Device> device) & noexcept {
    if (device) {
      device_ = *device;
      has_device_ = true;
    } else {
      has_device_ = false;
    }
  }

  /// Mutably set the dtype of `TensorOptions`.
  void set_dtype(c10::optional<caffe2::TypeMeta> dtype) & noexcept {
    if (dtype) {
      dtype_ = *dtype;
      has_dtype_ = true;
    } else {
      has_dtype_ = false;
    }
  }

  // legacy function to support ScalarType
  void set_dtype(c10::optional<ScalarType> dtype) & noexcept {
    if (dtype) {
      dtype_ = scalarTypeToTypeMeta(*dtype);
      has_dtype_ = true;
    } else {
      has_dtype_ = false;
    }
  }

  /// Mutably set the layout of `TensorOptions`.
  void set_layout(c10::optional<Layout> layout) & noexcept {
    if (layout) {
      layout_ = *layout;
      has_layout_ = true;
    } else {
      has_layout_ = false;
    }
  }

  /// Mutably set the `requires_grad` property of `TensorOptions`.
  void set_requires_grad(c10::optional<bool> requires_grad) & noexcept {
    if (requires_grad) {
      requires_grad_ = *requires_grad;
      has_requires_grad_ = true;
    } else {
      has_requires_grad_ = false;
    }
  }

  /// Mutably set the `pinned_memory` property of `TensorOptions`.
  void set_pinned_memory(c10::optional<bool> pinned_memory) & noexcept {
    if (pinned_memory) {
      pinned_memory_ = *pinned_memory;
      has_pinned_memory_ = true;
    } else {
      has_pinned_memory_ = false;
    }
  }

  /// Mutably set the `memory_Format` property of `TensorOptions`.
  void set_memory_format(c10::optional<MemoryFormat> memory_format) & noexcept {
    if (memory_format) {
      memory_format_ = *memory_format;
      has_memory_format_ = true;
    } else {
      has_memory_format_ = false;
    }
  }

  // WARNING: If you edit TensorOptions to add more options, you
  // may need to adjust the implementation of Tensor::options.
  // The criteria for whether or not Tensor::options must be adjusted
  // is whether or not the new option you added should preserved
  // by functions such as empty_like(); if it should be preserved,
  // you must adjust options().
  //
  // TODO: MemoryFormat is not implemented in this way

  // NB: We didn't use c10::optional here, because then we can't pack
  // the has_***_ boolean fields.

  Device device_ = at::kCPU; // 16-bit
  caffe2::TypeMeta dtype_ = caffe2::TypeMeta::Make<float>(); // 16-bit
  Layout layout_ = at::kStrided; // 8-bit
  MemoryFormat memory_format_ = MemoryFormat::Contiguous; // 8-bit

  // Bitmask required here to get this to fit inside 32 bits (or even 64 bits,
  // for that matter)

  bool requires_grad_ : 1;
  bool pinned_memory_ : 1;

  bool has_device_ : 1;
  bool has_dtype_ : 1;
  bool has_layout_ : 1;
  bool has_requires_grad_ : 1;
  bool has_pinned_memory_ : 1;
  bool has_memory_format_ : 1;
};

// We should aspire to fit in one machine-size word; but a size greater than two
// words is too much.  (We are doing terribly on 32-bit archs, where we require
// three machine size words to store tensor options.  Eek!)
static_assert(
    sizeof(TensorOptions) <= sizeof(int64_t) * 2,
    "TensorOptions must fit in 128-bits");

/// Convenience function that returns a `TensorOptions` object with the `dtype`
/// set to the given one.
inline TensorOptions dtype(caffe2::TypeMeta dtype) {
  return TensorOptions().dtype(dtype);
}

// legacy function to support ScalarType
inline TensorOptions dtype(ScalarType dtype) {
  return TensorOptions().dtype(scalarTypeToTypeMeta(dtype));
}

/// Convenience function that returns a `TensorOptions` object with the `layout`
/// set to the given one.
inline TensorOptions layout(Layout layout) {
  return TensorOptions().layout(layout);
}

/// Convenience function that returns a `TensorOptions` object with the `device`
/// set to the given one.
inline TensorOptions device(Device device) {
  return TensorOptions().device(device);
}

/// Convenience function that returns a `TensorOptions` object with the
/// `device` set to CUDA and the `device_index` set to the given one.
inline TensorOptions device_index(int16_t device_index) {
  return TensorOptions().device_index(
      static_cast<c10::DeviceIndex>(device_index));
}

/// Convenience function that returns a `TensorOptions` object with the
/// `requires_grad` set to the given one.
inline TensorOptions requires_grad(bool requires_grad = true) {
  return TensorOptions().requires_grad(requires_grad);
}

/// Convenience function that returns a `TensorOptions` object with the
/// `memory_format` set to the given one.
inline TensorOptions memory_format(MemoryFormat memory_format) {
  return TensorOptions().memory_format(memory_format);
}

C10_API std::ostream& operator<<(
    std::ostream& stream,
    const TensorOptions& options);

template <typename T>
inline TensorOptions dtype() {
  return dtype(caffe2::TypeMeta::Make<T>());
}

inline std::string toString(const TensorOptions& options) {
  std::ostringstream stream;
  stream << options;
  return stream.str();
}

// This is intended to be a centralized location by which we can determine
// what an appropriate DispatchKey for a tensor is.
inline DispatchKey computeDispatchKey(
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device) {
  const auto layout_ = layout_or_default(layout);
  const auto device_ = device_or_default(device);
  switch (layout_) {
    case Layout::Strided: {
      const auto dtype_ = dtype_or_default(dtype);
      switch (device_.type()) {
#define DO_CASE(device, _)                   \
  case DeviceType::device: {                 \
    if (isQIntType(dtype_)) {                \
      return DispatchKey::Quantized##device; \
    }                                        \
    return DispatchKey::device;              \
  }
        C10_FORALL_BACKEND_DEVICE_TYPES(DO_CASE, unused)
#undef DO_CASE
        case DeviceType::FPGA:
          return DispatchKey::FPGA;
        case DeviceType::ORT:
          return DispatchKey::ORT;
        case DeviceType::Vulkan:
          return DispatchKey::Vulkan;
        case DeviceType::Metal:
          return DispatchKey::Metal;
        case DeviceType::MKLDNN:
        case DeviceType::OPENGL:
        case DeviceType::OPENCL:
        case DeviceType::IDEEP:
          TORCH_INTERNAL_ASSERT(
              0,
              "This is a grandfathered Caffe2 device type ",
              device_.type(),
              ", it shouldn't ever convert to a DispatchKey.  File a bug describing what you were doing if you think this is in error.");
        default:
          TORCH_CHECK_NOT_IMPLEMENTED(
              false,
              "Unsupported device type for dense layout: ",
              device_.type());
      }
    }
    case Layout::Sparse:
      switch (device_.type()) {
#define DO_CASE(device, _)              \
  case DeviceType::device: {            \
    return DispatchKey::Sparse##device; \
  }
        C10_FORALL_BACKEND_DEVICE_TYPES(DO_CASE, unused)
#undef DO_CASE
        default:
          TORCH_CHECK_NOT_IMPLEMENTED(
              false,
              "Unsupported device type for sparse layout: ",
              device_.type());
      }
    case Layout::Mkldnn:
      switch (device_.type()) {
        case DeviceType::CPU:
          return DispatchKey::MkldnnCPU;
        default:
          TORCH_CHECK_NOT_IMPLEMENTED(
              false,
              "Unsupported device type for mkldnn layout: ",
              device_.type());
      }
    case Layout::SparseCsr:
    case Layout::SparseCsc:
    case Layout::SparseBsr:
    case Layout::SparseBsc:
      switch (device_.type()) {
        case DeviceType::CPU:
          return DispatchKey::SparseCsrCPU;
        case DeviceType::CUDA:
          return DispatchKey::SparseCsrCUDA;
        default:
          AT_ERROR(
              "Unsupported device type for ",
              layout_,
              " layout: ",
              device_.type());
      }
    default:
      TORCH_CHECK(false, "Unsupported layout: ", layout_);
  }
}

inline Layout dispatchKeyToLayout(DispatchKey dispatch_key) {
  switch (dispatch_key) {
#define DO_CASE(bc, _) case DispatchKey::Sparse##bc:
    C10_FORALL_BACKEND_COMPONENTS(DO_CASE, unused)
#undef DO_CASE
    return Layout::Sparse;
    case DispatchKey::SparseCsrCPU:
    case DispatchKey::SparseCsrCUDA:
      TORCH_CHECK(
          false,
          "Cannot map DispatchKey ",
          dispatch_key,
          " to a unique layout.");
    case DispatchKey::MkldnnCPU:
      return Layout::Mkldnn;
    default:
      return Layout::Strided;
  }
}

inline DeviceType dispatchKeyToDeviceType(DispatchKey dispatch_key) {
  switch (dispatch_key) {
    // stuff that's real
#define DO_CASE(suffix, prefix)     \
  case DispatchKey::prefix##suffix: \
    return DeviceType::suffix;
#define DO_CASES(_, prefix) C10_FORALL_BACKEND_DEVICE_TYPES(DO_CASE, prefix)
    C10_FORALL_FUNCTIONALITY_KEYS(DO_CASES)
#undef DO_CASES
#undef DO_CASE

    case DispatchKey::MkldnnCPU:
      return DeviceType::CPU;
    case DispatchKey::Vulkan:
      return DeviceType::Vulkan;

    case DispatchKey::ORT:
      return DeviceType::ORT;
    default:
      TORCH_CHECK(
          false,
          "DispatchKey ",
          dispatch_key,
          " doesn't correspond to a device");
  }
}

inline TensorOptions dispatchKeyToTensorOptions(DispatchKey dispatch_key) {
  return TensorOptions()
      .layout(dispatchKeyToLayout(dispatch_key))
      .device(dispatchKeyToDeviceType(dispatch_key));
}

namespace detail {
inline bool backend_supports_empty_operator(const TensorOptions& options) {
  // Quantized backends don't support at::empty().
  // They have separate operators like at::empty_quantized() that take in
  // extra information about how to quantize the tensor.
  return !isQIntType(typeMetaToScalarType(options.dtype()));
}

} // namespace detail

} // namespace c10
