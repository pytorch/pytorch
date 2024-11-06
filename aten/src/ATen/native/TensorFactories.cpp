#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/TensorFactories.h>

#include <ATen/core/Tensor.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Dispatch.h>
#include <ATen/EmptyTensor.h>
#include <ATen/ExpandUtils.h>
#include <ATen/Parallel.h>
#include <ATen/MapAllocator.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/TracerMode.h>
#include <ATen/TensorOperators.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/UnaryOps.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <c10/util/MathConstants.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_cast_Byte_native.h>
#include <ATen/ops/_cast_Char_native.h>
#include <ATen/ops/_cast_Double_native.h>
#include <ATen/ops/_cast_Float_native.h>
#include <ATen/ops/_cast_Half_native.h>
#include <ATen/ops/_cast_Int_native.h>
#include <ATen/ops/_cast_Long_native.h>
#include <ATen/ops/_cast_Short_native.h>
#include <ATen/ops/_dim_arange_native.h>
#include <ATen/ops/_efficientzerotensor_native.h>
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/_empty_per_channel_affine_quantized.h>
#include <ATen/ops/_sparse_compressed_tensor_with_dims_native.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/arange_native.h>
#include <ATen/ops/bartlett_window_native.h>
#include <ATen/ops/blackman_window_native.h>
#include <ATen/ops/clone_native.h>
#include <ATen/ops/complex.h>
#include <ATen/ops/complex_native.h>
#include <ATen/ops/cumprod.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty_like_native.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/empty_permuted_native.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/empty_strided_native.h>
#include <ATen/ops/eye.h>
#include <ATen/ops/eye_native.h>
#include <ATen/ops/fill_native.h>
#include <ATen/ops/flip.h>
#include <ATen/ops/from_file_native.h>
#include <ATen/ops/full_like_native.h>
#include <ATen/ops/full_native.h>
#include <ATen/ops/hamming_window_native.h>
#include <ATen/ops/hann_window_native.h>
#include <ATen/ops/kaiser_window_native.h>
#include <ATen/ops/linspace.h>
#include <ATen/ops/linspace_native.h>
#include <ATen/ops/logspace.h>
#include <ATen/ops/logspace_native.h>
#include <ATen/ops/new_empty_native.h>
#include <ATen/ops/new_empty_strided_native.h>
#include <ATen/ops/new_full_native.h>
#include <ATen/ops/new_ones_native.h>
#include <ATen/ops/new_zeros_native.h>
#include <ATen/ops/normal_native.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/ones_like_native.h>
#include <ATen/ops/ones_native.h>
#include <ATen/ops/polar.h>
#include <ATen/ops/polar_native.h>
#include <ATen/ops/promote_types.h>
#include <ATen/ops/rand_like_native.h>
#include <ATen/ops/rand_native.h>
#include <ATen/ops/randint_like_native.h>
#include <ATen/ops/randint_native.h>
#include <ATen/ops/randn_like_native.h>
#include <ATen/ops/randn_native.h>
#include <ATen/ops/randperm.h>
#include <ATen/ops/randperm_native.h>
#include <ATen/ops/range.h>
#include <ATen/ops/range_native.h>
#include <ATen/ops/scalar_tensor_native.h>
#include <ATen/ops/tril_indices_native.h>
#include <ATen/ops/triu_indices_native.h>
#include <ATen/ops/vander_native.h>
#include <ATen/ops/zeros_like_native.h>
#include <ATen/ops/zeros_like_ops.h>
#include <ATen/ops/zeros_native.h>
#endif

#include <c10/core/SymIntArrayRef.h>
#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>

namespace at::native {
namespace {
void window_function_checks(
    const char* function_name,
    const TensorOptions& options,
    int64_t window_length) {
  TORCH_CHECK(
      options.layout() != kSparse,
      function_name,
      " is not implemented for sparse types, got: ",
      options);
  TORCH_CHECK(
      at::isFloatingType(typeMetaToScalarType(options.dtype())) || at::isComplexType(typeMetaToScalarType(options.dtype())),
      function_name,
      " expects floating point dtypes, got: ",
      options);
  TORCH_CHECK(
      window_length >= 0,
      function_name,
      " requires non-negative window_length, got window_length=",
      window_length);
}

} // namespace

DEFINE_DISPATCH(complex_stub);
DEFINE_DISPATCH(polar_stub);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ arange ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor arange(const Scalar& end,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  return native::arange(/*start=*/0, end, dtype, layout, device, pin_memory);
}

Tensor arange(const Scalar& start, const Scalar& end,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  return native::arange(
      start, end, /*step=*/1, dtype, layout, device, pin_memory);
}

Tensor arange(
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  bool set_to_integral_dtype = !options.has_dtype() &&
       // bool inputs are considered integral
       start.isIntegral(true) &&
       end.isIntegral(true) &&
       step.isIntegral(true);

  Tensor result = set_to_integral_dtype
      ? at::empty({0}, options.dtype(at::ScalarType::Long))
      : at::empty({0}, options);
  return at::arange_out(result, start, end, step);
}

Tensor& arange_out(const Scalar& end, Tensor& result) {
  return at::arange_out(result, /*start=*/0, end, /*step=*/1);
}

Tensor _dim_arange(const Tensor& like, int64_t dim) {
  return at::arange(like.size(dim), like.options().dtype(at::kLong));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ complex / polar ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

static void complex_check_floating(const Tensor& a, const Tensor& b) {
  TORCH_CHECK((a.scalar_type() == kFloat || a.scalar_type() == kDouble || a.scalar_type() == kHalf) &&
              (b.scalar_type() == kFloat || b.scalar_type() == kDouble || b.scalar_type() == kHalf),
              "Expected both inputs to be Half, Float or Double tensors but got ",
              a.scalar_type(), " and ", b.scalar_type());
}

static void complex_check_dtype(
    const Tensor& result,
    const Tensor& a,
    const Tensor& b) {
  complex_check_floating(a, b);
  TORCH_CHECK(a.scalar_type() == b.scalar_type(),
              "Expected object of scalar type ", a.scalar_type(),
              " but got scalar type ", b.scalar_type(), " for second argument");
  TORCH_CHECK(result.scalar_type() == toComplexType(a.scalar_type()),
              "Expected object of scalar type ", toComplexType(a.scalar_type()),
              " but got scalar type ", result.scalar_type(),
              " for argument 'out'");
}

Tensor& complex_out(const Tensor& real, const Tensor& imag, Tensor& result) {
  complex_check_dtype(result, real, imag);
  auto iter = TensorIteratorConfig()
      .add_output(result)
      .add_const_input(real)
      .add_const_input(imag)
      .check_all_same_dtype(false)
      .build();
  complex_stub(iter.device_type(), iter);
  return result;
}

Tensor complex(const Tensor& real, const Tensor& imag) {
  complex_check_floating(real, imag);
  c10::TensorOptions options = real.options();
  options = options.dtype(toComplexType(real.scalar_type()));
  Tensor result = at::empty(0, options);
  return at::complex_out(result, real, imag);
}

Tensor& polar_out(const Tensor& abs, const Tensor& angle, Tensor& result) {
  complex_check_dtype(result, abs, angle);
  auto iter = TensorIteratorConfig()
      .add_output(result)
      .add_const_input(abs)
      .add_const_input(angle)
      .check_all_same_dtype(false)
      .build();
  polar_stub(iter.device_type(), iter);
  return result;
}

Tensor polar(const Tensor& abs, const Tensor& angle) {
  complex_check_floating(abs, angle);
  c10::TensorOptions options = abs.options();
  options = options.dtype(toComplexType(abs.scalar_type()));
  Tensor result = at::empty(0, options);
  return at::polar_out(result, abs, angle);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ empty ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tensor empty_cpu(IntArrayRef size, std::optional<ScalarType> dtype_opt, std::optional<Layout> layout_opt,
                 std::optional<Device> device_opt, std::optional<bool> pin_memory_opt, std::optional<c10::MemoryFormat> memory_format_opt) {
  Tensor result = at::detail::empty_cpu(size, dtype_opt, layout_opt, device_opt, pin_memory_opt, memory_format_opt);
  // See Note [Enabling Deterministic Operations]
  if (C10_UNLIKELY(at::globalContext().deterministicAlgorithms() && at::globalContext().deterministicFillUninitializedMemory())) {
    fill_empty_deterministic_(result);
  }
  return result;
}

Tensor empty_names(
    IntArrayRef size,
    std::optional<DimnameList> names,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    std::optional<MemoryFormat> optional_memory_format) {
  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  if (!names.has_value()) {
    return at::empty(size, options, optional_memory_format);
  }
  TORCH_CHECK(options.layout() == Layout::Strided,
      "NYI: named tensors only support strided layout");
  TORCH_CHECK(options.device().is_cpu() || options.device().is_cuda() || options.device().is_xpu() || options.device().is_privateuseone(),
      "NYI: named tensors only support CPU, CUDA, XPU or ", c10::get_privateuse1_backend(), " tensors.");
  auto result = at::empty(size, options, optional_memory_format);
  internal_set_names_inplace(result, names);
  return result;
}

Tensor empty_permuted_symint(SymIntArrayRef size, IntArrayRef physical_layout, std::optional<ScalarType> dtype_opt,
  std::optional<Layout> layout_opt, std::optional<Device> device_opt, std::optional<bool> pin_memory_opt
) {
  // size is logical; aka, the output size you'll get from the operation overall
  //
  // physical_layout follows NCHW/NHWC convention:
  // contiguous is [0,1,2,3], channels last is [0,2,3,1]
  //
  // this means if i is physical index, physical_layout[i] is logical index;
  // e.g., to find what is innermost physical dim (3), query NHWC[3] == 1
  // (aka it is channels)
  int64_t dim = static_cast<int64_t>(size.size());
  SymDimVector phys_size(dim);
  TORCH_CHECK(static_cast<int64_t>(physical_layout.size()) == dim,
    "Number of dimensions in size does not match the "
    "length of the physical_layout; i.e. len(size) = ", dim,
    " is not equal to len(physical_layout) = ", physical_layout.size());
  std::vector<bool> seen_dims(dim);
  for (const auto i : c10::irange(dim)) {
    TORCH_CHECK(physical_layout[i] >= 0 && physical_layout[i] < dim,
      "Dimension out of range (expected to be between 0 and ", dim - 1, ", but got ",
      physical_layout[i], " at index ", i, ").  NB: negative dims "
      "not currently supported; file an issue if you want it.");
    TORCH_CHECK(!seen_dims[physical_layout[i]], "Duplicate dim not allowed");
    phys_size[i] = size[physical_layout[i]];
    seen_dims[physical_layout[i]] = true;
  }
  // do a contiguous allocation
  Tensor phys_tensor = at::empty_symint(phys_size, dtype_opt, layout_opt, device_opt, pin_memory_opt, std::nullopt);
  SymIntArrayRef phys_strides = phys_tensor.sym_strides();
  // permute the strides (inverse permutation!  This is why this is
  // empty_permute*d*, not empty_permute; it's not an empty + permute)
  SymDimVector strides(dim);
  for (const auto i : c10::irange(dim)) {
    strides[physical_layout[i]] = phys_strides[i];
  }
  return phys_tensor.as_strided_symint(size, strides);
}

Tensor empty_strided_cpu(IntArrayRef size, IntArrayRef stride, std::optional<ScalarType> dtype_opt,
                         std::optional<Layout> layout_opt, std::optional<Device> device_opt, std::optional<bool> pin_memory_opt) {
  Tensor result = at::detail::empty_strided_cpu(size, stride, dtype_opt, layout_opt, device_opt, pin_memory_opt);
  // See Note [Enabling Deterministic Operations]
  if (C10_UNLIKELY(at::globalContext().deterministicAlgorithms() && at::globalContext().deterministicFillUninitializedMemory())) {
    fill_empty_deterministic_(result);
  }
  return result;
}

Tensor& empty_out(IntArrayRef size,
    std::optional<c10::MemoryFormat> optional_memory_format,
    Tensor& result) {
  // Preferably, this argument would not be accepted by _out, but the code
  // generator requires the out and non-out overloads to match exactly
  TORCH_CHECK(
      !optional_memory_format.has_value(),
      "'memory_format' argument is incompatible with 'out' tensor argument");
  check_size_nonnegative(size);
  if (result.is_sparse()) {
    result.sparse_resize_and_clear_(size, size.size(), 0);
  } else {
    result.resize_(size);
  }
  // See Note [Enabling Deterministic Operations]
  if (C10_UNLIKELY(at::globalContext().deterministicAlgorithms() && at::globalContext().deterministicFillUninitializedMemory())) {
    fill_empty_deterministic_(result);
  }
  return result;
}

// Temporary type cast operators. These are needed to trace type-casts now since
// Type's are not supported in the IR. Instead, we call down to these
// specialized operators for each datatype.
// TODO: remove when we have Type support in the IR

#define DEFINE_CAST_OP(_1, n)                                    \
  Tensor _cast_##n(const Tensor& self, bool non_blocking) {      \
    if (self.scalar_type() == ScalarType::n)                     \
      return self;                                               \
    return self.to(ScalarType::n, non_blocking);                 \
  }

// Some scalar types in CAST_OP have no declarations, they may be unused in Pytorch.
// But we keep them and ignore the warning here until verified in the future.
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wmissing-prototypes")
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, DEFINE_CAST_OP)
C10_DIAGNOSTIC_POP()

#undef DEFINE_CAST_OP

Tensor empty_like(
    const Tensor& self,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    std::optional<c10::MemoryFormat> optional_memory_format) {
  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options_ = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  TensorOptions options =
      self.options()
          .merge_in(options_)
          .merge_memory_format(optional_memory_format);

  TORCH_CHECK(
      !(options.layout() != kStrided &&
          optional_memory_format.has_value()),
      "memory format option is only supported by strided tensors");

  auto memory_format = options.memory_format_opt().value_or(MemoryFormat::Preserve);

  Tensor result;

  if (memory_format == MemoryFormat::Preserve) {
    if (self.is_non_overlapping_and_dense()) {
      result = at::empty_strided_symint(self.sym_sizes(), self.sym_strides(), options.memory_format(std::nullopt));
    } else if (self.unsafeGetTensorImpl()->support_as_strided() && self.layout() == kStrided) {
      // If input tensor is not dense and non-overlapping but strided, we will infer an output strides
      // which keeps the layout permutation of the input tensor.
      std::vector<int64_t> strides = infer_dense_strides(self.sizes(), self.strides());
      // See Note [Explicit nullopt MemoryFormat argument]
      result = at::empty_strided(self.sizes(), strides, options.memory_format(std::nullopt));
    } else {
      // See Note [Explicit nullopt MemoryFormat argument]
      result = at::empty_symint(self.sym_sizes(), options.memory_format(self.suggest_memory_format()), std::nullopt);
    }
  } else {
    // See Note [Explicit nullopt MemoryFormat argument]
    result = at::empty_symint(self.sym_sizes(), options.memory_format(memory_format), std::nullopt);
  }

  if (self.opt_names()) {
    namedinference::propagate_names(result, self.names());
  }

  // never propagate Conjugate, Negative, and ZeroTensor dispatch key
  result._set_conj(false);
  result._set_neg(false);
  result._set_zero(false);
  return result;
}

Tensor empty_like_quantized(
    const Tensor& self,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    std::optional<c10::MemoryFormat> optional_memory_format) {
  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options_ = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  TORCH_CHECK(
    !(options_.has_memory_format() && optional_memory_format.has_value()),
    "Cannot set memory_format both in TensorOptions and explicit argument; please delete "
    "the redundant setter.");

  TensorOptions options =
      self.options()
          .merge_in(options_)
          .merge_memory_format(optional_memory_format);

  TORCH_CHECK(
      !(options.layout() != kStrided &&
          optional_memory_format.has_value()),
      "memory format option is only supported by strided tensors");

  auto memory_format = options.memory_format_opt().value_or(MemoryFormat::Preserve);


  // TODO: To support all features of MemoryFormat::Preserve we need to add
  // _empty_affine_quantized_strided function and use it similarly to
  // Tensor clone(const Tensor& src, std::optional<c10::MemoryFormat> optional_memory_format)
  // if (self.is_non_overlapping_and_dense()) -> _empty_affine_quantized_strided
  if (memory_format == MemoryFormat::Preserve) {
    memory_format = self.suggest_memory_format();
  }


  // Note [Explicit nullopt MemoryFormat argument]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Some functions which we call default the OPTIONAL MemoryFormat
  // argument to something that's not nullopt.  If we pass the
  // MemoryFormat via TensorOptions, we must explicitly disable this
  // defaulting process, by explicitly passing nullopt for the MemoryFormat
  // argument.  When codegen is adjusted so we can delete this argument from
  // the method signature, the argument will just disappear entirely.
  //
  // BTW, there are a few places where the optional MemoryFormat is None,
  // but I still pass in nullopt for robustness.

  // We could check if dtype is still quantized?  But then should we shift/scale
  // the q_zero_point / q_scale or not?
  TORCH_CHECK(!options.has_dtype() || options.dtype() == self.dtype(),
              "It is currently not supported to specify a dtype that doesn't match "
              "the input tensor's dtype via empty_like.  Specified: ", options.dtype(),
              " Input tensor's dtype: ", self.dtype());
  auto qscheme = self.qscheme();
  if (qscheme == kPerTensorAffine) {
    return at::_empty_affine_quantized(self.sizes(), options.memory_format(memory_format),
                                        self.q_scale(),
                                        self.q_zero_point(),
                                        // See Note [Explicit nullopt MemoryFormat argument]
                                        std::nullopt);
  } else if (qscheme == kPerChannelAffine) {
    // Copy the tensors with channels to avoid accidental overrides
    return at::_empty_per_channel_affine_quantized(
        self.sizes(),
        self.q_per_channel_scales().clone(at::MemoryFormat::Preserve),
        self.q_per_channel_zero_points().clone(at::MemoryFormat::Preserve),
        self.q_per_channel_axis(),
        options.memory_format(memory_format),
        // See Note [Explicit nullopt MemoryFormat argument]
        std::nullopt);
  } else {
    TORCH_CHECK(false, "Unsupported qscheme: ", toString(qscheme));
  }
}

Tensor new_empty_symint(
    const Tensor& self,
    SymIntArrayRef size,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt
    ) {
  auto dtype = dtype_opt.has_value() ? dtype_opt : optTypeMetaToScalarType(self.options().dtype_opt());
  auto layout = layout_opt.has_value() ? layout_opt : self.options().layout_opt();
  auto device = device_opt.has_value() ? device_opt : self.options().device_opt();
  auto pin_memory = pin_memory_opt.has_value() ? pin_memory_opt : self.options().pinned_memory_opt();
  return at::empty_symint(size, dtype, layout, device, pin_memory, std::nullopt);
}

Tensor new_empty_strided_symint(
    const Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory
    ) {
  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  return at::empty_strided_symint(size, stride, self.options().merge_in(options));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ eye ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor eye(int64_t n,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // the default value of `m` equals to `n`
  return at::eye(n, n, dtype, layout, device, pin_memory);
}

Tensor eye(int64_t n, int64_t m,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  auto tensor = at::empty({0}, options); // to be resized
  return at::eye_out(tensor, n, m);
}

Tensor& eye_out_cpu(int64_t n, Tensor& result) {
  // the default value of `m` equals to `n`
  return native::eye_out_cpu(n, n, result);
}

Tensor& eye_out_cpu(int64_t n, int64_t m, Tensor& result) {
  TORCH_CHECK(n >= 0, "n must be greater or equal to 0, got ", n);
  TORCH_CHECK(m >= 0, "m must be greater or equal to 0, got ", m);

  result.resize_({n, m});

  if (result.is_meta()) return result;

  result.zero_();

  int64_t sz = std::min<int64_t>(n, m);
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBFloat16, kHalf, kBool, result.scalar_type(), "eye", [&]() -> void {
    scalar_t* result_data = result.data_ptr<scalar_t>();
    at::parallel_for(0, sz, internal::GRAIN_SIZE, [&](int64_t p_begin, int64_t p_end) {
      for (const auto i : c10::irange(p_begin, p_end))result_data[i*(result.strides()[0] + result.strides()[1])] = 1;
    });
  });

  return result;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ full ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace {

// Performs dtype inference for full
TensorOptions infer_full_options(
  const Scalar& fill_value,
  const TensorOptions& options) {

  if (!options.has_dtype()) {
    if (fill_value.isBoolean()) {
      return options.dtype(at::kBool);
    } else if (fill_value.isIntegral(false)) {
      return options.dtype(at::kLong);
    } else if (fill_value.isComplex()) {
      auto scalar_type = (get_default_dtype() == ScalarType::Double) ?
                            ScalarType::ComplexDouble :
                            ScalarType::ComplexFloat;
      return options.dtype(scalar_type);
    } else {
      return options.dtype(get_default_dtype());
    }
  }

  return options;
}

} // anonymous namespace

Tensor full(IntArrayRef size, const Scalar& fill_value,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  TORCH_CHECK(options.layout() != kSparse,
    "full(...) is not implemented for sparse layout");

  auto result = at::empty(size, infer_full_options(fill_value, options));
  return result.fill_(fill_value);
}

Tensor& full_out(IntArrayRef size, const Scalar& fill_value, Tensor& result) {
  TORCH_CHECK(!result.is_sparse(),
    "full(...) is not implemented for sparse layout");

  result.resize_(size);
  return result.fill_(fill_value);
}

Tensor full_like(
    const Tensor& self,
    const Scalar& fill_value,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    std::optional<c10::MemoryFormat> optional_memory_format) {
  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  auto result = at::empty_like(self, options, optional_memory_format);
  return result.fill_(fill_value);
}

Tensor new_full(
    const Tensor& self,
    IntArrayRef size,
    const Scalar& fill_value,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory
    ) {

  Tensor r = self.new_empty(size, TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory));
  r.fill_(fill_value);
  return r;
}

namespace {
TensorOptions linspace_logspace_infer_options(
    const Scalar& start,
    const Scalar& end,
    const TensorOptions& options,
    const char* fn_name) {
  if (start.isComplex() || end.isComplex()) {
    const auto default_complex_dtype = c10::get_default_complex_dtype();
    if (options.has_dtype()) {
      auto dtype = c10::typeMetaToScalarType(options.dtype());
      TORCH_CHECK(at::isComplexType(dtype),
          fn_name, ": inferred dtype ", default_complex_dtype, " can't be safely cast to passed dtype ", dtype);
    } else {
      return options.dtype(default_complex_dtype);
    }
  }

  return options.has_dtype() ? options : options.dtype(c10::get_default_dtype());
}
} // anonymous namespace

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ linspace ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor linspace(
    const Scalar& start,
    const Scalar& end,
    int64_t steps,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  TORCH_CHECK(steps >= 0, "number of steps must be non-negative");
  auto result_options = linspace_logspace_infer_options(start, end, options, "torch.linspace()");
  Tensor result = at::empty({steps}, result_options);
  return at::linspace_out(result, start, end, steps);
}

Tensor linspace(
    const Tensor& start,
    const Tensor& end,
    int64_t steps,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  TORCH_CHECK(start.dim() == 0 && end.dim() == 0, "linspace only supports 0-dimensional start and end tensors, "
    "but got start with ", start.dim(), " dimension(s) and end with ", end.dim()," dimension(s).");
  return at::linspace(start.item(), end.item(), steps, dtype, layout, device, pin_memory);
}

Tensor linspace(
    const Tensor& start,
    const Scalar& end,
    int64_t steps,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  TORCH_CHECK(start.dim() == 0, "linspace only supports 0-dimensional start and end tensors, "
    "but got start with ", start.dim(), " dimension(s).");
  return at::linspace(start.item(), end, steps, dtype, layout, device, pin_memory);
}

Tensor linspace(
    const Scalar& start,
    const Tensor& end,
    int64_t steps,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  TORCH_CHECK(end.dim() == 0, "linspace only supports 0-dimensional start and end tensors, "
    "but got end with ", end.dim()," dimension(s).");
  return at::linspace(start, end.item(), steps, dtype, layout, device, pin_memory);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ logspace ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor logspace(
    const Scalar& start,
    const Scalar& end,
    int64_t steps,
    double base,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  TORCH_CHECK(steps >= 0, "number of steps must be non-negative");
  auto result_options = linspace_logspace_infer_options(start, end, options, "torch.logspace()");
  Tensor result = at::empty({steps}, result_options);
  return at::logspace_out(result, start, end, steps, base);
}

Tensor logspace(
    const Tensor& start,
    const Tensor& end,
    int64_t steps,
    double base,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  TORCH_CHECK(start.dim() == 0 && end.dim() == 0, "logspace only supports 0-dimensional start and end tensors, "
    "but got start with ", start.dim(), " dimension(s) and end with ", end.dim()," dimension(s).");
  return at::logspace(start.item(), end.item(), steps, base, dtype, layout, device, pin_memory);
}

Tensor logspace(
    const Tensor& start,
    const Scalar& end,
    int64_t steps,
    double base,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  TORCH_CHECK(start.dim() == 0, "logspace only supports 0-dimensional start and end tensors, "
    "but got start with ", start.dim(), " dimension(s).");
  return at::logspace(start.item(), end, steps, base, dtype, layout, device, pin_memory);
}

Tensor logspace(
    const Scalar& start,
    const Tensor& end,
    int64_t steps,
    double base,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  TORCH_CHECK(end.dim() == 0, "logspace only supports 0-dimensional start and end tensors, "
    "but got end with ", end.dim()," dimension(s).");
  return at::logspace(start, end.item(), steps, base, dtype, layout, device, pin_memory);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ones ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor ones(IntArrayRef size,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  return native::full(size, /*fill_value=*/1., dtype, layout, device, pin_memory);
}

Tensor& ones_out(IntArrayRef size, Tensor& result) {
  return native::full_out(size, /*fill_value=*/1., result);
}

Tensor ones_like(
    const Tensor& self,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    std::optional<c10::MemoryFormat> optional_memory_format) {
  auto result = at::empty_like(self, dtype, layout, device, pin_memory, optional_memory_format);
  return result.fill_(1.);
}

Tensor new_ones(
    const Tensor& self,
    IntArrayRef size,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // See [Note: hacky wrapper removal for TensorOptions]
  Tensor r = self.new_empty(size, TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory));
  r.fill_(1.);
  return r;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ scalar_tensor ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor scalar_tensor(const Scalar& s,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  if (options.device() == at::kCPU) {
    // This is a fast track to skip device dispatch for making scalar tensor on CPU.
    // See https://github.com/pytorch/pytorch/pull/29915 for more detailed perf
    // difference.
    // In the future when we remove the overhead of device dispatch, we'll happily
    // revert this to following:
    //   auto result = at::empty({}, options);
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    at::AutoDispatchBelowAutograd mode;
    auto result = empty_cpu({}, optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
    at::native::fill_(result, s);
    return result;
  }
  return at::empty({}, options).fill_(s);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ rand ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor rand(IntArrayRef size,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  return native::rand(size, static_cast<std::optional<Generator>>(std::nullopt), dtype, layout, device, pin_memory);
}

Tensor rand(IntArrayRef size, std::optional<Generator> generator,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  auto result = at::empty(size, options);
  return result.uniform_(0, 1, std::move(generator));
}

Tensor& rand_out(IntArrayRef size, Tensor& result) {
  return native::rand_out(size, std::nullopt, result);
}

Tensor& rand_out(IntArrayRef size, std::optional<Generator> generator, Tensor& result) {
  result.resize_(size);
  return result.uniform_(0, 1, std::move(generator));
}

Tensor rand_like(
    const Tensor& self,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    std::optional<c10::MemoryFormat> optional_memory_format) {
  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  auto result = at::empty_like(self, options, optional_memory_format);
  return result.uniform_(0, 1, std::nullopt);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ randint ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor randint(int64_t high, IntArrayRef size,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  return native::randint(high, size, std::nullopt /* generator*/, dtype, layout, device, pin_memory);
}

Tensor randint(
    int64_t high,
    IntArrayRef size,
    std::optional<Generator> generator,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  return native::randint(0, high, size, std::move(generator), dtype, layout, device, pin_memory);
}

Tensor randint(
    int64_t low,
    int64_t high,
    IntArrayRef size,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  return native::randint(low, high, size, std::nullopt, dtype, layout, device, pin_memory);
}

Tensor randint(
    int64_t low,
    int64_t high,
    IntArrayRef size,
    std::optional<Generator> generator,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  auto result = at::empty(size, options);
  return result.random_(low, high, std::move(generator));
}

Tensor& randint_out(int64_t high, IntArrayRef size, Tensor& result) {
  return native::randint_out(high, size, std::nullopt, result);
}

Tensor& randint_out(int64_t high,
    IntArrayRef size,
    std::optional<Generator> generator,
    Tensor& result) {
  result.resize_(size);
  return result.random_(0, high, std::move(generator));
}

Tensor& randint_out(int64_t low, int64_t high, IntArrayRef size, Tensor& result) {
  return native::randint_out(low, high, size, std::nullopt, result);
}

Tensor& randint_out(int64_t low,
    int64_t high,
    IntArrayRef size,
    std::optional<Generator> generator,
    Tensor& result) {
  result.resize_(size);
  return result.random_(low, high, std::move(generator));
}

Tensor randint_like(
    const Tensor& self,
    int64_t high,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    std::optional<c10::MemoryFormat> optional_memory_format) {
  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  auto result = at::empty_like(self, options, optional_memory_format);
  return result.random_(0, high, std::nullopt);
}

Tensor randint_like(
    const Tensor& self,
    int64_t low,
    int64_t high,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    std::optional<c10::MemoryFormat> optional_memory_format) {
  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  auto result = at::empty_like(self, options, optional_memory_format);
  return result.random_(low, high, std::nullopt);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ randn ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor randn(IntArrayRef size,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  return native::randn(size, static_cast<std::optional<Generator>>(std::nullopt), dtype, layout, device, pin_memory);
}

Tensor randn(IntArrayRef size, std::optional<Generator> generator,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  auto result = at::empty(size, options);
  return result.normal_(0, 1, std::move(generator));
}

Tensor& randn_out(IntArrayRef size, Tensor& result) {
  return native::randn_out(size, std::nullopt, result);
}

Tensor& randn_out(IntArrayRef size, std::optional<Generator> generator, Tensor& result) {
  result.resize_(size);
  return result.normal_(0, 1, std::move(generator));
}

Tensor normal(double mean, double std, IntArrayRef size,
              std::optional<Generator> generator,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  auto result = at::empty(size, options);
  return result.normal_(mean, std, std::move(generator));
}

Tensor& normal_out(double mean, double std,
                   IntArrayRef size, std::optional<Generator> generator, Tensor& result) {
  result.resize_(size);
  return result.normal_(mean, std, std::move(generator));
}

Tensor randn_like(
    const Tensor& self,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    std::optional<c10::MemoryFormat> optional_memory_format) {
  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  auto result = at::empty_like(self, options, optional_memory_format);
  return result.normal_(0, 1, std::nullopt);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ randperm ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace {
template <typename scalar_t>
void randperm_cpu(Tensor& result, int64_t n, CPUGeneratorImpl* generator) {
  scalar_t *r__data = result.data_ptr<scalar_t>();

  result.resize_({n});
  int64_t r__stride_0 = result.stride(0);

  at::parallel_for(0, n, internal::GRAIN_SIZE,
                  [&r__data, &r__stride_0](int64_t p_begin, int64_t p_end) {
    for (const auto i : c10::irange(p_begin, p_end)) {
      r__data[i*r__stride_0] = static_cast<scalar_t>(i);
    }
  });

  for(int64_t i = 0; i < n - 1; i++)
  {
    // NOLINTNEXTLINE(clang-analyzer-security.insecureAPI.rand)
    int64_t z = generator->random() % (n-i);
    scalar_t sav = r__data[i*r__stride_0];
    r__data[i*r__stride_0] = r__data[(z+i)*r__stride_0];
    r__data[(z+i)*r__stride_0] = sav;
  }
}
} // namespace

Tensor randperm(int64_t n,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  return native::randperm(n, std::nullopt, dtype, layout, device, pin_memory);
}

Tensor randperm(int64_t n, std::optional<Generator> generator,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  if (!dtype.has_value()) {
    dtype = ScalarType::Long;
  }

  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  auto tensor = at::empty(n, options);
  return at::randperm_out(tensor, n, std::move(generator));
}

Tensor& randperm_out(int64_t n, Tensor& result) {
  return at::randperm_out(result, n, std::nullopt);
}

Tensor& randperm_out_cpu(int64_t n, std::optional<Generator> generator, Tensor& result) {
  TORCH_CHECK(n >= 0, "n must be non-negative, got", n);
  TORCH_CHECK(!generator.has_value() || (generator.has_value() && result.device() == generator->device()), "Expected a '", result.device(), "' generator device but found '", generator->device(), "'");
  check_supported_max_int_with_precision(n, result);
  result.resize_({n});
  auto gen = get_generator_or_default<CPUGeneratorImpl>(generator, detail::getDefaultCPUGenerator());
  // See Note [Acquire lock when using random generators]
  std::lock_guard<std::mutex> lock(gen->mutex_);
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, result.scalar_type(), "randperm", [&]() -> void {
    randperm_cpu<scalar_t>(result, n, gen);
  });

  return result;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ range ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor range(
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  Tensor result = at::empty({0}, options);
  return at::range_out(result, start, end, step);
}

Tensor range(
    const Scalar& start,
    const Scalar& end,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  return at::native::range(start, end, 1, dtype, layout, device, pin_memory);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ triangle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor tril_indices_cpu(
    int64_t row, int64_t col, int64_t offset, std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt, std::optional<Device> device_opt, std::optional<bool> pin_memory_opt) {
  if (!dtype_opt.has_value()) {
    dtype_opt = ScalarType::Long;
  }

  check_args(row, col, layout_opt);

  auto tril_size = get_tril_size(row, col, offset);

  // create an empty Tensor with correct size
  auto result = at::native::empty_cpu({2, tril_size}, dtype_opt, layout_opt, device_opt, pin_memory_opt);

  // The following three approaches result in very little performance
  // differences. Hence, the 2nd option is taken for simpler code, and to return
  // contiguous tensors. Refer to #14904 for more details.
  //
  // 1. sequential RAM access: fill row coordinates first, then columns. This
  //    results in two for-loop and more arithmetic operations.
  //
  // 2. interleaved RAM access: fill in index coordinates one by one, which
  //    jumps between the two output Tensor rows in every iteration.
  //
  // 3. sequential RAM + transpose: create an n X 2 Tensor, fill the Tensor
  //    sequentially, and then transpose it.
  AT_DISPATCH_INDEX_TYPES(result.scalar_type(), "tril_indices", [&]() -> void {
    // fill the Tensor with correct values
    index_t* result_data = result.data_ptr<index_t>();
    int64_t i = 0;

    index_t r = std::max<int64_t>(0, -offset), c = 0;
    while (i < tril_size) {
      result_data[i] = r;
      result_data[tril_size + i++] = c;

      // move to the next column and check if (r, c) is still in bound
      c += 1;
      if (c > r + offset || c >= col) {
        r += 1;
        c = 0;
        // NOTE: not necessary to check if r is less than row here, because i
        // and tril_size provide the guarantee
      }
    }
  });

  return result;
}

Tensor triu_indices_cpu(
    int64_t row, int64_t col, int64_t offset, std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt, std::optional<Device> device_opt, std::optional<bool> pin_memory_opt) {
  if (!dtype_opt.has_value()) {
    dtype_opt = ScalarType::Long;
  }

  check_args(row, col, layout_opt);

  auto triu_size = row * col - get_tril_size(row, col, offset - 1);

  // create an empty Tensor with correct size
  auto result = at::native::empty_cpu({2, triu_size}, dtype_opt, layout_opt, device_opt, pin_memory_opt);

  AT_DISPATCH_INDEX_TYPES(result.scalar_type(), "triu_indices", [&]() -> void {
    // fill the Tensor with correct values
    index_t* result_data = result.data_ptr<index_t>();
    int64_t i = 0;
    // not typing std::max with scalar_t as it could be an unsigned type
    // NOTE: no need to check if the returned value of std::max overflows
    // index_t, as i and triu_size act as a guard.
    index_t c = std::max<int64_t>(0, offset), r = 0;
    while (i < triu_size) {
      result_data[i] = r;
      result_data[triu_size + i++] = c;

      // move to the next column and check if (r, c) is still in bound
      c += 1;
      if (c >= col) {
        r += 1;
        // not typing std::max with scalar_t as it could be an unsigned type
        // NOTE: not necessary to check if c is less than col or overflows here,
        // because i and triu_size act as a guard.
        c = std::max<int64_t>(0, r + offset);
      }
    }
  });

  return result;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ zeros ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

static Tensor zeros_sparse_compressed_symint(c10::SymIntArrayRef size,
    std::optional<ScalarType> dtype,
    Layout layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  check_size_nonnegative(size);
  TORCH_CHECK(size.size() >= 2, "torch.zeros: Only batched sparse compressed (non-block) tensors are supported, but got size ", size);
  auto size_ = C10_AS_INTARRAYREF_SLOW(size);
  // torch.zeros cannot be used to create blocked tensors because its
  // API lacks a method to specify the block size.
  AT_DISPATCH_SPARSE_COMPRESSED_NONBLOCK_LAYOUTS(layout, "zeros_sparse_compressed", [&]{});

  int64_t nnz = 0;
  auto compressed_indices_size = DimVector(size_.slice(0, size.size() - 2));
  auto plain_indices_and_values_size = DimVector(size_.slice(0, size.size() - 2));
  compressed_indices_size.push_back(size_[at::sparse_csr::compressedDimension(layout, size_)] + 1);
  plain_indices_and_values_size.push_back(nnz);

  TensorOptions options = TensorOptions().dtype(ScalarType::Long).layout(Layout::Strided).device(device).pinned_memory(pin_memory);
  auto compressed_indices = at::empty(compressed_indices_size, options);
  compressed_indices.zero_();
  auto plain_indices = at::empty(plain_indices_and_values_size, options);
  auto values = at::empty(plain_indices_and_values_size, options.dtype(dtype));

  return at::_sparse_compressed_tensor_unsafe(compressed_indices,
                                              plain_indices,
                                              values,
                                              size_,
                                              dtype,
                                              layout,
                                              device,
                                              pin_memory);
}

Tensor zeros_symint(SymIntArrayRef size,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  Layout layout_ = layout.value_or(Layout::Strided);
  if (at::sparse_csr::is_sparse_compressed(layout_)) {
    return zeros_sparse_compressed_symint(size, dtype, layout_, device, pin_memory);
  }
  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);
  auto result = at::empty_symint(size, options);
  return result.zero_();
}

Tensor _efficientzerotensor(IntArrayRef size,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
    auto device_ = device_or_default(device);
    auto allocator = at::native::ZeroTensorAllocator(device_);
    auto dtype_ = dtype_or_default(dtype);
    auto zero_ks = at::DispatchKeySet(c10::DispatchKey::CPU) | at::DispatchKeySet(c10::DispatchKey::ZeroTensor);
    auto out = at::detail::empty_generic(size, &allocator, zero_ks, dtype_, std::nullopt);
    return out;
}

Tensor _efficientzerotensor_meta_symint(SymIntArrayRef size,
                                        std::optional<ScalarType> dtype,
                                        std::optional<Layout> layout,
                                        std::optional<Device> device,
                                        std::optional<bool> pin_memory) {
  auto device_ = device_or_default(device);
  auto allocator = at::native::ZeroTensorAllocator(device_);
  auto dtype_ = dtype_or_default(dtype);
  auto zero_ks = at::DispatchKeySet(c10::DispatchKey::Meta) | at::DispatchKeySet(c10::DispatchKey::ZeroTensor);
  auto out = at::detail::empty_generic_symint(size, &allocator, zero_ks, dtype_, std::nullopt);
  return out;
}

Tensor& zeros_sparse_out(IntArrayRef size, Tensor& result) {
  result.sparse_resize_and_clear_(size, size.size(), 0.);
  return result;
}

Tensor& zeros_out(IntArrayRef size, Tensor& result) {
  if (result.is_sparse()) {
    // TODO: I think this branch should be dead, but we don't have an easy
    // way to cover all sparse kernels with zeros_sparse_out, so retain this
    // for now
    result.sparse_resize_and_clear_(size, size.size(), 0.);
    return result;
  } else {
    result.resize_(size);
  }
  return result.zero_();
}

Tensor zeros_like(
    const Tensor& self,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    std::optional<c10::MemoryFormat> optional_memory_format) {
  // See [Note: hacky wrapper removal for TensorOptions]
  auto other_options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);
  // Prefer values passed in explicitly, but default to value from self.
  auto options = self.options().merge_in(other_options);

  if (options.layout() == kSparse) {
    TORCH_CHECK(
        !(optional_memory_format.has_value()),
        "memory format option is only supported by strided tensors");
    auto res = at::empty({0}, self.options().merge_in(options)); // to be resized

    if (self.is_sparse()) {
      res.sparse_resize_and_clear_(
          self.sizes(), self.sparse_dim(), self.dense_dim());
    } else if (at::sparse_csr::is_sparse_compressed(self)) {
      res.sparse_resize_and_clear_(
          self.sizes(), self.sizes().size() - self.dense_dim(), self.dense_dim());
    } else {
      res.sparse_resize_and_clear_(self.sizes(), self.sizes().size(), 0);
    }
    res._coalesced_(true);

    return res;
  } else if (at::sparse_csr::is_sparse_compressed(options.layout())) {
    int64_t nnz = 0;
    int64_t dense_dim = (self.layout() == kStrided ? self.dim() - 2: self.dense_dim());
    DimVector blocksize{};
    if (self.layout() == kSparseBsr || self.layout() == kSparseBsc) {
      blocksize.append(at::sparse_csr::getBlockSize(self));
    }
    ScalarType index_dtype = at::sparse_csr::getIndexDtype(self);
    auto res = at::native::sparse_compressed_tensor_with_dims(
      nnz, dense_dim, self.sizes(), blocksize, index_dtype,
      typeMetaToScalarType(options.dtype()), options.layout(), options.device(), options.pinned_memory());
    auto [compressed_indices, plain_indices] = at::sparse_csr::getCompressedPlainIndices(res);
    compressed_indices.zero_();
    return res;
  }
  auto result = at::empty_like(self, options, optional_memory_format);
  return result.zero_();
}

Tensor new_zeros(
    const Tensor& self,
    IntArrayRef size,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory
    ) {
  Tensor r = self.new_empty(size, TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory));
  r.zero_();
  return r;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~ bartlett_window ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor bartlett_window(int64_t window_length,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  return native::bartlett_window(
      window_length, /*periodic=*/true, dtype, layout, device, pin_memory);
}

Tensor bartlett_window(
    int64_t window_length,
    bool periodic,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // See [Note: hacky wrapper removal for TensorOptions]
  ScalarType dtype = c10::dtype_or_default(dtype_opt);
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  window_function_checks("bartlett_window", options, window_length);
  if (window_length == 0) {
    return at::empty({0}, options);
  }
  if (window_length == 1) {
    return native::ones({1}, dtype, layout, device, pin_memory);
  }
  if (periodic) {
    window_length += 1;
  }
  auto window = native::arange(window_length, dtype, layout, device, pin_memory)
                    .mul_(2. / static_cast<double>(window_length - 1));
  const int64_t first_half_size = ((window_length - 1) >> 1) + 1;
  window.narrow(0, first_half_size, window_length - first_half_size).mul_(-1).add_(2);
  return periodic ? window.narrow(0, 0, window_length - 1) : std::move(window);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~ blackman_window ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor blackman_window(int64_t window_length,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  return native::blackman_window(
      window_length, /*periodic=*/true, dtype, layout, device, pin_memory);
}

Tensor blackman_window(
    int64_t window_length,
    bool periodic,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // See [Note: hacky wrapper removal for TensorOptions]
  ScalarType dtype = c10::dtype_or_default(dtype_opt);
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  window_function_checks("blackman_window", options, window_length);
  if (window_length == 0) {
    return at::empty({0}, options);
  }
  if (window_length == 1) {
    return native::ones({1}, dtype, layout, device, pin_memory);
  }
  if (periodic) {
    window_length += 1;
  }
  // from https://en.wikipedia.org/wiki/Window_function#Blackman_window
  auto window =
      native::arange(window_length, dtype, layout, device, pin_memory)
          .mul_(c10::pi<double> / static_cast<double>(window_length - 1));
  window = window.mul(4).cos_().mul_(0.08) - window.mul(2).cos_().mul_(0.5) + 0.42;
  return periodic ? window.narrow(0, 0, window_length - 1) : std::move(window);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ hamming_window ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor hamming_window(int64_t window_length,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  return native::hamming_window(
      window_length, /*periodic=*/true, dtype, layout, device, pin_memory);
}

Tensor hamming_window(
    int64_t window_length,
    bool periodic,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  return native::hamming_window(
      window_length,
      periodic,
      /*alpha=*/0.54,
      dtype,
      layout,
      device,
      pin_memory);
}

Tensor hamming_window(
    int64_t window_length,
    bool periodic,
    double alpha,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  return native::hamming_window(
      window_length, periodic, alpha, /*beta=*/0.46, dtype, layout, device, pin_memory);
}

Tensor hamming_window(
    int64_t window_length,
    bool periodic,
    double alpha,
    double beta,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // See [Note: hacky wrapper removal for TensorOptions]
  ScalarType dtype = c10::dtype_or_default(dtype_opt);
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  window_function_checks("hamming_window", options, window_length);
  if (window_length == 0) {
    return at::empty({0}, options);
  }
  if (window_length == 1) {
    return native::ones({1}, dtype, layout, device, pin_memory);
  }
  if (periodic) {
    window_length += 1;
  }
  auto window = native::arange(window_length, dtype, layout, device, pin_memory);
  window.mul_(c10::pi<double> * 2. / static_cast<double>(window_length - 1)).cos_().mul_(-beta).add_(alpha);
  return periodic ? window.narrow(0, 0, window_length - 1) : std::move(window);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ hann_window ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor hann_window(int64_t window_length,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  return native::hann_window(window_length, /*periodic=*/true, dtype, layout, device, pin_memory);
}

Tensor hann_window(
    int64_t window_length,
    bool periodic,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  window_function_checks("hann_window", options, window_length);
  return native::hamming_window(
      window_length, periodic, /*alpha=*/0.5, /*beta=*/0.5, dtype, layout, device, pin_memory);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ kaiser_window ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor kaiser_window(int64_t window_length,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  return native::kaiser_window(
      window_length,
      /*periodic=*/true,
      /*beta=*/12.0,
      dtype,
      layout,
      device,
      pin_memory);
}

Tensor kaiser_window(int64_t window_length, bool periodic,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  return native::kaiser_window(window_length, periodic, /*beta=*/12.0, dtype, layout, device, pin_memory);
}

Tensor kaiser_window(
    int64_t window_length,
    bool periodic,
    double beta,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // See [Note: hacky wrapper removal for TensorOptions]
  ScalarType dtype = c10::dtype_or_default(dtype_opt);
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  window_function_checks("kaiser_window", options, window_length);
  // short-circuit for `meta`.
  if (device == kMeta) {
    return at::empty({window_length}, options);
  }

  if (window_length == 0) {
    return at::empty({0}, options);
  }
  if (window_length == 1) {
    return at::ones({1}, options);
  }
  if (periodic) {
    window_length += 1;
  }
  auto initial = at::arange(window_length, options);
  auto window = at::empty(window_length, options);
  auto iter = TensorIterator::unary_op(window, initial);
  kaiser_window_stub(iter.device_type(), iter, window_length, beta);
  return periodic ? window.narrow(0, 0, window_length - 1) : std::move(window);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~ vandermonde_matrix ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Tensor vander(const Tensor& x, std::optional<int64_t> N, bool increasing) {
  TORCH_CHECK(x.dim() == 1, "x must be a one-dimensional tensor.");

  // Acquires n, defaulting to size if not provided
  int64_t n = x.size(0);
  if (N.has_value()) {
    n = *N;
    TORCH_CHECK(n >= 0, "N must be non-negative.");
  }

  // Note: result is long if x is an integer tensor (like int8) because
  // cumprod promotes integer tensors to long
  auto result = at::empty({x.size(0), n}, x.options().dtype(at::promote_types(x.scalar_type(), c10::ScalarType::Long)));

  if (n > 0) {
    result.select(1, 0).fill_(1);
  }
  if (n > 1) {
    result.slice(1, 1).copy_(x.unsqueeze(1));
    result.slice(1, 1).copy_(at::cumprod(result.slice(1, 1), 1));
  }

  if (!increasing) {
    return at::flip(result, {1});
  }
  return result;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ tensor ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename T>
Tensor tensor_cpu(ArrayRef<T> values, const TensorOptions& options) {
  return at::detail::tensor_cpu(values, options);
}

template <typename T>
Tensor tensor_backend(ArrayRef<T> values, const TensorOptions& options) {
  return at::detail::tensor_backend(values, options);
}

template <typename T>
Tensor tensor_complex_cpu(ArrayRef<T> values, const TensorOptions& options) {
  return at::detail::tensor_complex_cpu(values, options);
}

template <typename T>
Tensor tensor_complex_backend(ArrayRef<T> values, const TensorOptions& options) {
  return at::detail::tensor_complex_backend(values, options);
}

Tensor from_file(std::string_view filename, std::optional<bool> shared, std::optional<int64_t> size,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

    TORCH_CHECK(!options.pinned_memory(), "tensors constructed from a file cannot be pinned");
    int64_t my_size = size.value_or(0);
    int flags = shared.value_or(false) ? ALLOCATOR_MAPPED_SHARED : 0;
    auto my_dtype = options.dtype();
    size_t size_bytes = my_size * my_dtype.itemsize();
    auto storage_impl = c10::make_intrusive<at::StorageImpl>(
        c10::StorageImpl::use_byte_size_t(),
        size_bytes,
        MapAllocator::makeDataPtr(
            std::string(filename), flags, size_bytes, nullptr),
        /*allocator=*/nullptr,
        /*resizable=*/false);
    auto tensor = detail::make_tensor<at::TensorImpl>(
        storage_impl, at::DispatchKey::CPU, my_dtype);
    tensor.unsafeGetTensorImpl()->set_sizes_contiguous({my_size});
    return tensor;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ clone ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor clone(const Tensor& src, std::optional<c10::MemoryFormat> optional_memory_format) {
  auto memory_format =
      optional_memory_format.value_or(MemoryFormat::Preserve);
  Tensor self;
  if (memory_format == MemoryFormat::Preserve) {
    if (src.is_non_overlapping_and_dense()) {
      // Copy all strides, this is marginally faster than calling empty_like
      self = at::empty_strided_symint(src.sym_sizes(), src.sym_strides(), src.options());
    } else {
      self = at::empty_like(src);
    }
  } else {
    self = at::empty_like(src, src.options(), memory_format);
  }

  if (src._is_zerotensor()) {
    self.zero_();
  } else {
    self.copy_(src);
  }
  return self;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~ named tensor overloads ~~~~~~~~~~~~~~~~~~~~~~~~~~~
// In the short term, these exist.
// In the long term, we should move DimnameList into TensorOptions to avoid
// having these overloads.

Tensor full(
    IntArrayRef size,
    const Scalar& fill_value,
    std::optional<DimnameList> names,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);


  TORCH_CHECK(options.layout() != kSparse,
    "full(...) is not implemented for sparse layout");

  auto result = at::empty(size, names, infer_full_options(fill_value, options));
  return result.fill_(fill_value);
}

Tensor ones(
    IntArrayRef size,
    std::optional<DimnameList> names,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // See [Note: hacky wrapper removal for TensorOptions]

  return native::full(
      size, /*fill_value=*/1., names, dtype, layout, device, pin_memory);
}

Tensor zeros(
    IntArrayRef size,
    std::optional<DimnameList> names,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  return native::full(size, /*fill_value=*/0., names, dtype, layout, device, pin_memory);
}

Tensor randn(
    IntArrayRef size,
    std::optional<DimnameList> names,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  return native::randn(size, std::nullopt, names, dtype, layout, device, pin_memory);
}

Tensor randn(
    IntArrayRef size,
    std::optional<Generator> generator,
    std::optional<DimnameList> names,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  auto result = at::empty(size, names, options);
  return result.normal_(0, 1, std::move(generator));
}

Tensor rand(
    IntArrayRef size,
    std::optional<DimnameList> names,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  return native::rand(size, std::nullopt, names, dtype, layout, device, pin_memory);
}

Tensor rand(
    IntArrayRef size,
    std::optional<Generator> generator,
    std::optional<DimnameList> names,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  auto result = at::empty(size, names, options);
  return result.uniform_(0, 1, std::move(generator));
}


DEFINE_DISPATCH(kaiser_window_stub);

} // namespace at::native
