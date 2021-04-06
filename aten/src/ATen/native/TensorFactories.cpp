#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Utils.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TracerMode.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Deprecated.h>
#include <ATen/native/Math.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorFactories.h>
#include <c10/core/TensorOptions.h>
#include <TH/THAllocator.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <c10/util/Exception.h>
#include <ATen/NamedTensorUtils.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <string>

namespace at {
namespace native {
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

Tensor arange(const Scalar& end, const TensorOptions& options) {
  return native::arange(/*start=*/0, end, options);
}

Tensor arange(const Scalar& start, const Scalar& end, const TensorOptions& options) {
  return native::arange(start, end, /*step=*/1, options);
}

Tensor arange(
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    const TensorOptions& options) {
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
  return at::arange_out(result, /*start=*/0, end);
}

Tensor& arange_out(Tensor& result, const Scalar& start, const Scalar& end) {
  return at::arange_out(result, start, end, /*step=*/1);
}

Tensor _dim_arange(const Tensor& like, int64_t dim) {
  return at::arange(like.size(dim), like.options().dtype(at::kLong));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ complex / polar ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void complex_check_floating(const Tensor& a, const Tensor& b) {
  TORCH_CHECK((a.scalar_type() == kFloat || a.scalar_type() == kDouble) &&
              (b.scalar_type() == kFloat || b.scalar_type() == kDouble),
              "Expected both inputs to be Float or Double tensors but got ",
              a.scalar_type(), " and ", b.scalar_type());
}

void complex_check_dtype(
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
      .add_input(real)
      .add_input(imag)
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
      .add_input(abs)
      .add_input(angle)
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
Tensor empty_cpu(IntArrayRef size, c10::optional<ScalarType> dtype_opt, c10::optional<Layout> layout_opt,
                 c10::optional<Device> device_opt, c10::optional<bool> pin_memory_opt, c10::optional<c10::MemoryFormat> memory_format_opt) {
  return at::detail::empty_cpu(size, dtype_opt, layout_opt, device_opt, pin_memory_opt, memory_format_opt);
}

Tensor empty(
    IntArrayRef size,
    at::optional<DimnameList> names,
    const TensorOptions& options,
    optional<MemoryFormat> optional_memory_format) {
  if (!names.has_value()) {
    return at::empty(size, options, optional_memory_format);
  }
  TORCH_CHECK(options.layout() == Layout::Strided,
      "NYI: named tensors only support strided layout");
  TORCH_CHECK(options.device().is_cpu() || options.device().is_cuda(),
      "NYI: named tensors only support CPU and CUDA tensors");
  auto result = at::empty(size, options, optional_memory_format);
  internal_set_names_inplace(result, names);
  return result;
}

Tensor empty_strided_cpu(IntArrayRef size, IntArrayRef stride, c10::optional<ScalarType> dtype_opt,
                         c10::optional<Layout> layout_opt, c10::optional<Device> device_opt, c10::optional<bool> pin_memory_opt) {
  check_size_nonnegative(size);
  auto t = at::native::empty_cpu({0}, dtype_opt, layout_opt, device_opt, pin_memory_opt);
  at::native::resize_impl_cpu_(t.unsafeGetTensorImpl(), size, stride);
  return t;
}

Tensor& empty_out(IntArrayRef size,
    c10::optional<c10::MemoryFormat> optional_memory_format,
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

AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, DEFINE_CAST_OP)

#undef DEFINE_CAST_OP

Tensor empty_like(
    const Tensor& self,
    const TensorOptions& options_,
    c10::optional<c10::MemoryFormat> optional_memory_format) {

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
  if (options.layout() == kSparse && self.is_sparse()) {
    auto result = at::empty({0}, options); // to be resized
    result.sparse_resize_and_clear_(
        self.sizes(), self.sparse_dim(), self.dense_dim());
    return result;
  }

  auto memory_format = options.memory_format_opt().value_or(MemoryFormat::Preserve);

  if (self.is_quantized()) {

    // TODO: To support all features of MemoryFormat::Preserve we need to add
    // _empty_affine_quantized_strided function and use it similarly to
    // Tensor clone(const Tensor& src, c10::optional<c10::MemoryFormat> optional_memory_format)
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
                                         c10::nullopt);
    } else if (qscheme == kPerChannelAffine) {
      // Copy the tensors with channels to avoid accidental overrides
      return at::_empty_per_channel_affine_quantized(
          self.sizes(),
          self.q_per_channel_scales().clone(at::MemoryFormat::Preserve),
          self.q_per_channel_zero_points().clone(at::MemoryFormat::Preserve),
          self.q_per_channel_axis(),
          options.memory_format(memory_format),
          // See Note [Explicit nullopt MemoryFormat argument]
          c10::nullopt);
    } else {
      TORCH_CHECK(false, "Unsupported qscheme: ", toString(qscheme));
    }
  }

  Tensor result;

  if (memory_format == MemoryFormat::Preserve) {
    if (self.is_non_overlapping_and_dense()) {
      result = at::empty_strided(self.sizes(), self.strides(), options.memory_format(c10::nullopt));
    } else if (self.unsafeGetTensorImpl()->support_as_strided() && self.layout() == kStrided) {
      // If input tensor is not dense and non-overlapping but strided, we will infer an output strides
      // which keeps the layout permutation of the input tensor.
      std::vector<int64_t> strides = infer_dense_strides(self.sizes(), self.strides());
      // See Note [Explicit nullopt MemoryFormat argument]
      result = at::empty_strided(self.sizes(), strides, options.memory_format(c10::nullopt));
    } else {
      // See Note [Explicit nullopt MemoryFormat argument]
      result = at::empty(self.sizes(), options.memory_format(self.suggest_memory_format()), c10::nullopt);
    }
  } else {
    // See Note [Explicit nullopt MemoryFormat argument]
    result = at::empty(self.sizes(), options.memory_format(memory_format), c10::nullopt);
  }

  if (self.opt_names()) {
    namedinference::propagate_names(result, self.names());
  }

  return result;
}

Tensor new_empty(
    const Tensor& self,
    IntArrayRef size,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt
    ) {
  auto dtype = dtype_opt.has_value() ? dtype_opt : optTypeMetaToScalarType(self.options().dtype_opt());
  auto layout = layout_opt.has_value() ? layout_opt : self.options().layout_opt();
  auto device = device_opt.has_value() ? device_opt : self.options().device_opt();
  auto pin_memory = pin_memory_opt.has_value() ? pin_memory_opt : self.options().pinned_memory_opt();
  return at::empty(size, dtype, layout, device, pin_memory, c10::nullopt);
}

Tensor new_empty_strided(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions& options
    ) {
  return at::empty_strided(size, stride, self.options().merge_in(options));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ eye ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor eye(int64_t n, const TensorOptions& options) {
  // the default value of `m` equals to `n`
  return native::eye(n, n, options);
}

Tensor eye(int64_t n, int64_t m, const TensorOptions& options) {
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
  result.zero_();

  int64_t sz = std::min<int64_t>(n, m);
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(at::ScalarType::Half, at::ScalarType::Bool, result.scalar_type(), "eye", [&]() -> void {
    scalar_t* result_data = result.data_ptr<scalar_t>();
    at::parallel_for(0, sz, internal::GRAIN_SIZE, [&](int64_t p_begin, int64_t p_end) {
      for(int64_t i = p_begin; i < p_end; i++)
        result_data[i*(result.strides()[0] + result.strides()[1])] = 1;
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

Tensor full(IntArrayRef size, const Scalar& fill_value, const TensorOptions& options) {
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
    const TensorOptions& options,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  auto result = at::empty_like(self, options, optional_memory_format);
  return result.fill_(fill_value);
}

Tensor new_full(
    const Tensor& self,
    IntArrayRef size,
    const Scalar& fill_value,
    const TensorOptions& options
    ) {
  return at::full(size, fill_value, self.options().merge_in(options));
}

namespace {
TensorOptions linspace_logspace_infer_options(
    const Scalar& start,
    const Scalar& end,
    const TensorOptions& options) {
  auto result_options = options;
  if (start.isComplex() || end.isComplex()) {
    // Since result_options.has_dtype() returns true (dtype is default type),
    // even if the user hasn't specified the dtype.
    // We just check to see if either `start` or `end` is complex,
    // and if the `result_dtype` is not complex (be it default float type or
    // user provided), we cast it to default complex dtype with a Warning!.
    auto result_dtype = c10::typeMetaToScalarType(options.dtype());
    if (!at::isComplexType(result_dtype)) {
      TORCH_WARN(
          "As either `start` or `stop` is complex, return type will be the complex dtype corresponding to default dtype.",
          "In future, this may throw an error when a non-complex dtype arg is passed as input along ",
          "with complex valued start or end value.");
      result_options = result_options.dtype(c10::get_default_complex_dtype());
    }
  }

  return result_options;
}
} // anonymous namespace

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ linspace ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor linspace(
    const Scalar& start,
    const Scalar& end,
    c10::optional<int64_t> steps,
    const TensorOptions& options) {
  const auto steps_ = steps.value_or(100);
  TORCH_CHECK(steps_ >= 0, "number of steps must be non-negative");
  auto result_options = linspace_logspace_infer_options(start, end, options);
  Tensor result = at::empty({steps_}, result_options);
  return at::linspace_out(result, start, end, steps);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ logspace ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor logspace(
    const Scalar& start,
    const Scalar& end,
    c10::optional<int64_t> steps,
    double base,
    const TensorOptions& options) {
  const auto steps_ = steps.value_or(100);
  TORCH_CHECK(steps_ >= 0, "number of steps must be non-negative");
  auto result_options = linspace_logspace_infer_options(start, end, options);
  Tensor result = at::empty({steps_}, result_options);
  return at::logspace_out(result, start, end, steps, base);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ones ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor ones(IntArrayRef size, const TensorOptions& options) {
  return native::full(size, /*fill_value=*/1., options);
}

Tensor& ones_out(Tensor& result, IntArrayRef size) {
  return native::full_out(size, /*fill_value=*/1., result);
}

Tensor ones_like(
    const Tensor& self,
    const TensorOptions& options,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  auto result = at::empty_like(self, options, optional_memory_format);
  return result.fill_(1.);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ scalar_tensor ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor scalar_tensor(const Scalar& s, const TensorOptions& options) {
  if (options.device() == at::kCPU) {
    // This is a fast track to skip device dispatch for making scalar tensor on CPU.
    // See https://github.com/pytorch/pytorch/pull/29915 for more detailed perf
    // difference.
    // In the future when we remove the overhead of device dispatch, we'll happily
    // revert this to following:
    //   auto result = at::empty({}, options);
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    auto result = empty_cpu({}, optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt());
    at::native::fill_(result, s);
    return result;
  }
  return at::empty({}, options).fill_(s);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ rand ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor rand(IntArrayRef size, const TensorOptions& options) {
  return native::rand(size, static_cast<c10::optional<Generator>>(c10::nullopt), options);
}

Tensor rand(IntArrayRef size, c10::optional<Generator> generator, const TensorOptions& options) {
  auto result = at::empty(size, options);
  return result.uniform_(0, 1, generator);
}

Tensor& rand_out(Tensor& result, IntArrayRef size) {
  return native::rand_out(result, size, c10::nullopt);
}

Tensor& rand_out(Tensor& result, IntArrayRef size, c10::optional<Generator> generator) {
  result.resize_(size);
  return result.uniform_(0, 1, generator);
}

Tensor rand_like(
    const Tensor& self,
    const TensorOptions& options,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  auto result = at::empty_like(self, options, optional_memory_format);
  return result.uniform_(0, 1, c10::nullopt);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ randint ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor randint(int64_t high, IntArrayRef size, const TensorOptions& options) {
  return native::randint(high, size, c10::nullopt, options);
}

Tensor randint(
    int64_t high,
    IntArrayRef size,
    c10::optional<Generator> generator,
    const TensorOptions& options) {
  return native::randint(0, high, size, generator, options);
}

Tensor randint(
    int64_t low,
    int64_t high,
    IntArrayRef size,
    const TensorOptions& options) {
  return native::randint(low, high, size, c10::nullopt, options);
}

Tensor randint(
    int64_t low,
    int64_t high,
    IntArrayRef size,
    c10::optional<Generator> generator,
    const TensorOptions& options) {
  auto result = at::empty(size, options);
  return result.random_(low, high, generator);
}

Tensor& randint_out(Tensor& result, int64_t high, IntArrayRef size) {
  return native::randint_out(result, high, size, c10::nullopt);
}

Tensor& randint_out(
    Tensor& result,
    int64_t high,
    IntArrayRef size,
    c10::optional<Generator> generator) {
  result.resize_(size);
  return result.random_(0, high, generator);
}

Tensor& randint_out(Tensor& result, int64_t low, int64_t high, IntArrayRef size) {
  return native::randint_out(result, low, high, size, c10::nullopt);
}

Tensor& randint_out(
    Tensor& result,
    int64_t low,
    int64_t high,
    IntArrayRef size,
    c10::optional<Generator> generator) {
  result.resize_(size);
  return result.random_(low, high, generator);
}

Tensor randint_like(
    const Tensor& self,
    int64_t high,
    const TensorOptions& options,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  auto result = at::empty_like(self, options, optional_memory_format);
  return result.random_(0, high, c10::nullopt);
}

Tensor randint_like(
    const Tensor& self,
    int64_t low,
    int64_t high,
    const TensorOptions& options,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  auto result = at::empty_like(self, options, optional_memory_format);
  return result.random_(low, high, c10::nullopt);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ randn ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor randn(IntArrayRef size, const TensorOptions& options) {
  return native::randn(size, static_cast<c10::optional<Generator>>(c10::nullopt), options);
}

Tensor randn(IntArrayRef size, c10::optional<Generator> generator, const TensorOptions& options) {
  auto result = at::empty(size, options);
  return result.normal_(0, 1, generator);
}

Tensor& randn_out(Tensor& result, IntArrayRef size) {
  return native::randn_out(result, size, c10::nullopt);
}

Tensor& randn_out(Tensor& result, IntArrayRef size, c10::optional<Generator> generator) {
  result.resize_(size);
  return result.normal_(0, 1, generator);
}

Tensor normal(double mean, double std, IntArrayRef size,
              c10::optional<Generator> generator, const TensorOptions& options) {
  auto result = at::empty(size, options);
  return result.normal_(mean, std, generator);
}

Tensor& normal_out(Tensor& result, double mean, double std,
                   IntArrayRef size, c10::optional<Generator> generator) {
  result.resize_(size);
  return result.normal_(mean, std, generator);
}

Tensor randn_like(
    const Tensor& self,
    const TensorOptions& options,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  auto result = at::empty_like(self, options, optional_memory_format);
  return result.normal_(0, 1, c10::nullopt);
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
    for(int64_t i = p_begin; i < p_end; i++)
      r__data[i*r__stride_0] = static_cast<scalar_t>(i);
  });

  for(int64_t i = 0; i < n - 1; i++)
  {
    int64_t z = generator->random() % (n-i);
    scalar_t sav = r__data[i*r__stride_0];
    r__data[i*r__stride_0] = r__data[(z+i)*r__stride_0];
    r__data[(z+i)*r__stride_0] = sav;
  }
}
} // namespace

Tensor randperm(int64_t n, const TensorOptions& options) {
  return native::randperm(n, c10::nullopt, options);
}

Tensor randperm(int64_t n, c10::optional<Generator> generator, const TensorOptions& options) {
  auto tensor = at::empty(n, options);
  return at::randperm_out(tensor, n, generator);
}

Tensor& randperm_out(Tensor& result, int64_t n) {
  return at::randperm_out(result, n, c10::nullopt);
}

Tensor& randperm_out_cpu(Tensor& result, int64_t n, c10::optional<Generator> generator) {
  TORCH_CHECK(n >= 0, "n must be non-negative, got", n);
  TORCH_CHECK(!generator.has_value() || (generator.has_value() && result.device() == generator->device()), "Expected a '", result.device(), "' generator device but found '", generator->device(), "'");
  check_supported_max_int_with_precision(n, result);
  result.resize_({n});
  auto gen = get_generator_or_default<CPUGeneratorImpl>(generator, detail::getDefaultCPUGenerator());
  // See Note [Acquire lock when using random generators]
  std::lock_guard<std::mutex> lock(gen->mutex_);
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, result.scalar_type(), "randperm", [&]() -> void {
    randperm_cpu<scalar_t>(result, n, gen);
  });

  return result;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ range ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor range(
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    const TensorOptions& options) {
  Tensor result = at::empty({0}, options);
  return at::range_out(result, start, end, step);
}

Tensor range(
    const Scalar& start,
    const Scalar& end,
    const TensorOptions& options) {
  return at::native::range(start, end, 1, options);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ triangle ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor tril_indices_cpu(
    int64_t row, int64_t col, int64_t offset, c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt, c10::optional<Device> device_opt, c10::optional<bool> pin_memory_opt) {
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
  AT_DISPATCH_ALL_TYPES(result.scalar_type(), "tril_indices", [&]() -> void {
    // fill the Tensor with correct values
    scalar_t* result_data = result.data_ptr<scalar_t>();
    int64_t i = 0;

    scalar_t r = std::max<int64_t>(0, -offset), c = 0;
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
    int64_t row, int64_t col, int64_t offset, c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt, c10::optional<Device> device_opt, c10::optional<bool> pin_memory_opt) {
  check_args(row, col, layout_opt);

  auto triu_size = row * col - get_tril_size(row, col, offset - 1);

  // create an empty Tensor with correct size
  auto result = at::native::empty_cpu({2, triu_size}, dtype_opt, layout_opt, device_opt, pin_memory_opt);

  AT_DISPATCH_ALL_TYPES(result.scalar_type(), "triu_indices", [&]() -> void {
    // fill the Tensor with correct values
    scalar_t* result_data = result.data_ptr<scalar_t>();
    int64_t i = 0;
    // not typing std::max with scalar_t as it could be an unsigned type
    // NOTE: no need to check if the returned value of std::max overflows
    // scalar_t, as i and triu_size act as a guard.
    scalar_t c = std::max<int64_t>(0, offset), r = 0;
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

Tensor zeros(IntArrayRef size, const TensorOptions& options) {
  auto result = at::empty(size, options);
  return result.zero_();
}

Tensor& zeros_out(Tensor& result, IntArrayRef size) {
  if (result.is_sparse()) {
    result.sparse_resize_and_clear_(size, size.size(), 0.);
    return result;
  } else {
    result.resize_(size);
  }
  return result.zero_();
}

Tensor zeros_like(
    const Tensor& self,
    const TensorOptions& options,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  if (options.layout() == kSparse && self.is_sparse()) {
    TORCH_CHECK(
        !(optional_memory_format.has_value()),
        "memory format option is only supported by strided tensors");
    auto res = at::empty({0}, options); // to be resized
    res.sparse_resize_and_clear_(
        self.sizes(), self.sparse_dim(), self.dense_dim());
    return res;
  }
  auto result = at::empty_like(self, options, optional_memory_format);
  return result.zero_();
}

Tensor new_zeros(
    const Tensor& self,
    IntArrayRef size,
    const TensorOptions& options
    ) {
  return at::zeros(size, self.options().merge_in(options));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~ bartlett_window ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor bartlett_window(int64_t window_length, const TensorOptions& options) {
  return native::bartlett_window(window_length, /*periodic=*/true, options);
}

Tensor bartlett_window(
    int64_t window_length,
    bool periodic,
    const TensorOptions& options) {
  window_function_checks("bartlett_window", options, window_length);
  if (window_length == 0) {
    return at::empty({0}, options);
  }
  if (window_length == 1) {
    return native::ones({1}, options);
  }
  if (periodic) {
    window_length += 1;
  }
  auto window = native::arange(window_length, options).mul_(2. / static_cast<double>(window_length - 1));
  const int64_t first_half_size = ((window_length - 1) >> 1) + 1;
  window.narrow(0, first_half_size, window_length - first_half_size).mul_(-1).add_(2);
  return periodic ? window.narrow(0, 0, window_length - 1) : window;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~ blackman_window ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor blackman_window(int64_t window_length, const TensorOptions& options) {
  return native::blackman_window(window_length, /*periodic=*/true, options);
}

Tensor blackman_window(
    int64_t window_length,
    bool periodic,
    const TensorOptions& options) {
  window_function_checks("blackman_window", options, window_length);
  if (window_length == 0) {
    return at::empty({0}, options);
  }
  if (window_length == 1) {
    return native::ones({1}, options);
  }
  if (periodic) {
    window_length += 1;
  }
  // from https://en.wikipedia.org/wiki/Window_function#Blackman_window
  auto window = native::arange(window_length, options).mul_(c10::pi<double> / static_cast<double>(window_length - 1));
  window = window.mul(4).cos_().mul_(0.08) - window.mul(2).cos_().mul_(0.5) + 0.42;
  return periodic ? window.narrow(0, 0, window_length - 1) : window;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ hamming_window ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor hamming_window(int64_t window_length, const TensorOptions& options) {
  return native::hamming_window(window_length, /*periodic=*/true, options);
}

Tensor hamming_window(
    int64_t window_length,
    bool periodic,
    const TensorOptions& options) {
  return native::hamming_window(
      window_length, periodic, /*alpha=*/0.54, options);
}

Tensor hamming_window(
    int64_t window_length,
    bool periodic,
    double alpha,
    const TensorOptions& options) {
  return native::hamming_window(
      window_length, periodic, alpha, /*beta=*/0.46, options);
}

Tensor hamming_window(
    int64_t window_length,
    bool periodic,
    double alpha,
    double beta,
    const TensorOptions& options) {
  window_function_checks("hamming_window", options, window_length);
  if (window_length == 0) {
    return at::empty({0}, options);
  }
  if (window_length == 1) {
    return native::ones({1}, options);
  }
  if (periodic) {
    window_length += 1;
  }
  auto window = native::arange(window_length, options);
  window.mul_(c10::pi<double> * 2. / static_cast<double>(window_length - 1)).cos_().mul_(-beta).add_(alpha);
  return periodic ? window.narrow(0, 0, window_length - 1) : window;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ hann_window ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor hann_window(int64_t window_length, const TensorOptions& options) {
  return native::hann_window(window_length, /*periodic=*/true, options);
}

Tensor hann_window(
    int64_t window_length,
    bool periodic,
    const TensorOptions& options) {
  window_function_checks("hann_window", options, window_length);
  return native::hamming_window(
      window_length, periodic, /*alpha=*/0.5, /*beta=*/0.5, options);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ kaiser_window ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor kaiser_window(int64_t window_length, const TensorOptions& options) {
  return native::kaiser_window(window_length, /*periodic=*/true, /*beta=*/12.0, options);
}

Tensor kaiser_window(int64_t window_length, bool periodic, const TensorOptions& options) {
  return native::kaiser_window(window_length, periodic, /*beta=*/12.0, options);
}

Tensor kaiser_window(
    int64_t window_length,
    bool periodic,
    double beta,
    const TensorOptions& options) {
  window_function_checks("kaiser_window", options, window_length);
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
  return periodic ? window.narrow(0, 0, window_length - 1) : window;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~ vandermonde_matrix ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Tensor vander(const Tensor& x, c10::optional<int64_t> N, bool increasing) {
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

Tensor from_file(std::string filename, c10::optional<bool> shared, c10::optional<int64_t> size, const TensorOptions& options) {
    TORCH_CHECK(!options.pinned_memory(), "tensors constructed from a file cannot be pinned");
    int64_t my_size = size.value_or(0);
    int flags = shared.value_or(false) ? TH_ALLOCATOR_MAPPED_SHARED : 0;
    auto dtype = options.dtype();
    size_t size_bytes = my_size * dtype.itemsize();
    auto storage_impl = c10::make_intrusive<at::StorageImpl>(
        c10::StorageImpl::use_byte_size_t(),
        size_bytes,
        THMapAllocator::makeDataPtr(
            filename.c_str(), flags, size_bytes, nullptr),
        /*allocator=*/nullptr,
        /*resizable=*/false);
    auto tensor = detail::make_tensor<at::TensorImpl>(
        storage_impl, at::DispatchKey::CPU, dtype);
    tensor.unsafeGetTensorImpl()->set_sizes_contiguous({my_size});
    return tensor;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ clone ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor clone(const Tensor& src, c10::optional<c10::MemoryFormat> optional_memory_format) {
  auto memory_format =
      optional_memory_format.value_or(MemoryFormat::Preserve);
  if (memory_format == MemoryFormat::Preserve) {
    if (src.is_non_overlapping_and_dense()) {
      // Copy all strides
      auto self = at::empty_strided(src.sizes(), src.strides(), src.options());
      self.copy_(src);
      return self;
    } else {
      memory_format = src.suggest_memory_format();
    }
  }
  auto self = at::empty_like(src, src.options(), memory_format);
  self.copy_(src);
  return self;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~ named tensor overloads ~~~~~~~~~~~~~~~~~~~~~~~~~~~
// In the short term, these exist.
// In the long term, we should move DimnameList into TensorOptions to avoid
// having these overloads.

Tensor full(
    IntArrayRef size,
    const Scalar& fill_value,
    optional<DimnameList> names,
    const TensorOptions& options) {

  TORCH_CHECK(options.layout() != kSparse,
    "full(...) is not implemented for sparse layout");

  auto result = at::empty(size, names, infer_full_options(fill_value, options));
  return result.fill_(fill_value);
}

Tensor ones(
    IntArrayRef size,
    optional<DimnameList> names,
    const TensorOptions& options) {
  return native::full(size, /*fill_value=*/1., names, options);
}

Tensor zeros(
    IntArrayRef size,
    optional<DimnameList> names,
    const TensorOptions& options) {
  return native::full(size, /*fill_value=*/0., names, options);
}

Tensor randn(
    IntArrayRef size,
    optional<DimnameList> names,
    const TensorOptions& options) {
  return native::randn(size, c10::nullopt, names, options);
}

Tensor randn(
    IntArrayRef size,
    c10::optional<Generator> generator,
    optional<DimnameList> names,
    const TensorOptions& options) {
  auto result = at::empty(size, names, options);
  return result.normal_(0, 1, generator);
}

Tensor rand(
    IntArrayRef size,
    optional<DimnameList> names,
    const TensorOptions& options) {
  return native::rand(size, c10::nullopt, names, options);
}

Tensor rand(
    IntArrayRef size,
    c10::optional<Generator> generator,
    optional<DimnameList> names,
    const TensorOptions& options) {
  auto result = at::empty(size, names, options);
  return result.uniform_(0, 1, generator);
}


DEFINE_DISPATCH(kaiser_window_stub);

} // namespace native
} // namespace at
