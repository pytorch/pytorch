#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Config.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/native/SpectralOpsUtils.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorOperators.h>
#include <ATen/WrapDimUtils.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_cufft_clear_plan_cache_native.h>
#include <ATen/ops/_cufft_get_plan_cache_max_size_native.h>
#include <ATen/ops/_cufft_get_plan_cache_size_native.h>
#include <ATen/ops/_cufft_set_plan_cache_max_size_native.h>
#include <ATen/ops/_fft_c2c.h>
#include <ATen/ops/_fft_c2r.h>
#include <ATen/ops/_fft_r2c.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/arange_native.h>
#include <ATen/ops/conj.h>
#include <ATen/ops/conj_physical.h>
#include <ATen/ops/constant_pad_nd.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/fft_fft2_native.h>
#include <ATen/ops/fft_fft_native.h>
#include <ATen/ops/fft_fftfreq_native.h>
#include <ATen/ops/fft_fftn_native.h>
#include <ATen/ops/fft_fftshift_native.h>
#include <ATen/ops/fft_hfft2_native.h>
#include <ATen/ops/fft_hfft_native.h>
#include <ATen/ops/fft_hfftn_native.h>
#include <ATen/ops/fft_ifft2_native.h>
#include <ATen/ops/fft_ifft_native.h>
#include <ATen/ops/fft_ifftn_native.h>
#include <ATen/ops/fft_ifftshift_native.h>
#include <ATen/ops/fft_ihfft2_native.h>
#include <ATen/ops/fft_ihfft_native.h>
#include <ATen/ops/fft_ihfftn_native.h>
#include <ATen/ops/fft_irfft2_native.h>
#include <ATen/ops/fft_irfft_native.h>
#include <ATen/ops/fft_irfftn_native.h>
#include <ATen/ops/fft_rfft2_native.h>
#include <ATen/ops/fft_rfft_native.h>
#include <ATen/ops/fft_rfftfreq_native.h>
#include <ATen/ops/fft_rfftn_native.h>
#include <ATen/ops/istft_native.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/pad.h>
#include <ATen/ops/roll.h>
#include <ATen/ops/stft.h>
#include <ATen/ops/stft_native.h>
#include <ATen/ops/unfold_backward.h>
#include <ATen/ops/view_as_complex.h>
#include <ATen/ops/view_as_real.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like_ops.h>
#endif

#include <algorithm>

namespace at { namespace native {

namespace {

// Promote inputs to FFT functions
// * Integers are promoted to the default floating type
// * If require_complex=True, all types are promoted to complex
// * Raises an error for half-precision dtypes to allow future support
ScalarType promote_type_fft(ScalarType type, bool require_complex, Device device) {
  if (at::isComplexType(type)) {
    return type;
  }
  // Promote integral to default float type
  if (!at::isFloatingType(type)) {
    type = c10::typeMetaToScalarType(c10::get_default_dtype());
  }

  const bool maybe_support_half = (
    // Only CUDA supports half precision, but since meta tensors don't have a
    // device we err on the side of accepting it
    (device.is_cuda() || device.is_meta()) &&
    !at::detail::getCUDAHooks().hasROCM()
  );
  if (maybe_support_half) {
    TORCH_CHECK(type == kHalf || type == kFloat || type == kDouble, "Unsupported dtype ", type);
  } else {
    TORCH_CHECK(type == kFloat || type == kDouble, "Unsupported dtype ", type);
  }

  if (!require_complex) {
    return type;
  }

  // Promote to complex
  switch (type) {
  case kHalf: return kComplexHalf;
  case kFloat: return kComplexFloat;
  case kDouble: return kComplexDouble;
  default: TORCH_INTERNAL_ASSERT(false, "Unhandled dtype");
  }
}

// Promote a tensor's dtype according to promote_type_fft
Tensor promote_tensor_fft(const Tensor& t, bool require_complex=false) {
  auto cur_type = t.scalar_type();
  auto new_type = promote_type_fft(cur_type, require_complex, t.device());
  return (cur_type == new_type) ? t : t.to(new_type);
}

// Convert NumPy compatible normalization mode string to enum values
// NOTE: NumPy's normalization modes have direction-specific meanings. For example,
// "forward" translates to `by_n` for a forward transform and `none` for backward.
fft_norm_mode norm_from_string(c10::optional<c10::string_view> norm, bool forward) {
  if (!norm || *norm == "backward") {
    return forward ? fft_norm_mode::none : fft_norm_mode::by_n;
  }

  if (*norm == "forward") {
    return forward ? fft_norm_mode::by_n : fft_norm_mode::none;
  }

  if (*norm == "ortho") {
    return fft_norm_mode::by_root_n;
  }

  TORCH_CHECK(false, "Invalid normalization mode: \"", *norm, "\"")
}

// Fixes the shape of x such that x.size(dims[i]) == sizes[i],
// either by zero-padding, or by slicing x starting from 0.
Tensor resize_fft_input(Tensor x, IntArrayRef dims, SymIntArrayRef sizes) {
  TORCH_INTERNAL_ASSERT(dims.size() == sizes.size());
  bool must_copy = false;
  auto x_sizes = x.sym_sizes();
  SymDimVector pad_amount(x_sizes.size() * 2);
  for (const auto i : c10::irange(dims.size())) {
    if (sizes[i] == -1) {
      continue;
    }

    if (x_sizes[dims[i]] < sizes[i]) {
      must_copy = true;
      auto pad_idx = pad_amount.size() - 2 * dims[i] - 1;
      pad_amount[pad_idx] = sizes[i] - x_sizes[dims[i]];
    }

    if (x_sizes[dims[i]] > sizes[i]) {
      x = x.slice_symint(dims[i], 0, sizes[i]);
    }
  }

  // Only call pad if necessary since pad copies the entire tensor
  return must_copy ? at::constant_pad_nd_symint(x, pad_amount) : x;
}

Tensor fft_r2c_maybe_out(
    c10::string_view fname, const Tensor& out, const Tensor& input,
    IntArrayRef dim, int64_t norm, bool onesided) {
  if (out.defined()) {
    TORCH_CHECK(out.is_complex(), fname,
                " expects a complex output tensor, but got ", out.scalar_type());
    auto out_mut = out;
    return at::_fft_r2c_outf(input, dim, norm, onesided, out_mut);
  }
  return at::_fft_r2c(input, dim, norm, onesided);
}

Tensor fft_c2r_maybe_out(
    c10::string_view fname, const Tensor& out, const Tensor& input,
    IntArrayRef dim, int64_t norm, SymInt last_dim_size) {
  // Support out argument if defined, otherwise call functional
  // variant so autograd works properly.
  if (out.defined()) {
    TORCH_CHECK(out.is_floating_point(), fname,
                " expects a floating point output tensor, but got ", out.scalar_type());
    auto out_mut = out;
    return at::_fft_c2r_symint_outf(input, dim, norm, last_dim_size, out_mut);
  }
  return at::_fft_c2r_symint(input, dim, norm, last_dim_size);
}

Tensor fft_c2c_maybe_out(
    c10::string_view fname, const Tensor& out, const Tensor& input,
    IntArrayRef dim, int64_t norm, bool forward) {
  if (out.defined()) {
    TORCH_CHECK(out.is_complex(), fname,
                " expects a complex output tensor, but got ", out.scalar_type());
    auto out_mut = out;
    return at::_fft_c2c_outf(input, dim, norm, forward, out_mut);
  }
  return at::_fft_c2c(input, dim, norm, forward);
}

// Complex to real FFT
Tensor fft_c2r(c10::string_view function_name,
               Tensor out, Tensor input, c10::optional<SymInt> n_opt,
               int64_t unwrapped_dim, c10::optional<c10::string_view> norm_str,
               bool forward) {
  TORCH_CHECK(!out.defined() || out.is_floating_point(), function_name,
              " expects a floating point output tensor, but got ", out.scalar_type());
  input = promote_tensor_fft(input, /*require_complex=*/true);
  const auto input_dim = input.dim();
  const auto dim = maybe_wrap_dim(unwrapped_dim, input_dim, /*wrap_scalar=*/false);
  const auto n = n_opt.value_or(2*(input.sym_sizes()[dim] - 1));
  TORCH_CHECK(n >= 1, "Invalid number of data points (", n, ") specified");
  if (n_opt) {
    input = resize_fft_input(input, dim, n/2 + 1);
  }
  const auto norm = norm_from_string(norm_str, forward);
  if (forward) {
    // FIXME: _fft does not support complex_output=false with inverse=false
    input = input.conj();
  }
  return fft_c2r_maybe_out(
      function_name, out, input, dim, static_cast<int64_t>(norm), n);
}

// Real to complex FFT
Tensor fft_r2c(c10::string_view function_name,
               Tensor out, Tensor input, c10::optional<SymInt> n_opt,
               int64_t unwrapped_dim, c10::optional<c10::string_view> norm_str,
               bool forward, bool onesided) {
  TORCH_CHECK(!input.is_complex(), function_name,
              " expects a real input tensor, but got ", input.scalar_type());
  TORCH_CHECK(!out.defined() || out.is_complex(), function_name,
              " expects a complex output tensor, but got ", out.scalar_type());
  input = promote_tensor_fft(input);
  const auto input_dim = input.dim();
  const auto dim = maybe_wrap_dim(unwrapped_dim, input_dim, /*wrap_scalar=*/false);
  const auto n = n_opt.value_or(input.sym_sizes()[dim]);
  TORCH_CHECK(n >= 1, "Invalid number of data points (", n, ") specified");
  if (n_opt) {
    input = resize_fft_input(input, dim, n);
  }

  const auto norm = norm_from_string(norm_str, forward);

  Tensor ret;
  if (out.defined() && forward) {
    ret = at::_fft_r2c_out(out, input, dim, static_cast<int64_t>(norm), onesided);
  } else {
    ret = at::_fft_r2c(input, dim, static_cast<int64_t>(norm), onesided);
  }

  if (!forward) {
    // FIXME: _fft_r2c doesn't support native r2c IFFT
    return out.defined() ? at::conj_physical_out(out, ret) : ret.conj();
  } else {
    return ret;
  }
}

// Complex to complex FFT
Tensor fft_c2c(c10::string_view function_name,
               Tensor out, Tensor input, c10::optional<SymInt> n_opt,
               int64_t unwrapped_dim, c10::optional<c10::string_view> norm_str,
               bool forward) {
  TORCH_CHECK(input.is_complex(), function_name,
              " expects a complex input tensor, but got ", input.scalar_type());
  const auto input_dim = input.dim();
  const auto dim = maybe_wrap_dim(unwrapped_dim, input_dim, /*wrap_scalar=*/false);
  const auto n = n_opt.value_or(input.sym_sizes()[dim]);
  TORCH_CHECK(n >= 1, "Invalid number of data points (", n, ") specified");
  if (n_opt) {
    input = resize_fft_input(input, dim, n);
  }
  const auto norm = static_cast<int64_t>(norm_from_string(norm_str, forward));
  return fft_c2c_maybe_out(function_name, out, input, dim, norm, forward);
}

// Dimensions to transform, and the signal shape in those dimensions
struct ShapeAndDims {
  SymDimVector shape;
  DimVector dim;
};

// Pre-process n-dimensional fft's `s` and `dim` arguments.
// Wraps dimensions and applies defaulting behavior.
// Also checks transform dims are unique and transform shape is non-empty.
ShapeAndDims canonicalize_fft_shape_and_dim_args(
    Tensor input, at::OptionalSymIntArrayRef shape, at::OptionalIntArrayRef dim) {
  const int64_t input_dim = input.dim();
  const SymIntArrayRef input_sizes = input.sym_sizes();
  ShapeAndDims ret;

  if (dim) {
    ret.dim.resize(dim->size());
    std::copy(dim->begin(), dim->end(), ret.dim.begin());
    maybe_wrap_dims(ret.dim, input_dim, /*wrap_scalars=*/false);

    // Check dims are unique
    DimVector copy = ret.dim;
    std::sort(copy.begin(), copy.end());
    auto duplicate = std::adjacent_find(copy.begin(), copy.end());
    TORCH_CHECK(duplicate == copy.end(), "FFT dims must be unique");
  }

  if (shape) {
    // Has shape, may have dim
    TORCH_CHECK(!dim ||
                dim->size() == shape->size(),
                "When given, dim and shape arguments must have the same length");
    TORCH_CHECK(static_cast<int64_t>(shape->size()) <= input_dim,
                "Got shape with ", shape->size(), " values but input tensor "
                "only has ", input_dim, " dimensions.");
    const int64_t transform_ndim = shape->size();
    // If shape is given, dims defaults to the last shape.size() dimensions
    if (!dim) {
      ret.dim.resize(transform_ndim);
      std::iota(ret.dim.begin(), ret.dim.end(), input_dim - transform_ndim);
    }

    // Translate shape of -1 to the default length
    ret.shape.resize(transform_ndim);
    for (const auto i : c10::irange(transform_ndim)) {
      const auto n = (*shape)[i];
      ret.shape[i] = n == -1 ? input_sizes[ret.dim[i]] : n;
    }
  } else if (!dim) {
    // No shape, no dim
    ret.dim.resize(input_dim);
    std::iota(ret.dim.begin(), ret.dim.end(), int64_t{0});
    ret.shape.resize(input_dim);
    std::copy(input_sizes.begin(), input_sizes.end(), ret.shape.begin());
  } else {
    // No shape, has dim
    ret.shape.resize(ret.dim.size());
    for (const auto i : c10::irange(ret.dim.size())) {
      ret.shape[i] = input_sizes[ret.dim[i]];
    }
  }

  for (const auto & shape : ret.shape) {
    TORCH_CHECK(shape > 0,
                "Invalid number of data points (", shape, ") specified");
  }

  return ret;
}

// Complex to complex n-dimensional fft
Tensor fftn_c2c(
    c10::string_view function_name,
    Tensor out, const Tensor& input, SymIntArrayRef shape,
    IntArrayRef dim, c10::optional<c10::string_view> norm_str, bool forward) {
  TORCH_CHECK(input.is_complex(), function_name, " expects a complex input tensor, but got", input.scalar_type());
  Tensor x = resize_fft_input(input, dim, shape);
  const auto norm = static_cast<int64_t>(norm_from_string(norm_str, forward));
  constexpr c10::string_view fname = "fftn";
  return fft_c2c_maybe_out(fname, out, x, dim, norm, forward);
}

}  // namespace (anonymous)

// torch.fft.fft, analogous to NumPy's numpy.fft.fft
Tensor fft_fft_symint(const Tensor& self, c10::optional<SymInt> n, int64_t dim,
               c10::optional<c10::string_view> norm) {
  return self.is_complex() ?
    fft_c2c("fft", {}, self, n, dim, norm, /*forward=*/true) :
    fft_r2c("fft", {}, self, n, dim, norm, /*forward=*/true, /*onesided=*/false);
}

Tensor& fft_fft_symint_out(const Tensor& self, c10::optional<SymInt> n,
                    int64_t dim, c10::optional<c10::string_view> norm, Tensor& out) {
  if (self.is_complex()) {
    fft_c2c("fft", out, self, n, dim, norm, /*forward=*/true);
  } else {
    fft_r2c("fft", out, self, n, dim, norm, /*forward=*/true, /*onesided=*/false);
  }
  return out;
}

Tensor fft_ifft_symint(const Tensor& self, c10::optional<SymInt> n, int64_t dim,
                c10::optional<c10::string_view> norm) {
  return self.is_complex() ?
    fft_c2c("ifft", {}, self, n, dim, norm, /*forward=*/false) :
    fft_r2c("ifft", {}, self, n, dim, norm, /*forward=*/false, /*onesided=*/false);
}

Tensor& fft_ifft_symint_out(const Tensor& self, c10::optional<SymInt> n,
                     int64_t dim, c10::optional<c10::string_view> norm, Tensor& out) {
  if (self.is_complex()) {
    fft_c2c("ifft", out, self, n, dim, norm, /*forward=*/false);
  } else {
    fft_r2c("ifft", out, self, n, dim, norm, /*forward=*/false, /*onesided=*/false);
  }
  return out;
}

Tensor fft_rfft_symint(const Tensor& self, c10::optional<SymInt> n, int64_t dim,
                c10::optional<c10::string_view> norm) {
  return fft_r2c("rfft", {}, self, n, dim, norm, /*forward=*/true, /*onesided=*/true);
}

Tensor& fft_rfft_symint_out(const Tensor& self, c10::optional<SymInt> n,
                     int64_t dim, c10::optional<c10::string_view> norm, Tensor& out) {
  fft_r2c("rfft", out, self, n, dim, norm, /*forward=*/true, /*onesided=*/true);
  return out;
}

Tensor fft_irfft_symint(const Tensor& self, c10::optional<SymInt> n, int64_t dim,
                 c10::optional<c10::string_view> norm) {
  return fft_c2r("irfft", {}, self, n, dim, norm, /*forward=*/false);
}

Tensor& fft_irfft_symint_out(const Tensor& self, c10::optional<SymInt> n,
                  int64_t dim, c10::optional<c10::string_view> norm, Tensor& out) {
  fft_c2r("irfft", out, self, n, dim, norm, /*forward=*/false);
  return out;
}

Tensor fft_hfft_symint(const Tensor& self, c10::optional<SymInt> n, int64_t dim,
                c10::optional<c10::string_view> norm) {
  return fft_c2r("hfft", {}, self, n, dim, norm, /*forward=*/true);
}

Tensor& fft_hfft_symint_out(const Tensor& self, c10::optional<SymInt> n,
                     int64_t dim, c10::optional<c10::string_view> norm, Tensor& out) {
  fft_c2r("hfft", out, self, n, dim, norm, /*forward=*/true);
  return out;
}

Tensor fft_ihfft_symint(const Tensor& self, c10::optional<SymInt> n, int64_t dim,
                 c10::optional<c10::string_view> norm) {
  return fft_r2c("ihfft", {}, self, n, dim, norm, /*forward=*/false, /*onesided=*/true);
}

Tensor& fft_ihfft_symint_out(const Tensor& self, c10::optional<SymInt> n,
                     int64_t dim, c10::optional<c10::string_view> norm, Tensor& out) {
  fft_r2c("ihfft", out, self, n, dim, norm, /*forward=*/false, /*onesided=*/true);
  return out;
}

Tensor fft_fftn_symint(const Tensor& self, at::OptionalSymIntArrayRef s,
                at::OptionalIntArrayRef dim,
                c10::optional<c10::string_view> norm) {
  auto desc = canonicalize_fft_shape_and_dim_args(self, s, dim);
  // TODO: For real input, perform rfftn then mirror with conjugate symmetry
  Tensor input = promote_tensor_fft(self, /*require_complex=*/true);
  return fftn_c2c("fftn", {}, input, desc.shape, desc.dim, norm, /*forward=*/true);
}

Tensor& fft_fftn_symint_out(const Tensor& self,
                     at::OptionalSymIntArrayRef s,
                     at::OptionalIntArrayRef dim,
                     c10::optional<c10::string_view> norm, Tensor& out) {
  auto desc = canonicalize_fft_shape_and_dim_args(self, s, dim);
  // TODO: For real input, perform rfftn then mirror with conjugate symmetry
  Tensor input = promote_tensor_fft(self, /*require_complex=*/true);
  fftn_c2c("fftn", out, input, desc.shape, desc.dim, norm, /*forward=*/true);
  return out;
}

Tensor fft_ifftn_symint(const Tensor& self, at::OptionalSymIntArrayRef s,
                at::OptionalIntArrayRef dim,
                c10::optional<c10::string_view> norm) {
  auto desc = canonicalize_fft_shape_and_dim_args(self, s, dim);
  Tensor input = promote_tensor_fft(self, /*require_complex=*/true);
  return fftn_c2c("ifftn", {}, input, desc.shape, desc.dim, norm, /*forward=*/false);
}

Tensor& fft_ifftn_symint_out(const Tensor& self,
                      at::OptionalSymIntArrayRef s,
                      at::OptionalIntArrayRef dim,
                      c10::optional<c10::string_view> norm, Tensor& out) {
  auto desc = canonicalize_fft_shape_and_dim_args(self, s, dim);
  Tensor input = promote_tensor_fft(self, /*require_complex=*/true);
  fftn_c2c("ifftn", out, input, desc.shape, desc.dim, norm, /*forward=*/false);
  return out;
}

static Tensor fft_rfftn_impl(Tensor out, const Tensor& self,
                             at::OptionalSymIntArrayRef s,
                             at::OptionalIntArrayRef dim,
                             const c10::optional<c10::string_view>& norm_str) {
  TORCH_CHECK(!self.is_complex(), "rfftn expects a real-valued input tensor, but got ", self.scalar_type());
  auto desc = canonicalize_fft_shape_and_dim_args(self, s, dim);
  TORCH_CHECK(!desc.shape.empty(), "rfftn must transform at least one axis");
  Tensor input = promote_tensor_fft(self, /*require_complex=*/false);
  Tensor x = resize_fft_input(input, desc.dim, desc.shape);
  const auto norm = static_cast<int64_t>(norm_from_string(norm_str, /*forward=*/true));
  constexpr c10::string_view fname = "rfftn";
  return fft_r2c_maybe_out(fname, out, x, desc.dim, norm, /*onesided=*/true);
}

Tensor fft_rfftn_symint(const Tensor& self, at::OptionalSymIntArrayRef s,
                at::OptionalIntArrayRef dim,
                c10::optional<c10::string_view> norm_str) {
  return fft_rfftn_impl({}, self, s, dim, norm_str);
}

Tensor& fft_rfftn_symint_out(const Tensor& self,
                      at::OptionalSymIntArrayRef s,
                      at::OptionalIntArrayRef dim,
                      c10::optional<c10::string_view> norm_str, Tensor& out) {
  fft_rfftn_impl(out, self, s, dim, norm_str);
  return out;
}

static ShapeAndDims canonicalize_fft_c2r_shape_and_dim_args(
    c10::string_view fname, const Tensor& self,
    const at::OptionalSymIntArrayRef& s,
    const at::OptionalIntArrayRef& dims,
    SymInt& last_dim_size) {
  auto desc = canonicalize_fft_shape_and_dim_args(self, s, dims);
  TORCH_CHECK(!desc.shape.empty(), fname, " must transform at least one axis");

  // Expected output size of the hermitian-symmetric dimension
  last_dim_size = [&] {
    // Fixup default shape handling in the last dimension,
    if (!s.has_value() || (s->back() == -1)) {
      const auto last_dim = desc.dim.back();
      return 2 * (self.sym_sizes()[last_dim] - 1);
    }
    return desc.shape.back();
  }();
  TORCH_CHECK(last_dim_size >= 1, "Invalid number of data points (", last_dim_size, ") specified");

  // Expected input size of the complex-hermitian data
  desc.shape.back() = last_dim_size / 2 + 1;
  return desc;
}

static Tensor fft_irfftn_impl(Tensor out, const Tensor& self,
                              at::OptionalSymIntArrayRef s,
                              at::OptionalIntArrayRef dim,
                              const c10::optional<c10::string_view>& norm_str) {
  SymInt last_dim_size = 0;
  auto desc = canonicalize_fft_c2r_shape_and_dim_args(
      "irfftn", self, s, dim, last_dim_size);
  Tensor input = promote_tensor_fft(self, /*require_complex=*/true);
  Tensor x = resize_fft_input(input, desc.dim, desc.shape);
  const auto norm = static_cast<int64_t>(norm_from_string(norm_str, /*forward=*/false));
  constexpr c10::string_view fname = "irfftn";
  return fft_c2r_maybe_out(fname, out, x, desc.dim, norm, last_dim_size);
}

Tensor fft_irfftn_symint(const Tensor& self,
                  at::OptionalSymIntArrayRef s,
                  at::OptionalIntArrayRef dim,
                  c10::optional<c10::string_view> norm_str) {
  return fft_irfftn_impl({}, self, s, dim, norm_str);
}

Tensor& fft_irfftn_symint_out(const Tensor& self,
                       at::OptionalSymIntArrayRef s,
                       at::OptionalIntArrayRef dim,
                       c10::optional<c10::string_view> norm_str, Tensor& out) {
  fft_irfftn_impl(out, self, s, dim, norm_str);
  return out;
}

static Tensor fft_hfftn_impl(
    const Tensor& self,
    at::OptionalSymIntArrayRef s,
    at::OptionalIntArrayRef dim,
    c10::optional<c10::string_view> norm_str,
    const Tensor& out) {
  constexpr c10::string_view fname = "hfftn";
  SymInt last_dim_size = 0;
  auto desc = canonicalize_fft_c2r_shape_and_dim_args(
      fname, self, s, dim, last_dim_size);
  auto input = promote_tensor_fft(self, /*require_complex=*/true);
  auto x = resize_fft_input(input, desc.dim, desc.shape);
  const auto norm = static_cast<int64_t>(
      norm_from_string(norm_str, /*forward=*/true));

  Tensor tmp;
  if (desc.dim.size() > 1) {
    auto c2c_dims = IntArrayRef(desc.dim).slice(0, desc.dim.size() - 1);
    tmp = at::_fft_c2c(x, c2c_dims, norm, /*forward=*/true);
  } else {
    tmp = x;
  }

  const auto last_dim = desc.dim.back();
  tmp = tmp.conj();
  return fft_c2r_maybe_out(fname, out, tmp, last_dim, norm, last_dim_size);
}

Tensor fft_hfftn_symint(
    const Tensor& self,
    at::OptionalSymIntArrayRef s,
    at::OptionalIntArrayRef dim,
    c10::optional<c10::string_view> norm) {
  return fft_hfftn_impl(self, s, dim, norm, {});
}

const Tensor& fft_hfftn_symint_out(
    const Tensor& self,
    at::OptionalSymIntArrayRef s,
    at::OptionalIntArrayRef dim, c10::optional<c10::string_view> norm,
    const Tensor& out) {
  fft_hfftn_impl(self, s, dim, norm, out);
  return out;
}

static Tensor fft_ihfftn_impl(
    const Tensor& self,
    const at::OptionalSymIntArrayRef& s,
    const at::OptionalIntArrayRef& dim,
    const c10::optional<c10::string_view>& norm_str,
    const Tensor& out) {
  constexpr c10::string_view fname = "ihfftn";
  auto desc = canonicalize_fft_shape_and_dim_args(self, s, dim);
  TORCH_CHECK(!desc.shape.empty(), "ihfftn must transform at least one axis");
  auto input = promote_tensor_fft(self, /*require_complex=*/false);
  auto x = resize_fft_input(input, desc.dim, desc.shape);
  const auto norm = static_cast<int64_t>(
      norm_from_string(norm_str, /*forward=*/false));

  const auto last_dim = desc.dim.back();
  auto tmp = at::_fft_r2c(x, last_dim, norm, /*onesided=*/true);
  if (desc.dim.size() == 1) {
    return out.defined() ? at::conj_physical_out(tmp, out) : tmp.conj();
  }

  tmp = at::conj_physical(tmp);
  auto c2c_dims = IntArrayRef(desc.dim).slice(0, desc.dim.size() - 1);
  return fft_c2c_maybe_out(fname, out, tmp, c2c_dims, norm, /*forward=*/false);
}

Tensor fft_ihfftn_symint(
    const Tensor& self,
    at::OptionalSymIntArrayRef s,
    at::OptionalIntArrayRef dim,
    c10::optional<c10::string_view> norm) {
  return fft_ihfftn_impl(self, s, dim, norm, {});
}

const Tensor& fft_ihfftn_symint_out(
    const Tensor& self,
    at::OptionalSymIntArrayRef s,
    at::OptionalIntArrayRef dim,
    c10::optional<c10::string_view> norm,
    const Tensor& out) {
  fft_ihfftn_impl(self, s, dim, norm, out);
  return out;
}

Tensor fft_fft2_symint(const Tensor& self, at::OptionalSymIntArrayRef s,
                IntArrayRef dim, c10::optional<c10::string_view> norm) {
  return native::fft_fftn_symint(self, s, dim, std::move(norm));
}

Tensor& fft_fft2_symint_out(const Tensor& self, at::OptionalSymIntArrayRef s,
                     IntArrayRef dim, c10::optional<c10::string_view> norm, Tensor& out) {
  return native::fft_fftn_symint_out(self, s, dim, std::move(norm), out);
}

Tensor fft_ifft2_symint(const Tensor& self, at::OptionalSymIntArrayRef s,
                IntArrayRef dim, c10::optional<c10::string_view> norm) {
  return native::fft_ifftn_symint(self, s, dim, std::move(norm));
}

Tensor& fft_ifft2_symint_out(const Tensor& self, at::OptionalSymIntArrayRef s,
                      IntArrayRef dim, c10::optional<c10::string_view> norm, Tensor& out) {
  return native::fft_ifftn_symint_out(self, s, dim, std::move(norm), out);
}

Tensor fft_rfft2_symint(const Tensor& self, at::OptionalSymIntArrayRef s,
                IntArrayRef dim, c10::optional<c10::string_view> norm) {
  return native::fft_rfftn_symint(self, s, dim, std::move(norm));
}

Tensor& fft_rfft2_symint_out(const Tensor& self, at::OptionalSymIntArrayRef s,
                      IntArrayRef dim, c10::optional<c10::string_view> norm, Tensor& out) {
  return native::fft_rfftn_symint_out(self, s, dim, std::move(norm), out);
}

Tensor fft_irfft2_symint(const Tensor& self, at::OptionalSymIntArrayRef s,
                  IntArrayRef dim, c10::optional<c10::string_view> norm) {
  return native::fft_irfftn_symint(self, s, dim, std::move(norm));
}

Tensor& fft_irfft2_symint_out(const Tensor& self, at::OptionalSymIntArrayRef s,
                       IntArrayRef dim, c10::optional<c10::string_view> norm, Tensor& out) {
  return native::fft_irfftn_symint_out(self, s, dim, std::move(norm), out);
}

const Tensor& fft_hfft2_symint_out(
    const Tensor& self, at::OptionalSymIntArrayRef s, IntArrayRef dim,
    c10::optional<c10::string_view> norm, const Tensor& out) {
  return native::fft_hfftn_symint_out(self, s, dim, std::move(norm), out);
}

Tensor fft_hfft2_symint(const Tensor& self, at::OptionalSymIntArrayRef s,
                 IntArrayRef dim, c10::optional<c10::string_view> norm) {
  return native::fft_hfftn_symint(self, s, dim, std::move(norm));
}

const Tensor& fft_ihfft2_symint_out(
    const Tensor& self, at::OptionalSymIntArrayRef s, IntArrayRef dim,
    c10::optional<c10::string_view> norm, const Tensor& out) {
  return native::fft_ihfftn_symint_out(self, s, dim, std::move(norm), out);
}

Tensor fft_ihfft2_symint(const Tensor& self, at::OptionalSymIntArrayRef s,
                  IntArrayRef dim, c10::optional<c10::string_view> norm) {
  return native::fft_ihfftn_symint(self, s, dim, std::move(norm));
}

Tensor& fft_fftfreq_out(int64_t n, double d, Tensor& out) {
  ScalarType dtype = out.scalar_type();
  TORCH_CHECK(at::isFloatingType(dtype) || at::isComplexType(dtype),
              "fftfreq requires a floating point or complex dtype");
  // TODO: arange doesn't have complex support
  at::arange_out(out, n);
  auto right_slice = out.slice(0, (n + 1) / 2, 0);
  at::arange_out(right_slice, -(n/2), 0, 1);
  return out.mul_(1.0 / (n * d));  // Slightly faster than div_(n*d)
}

Tensor fft_fftfreq(int64_t n, double d,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory) {
  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  auto out = at::empty({n}, options);
  return native::fft_fftfreq_out(n, d, out);
}

Tensor& fft_rfftfreq_out(int64_t n, double d, Tensor& out) {
  ScalarType dtype = out.scalar_type();
  TORCH_CHECK(at::isFloatingType(dtype) || at::isComplexType(dtype),
              "rfftfreq requires a floating point or complex dtype");
  // TODO: arange doesn't have complex support
  native::arange_out(n/2 + 1, out);
  return out.mul_(1.0 / (n * d));  // Slightly faster than div_(n*d)
}

Tensor fft_rfftfreq(int64_t n, double d,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory) {
  // See [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  auto out = at::empty({n/2 + 1}, options);
  return native::fft_rfftfreq_out(n, d, out);
}

// If an array dim is specified, wraps them according to self.dim().
// Otherwise returns a vector of all dims.
static DimVector default_alldims(const Tensor& self, at::OptionalIntArrayRef dim_opt) {
  DimVector dim;
  if (dim_opt) {
    IntArrayRef dim_unwrapped = *dim_opt;
    dim.resize(dim_unwrapped.size());
    for (const auto i : c10::irange(dim.size())) {
      dim[i] = maybe_wrap_dim(dim_unwrapped[i], self.dim(), /*wrap_scalars=*/false);
    }
  } else {
    dim.resize(self.dim());
    std::iota(dim.begin(), dim.end(), 0);
  }
  return dim;
}

Tensor fft_fftshift(const Tensor& x, at::OptionalIntArrayRef dim_opt) {
  auto dim = default_alldims(x, dim_opt);

  SymIntArrayRef x_sizes = x.sym_sizes();
  SymDimVector shift(dim.size());
  for (const auto i : c10::irange(dim.size())) {
    shift[i] = x_sizes[dim[i]] / 2;
  }

  return at::roll_symint(x, shift, dim);
}

Tensor fft_ifftshift(const Tensor& x, at::OptionalIntArrayRef dim_opt) {
  auto dim = default_alldims(x, dim_opt);

  SymIntArrayRef x_sizes = x.sym_sizes();
  SymDimVector shift(dim.size());
  for (const auto i : c10::irange(dim.size())) {
    shift[i] = (x_sizes[dim[i]] + 1) / 2;
  }

  return at::roll_symint(x, shift, dim);
}


// We call the following methods via CUDA hooks because they are really only
// valid when CUDA is available. See native/cuda/CuFFTPlanCache.h for more details.
int64_t _cufft_get_plan_cache_max_size(DeviceIndex device_index) {
  return detail::getCUDAHooks().cuFFTGetPlanCacheMaxSize(device_index);
}

void _cufft_set_plan_cache_max_size(DeviceIndex device_index, int64_t max_size) {
  detail::getCUDAHooks().cuFFTSetPlanCacheMaxSize(device_index, max_size);
}

int64_t _cufft_get_plan_cache_size(DeviceIndex device_index) {
  return detail::getCUDAHooks().cuFFTGetPlanCacheSize(device_index);
}

void _cufft_clear_plan_cache(DeviceIndex device_index) {
  detail::getCUDAHooks().cuFFTClearPlanCache(device_index);
}

template <typename Stream, typename T>
static Stream& write_opt(Stream& SS, const optional<T>& value) {
  if (value) {
    SS << *value;
  } else {
    SS << "None";
  }
  return SS;
}

/* Short-time Fourier Transform, for signal analysis.
 *
 * This is modeled after librosa but with support for complex time-domain
 * signals and complex windows.
 */
Tensor stft(const Tensor& self, const int64_t n_fft, const optional<int64_t> hop_lengthOpt,
            const optional<int64_t> win_lengthOpt, const c10::optional<Tensor>& window_opt,
            const bool center, c10::string_view mode, const bool normalized,
            const optional<bool> onesidedOpt, const optional<bool> return_complexOpt) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> window_maybe_owned = at::borrow_from_optional_tensor(window_opt);
  const Tensor& window = *window_maybe_owned;

  #define REPR(SS) \
    SS << "stft(" << self.toString() << self.sizes() << ", n_fft=" << n_fft \
       << ", hop_length=" << hop_length << ", win_length=" << win_length \
       << ", window="; \
    if (window.defined()) { \
      SS << window.toString() << "{" << window.sizes() << "}"; \
    } else { \
      SS << "None"; \
    } \
    SS << ", normalized=" << normalized << ", onesided="; \
    write_opt(SS, onesidedOpt) << ", return_complex="; \
    write_opt(SS, return_complexOpt) << ") "

  TORCH_CHECK(!window.defined() || window.device() == self.device(),
              "stft input and window must be on the same device but got self on ",
              self.device(), " and window on ", window.device())

  // default_init hop_length and win_length
  auto hop_length = hop_lengthOpt.value_or(n_fft >> 2);
  auto win_length = win_lengthOpt.value_or(n_fft);
  const bool return_complex = return_complexOpt.value_or(
      self.is_complex() || (window.defined() && window.is_complex()));
  if (!return_complex) {
    TORCH_CHECK(return_complexOpt.has_value(),
        "stft requires the return_complex parameter be given for real inputs, "
        "and will further require that return_complex=True in a future PyTorch release.");


    TORCH_WARN_ONCE(
        "stft with return_complex=False is deprecated. In a future pytorch "
        "release, stft will return complex tensors for all inputs, and "
        "return_complex=False will raise an error.\n"
        "Note: you can still call torch.view_as_real on the complex output to "
        "recover the old return format.");
  }

  if (!at::isFloatingType(self.scalar_type()) && !at::isComplexType(self.scalar_type())) {
    std::ostringstream ss;
    REPR(ss) << ": expected a tensor of floating point or complex values";
    AT_ERROR(ss.str());
  }
  if (self.dim() > 2 || self.dim() < 1) {
    std::ostringstream ss;
    REPR(ss) << ": expected a 1D or 2D tensor";
    AT_ERROR(ss.str());
  }
  Tensor input = self;
  if (self.dim() == 1) {
    input = input.unsqueeze(0);
  }

  if (center) {
    const auto input_shape = input.sizes();
    const auto input_dim = input_shape.size();
    const auto extra_dims = std::max(size_t{3}, input_dim) - input_dim;
    const auto pad_amount = n_fft / 2;

    DimVector extended_shape(extra_dims, 1);
    extended_shape.append(input_shape.begin(), input_shape.end());
    input = at::pad(input.view(extended_shape), {pad_amount, pad_amount}, mode);
    input = input.view(IntArrayRef(input.sizes()).slice(extra_dims));
  }

  int64_t batch = input.size(0);
  int64_t len = input.size(1);
  if (n_fft <= 0 || n_fft > len) {
    std::ostringstream ss;
    REPR(ss) << ": expected 0 < n_fft < " << len
             << ", but got n_fft=" << win_length;
    AT_ERROR(ss.str());
  }
  if (hop_length <= 0) {
    std::ostringstream ss;
    REPR(ss) << ": expected hop_length > 0, but got hop_length=" << hop_length;
    AT_ERROR(ss.str());
  }
  if (win_length <= 0 || win_length > n_fft) {
    std::ostringstream ss;
    REPR(ss) << ": expected 0 < win_length <= n_fft, but got win_length="
             << win_length;
    AT_ERROR(ss.str());
  }
  if (window.defined() && (window.dim() != 1 || window.size(0) != win_length)) {
    std::ostringstream ss;
    REPR(ss) << ": expected a 1D window tensor of size equal to win_length="
             << win_length << ", but got window with size " << window.sizes();
    AT_ERROR(ss.str());
  }
  #undef REPR
  auto window_ = window;
  if (win_length < n_fft) {
    // pad center
    auto left = (n_fft - win_length) / 2;
    if (window.defined()) {
      window_ = at::zeros({n_fft}, window.options());
      window_.narrow(0, left, win_length).copy_(window);
    } else {
      window_ = at::zeros({n_fft}, self.options());
      window_.narrow(0, left, win_length).fill_(1);
    }
  }
  int64_t n_frames = 1 + (len - n_fft) / hop_length;
  // time2col
  input = input.as_strided(
    {batch, n_frames, n_fft},
    {input.stride(0), hop_length * input.stride(1), input.stride(1)}
  );
  if (window_.defined()) {
    input = input.mul(window_);
  }

  // FFT and transpose to get (batch x fft_size x num_frames)
  const bool complex_fft = input.is_complex();
  const auto onesided = onesidedOpt.value_or(!complex_fft);

  const fft_norm_mode norm = normalized ? fft_norm_mode::by_root_n : fft_norm_mode::none;
  Tensor out;
  if (complex_fft) {
    TORCH_CHECK(!onesided, "Cannot have onesided output if window or input is complex");
    out = at::_fft_c2c(input, input.dim() - 1, static_cast<int64_t>(norm), /*forward=*/true);
  } else {
    out = at::_fft_r2c(input, input.dim() - 1, static_cast<int64_t>(norm), onesided);
  }
  out.transpose_(1, 2);

  if (self.dim() == 1) {
    out.squeeze_(0);
  }

  if (return_complex) {
    return out;
  } else {
    return at::view_as_real(out);
  }
}

Tensor stft(
    const Tensor& self, const int64_t n_fft, const optional<int64_t> hop_lengthOpt,
    const optional<int64_t> win_lengthOpt, const c10::optional<Tensor>& window_opt,
    const bool normalized,
    const optional<bool> onesidedOpt, const optional<bool> return_complexOpt) {
  return at::stft(
      self, n_fft, hop_lengthOpt, win_lengthOpt, window_opt,
      /*center=*/false, /*mode=*/"constant", normalized, onesidedOpt,
      return_complexOpt);
}

// Create complex tensor from the old style of real tensor with size=(..., 2)
// This is to support istft in the transition to requiring complex input.
// NOTE: This may return a view of the input tensor, or might clone if necessary
static Tensor as_complex(const Tensor& self) {
  const bool can_view_as_complex = [&]{
    auto strides = self.strides();
    for (const auto i : c10::irange(static_cast<int64_t>(strides.size()) - 1)) {
      if (strides[i] % 2 != 0) {
        return false;
      }
    }
    return strides.back() == 1 && self.storage_offset() % 2 == 0;
  }();
  return at::view_as_complex(can_view_as_complex ? self : self.clone(MemoryFormat::Contiguous));
}

/* Inverse Short-time Fourier Transform
 *
 * This is modeled after librosa but with support for complex time-domain
 * signals and complex windows.
 */
Tensor istft(const Tensor& self, const int64_t n_fft, const optional<int64_t> hop_lengthOpt,
             const optional<int64_t> win_lengthOpt, const c10::optional<Tensor>& window_opt,
             const bool center, const bool normalized, const c10::optional<bool> onesidedOpt,
             const optional<int64_t> lengthOpt, const bool return_complex) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> window_maybe_owned = at::borrow_from_optional_tensor(window_opt);
  const Tensor& window = *window_maybe_owned;

  #define REPR(SS) \
    SS << "istft(" << self.toString() << self.sizes() << ", n_fft=" << n_fft \
       << ", hop_length=" << hop_length << ", win_length=" << win_length \
       << ", window="; \
    if (window.defined()) { \
      SS << window.toString() << "{" << window.sizes() << "}"; \
    } else { \
      SS << "None"; \
    } \
    SS << ", center=" << center << ", normalized=" << normalized << ", onesided="; \
    write_opt(SS, onesidedOpt) << ", length="; \
    write_opt(SS, lengthOpt) << ", return_complex=" << return_complex << ") "

  TORCH_CHECK(!window.defined() || window.device() == self.device(),
              "istft input and window must be on the same device but got self on ",
              self.device(), " and window on ", window.device())

  // default_init hop_length and win_length
  const auto hop_length = hop_lengthOpt.value_or(n_fft >> 2);
  const auto win_length = win_lengthOpt.value_or(n_fft);

  TORCH_CHECK(self.is_complex(),
              "istft requires a complex-valued input tensor matching the "
              "output from stft with return_complex=True.");
  Tensor input = at::view_as_real(self.resolve_conj());
  const auto input_dim = input.dim();
  const auto n_frames = input.size(-2);
  const auto fft_size = input.size(-3);

  const auto expected_output_signal_len = n_fft + hop_length * (n_frames - 1);

  const auto options = at::device(input.device()).dtype(input.dtype());
  if (input.numel() == 0) {
    std::ostringstream ss;
    REPR(ss) << ": input tensor cannot be empty.";
    AT_ERROR(ss.str());
  }
  if (input_dim != 3 && input_dim != 4) {
    std::ostringstream ss;
    REPR(ss) << ": expected a tensor with 3 or 4 dimensions, but got " << input_dim;
    AT_ERROR(ss.str());
  }
  if (input.size(-1) != 2) {
    std::ostringstream ss;
    REPR(ss) << ": expected the last dimension to be 2 (corresponding to real and imaginary parts), but got " << self.size(-1);
    AT_ERROR(ss.str());
  }

  const bool onesided = onesidedOpt.value_or(fft_size != n_fft);
  if (onesided) {
    if (n_fft / 2 + 1 != fft_size) {
      std::ostringstream ss;
      REPR(ss) << ": expected the frequency dimension (3rd to the last) of the input tensor to match n_fft / 2 + 1 when onesided=True, but got " << fft_size;
      AT_ERROR(ss.str());
    }
  } else {
    if (n_fft != fft_size) {
      std::ostringstream ss;
      REPR(ss) << ": expected the frequency dimension (3rd to the last) of the input tensor to match n_fft when onesided=False, but got " << fft_size;
      AT_ERROR(ss.str());
    }
  }

  if (!(0 < hop_length && hop_length <= win_length)) {
    std::ostringstream ss;
    REPR(ss) << ": expected 0 < hop_length <= win_length";
    AT_ERROR(ss.str());
  }

  if (!(0 < win_length && win_length <= n_fft)) {
    std::ostringstream ss;
    REPR(ss) << ": expected 0 < win_length <= n_fft";
    AT_ERROR(ss.str());
  }
  if (window.defined()) {
    if (window.dim() != 1 || window.size(0) != win_length) {
      std::ostringstream ss;
      REPR(ss) << ": Invalid window shape. window has to be 1D and length of `win_length`";
      AT_ERROR(ss.str());
    }
  }

  Tensor window_tmp = window.defined() ? window : at::ones({win_length,}, options);
  if (win_length != n_fft) {
    // center window by padding zeros on right and left side
    int64_t left = (n_fft - win_length) / 2;
    window_tmp = at::constant_pad_nd(window_tmp, {left, n_fft - win_length - left}, 0);
    TORCH_INTERNAL_ASSERT(window_tmp.size(0) == n_fft);
  }

  if (input_dim == 3) {
    input = input.unsqueeze(0);
  }

  input = as_complex(input.transpose(1, 2));  // size: (channel, n_frames, fft_size)

  const fft_norm_mode norm = normalized ? fft_norm_mode::by_root_n : fft_norm_mode::by_n;
  if (return_complex) {
    TORCH_CHECK(!onesided, "Cannot have onesided output if window or input is complex");
    input = at::_fft_c2c(input, input.dim() - 1, static_cast<int64_t>(norm), /*forward=*/false);  // size: (channel, n_frames, n_fft)
  } else {
    TORCH_CHECK(!window.defined() || !window.is_complex(),
                "Complex windows are incompatible with return_complex=False");
    if (!onesided) {
      input = input.slice(-1, 0, n_fft / 2 + 1);
    }
    input = at::_fft_c2r(input, input.dim() - 1, static_cast<int64_t>(norm), n_fft);  // size: (channel, n_frames, n_fft)
  }
  TORCH_INTERNAL_ASSERT(input.size(2) == n_fft);

  Tensor y_tmp = input * window_tmp.view({1, 1, n_fft});  // size: (channel, n_frames, n_fft)

  Tensor y = at::unfold_backward(
    y_tmp,
    /*input_sizes=*/{y_tmp.size(0), expected_output_signal_len},
    /*dim=*/1,
    /*size=*/n_fft,
    /*step=*/hop_length);
  window_tmp = window_tmp.pow(2).expand({1, n_frames, n_fft});  // size: (1, n_frames, n_fft)
  Tensor window_envelop = at::unfold_backward(
    window_tmp,
    /*input_sizes=*/{1, expected_output_signal_len},
    /*dim=*/1,
    /*size=*/n_fft,
    /*step=*/hop_length); // size: (1, expected_output_signal_len)

  TORCH_INTERNAL_ASSERT(expected_output_signal_len == y.size(1));
  TORCH_INTERNAL_ASSERT(expected_output_signal_len == window_envelop.size(1));

  // We need to trim the front padding away if centered
  const auto start = center ? n_fft / 2 : 0;
  const auto end = [&] () -> int64_t {
    if (lengthOpt.has_value()) {
      return start + *lengthOpt;
    }
    if (center) {
      return -(n_fft / 2);
    }
    return expected_output_signal_len;
  }();

  y = y.slice(1, start, end, 1);
  window_envelop = window_envelop.slice(1, start, end, 1);
  const auto window_envelop_lowest = window_envelop.abs().min().lt(1e-11);
  if (at::is_scalar_tensor_true(window_envelop_lowest)) {
    std::ostringstream ss;
    REPR(ss) << "window overlap add min: " << window_envelop_lowest;
    AT_ERROR(ss.str());
  }

  y = (y / window_envelop);  // size: (channel, expected_output_signal_len)
  if (input_dim == 3) {
    y = y.squeeze(0);
  }
  // zero padding if the given lengthOpt is longer than expected
  if(end > expected_output_signal_len) {
    TORCH_WARN_ONCE(
      "The length of signal is shorter than the length parameter. Result is being padded with zeros in the tail. "
      "Please check your center and hop_length settings."
    );
    y = at::constant_pad_nd(y, {0, end - expected_output_signal_len}, 0);
  }
  return y;

#undef REPR
}

static Tensor istft(const Tensor& self, const int64_t n_fft, const optional<int64_t> hop_lengthOpt,
             const optional<int64_t> win_lengthOpt, const Tensor& window,
             const bool center, const bool normalized, const optional<bool> onesidedOpt,
             const optional<int64_t> lengthOpt) {
  return at::native::istft(
      self, n_fft, hop_lengthOpt, win_lengthOpt, window, center, normalized,
      onesidedOpt, lengthOpt, /*return_complex=*/false);
}

void _fft_fill_with_conjugate_symmetry_(const Tensor& input, IntArrayRef dim_) {
  const auto input_sizes = input.sizes();
  const auto input_strides = input.strides();
  TORCH_CHECK(!dim_.empty());
  DimVector dim(dim_.begin(), dim_.end());
  at::maybe_wrap_dims(dim, input_strides.size(), /*wrap_scalars=*/false);

  if (input.numel() == 0 || input_sizes[dim.back()] <= 2) {
    return;  // No elements need writing
  }

  // Small dimensions may be treated as batch dims since they don't get mirrored
  dim.erase(
      std::remove_if(dim.begin(), dim.end(), [&](int64_t dim) {
        return (input_sizes[dim] <= 2);
      }),
      dim.end());

  // Use TensorIterator to coalesce batch dimensions
  // NOTE: Can't use TensorIterator loops because we need negative strides
  auto iter = TensorIteratorConfig()
      .add_output(input)
      .add_input(input)
      .resize_outputs(false)
      .declare_static_shape(input_sizes, dim)
      .build();

  const auto iter_strides = iter.strides(0);
  const auto iter_sizes = iter.shape();
  const auto ndim = static_cast<int64_t>(iter_strides.size() + dim.size());
  DimVector in_strides(ndim), signal_half_sizes(ndim);
  // Take coalesced batch dimensions from TensorIterator
  std::copy(iter_strides.begin(), iter_strides.end(), in_strides.begin());
  std::copy(iter_sizes.begin(), iter_sizes.end(), signal_half_sizes.begin());

  // Take transformed dimensions directly from the input
  const auto element_size = iter.element_size(0);
  for (const auto i : c10::irange(dim.size())) {
    // Convert to byte strides to match TensorIterator
    in_strides[iter_strides.size() + i] = input_strides[dim[i]] * element_size;
    signal_half_sizes[iter_strides.size() + i] = input_sizes[dim[i]];
  }

  // For the last dimension, use negative strides to perform the mirroring
  signal_half_sizes.back() = (input_sizes[dim.back()] - 1) / 2;
  auto out_strides = in_strides;
  out_strides.back() *= -1;

  auto* data_ptr = static_cast<char*>(input.data_ptr());
  const auto* in_data = data_ptr + input_strides[dim.back()] * element_size;
  auto* out_data = data_ptr + (
      input_strides[dim.back()] * (input_sizes[dim.back()] - 1) * element_size);

  // Reorder dimensions by stride to maximize data locality
  DimVector dim_permute(ndim);
  std::iota(dim_permute.begin(), dim_permute.end(), 0);
  std::sort(dim_permute.begin(), dim_permute.end(),
      [&](auto dim1, auto dim2) {
        return in_strides[dim1] < in_strides[dim2];
      });

  DimVector temp(ndim);
  auto apply_permutation = [&] (DimVector & vec) {
    // Do permuted index copy into a temporary, then copy back
    for (const auto i : c10::irange(ndim)) {
      temp[i] = vec[dim_permute[i]];
    }
    vec = temp;
  };
  apply_permutation(in_strides);
  apply_permutation(out_strides);
  apply_permutation(signal_half_sizes);

  // Find dims.slice(dims.size() - 1) in the new permuted order.
  // These are the dimensions that need explicit Hermitian mirroring
  DimVector mirror_dims;
  mirror_dims.reserve(dim.size() - 1);
  for (const auto i : c10::irange(ndim)) {
    if (dim_permute[i] >= static_cast<int64_t>(iter_strides.size()) &&  // Not a batch dimension
        dim_permute[i] != ndim - 1) {  // Not the last dim, which is mirrored separately with negative strides
      mirror_dims.push_back(i);
    }
  }
  TORCH_INTERNAL_ASSERT(mirror_dims.size() == dim.size() - 1);

  // Dispatch to CPU or CUDA kernel to do the actual conjugate mirroring
  fft_fill_with_conjugate_symmetry_stub(
      input.device().type(), input.scalar_type(),
      mirror_dims, signal_half_sizes, in_strides, in_data, out_strides, out_data);
}

DEFINE_DISPATCH(fft_fill_with_conjugate_symmetry_stub);

}} // at::native
