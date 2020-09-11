// define constants like M_PI and C keywords for MSVC
#ifdef _MSC_VER
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>
#endif

#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/native/SpectralOpsUtils.h>

#include <algorithm>
#include <vector>
#include <cmath>

namespace at { namespace native {

// Common code for all FFT functions
static inline Tensor _fft(
    const Tensor &self, int64_t signal_ndim, bool complex_input,
    const bool complex_output, bool inverse, IntArrayRef signal_sizes,
    fft_norm_mode normalization, bool onesided);

namespace {

// Promote inputs to FFT functions
// * Integers are promoted to the default floating type
// * If require_complex=True, all types are promoted to complex
// * Raises an error for half-precision dtypes to allow future support
ScalarType promote_type_fft(ScalarType type, bool require_complex) {
  if (at::isComplexType(type)) {
    return type;
  }
  // Promote integral to default float type
  if (!at::isFloatingType(type)) {
    type = c10::typeMetaToScalarType(c10::get_default_dtype());
  }

  TORCH_CHECK(type == kFloat || type == kDouble, "Unsupported dtype ", type);

  if (!require_complex) {
    return type;
  }

  // Promote to complex
  switch (type) {
  case kFloat: return kComplexFloat;
  case kDouble: return kComplexDouble;
  default: TORCH_INTERNAL_ASSERT(false, "Unhandled dtype");
  }
}

// Promote a tensor's dtype according to promote_type_fft
Tensor promote_tensor_fft(const Tensor& t, bool require_complex=false) {
  auto cur_type = t.scalar_type();
  auto new_type = promote_type_fft(cur_type, require_complex);
  return (cur_type == new_type) ? t : t.to(new_type);
}

// Convert NumPy compatible normalization mode string to enum values
// NOTE: NumPy's normalization modes have direction-specific meanings. For example,
// "forward" translates to `by_n` for a forward transform and `none` for backward.
fft_norm_mode norm_from_string(c10::optional<std::string> norm, bool forward) {
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
Tensor resize_fft_input(Tensor x, IntArrayRef dims, IntArrayRef sizes) {
  TORCH_INTERNAL_ASSERT(dims.size() == sizes.size());
  bool must_copy = false;
  auto x_sizes = x.sizes();
  DimVector pad_amount(x_sizes.size() * 2);
  for (int64_t i = 0; i < dims.size(); ++i) {
    if (sizes[i] == -1) {
      continue;
    }

    if (x_sizes[dims[i]] < sizes[i]) {
      must_copy = true;
      auto pad_idx = pad_amount.size() - 2 * dims[i] - 1;
      pad_amount[pad_idx] = sizes[i] - x_sizes[dims[i]];
    }

    if (x_sizes[dims[i]] > sizes[i]) {
      x = x.slice(dims[i], 0, sizes[i]);
    }
  }

  // Only call pad if necessary since pad copies the entire tensor
  return must_copy ? at::constant_pad_nd(x, pad_amount) : x;
}

// Complex to real FFT
Tensor fft_c2r(Tensor input, c10::optional<int64_t> n_opt,
               int64_t unwrapped_dim, c10::optional<std::string> norm_str,
               bool forward) {
  input = promote_tensor_fft(input, /*require_complex=*/true);
  const auto input_dim = input.dim();
  const auto dim = maybe_wrap_dim(unwrapped_dim, input_dim);
  const auto n = n_opt.value_or(2*(input.sizes()[dim] - 1));
  TORCH_CHECK(n >= 1, "Invalid number of data points (", n, ") specified");
  if (n_opt) {
    input = resize_fft_input(input, dim, n/2 + 1);
  }
  // _fft only operates on the last dim, so transpose the selected dim to the end
  const bool must_transpose = (dim != input_dim - 1);
  if (must_transpose) {
    input = at::transpose(input, -1, dim);
  }
  const auto norm = norm_from_string(norm_str, forward);
  if (forward) {
    // FIXME: _fft does not support complex_output=false with inverse=false
    input = at::conj(input);
  }
  auto out = _fft(at::view_as_real(input),
                  /*signal_ndim=*/1, /*complex_input=*/true,
                  /*complex_output=*/false, /*inverse=*/true,
                  /*signal_sizes=*/{n}, /*normalization=*/norm,
                  /*onesided=*/true);
  if (must_transpose) {
    out = at::transpose(out, -1, dim);
  }
  return out;
}

// Real to complex FFT
Tensor fft_r2c(Tensor input, c10::optional<int64_t> n_opt,
               int64_t unwrapped_dim, c10::optional<std::string> norm_str,
               bool forward, bool onesided) {
  TORCH_CHECK(!input.is_complex(), "Expected a real input tensor to FFT");
  input = promote_tensor_fft(input);
  const auto input_dim = input.dim();
  const auto dim = maybe_wrap_dim(unwrapped_dim, input_dim);
  const auto n = n_opt.value_or(input.sizes()[dim]);
  TORCH_CHECK(n >= 1, "Invalid number of data points (", n, ") specified");
  if (n_opt) {
    input = resize_fft_input(input, dim, n);
  }
  // _fft only operates on the last dim, so transpose the selected dim to the end
  const bool must_transpose = (dim != input_dim - 1);
  if (must_transpose) {
    input = at::transpose(input, -1, dim);
  }
  const auto norm = norm_from_string(norm_str, forward);
  auto out = _fft(input, /*signal_ndim=*/1, /*complex_input=*/false,
                  /*complex_output=*/true, /*inverse=*/false,
                  /*signal_sizes=*/{n}, /*normalization=*/norm,
                  /*onesided=*/onesided);
  out = at::view_as_complex(out);
  if (must_transpose) {
    out = at::transpose(out, -1, dim);
  }
  if (!forward) {
    // FIXME: _fft does not support complex_input=false with inverse=true
    out = at::conj(out);
  }
  return out;
}

// Complex to complex FFT
Tensor fft_c2c(Tensor input, c10::optional<int64_t> n_opt,
               int64_t unwrapped_dim, c10::optional<std::string> norm_str,
               bool forward) {
  TORCH_CHECK(input.is_complex(), "Expected a complex input tensor to FFT");
  const auto input_dim = input.dim();
  const auto dim = maybe_wrap_dim(unwrapped_dim, input_dim);
  const auto n = n_opt.value_or(input.sizes()[dim]);
  TORCH_CHECK(n >= 1, "Invalid number of data points (", n, ") specified");
  if (n_opt) {
    input = resize_fft_input(input, dim, n);
  }
  // _fft only operates on the last dim, so transpose the selected dim to the end
  const bool must_transpose = (dim != input_dim - 1);
  if (must_transpose) {
    input = at::transpose(input, -1, dim);
  }
  const auto norm = norm_from_string(norm_str, forward);
  auto out = _fft(at::view_as_real(input),
                  /*signal_ndim=*/1, /*complex_input=*/true,
                  /*complex_output=*/true, /*inverse=*/!forward,
                  /*signal_sizes=*/{}, /*normalization=*/norm,
                  /*onesided=*/false);
  out = at::view_as_complex(out);
  if (must_transpose) {
    out = at::transpose(out, -1, dim);
  }
  return out;
}

}

// torch.fft.fft, analogous to NumPy's numpy.fft.fft
Tensor fft_fft(const Tensor& self, c10::optional<int64_t> n, int64_t dim,
               c10::optional<std::string> norm) {
  return self.is_complex() ? 
    fft_c2c(self, n, dim, norm, /*forward=*/true) :
    fft_r2c(self, n, dim, norm, /*forward=*/true, /*onesided=*/false);
}

Tensor fft_ifft(const Tensor& self, c10::optional<int64_t> n, int64_t dim,
                c10::optional<std::string> norm) {
  return self.is_complex() ? 
    fft_c2c(self, n, dim, norm, /*forward=*/false) :
    fft_r2c(self, n, dim, norm, /*forward=*/false, /*onesided=*/false);
}

Tensor fft_rfft(const Tensor& self, c10::optional<int64_t> n, int64_t dim,
                c10::optional<std::string> norm) {
  return fft_r2c(self, n, dim, norm, /*forward=*/true, /*onesided=*/true);
}

Tensor fft_irfft(const Tensor& self, c10::optional<int64_t> n, int64_t dim,
                 c10::optional<std::string> norm) {
  return fft_c2r(self, n, dim, norm, /*forward=*/false);
}

Tensor fft_hfft(const Tensor& self, c10::optional<int64_t> n, int64_t dim,
                c10::optional<std::string> norm) {
  return fft_c2r(self, n, dim, norm, /*forward=*/true);
}

Tensor fft_ihfft(const Tensor& self, c10::optional<int64_t> n, int64_t dim,
                 c10::optional<std::string> norm) {
  return fft_r2c(self, n, dim, norm, /*forward=*/false, /*onesided=*/true);
}


// This is a pass-through wrapper function that does the size check and
// inferences. The actual forward implementation function is called
// at::_fft_with_size which dispatches to _fft_cufft (CUDA) or _fft_mkl (CPU).
static inline Tensor _fft(const Tensor &self, const int64_t signal_ndim,
           const bool complex_input, const bool complex_output,
           const bool inverse, IntArrayRef signal_sizes,
           const fft_norm_mode normalization, const bool onesided) {

  TORCH_CHECK(signal_ndim >= 1 && signal_ndim <= 3,
           "Expected signal_ndim to be 1, 2, or 3, but got signal_ndim=",
           signal_ndim);
  TORCH_CHECK(at::isFloatingType(self.scalar_type()),
           "Expected an input tensor of floating types, but got input=",
           self.toString(), self.sizes());

  auto signal_tensor_ndim = signal_ndim + static_cast<int64_t>(complex_input);  // add complex dim
  if (self.dim() < signal_tensor_ndim) {
    std::ostringstream ss;
    ss << "Given signal_ndim=" << signal_ndim << ", expected an input tensor "
       << "of at least " << signal_tensor_ndim << "D";
    if (complex_input) {
      ss << " (complex input adds an extra dimension)";
    }
    ss << ", but got input=" << self.toString() << self.sizes();
    AT_ERROR(ss.str());
  }

  auto self_shape = self.sizes();
  auto batch_ndim = self.dim() - signal_tensor_ndim;

  Tensor input = self;
  // flatten the batch dims
  if (batch_ndim == 0) {
    // slightly faster path for non-batch mode
    input = input.unsqueeze(0);
  } else if (batch_ndim > 1) {
    std::vector<int64_t> flatten_input_shape(signal_tensor_ndim + 1);
    std::copy(self_shape.begin() + batch_ndim, self_shape.end(), flatten_input_shape.begin() + 1);
    flatten_input_shape[0] = -1;
    input = input.reshape(flatten_input_shape);

  }

  // now we assume that input is batched as [ B x signal_dims... ]

  if (complex_input) {
    TORCH_CHECK(input.size(signal_ndim + 1) == 2,
             "Expected an input tensor with a last dimension of size 2 "
             "representing real + imaginary components, but got input ",
             self.toString(), self.sizes());
  }

  // build signal_sizes and output_size
  TORCH_CHECK(signal_sizes.size() == 0 || static_cast<int64_t>(signal_sizes.size()) == signal_ndim,
           "Expected signal_sizes to be empty (default) or of signal_ndim=",
           signal_ndim, "D, but got signal_sizes=", signal_sizes);
  std::vector<int64_t> output_sizes(signal_ndim + 1 + static_cast<int64_t>(complex_output));
  output_sizes[0] = input.size(0);  // batch size
  std::vector<int64_t> checked_signal_sizes(signal_ndim);
  for (int64_t i = 0; i < signal_ndim; i++) {
    int64_t input_size = input.size(i + 1);
    if (i == signal_ndim - 1 && onesided && complex_input && !complex_output) {
      // If last dim and complex-to-real onesided, input is only half of
      // signal, and we need to infer basing on signal_sizes, if given
      // See native/SpectralOpsUtils.h for detailed description.
      int64_t inferred_size;
      if (signal_sizes.size() > 0) {
        inferred_size = infer_ft_complex_to_real_onesided_size(input_size, signal_sizes[i]);
      } else {
        inferred_size = infer_ft_complex_to_real_onesided_size(input_size);
      }
      checked_signal_sizes[i] = inferred_size;
      output_sizes[i + 1] = inferred_size;
    } else {
      if (i == signal_ndim - 1 && onesided && !complex_input && complex_output) {
        // if last dim and real-to-complex onesided, output should be only
        // half of the signal, and we need to infer using input_size
        output_sizes[i + 1] = infer_ft_real_to_complex_onesided_size(input_size);
      } else {
        output_sizes[i + 1] = input_size;
      }
      checked_signal_sizes[i] = input_size;
      TORCH_CHECK(signal_sizes.size() == 0 || signal_sizes[i] == checked_signal_sizes[i],
               "Expected given signal_sizes=", signal_sizes," to have same "
               "shape with input at signal dimension ", i, ", but got "
               "signal_sizes=", signal_sizes, " and input=", self.toString(),
               self.sizes());
    }
  }
  if (complex_output) {
    output_sizes[signal_ndim + 1] = 2;
  }

  Tensor output = at::_fft_with_size(input, signal_ndim, complex_input,
                                     complex_output, inverse,
                                     checked_signal_sizes,
                                     static_cast<int64_t>(normalization),
                                     onesided,
                                     output_sizes);

  // unflatten the batch dims
  if (batch_ndim == 0) {
    // slightly faster path for non-batch mode
    output = output.squeeze(0);
  } else if (batch_ndim > 1) {
    auto output_ndim = self.dim() + static_cast<int64_t>(complex_output) - static_cast<int64_t>(complex_input);
    std::vector<int64_t> unflatten_output_shape(output_ndim);
    std::copy(self_shape.begin(), self_shape.begin() + batch_ndim, unflatten_output_shape.begin());
    std::copy(output_sizes.begin() + 1, output_sizes.end(), unflatten_output_shape.begin() + batch_ndim);
    output = output.reshape(unflatten_output_shape);
  }
  return output;
}

// Wrapper to preserve the historic signature of _fft_with_size
// NOTE: This is only used for torchscript backwards compatibility and the new
// signature with normalization modes should be used in all other cases
Tensor _fft_with_size(const Tensor& input, int64_t signal_ndim,
                      bool complex_input, bool complex_output,
                      bool inverse, IntArrayRef checked_signal_sizes,
                      bool normalized, bool onesided,
                      IntArrayRef output_sizes) {
  fft_norm_mode norm;
  if (normalized) {
    norm = fft_norm_mode::by_root_n;
  } else {
    norm = inverse ? fft_norm_mode::by_n : fft_norm_mode::none;
  }
  return at::_fft_with_size(
      input, signal_ndim, complex_input, complex_output, inverse,
      checked_signal_sizes, static_cast<int64_t>(norm), onesided, output_sizes);
}

// We call the following methods via CUDA hooks because they are really only
// valid when CUDA is available. See native/cuda/CuFFTPlanCache.h for more details.
int64_t _cufft_get_plan_cache_max_size(int64_t device_index) {
  return detail::getCUDAHooks().cuFFTGetPlanCacheMaxSize(device_index);
}

void _cufft_set_plan_cache_max_size(int64_t device_index, int64_t max_size) {
  detail::getCUDAHooks().cuFFTSetPlanCacheMaxSize(device_index, max_size);
}

int64_t _cufft_get_plan_cache_size(int64_t device_index) {
  return detail::getCUDAHooks().cuFFTGetPlanCacheSize(device_index);
}

void _cufft_clear_plan_cache(int64_t device_index) {
  detail::getCUDAHooks().cuFFTClearPlanCache(device_index);
}

Tensor fft(const Tensor& self, const int64_t signal_ndim, const bool normalized) {
  return _fft(self, signal_ndim, /* complex_input */ true,
              /* complex_output */ true, /* inverse */ false, {},
              normalized ? fft_norm_mode::by_root_n : fft_norm_mode::none,
              /* onesided */ false);
}

Tensor ifft(const Tensor& self, const int64_t signal_ndim, const bool normalized) {
  return _fft(self, signal_ndim, /* complex_input */ true,
              /* complex_output */ true, /* inverse */ true, {},
              normalized ? fft_norm_mode::by_root_n : fft_norm_mode::by_n,
              /* onesided */ false);
}

Tensor rfft(const Tensor& self, const int64_t signal_ndim, const bool normalized,
            const bool onesided) {
  return _fft(self, signal_ndim, /* complex_input */ false,
              /* complex_output */ true, /* inverse */ false, {},
              normalized ? fft_norm_mode::by_root_n : fft_norm_mode::none,
              onesided);
}

Tensor irfft(const Tensor& self, const int64_t signal_ndim, const bool normalized,
             const bool onesided,  IntArrayRef signal_sizes) {
  return _fft(self, signal_ndim, /* complex_input */ true,
              /* complex_output */ false, /* inverse */ true, signal_sizes,
              normalized ? fft_norm_mode::by_root_n : fft_norm_mode::by_n,
              onesided);
}


Tensor stft(const Tensor& self, const int64_t n_fft, const optional<int64_t> hop_lengthOpt,
            const optional<int64_t> win_lengthOpt, const Tensor& window,
            const bool normalized, const bool onesided) {
  #define REPR(SS) \
    SS << "stft(" << self.toString() << self.sizes() << ", n_fft=" << n_fft \
       << ", hop_length=" << hop_length << ", win_length=" << win_length \
       << ", window="; \
    if (window.defined()) { \
      SS << window.toString() << "{" << window.sizes() << "}"; \
    } else { \
      SS << "None"; \
    } \
    SS << ", normalized=" << normalized << ", onesided=" << onesided << ")"

  // default_init hop_length and win_length
  auto hop_length = hop_lengthOpt.value_or(n_fft >> 2);
  auto win_length = win_lengthOpt.value_or(n_fft);

  if (!at::isFloatingType(self.scalar_type()) || self.dim() > 2 || self.dim() < 1) {
    std::ostringstream ss;
    REPR(ss) << ": expected a 1D or 2D tensor of floating types";
    AT_ERROR(ss.str());
  }
  Tensor input = self;
  if (self.dim() == 1) {
    input = input.unsqueeze(0);
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
    window_ = at::zeros({n_fft}, self.options());
    auto left = (n_fft - win_length) / 2;
    if (window.defined()) {
      window_.narrow(0, left, win_length).copy_(window);
    } else {
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
  // rfft and transpose to get (batch x fft_size x num_frames)
  auto out = input.rfft(1, normalized, onesided).transpose_(1, 2);
  if (self.dim() == 1) {
    return out.squeeze_(0);
  } else {
    return out;
  }
}

Tensor istft(const Tensor& self, const int64_t n_fft, const optional<int64_t> hop_lengthOpt,
             const optional<int64_t> win_lengthOpt, const Tensor& window,
             const bool center, const bool normalized, const bool onesided,
             const optional<int64_t> lengthOpt) {
  #define REPR(SS) \
    SS << "istft(" << self.toString() << self.sizes() << ", n_fft=" << n_fft \
       << ", hop_length=" << hop_length << ", win_length=" << win_length \
       << ", window="; \
    if (window.defined()) { \
      SS << window.toString() << "{" << window.sizes() << "}"; \
    } else { \
      SS << "None"; \
    } \
    SS << ", center=" << center << ", normalized=" << normalized << ", onesided=" << onesided << ", length="; \
    if (lengthOpt.has_value()) { \
      SS << lengthOpt.value(); \
    } else { \
      SS << "None"; \
    } \
    SS << ")"

  // default_init hop_length and win_length
  const auto hop_length = hop_lengthOpt.value_or(n_fft >> 2);
  const auto win_length = win_lengthOpt.value_or(n_fft);

  const auto input_dim = self.dim();
  const auto n_frames = self.size(-2);
  const auto fft_size = self.size(-3);

  const auto expected_output_signal_len = n_fft + hop_length * (n_frames - 1);

  const auto options = at::device(self.device()).dtype(self.dtype());
  if (self.numel() == 0) {
    std::ostringstream ss;
    REPR(ss) << ": input tensor cannot be empty.";
    AT_ERROR(ss.str());
  }
  if (input_dim != 3 && input_dim != 4) {
    std::ostringstream ss;
    REPR(ss) << ": expected a tensor with 3 or 4 dimensions, but got " << input_dim;
    AT_ERROR(ss.str());
  }
 if (self.size(-1) != 2) {
    std::ostringstream ss;
    REPR(ss) << ": expected the last dimension to be 2 (corresponding to real and imaginary parts), but got " << self.size(-1);
    AT_ERROR(ss.str());
  }

  if (onesided) {
    if (n_fft / 2 + 1 != fft_size) {
      std::ostringstream ss;
      REPR(ss) << ": expected the frequency dimension (3rd to the last) of the input tensor to match n_fft / 2 + 1 when onsided=True, but got " << fft_size;
      AT_ERROR(ss.str());
    }
  } else {
    if (n_fft != fft_size) {
      std::ostringstream ss;
      REPR(ss) << ": expected the frequency dimension (3rd to the last) of the input tensor to match n_fft when onsided=False, but got " << fft_size;
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

  Tensor input = self;
  if (input_dim == 3) {
    input = input.unsqueeze(0);
  }

  input = input.transpose(1, 2);  // size: (channel, n_frames, fft_size, 2)
  input = at::native::irfft(input, 1, normalized, onesided, {n_fft, });  // size: (channel, n_frames, n_fft)
  TORCH_INTERNAL_ASSERT(input.size(2) == n_fft);

  Tensor y_tmp = input * window_tmp.view({1, 1, n_fft});  // size: (channel, n_frames, n_fft)
  y_tmp = y_tmp.transpose(1, 2);  // size: (channel, n_fft, frame)

  Tensor y = at::col2im(y_tmp,
                                  /*output_size*/ {1, (n_frames - 1) * hop_length + n_fft},
                                  /*kernel_size*/ {1, n_fft},
                                  /*dilation*/    {1, 1},
                                  /*padding*/     {0, 0},
                                  /*stride*/      {1, hop_length}
                                 ).squeeze(2);
  window_tmp = window_tmp.pow(2).view({n_fft, 1}).repeat({1, n_frames}).unsqueeze(0);  // size: (1, n_fft, n_frames)
  Tensor window_envelop = at::col2im(window_tmp,
                                  /*output_size*/ {1, (n_frames - 1) * hop_length + n_fft},
                                  /*kernel_size*/ {1, n_fft},
                                  /*dilation*/    {1, 1},
                                  /*padding*/     {0, 0},
                                  /*stride*/      {1, hop_length}
                                 ).squeeze(2); // size: (1, 1, expected_output_signal_len)

  TORCH_INTERNAL_ASSERT(expected_output_signal_len == y.size(2));
  TORCH_INTERNAL_ASSERT(expected_output_signal_len == window_envelop.size(2));

  // We need to trim the front padding away if centered
  const auto start = center ? n_fft / 2 : 0;
  const auto end = lengthOpt.has_value()? start + lengthOpt.value() : - n_fft / 2;

  y = y.slice(2, start, end, 1);
  window_envelop = window_envelop.slice(2, start, end, 1);
  const auto window_envelop_lowest = window_envelop.abs().min().item().toDouble();
  if (window_envelop_lowest < 1e-11) {
    std::ostringstream ss;
    REPR(ss) << "window overlap add min: " << window_envelop_lowest;
    AT_ERROR(ss.str());
  }

  y = (y / window_envelop).squeeze(1);  // size: (channel, expected_output_signal_len)
  if (input_dim == 3) {
    y = y.squeeze(0);
  }
  return y;

  #undef REPR
}

}} // at::native
