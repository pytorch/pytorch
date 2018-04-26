// define constants like M_PI and C keywords for MSVC
#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#include <math.h>
#endif

#include "ATen/ATen.h"
#include "ATen/Config.h"
#include "ATen/NativeFunctions.h"
#include "ATen/native/SpectralOpsUtils.h"

#include <algorithm>
#include <vector>
#include <cmath>

namespace at { namespace native {

// This is a pass-through wrapper function that does the size check and
// inferences. The actual forward implementation function is called
// at::_fft_with_size which dispatches to _fft_cufft (CUDA) or _fft_mkl (CPU).
static inline Tensor _fft(const Tensor &self, const int64_t signal_ndim,
           const bool complex_input, const bool complex_output,
           const bool inverse, IntList signal_sizes, const bool normalized,
           const bool onesided) {

  if (signal_ndim < 1 || signal_ndim > 3) {
    std::ostringstream ss;
    ss << "Expected signal_ndim to be 1, 2, or 3, but got signal_ndim="
       << signal_ndim;
    throw std::runtime_error(ss.str());
  }
  if (!at::isFloatingType(self.type().scalarType())) {
    std::ostringstream ss;
    ss << "Expected an input tensor of floating types, but got input="
       << self.type() << self.sizes();
    throw std::runtime_error(ss.str());
  }

  auto signal_tensor_ndim = signal_ndim + static_cast<int>(complex_input);  // add complex dim
  if (self.dim() < signal_tensor_ndim) {
    std::ostringstream ss;
    ss << "Given signal_ndim=" << signal_ndim << ", expected an input tensor "
       << "of at least" << signal_tensor_ndim << "D";
    if (complex_input) {
      ss << " (complex input adds an extra dimension)";
    }
    ss << ", but got input=" << self.type() << self.sizes();
    throw std::runtime_error(ss.str());
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
    if (input.size(signal_ndim + 1) != 2) {
      std::ostringstream ss;
      ss << "Expected an input tensor with a last dimension of size 2 "
         << "representing real + imaginary components, but got input "
         << self.type() << self.sizes();
      throw std::runtime_error(ss.str());
    }
  }

  // build signal_sizes and output_size
  if (signal_sizes.size() > 0 && static_cast<int64_t>(signal_sizes.size()) != signal_ndim) {
    std::ostringstream ss;
    ss << "Expected signal_sizes to be empty (default) or of signal_ndim="
       << signal_ndim << "D, but got signal_sizes=" << signal_sizes;
    throw std::runtime_error(ss.str());
  }
  std::vector<int64_t> output_sizes(signal_ndim + 1 + static_cast<int>(complex_output));
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
      if (signal_sizes.size() > 0 && signal_sizes[i] != checked_signal_sizes[i]) {
        std::ostringstream ss;
        ss << "Expected given signal_sizes=" << signal_sizes << " to have same "
           << "shape with input at signal dimension " << i << ", but got "
           << "signal_sizes=" << signal_sizes << " and input=" << self.type()
           << self.sizes();
        throw std::runtime_error(ss.str());
      }
    }
  }
  if (complex_output) {
    output_sizes[signal_ndim + 1] = 2;
  }

  Tensor output = at::_fft_with_size(input, signal_ndim, complex_input,
                                     complex_output, inverse,
                                     checked_signal_sizes, normalized, onesided,
                                     output_sizes);

  // unflatten the batch dims
  if (batch_ndim == 0) {
    // slightly faster path for non-batch mode
    output = output.squeeze(0);
  } else if (batch_ndim > 1) {
    auto output_ndim = self.dim() + static_cast<int>(complex_output) - static_cast<int>(complex_input);
    std::vector<int64_t> unflatten_output_shape(output_ndim);
    std::copy(self_shape.begin(), self_shape.begin() + batch_ndim, unflatten_output_shape.begin());
    std::copy(output_sizes.begin() + 1, output_sizes.end(), unflatten_output_shape.begin() + batch_ndim);
    output = output.reshape(unflatten_output_shape);
  }
  return output;
}

Tensor fft(const Tensor& self, const int64_t signal_ndim, const bool normalized) {
  return _fft(self, signal_ndim, /* complex_input */ true,
              /* complex_output */ true, /* inverse */ false, {}, normalized,
              /* onesided */ false);
}

Tensor ifft(const Tensor& self, const int64_t signal_ndim, const bool normalized) {
  return _fft(self, signal_ndim, /* complex_input */ true,
              /* complex_output */ true, /* inverse */ true, {}, normalized,
              /* onesided */ false);
}

Tensor rfft(const Tensor& self, const int64_t signal_ndim, const bool normalized,
            const bool onesided) {
  return _fft(self, signal_ndim, /* complex_input */ false,
              /* complex_output */ true, /* inverse */ false, {}, normalized,
              onesided);
}

Tensor irfft(const Tensor& self, const int64_t signal_ndim, const bool normalized,
             const bool onesided,  IntList signal_sizes) {
  return _fft(self, signal_ndim, /* complex_input */ true,
              /* complex_output */ false, /* inverse */ true, signal_sizes,
              normalized, onesided);
}


Tensor stft(const Tensor& self, const int64_t frame_length,
                                const int64_t hop, const int64_t fft_size,
                                const bool normalized, const bool onesided,
                                const Tensor& window, const int64_t pad_end) {
  #define REPR(SS) \
    SS << "stft(" << self.type() << self.sizes() << ", frame_length=" \
       << frame_length << ", hop=" << hop << ", fft_size=" << fft_size \
       << ", normalized=" << normalized << ", onesided=" << onesided << \
       ", window="; \
    if (window.defined()) { \
      SS << window.type() << "{" << window.sizes() << "}"; \
    } else { \
      SS << "None"; \
    } \
    SS << ", pad_end=" << pad_end << ")"

  if (!at::isFloatingType(self.type().scalarType()) || self.dim() > 2 || self.dim() < 1) {
    std::ostringstream ss;
    REPR(ss) << ": expected a 1D or 2D tensor of floating types";
    throw std::runtime_error(ss.str());
  }
  Tensor input = self;
  if (self.dim() == 1) {
    input = input.unsqueeze(0);
  }
  int64_t batch = input.size(0);
  int64_t len = input.size(1);
  if (pad_end < 0) {
    std::ostringstream ss;
    REPR(ss) << ": expected pad_end >= 0, but got pad_end=" << pad_end;
    throw std::runtime_error(ss.str());
  }
  // pad zeros
  if (pad_end != 0) {
    Tensor padded_input = at::zeros(self.type(), {batch, len + pad_end});
    padded_input.narrow(1, 0, len).copy_(input);
    len += pad_end;
    input = padded_input;
  }
  if (frame_length <= 0 || frame_length > len) {
    std::ostringstream ss;
    REPR(ss) << ": expected 0 < frame_length < " << len
             << ", but got frame_length=" << frame_length;
    throw std::runtime_error(ss.str());
  }
  if (hop <= 0) {
    std::ostringstream ss;
    REPR(ss) << " expected hop > 0, but got hop=" << hop;
    throw std::runtime_error(ss.str());
  }
  if (fft_size <= 0) {
    std::ostringstream ss;
    REPR(ss) << " expected fft_size > 0, but got fft_size=" << fft_size;
    throw std::runtime_error(ss.str());
  }
  if (window.defined() && (window.dim() != 1 || window.size(0) != frame_length)) {
    std::ostringstream ss;
    REPR(ss) << ": expected a 1D window tensor of size equal to "
             << "frame_length=" << frame_length
             << ", but got window with size " << window.sizes();
    throw std::runtime_error(ss.str());
  }
  #undef REPR
  int64_t return_size = onesided ? infer_ft_real_to_complex_onesided_size(fft_size) : fft_size;
  // build ft kernel
  // k[omega, t] = cos (2 pi omega t / N) - j sin (2 pi omega t / N)
  double N = static_cast<double>(fft_size);
  auto freq_arange = at::arange(self.type(), 0, return_size).mul_(M_PI * 2. / N);
  auto time_arange = at::arange(self.type(), 0, frame_length);
  auto arange_2d = at::ger(freq_arange, time_arange);
  auto re_kernel = arange_2d.cos();
  auto im_kernel = arange_2d.sin().neg_();
  auto kernel = at::cat({re_kernel, im_kernel}, 0);
  if (window.defined()) {
    kernel *= window.view({1, -1});
  }
  if (normalized) {
    double T = static_cast<double>(frame_length);
    kernel.div_(std::sqrt(T));
  }
  // prepare for conv1d
  input = input.view({batch, 1, len});
  kernel = kernel.view({return_size * 2, 1, frame_length});
  // conv is actually correlation, so we are good
  auto conv_out = at::conv1d(input, kernel, {}, hop).squeeze_(-1);
  // transpose to [batch x time x freq x (re/im)]
  auto out = conv_out.view({batch, 2, return_size, -1}).transpose_(1, -1);
  if (self.dim() == 1) {
    return out.squeeze_(0);
  } else {
    return out;
  }
}

}} // at::native
