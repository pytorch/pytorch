// define constants like M_PI and C keywords for MSVC
#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#include <math.h>
#endif

#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"
#include "ATen/WrapDimUtils.h"
#include "ATen/ExpandUtils.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>


namespace at { namespace native {

// Since real-to-complex satisfy the Hermitian symmetry, i.e.,
// X[m, \omega] = X[m, N - \omega]*. We return only the first floor(N / 2) + 1
// values by default. This is also the assumption in libraries including cuFFT.
static inline int64_t infer_ft_complex_length(int64_t real_length) {
  return (real_length >> 1) + 1;
}

Tensor stft(const Tensor& self, const int64_t frame_length,
                                const int64_t hop, const int64_t fft_size,
                                const bool return_onesided,
                                const Tensor& window, const int64_t pad_end) {
  #define REPR(SS) \
    SS << "stft(" << self.type() << "{" << self.sizes() << "}, frame_length=" \
       << frame_length << ", hop=" << hop << ", fft_size=" << fft_size \
       << ", return_onesided=" << return_onesided << ", window="; \
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
    REPR(ss) << ": expected pad_end >= 0, but get pad_end=" << pad_end;
    throw std::runtime_error(ss.str());
  }
  // pad zeros
  if (pad_end != 0) {
    Tensor padded_input = self.type().zeros({batch, len + pad_end});
    padded_input.narrow(1, 0, len).copy_(input);
    len += pad_end;
    input = padded_input;
  }
  if (frame_length <= 0 || frame_length > len) {
    std::ostringstream ss;
    REPR(ss) << ": expected 0 < frame_length < " << len
             << ", but get frame_length=" << frame_length;
    throw std::runtime_error(ss.str());
  }
  if (hop <= 0) {
    std::ostringstream ss;
    REPR(ss) << " expected hop > 0, but get hop=" << hop;
    throw std::runtime_error(ss.str());
  }
  if (fft_size <= 0) {
    std::ostringstream ss;
    REPR(ss) << " expected fft_size > 0, but get fft_size=" << fft_size;
    throw std::runtime_error(ss.str());
  }
  if (window.defined() && (window.dim() != 1 || window.size(0) != frame_length)) {
    std::ostringstream ss;
    REPR(ss) << ": expected a 1D window tensor of size equal to "
             << "frame_length=" << frame_length
             << ", but get window with size {" << window.sizes() << "}";
    throw std::runtime_error(ss.str());
  }
  #undef REPR
  int64_t return_size = return_onesided ? infer_ft_complex_length(fft_size) : fft_size;
  // build ft kernel
  // k[omega, t] = cos (2 pi omega t / N) - j sin (2 pi omega t / N)
  double N = static_cast<double>(fft_size);
  auto freq_arange = self.type().arange(0, return_size).mul_(M_PI * 2. / N);
  auto time_arange = self.type().arange(0, frame_length);
  auto arange_2d = at::ger(freq_arange, time_arange);
  auto re_kernel = arange_2d.cos();
  auto im_kernel = arange_2d.sin().mul_(-1);
  auto kernel = at::cat({re_kernel, im_kernel}, 0);
  if (window.defined()) {
    kernel *= window.view({1, -1});
  }
  // prepare for conv2d
  input = input.view({batch, 1, len, 1});
  kernel = kernel.view({return_size * 2, 1, frame_length, 1});
  // conv is actually correlation, so we are good
  auto conv_out = at::conv2d(input, kernel, {}, hop).squeeze_(-1);
  // transpose to [batch x time x freq x (re/im)]
  auto out = conv_out.view({batch, 2, return_size, -1}).transpose_(1, -1);
  if (self.dim() == 1) {
    return out.squeeze_(0);
  } else {
    return out;
  }
}

}} // at::native
