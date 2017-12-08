#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"
#include "ATen/WrapDimUtils.h"
#include "ATen/ExpandUtils.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>

namespace at { namespace native {

std::tuple<Tensor, Tensor> stft(const Tensor& self, const int64_t frame_length,
                                const int64_t hop, const Tensor& window,
                                const int64_t pad_end) {
  #define REPR(SS) SS << "stft(" << self.type() << "{" << self.sizes() \
                      << "}, frame_length=" << frame_length << ", hop=" << hop \
                      << ", window=" << window.type() << "{" << window.sizes() \
                      << "}, pad_end=" << pad_end << ")"
  if (!at::isFloatingType(self.type().scalarType()) || self.dim() > 2) {
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
  if (window.defined() && (window.dim() != 1 || window.size(0) != frame_length)) {
    std::ostringstream ss;
    REPR(ss) << ": expected a 1D window tensor of size equal to "
             << "frame_length=" << frame_length
             << ", but get window with size {" << window.sizes() << "}";
    throw std::runtime_error(ss.str());
  }
  if (pad_end < 0) {
    std::ostringstream ss;
    REPR(ss) << ": expected pad_end >= 0, but get pad_end=" << pad_end;
    throw std::runtime_error(ss.str());
  }
  #undef REPR
  // pad zeros
  if (pad_end != 0) {
    Tensor padded_input = self.type().zeros({batch, len + pad_end});
    padded_input.narrow(1, 0, len).copy_(input);
    input = padded_input;
    len += pad_end;
  }
  // build ft kernel
  // k[omega, t] = cos (2 pi omega t / N) - j sin (2 pi omega t / N)
  double N = static_cast<double>(frame_length);
  auto arange = self.type().arange(0, frame_length).unsqueeze_(1);
  auto arange_2d = arange.mm(arange.t()).mul_(M_PI * 2. / N);
  auto re_kernel = arange_2d.cos();
  auto im_kernel = arange_2d.sin();
  auto kernel = at::cat({re_kernel, im_kernel}, 0);
  if (window.defined()) {
    kernel = kernel.mul_(window.view({1, -1}).expand_as(kernel));
  }
  // prepare for conv2d
  input = input.view({batch, 1, len, 1});
  kernel = kernel.view({frame_length * 2, 1, frame_length, 1});
  // conv is actually correlation, so we are good
  auto conv_out = at::conv2d(input, kernel, {frame_length, 1}, {}, hop).squeeze_(-1);
  auto re_out = conv_out.narrow(1, 0, frame_length);
  auto im_out = conv_out.narrow(1, frame_length, frame_length);
  auto magnitude = re_out.pow(2).add_(im_out.pow(2)).sqrt_().transpose(1, 2);
  auto phase = im_out.mul_(-1).atan2(re_out).transpose(1, 2);
  if (self.dim() == 1) {
    return std::make_tuple(magnitude.squeeze_(0), phase.squeeze_(0));
  } else {
    return std::make_tuple(magnitude, phase);
  }
}

}} // at::native
