#include <torch/nn/modules/conv.h>

#include <ATen/Error.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>

namespace torch { namespace nn {
namespace {
IntVec makeTup(size_t number_of_dimensions, int x, int def = 0) {
  IntVec ret;
  if (number_of_dimensions == 1) {
    ret.push_back(x);
    ret.push_back(def);
  } else {
    for (size_t i = 0; i < number_of_dimensions; i++) {
      ret.push_back(x);
    }
  }
  return ret;
}
} // namespace

Conv::Conv(
    uint32_t Nd,
    uint32_t in_chan,
    uint32_t out_chan,
    IntVec ks,
    bool transposed,
    bool with_bias,
    int groups)
    : Nd_(Nd),
      in_channels_(in_chan),
      out_channels_(out_chan),
      transposed_(transposed),
      groups_(groups),
      ks_(std::move(ks)),
      stride_(makeTup(Nd, 1, 1)),
      padding_(makeTup(Nd, 0)),
      dilation_(makeTup(Nd, 1, 1)),
      dilated_(false),
      output_padding_(makeTup(Nd, 0)) {
  AT_CHECK(
      (Nd == 1) ? ks_.size() == 2 : ks_.size() == Nd,
      "Kernel rank (",
      ks_.size(),
      ") must match number of dimensions (",
      Nd,
      ")");

  if (!transposed_) {
    for (auto pad : output_padding_) {
      AT_CHECK(
          pad == 0, "Only transposed convolutions support output padding!");
    }
  }

  IntVec wsize;
  if (transposed_) {
    wsize.push_back(in_channels_);
    wsize.push_back(out_channels_ / groups_);
  } else {
    wsize.push_back(out_channels_);
    wsize.push_back(in_channels_ / groups_);
  }
  wsize.insert(wsize.end(), ks_.begin(), ks_.end());
  AT_ASSERT(wsize.size() == 2 + ks_.size());

  weight = add(Var(at::CPU(at::kFloat).empty(wsize)), "weight");
  if (with_bias) {
    bias = add(Var(at::CPU(at::kFloat).empty(out_channels_)), "bias");
  } else {
    AT_ASSERT(!bias.defined());
  }

  const auto number_of_features = std::accumulate(
      ks_.begin(), ks_.end(), in_channels_, std::multiplies<uint32_t>{});
  const auto stdv = 1.0 / std::sqrt(number_of_features);
  for (auto& p : parameters()) {
    p.second.data().uniform_(-stdv, stdv);
  }
}

Conv::Conv(
    uint32_t Nd,
    uint32_t in_chan,
    uint32_t out_chan,
    int ks,
    bool transposed,
    bool with_bias,
    int groups)
    : Conv(
          Nd,
          in_chan,
          out_chan,
          makeTup(Nd, ks, 1),
          transposed,
          with_bias,
          groups) {}

variable_list Conv::forward(variable_list input) {
  auto x = input[0];
  if (Nd_ == 1) {
    AT_ASSERT(x.ndimension() == 3);
    x = x.unsqueeze(-1); // TODO: Use conv1d once available
  } else if (Nd_ == 2) {
    AT_ASSERT(x.ndimension() == 4);
  } else if (Nd_ == 3) {
    AT_ASSERT(x.ndimension() == 5);
  } else {
    AT_ERROR("Only Conv{1,2,3}d are supported");
  }

  Variable out;
  if (Nd_ == 1 || Nd_ == 2) {
    if (transposed_) {
      out = at::conv_transpose2d(
          x,
          weight,
          bias,
          stride_,
          padding_,
          output_padding_,
          groups_,
          dilation_);
    } else {
      out = at::conv2d(x, weight, bias, stride_, padding_, dilation_, groups_);
    }
  } else if (Nd_ == 3) {
    if (transposed_) {
      out = at::conv_transpose3d(
          x,
          weight,
          bias,
          stride_,
          padding_,
          output_padding_,
          groups_,
          dilation_);
    } else {
      out = at::conv3d(x, weight, bias, stride_, padding_, dilation_, groups_);
    }
  }

  return variable_list({out});
}

Conv& Conv::stride(size_t value) {
  stride_ = makeTup(Nd_, value, 1);
  return *this;
}
Conv& Conv::padding(size_t value) {
  padding_ = makeTup(Nd_, value);
  return *this;
}
Conv& Conv::dilation(size_t value) {
  dilation_ = makeTup(Nd_, value, 1);
  return *this;
}
Conv& Conv::output_padding(size_t value) {
  output_padding_ = makeTup(Nd_, value);
  return *this;
}

}} // namespace torch::nn
