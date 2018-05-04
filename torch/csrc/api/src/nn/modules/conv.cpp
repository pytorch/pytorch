#include <torch/nn/modules/conv.h>

#include <cassert>
#include <stdexcept>

namespace torch { namespace nn {
void Conv::initialize_parameters() {
  if (!transposed_) {
    for (auto pad : output_padding_) {
      if (pad != 0) {
        throw std::runtime_error(
            "Only transposed convolutions support output padding!");
      }
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
  weight =
      this->add(Var(DefaultTensor(at::kFloat).tensor(wsize), true), "weight");
  if (!no_bias_) {
    bias = this->add(
        Var(DefaultTensor(at::kFloat).tensor({out_channels_}), true), "bias");
  } else {
    assert(!bias.defined());
  }
}

void Conv::reset_parameters() {
  auto n = in_channels_;
  for (auto k : ks_)
    n *= k;
  auto stdv = 1.0 / std::sqrt(n);
  for (auto& p : parameters()) {
    p.second.data().uniform_(-stdv, stdv);
  }
}

variable_list Conv::forward(variable_list input) {
  auto x = input[0];
  if (Nd_ == 1) {
    assert(x.ndimension() == 3);
    x = x.unsqueeze(-1); // TODO: Use conv1d once available
  } else if (Nd_ == 2) {
    assert(x.ndimension() == 4);
  } else if (Nd_ == 3) {
    assert(x.ndimension() == 5);
  } else {
    throw std::runtime_error("Only Conv{1,2,3}d are supported");
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
}} // namespace torch::nn
