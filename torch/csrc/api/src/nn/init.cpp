#include <torch/nn/init.h>

#include <torch/types.h>
#include <torch/utils.h>

#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

#include <algorithm>
#include <cmath>
#include <tuple>

namespace torch::nn::init {
namespace {
struct Fan {
  explicit Fan(Tensor& tensor) {
    const auto dimensions = tensor.ndimension();
    TORCH_CHECK(
        dimensions >= 2,
        "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions");

    if (dimensions == 2) {
      in = tensor.size(1);
      out = tensor.size(0);
    } else {
      in = tensor.size(1) * tensor[0][0].numel();
      out = tensor.size(0) * tensor[0][0].numel();
    }
  }

  int64_t in;
  int64_t out;
};

double calculate_kaiming_std(
    Tensor tensor,
    double a,
    FanModeType mode,
    NonlinearityType nonlinearity) {
  NoGradGuard guard;
  Fan fan(tensor);
  const auto gain = calculate_gain(nonlinearity, a);
  double std = 0.0;

  if (std::holds_alternative<enumtype::kFanIn>(mode)) {
    std = gain / std::sqrt(fan.in);
  } else {
    std = gain / std::sqrt(fan.out);
  }
  return std;
}
} // namespace

double calculate_gain(NonlinearityType nonlinearity, double param) {
  if (std::holds_alternative<enumtype::kTanh>(nonlinearity)) {
    return 5.0 / 3.0;
  } else if (std::holds_alternative<enumtype::kReLU>(nonlinearity)) {
    return std::sqrt(2.0);
  } else if (std::holds_alternative<enumtype::kLeakyReLU>(nonlinearity)) {
    return std::sqrt(2.0 / (1 + pow(param, 2)));
  }

  return 1.0;
}

// NOLINTNEXTLINE(performance-unnecessary-value-param)
Tensor constant_(Tensor tensor, Scalar value) {
  NoGradGuard guard;
  return tensor.fill_(value);
}

Tensor dirac_(Tensor tensor) {
  NoGradGuard guard;

  TORCH_CHECK(
      tensor.ndimension() >= 3 && tensor.ndimension() <= 5,
      "Only tensors with 3, 4, or 5 dimensions are supported");

  const auto sizes = tensor.sizes();
  const auto min_dim = std::min(sizes[0], sizes[1]);

  tensor.zero_();
  for (const auto d : c10::irange(min_dim)) {
    // NOLINTNEXTLINE(bugprone-switch-missing-default-case)
    switch (tensor.ndimension()) {
      case 3: // Temporal convolution
        tensor[d][d][sizes[2] / 2] = 1;
        break;
      case 4: // Spatial convolution
        tensor[d][d][sizes[2] / 2][sizes[3] / 2] = 1;
        break;
      case 5: // Volumetric convolution
        tensor[d][d][sizes[2] / 2][sizes[3] / 2][sizes[4] / 2] = 1;
        break;
    }
  }

  return tensor;
}

Tensor eye_(Tensor matrix) {
  NoGradGuard guard;
  TORCH_CHECK(
      matrix.ndimension() == 2, "Only tensors with 2 dimensions are supported");
  return torch::eye_out(matrix, matrix.size(0), matrix.size(1));
}

// NOLINTNEXTLINE(performance-unnecessary-value-param)
Tensor normal_(Tensor tensor, double mean, double std) {
  NoGradGuard guard;
  return tensor.normal_(mean, std);
}

// NOLINTNEXTLINE(performance-unnecessary-value-param)
Tensor ones_(Tensor tensor) {
  NoGradGuard guard;
  return tensor.fill_(1);
}

Tensor orthogonal_(Tensor tensor, double gain) {
  NoGradGuard guard;

  TORCH_CHECK(
      tensor.ndimension() >= 2,
      "Only tensors with 2 or more dimensions are supported");

  const auto rows = tensor.size(0);
  const auto columns = tensor.numel() / rows;
  auto flattened = torch::randn({rows, columns});

  if (rows < columns) {
    flattened.t_();
  }

  // Compute the qr factorization
  auto [q, r] = torch::linalg_qr(flattened);
  // Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
  auto d = torch::diag(r, 0);
  auto ph = d.sign();
  q *= ph;

  if (rows < columns) {
    q.t_();
  }

  tensor.view_as(q).copy_(q);
  tensor.mul_(gain);

  return tensor;
}

Tensor sparse_(Tensor tensor, double sparsity, double std) {
  NoGradGuard guard;

  TORCH_CHECK(
      tensor.ndimension() == 2, "Only tensors with 2 dimensions are supported");

  const auto rows = tensor.size(0);
  const auto columns = tensor.size(1);
  const int64_t num_zeros = std::ceil(sparsity * static_cast<double>(rows));
  tensor.normal_(0, std);
  for (const auto column : c10::irange(columns)) {
    auto row_indices = torch::randperm(rows, tensor.options().dtype(kLong));
    auto zero_indices =
        row_indices.slice(/*dim=*/0, /*start=*/0, /*end=*/num_zeros);
    tensor.index_put_(
        {zero_indices, torch::tensor(column, tensor.options().dtype(kLong))},
        torch::zeros(num_zeros, tensor.options()));
  }

  return tensor;
}

// NOLINTNEXTLINE(performance-unnecessary-value-param)
Tensor uniform_(Tensor tensor, double low, double high) {
  NoGradGuard guard;
  return tensor.uniform_(low, high);
}

Tensor kaiming_uniform_(
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    Tensor tensor,
    double a,
    FanModeType mode,
    NonlinearityType nonlinearity) {
  NoGradGuard guard;
  auto std = calculate_kaiming_std(tensor, a, mode, nonlinearity);
  // Calculate uniform bounds from standard deviation
  const auto bound = std::sqrt(3.0) * std;
  return tensor.uniform_(-bound, bound);
}

Tensor kaiming_normal_(
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    Tensor tensor,
    double a,
    FanModeType mode,
    NonlinearityType nonlinearity) {
  NoGradGuard guard;

  auto std = calculate_kaiming_std(tensor, a, mode, nonlinearity);
  return tensor.normal_(0, std);
}

Tensor xavier_normal_(Tensor tensor, double gain) {
  NoGradGuard guard;

  Fan fan(tensor);
  const auto std =
      gain * std::sqrt(2.0 / static_cast<double>(fan.in + fan.out));
  return tensor.normal_(0, std);
}

Tensor xavier_uniform_(Tensor tensor, double gain) {
  NoGradGuard guard;
  Fan fan(tensor);
  const auto std =
      gain * std::sqrt(2.0 / static_cast<double>(fan.in + fan.out));
  // Calculate uniform bounds from standard deviation with
  const auto a = std::sqrt(3.0) * std;
  return tensor.uniform_(-a, a);
}

// NOLINTNEXTLINE(performance-unnecessary-value-param)
Tensor zeros_(Tensor tensor) {
  NoGradGuard guard;
  return tensor.zero_();
}

std::tuple<int64_t, int64_t> _calculate_fan_in_and_fan_out(
    const Tensor& tensor) {
  const auto dimensions = tensor.dim();
  TORCH_CHECK(
      dimensions >= 2,
      "Fan in and fan out can not be computed "
      "for tensor with fewer than 2 dimensions")

  int64_t fan_in = 0, fan_out = 0;
  if (dimensions == 2) { // Linear
    fan_in = tensor.size(1);
    fan_out = tensor.size(0);
  } else {
    const auto num_input_fmaps = tensor.size(1);
    const auto num_output_fmaps = tensor.size(0);
    int64_t receptive_field_size = 1;
    if (tensor.dim() > 2) {
      receptive_field_size = tensor[0][0].numel();
    }
    fan_in = num_input_fmaps * receptive_field_size;
    fan_out = num_output_fmaps * receptive_field_size;
  }
  return std::tie(fan_in, fan_out);
}

} // namespace torch::nn::init
