#include <ATen/ATen.h>
#include <ATen/Error.h>

#include <algorithm>
#include <cmath>
#include <tuple>

namespace at {
namespace native {
namespace {
struct Fan {
  explicit Fan(Tensor& self) {
    const auto dimensions = self.ndimension();
    AT_CHECK(
        dimensions >= 2,
        "Fan in and fan out can not be computed for self with less than 2 dimensions");

    if (dimensions == 2) {
      in = self.size(1);
      out = self.size(0);
    } else {
      in = self.size(1) * self[0][0].numel();
      out = self.size(0) * self[0][0].numel();
    }
  }

  int64_t in;
  int64_t out;
};
} // namespace

Tensor& dirac_(Tensor& self) {
  AT_CHECK(
      self.ndimension() >= 3 && self.ndimension() <= 5,
      "Only tensors with 3, 4, or 5 dimensions are supported");

  const auto sizes = self.sizes();
  const auto min_dim = std::min(sizes[0], sizes[1]);

  self.zero_();
  for (int64_t d = 0; d < min_dim; ++d) {
    switch (self.ndimension()) {
      case 3: // Temporal convolution
        self[d][d][sizes[2] / 2] = 1;
        break;
      case 4: // Spatial convolution
        self[d][d][sizes[2] / 2][sizes[3] / 2] = 1;
        break;
      case 5: // Volumetric convolution
        self[d][d][sizes[2] / 2][sizes[3] / 2][sizes[4] / 2] = 1;
        break;
    }
  }

  return self;
}

Tensor& eye_(Tensor& self) {
  AT_CHECK(
      self.ndimension() == 2, "Only tensors with 2 dimensions are supported");
  return eye_out(self, self.size(0), self.size(1));
}

Tensor& orthogonal_(Tensor& self, double gain) {
  AT_CHECK(
      self.ndimension() >= 2,
      "Only tensors with 2 or more dimensions are supported");

  const auto rows = self.size(0);
  const auto columns = self.size(1);
  auto flattened = at::randn({rows, columns});

  if (rows < columns) {
    flattened.t_();
  }

  // Compute the qr factorization
  Tensor q, r;
  std::tie(q, r) = at::qr(flattened);
  // Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
  auto d = at::diag(r, 0);
  auto ph = d.sign();
  q *= ph;

  if (rows < columns) {
    q.t_();
  }

  self.view_as(q).copy_(q);
  self.mul_(gain);

  return self;
}

Tensor& sparse_(Tensor& self, double sparsity, double std) {
  AT_CHECK(
      self.ndimension() == 2, "Only tensors with 2 dimensions are supported");

  const auto rows = self.size(0);
  const auto columns = self.size(1);
  const int64_t num_zeros = std::ceil(sparsity * rows);
  self.normal_(0, std);
  for (int64_t column = 0; column < columns; ++column) {
    auto row_indices = at::randperm(rows, self.options().dtype(kLong));
    auto zero_indices =
        row_indices.slice(/*dim=*/0, /*start=*/0, /*end=*/num_zeros);
    self.index_put_(
        {zero_indices, at::tensor(column, self.options().dtype(kLong))},
        at::zeros(num_zeros, self.options()));
  }

  return self;
}

Tensor& xavier_uniform_(Tensor& self, double gain) {
  Fan fan(self);
  const auto std = gain * std::sqrt(2.0 / (fan.in + fan.out));
  // Calculate uniform bounds from standard deviation with
  const auto a = std::sqrt(3.0) * std;
  return self.uniform_(-a, a);
}

Tensor& xavier_normal_(Tensor& self, double gain) {
  Fan fan(self);
  const auto std = gain * std::sqrt(2.0 / (fan.in + fan.out));
  return self.normal_(0, std);
}

} // namespace native
} // namespace at
