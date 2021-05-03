#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>

#include <ATen/Parallel.h>
#include <ATen/native/TriangularOpsUtils.h>

namespace at {
namespace native {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ triu/tril ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t, bool upper>
static void apply_triu_tril_single(
    scalar_t* result, scalar_t* self, bool inplace,
    int64_t k, int64_t n, int64_t m,
    int64_t res_row_stride, int64_t res_col_stride,
    int64_t self_row_stride, int64_t self_col_stride) {

  constexpr int64_t zero = 0;

  if (upper) {
    at::parallel_for(0, n, 0, [&](int64_t start, int64_t end) {
      for (auto i = start; i < end; i++) {
        for (int64_t j = 0; j < std::min(m, i + k); j++) {
          result[i * res_row_stride + j * res_col_stride] = 0;
        }
        if (!inplace) {  // copy the rest of the self if not inplace
          for (int64_t j = std::max(zero, i + k); j < m; j++) {
            result[i * res_row_stride + j * res_col_stride] = self[i * self_row_stride + j * self_col_stride];
          }
        }
      }
    });
  } else {
    at::parallel_for(0, n, 0, [&](int64_t start, int64_t end) {
      for (auto i = start; i < end; i++) {
        for (int64_t j = std::max(zero, i + k + 1); j < m; j++) {
          result[i * res_row_stride + j * res_col_stride] = 0;
        }
        if (!inplace) {  // copy the rest of the self if not inplace
          for (int64_t j = zero; j < std::min(m, i + k + 1); j++) {
            result[i * res_row_stride + j * res_col_stride] = self[i * self_row_stride + j * self_col_stride];
          }
        }
      }
    });
  }
}

template <typename scalar_t, bool upper>
void apply_triu_tril(Tensor& result, const Tensor& self, bool inplace, int64_t k) {
  auto n = self.size(-2);
  auto m = self.size(-1);
  auto self_data = self.data_ptr<scalar_t>();
  auto self_stride = (self.dim() > 2 && self.stride(-3) > 0) ? self.stride(-3) : 1;
  auto batchsize = batchCountTrilTriu(result);
  auto self_row_stride = self.stride(-2);
  auto self_column_stride = self.stride(-1);

  auto result_data = result.data_ptr<scalar_t>();
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t result_stride, result_row_stride, result_column_stride;
  if (result_data != self_data) {
    result_stride = (result.dim() > 2 && result.stride(-3) > 0) ? result.stride(-3) : 1;
    result_row_stride = result.stride(-2);
    result_column_stride = result.stride(-1);
  } else {
    result_stride = self_stride;
    result_row_stride = self_row_stride;
    result_column_stride = self_column_stride;
  }

  at::parallel_for(0, batchsize, 0, [&](int64_t start, int64_t end) {
    for (auto b = start; b < end; b++) {
      scalar_t* self_batch = &self_data[b * self_stride];
      scalar_t* result_batch = &result_data[b * result_stride];
      apply_triu_tril_single<scalar_t, upper>(
          result_batch, self_batch, inplace, k, n, m,
          result_row_stride, result_column_stride, self_row_stride, self_column_stride);
    }
  });
}

Tensor tril(const Tensor& self, int64_t k) {
  Tensor result = at::empty({0}, self.options());
  at::tril_out(result, self, k);
  return result;
}

Tensor& tril_cpu_(Tensor &self, int64_t k) {
  if (self.numel() == 0) {
    return self;
  }
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  bool inplace;
  Tensor self_c;
  std::tie(inplace, self_c) = checkTrilTriuBatchContiguous(self, true);
  Tensor result = inplace ? self : at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(at::ScalarType::Half, at::ScalarType::Bool, self.scalar_type(), "tril", [&]{
    apply_triu_tril<scalar_t, false>(result, self_c, inplace, k);
  });
  if (!inplace) self.copy_(result);
  return self;
}

Tensor& tril_cpu_out(const Tensor& self, int64_t k, Tensor &result) {
  if (result.sizes() != self.sizes()) {
    result.resize_as_(self);
  }
  if (self.numel() == 0) {
    return result;
  }
  Tensor self_c;
  std::tie(std::ignore, self_c) = checkTrilTriuBatchContiguous(self, false);
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(at::ScalarType::Half, at::ScalarType::Bool, self.scalar_type(), "tril", [&]{
    apply_triu_tril<scalar_t, false>(result, self_c, false, k);
  });
  return result;
}

Tensor triu(const Tensor& self, int64_t k) {
  Tensor result = at::empty({0}, self.options());
  at::triu_out(result, self, k);
  return result;
}

Tensor& triu_cpu_(Tensor &self, int64_t k) {
  if (self.numel() == 0) {
    return self;
  }
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  bool inplace;
  Tensor self_c;
  std::tie(inplace, self_c) = checkTrilTriuBatchContiguous(self, true);
  Tensor result = inplace ? self : at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(at::ScalarType::Half, at::ScalarType::Bool, self.scalar_type(), "triu", [&]{
    apply_triu_tril<scalar_t, true>(result, self_c, inplace, k);
  });
  if (!inplace) self.copy_(result);
  return self;
}

Tensor& triu_cpu_out(const Tensor& self, int64_t k, Tensor &result) {
  if (result.sizes() != self.sizes()) {
    result.resize_as_(self);
  }
  if (self.numel() == 0) {
    return result;
  }
  Tensor self_c;
  std::tie(std::ignore, self_c) = checkTrilTriuBatchContiguous(self, false);
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(at::ScalarType::Half, at::ScalarType::Bool, self.scalar_type(), "triu", [&]{
    apply_triu_tril<scalar_t, true>(result, self_c, false, k);
  });
  return result;
}

Tensor trace_backward(const Tensor& grad, IntArrayRef sizes) {
  if (sizes.size() != 2) {
    throw std::runtime_error("expected matrix input");
  }

  auto grad_input = at::zeros(sizes[0] * sizes[1], grad.options());
  auto indices = at::arange(0, grad_input.numel(), sizes[1] + 1, grad.options().dtype(at::kLong));
  grad_input.index_fill_(0, indices, grad);
  return grad_input.view(sizes);
}


}  // namespace native
}  // namespace at
