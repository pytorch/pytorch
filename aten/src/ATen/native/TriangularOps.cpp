#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorMeta.h>
#include <ATen/native/TriangularOpsUtils.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/arange.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/trace_backward_native.h>
#include <ATen/ops/tril_native.h>
#include <ATen/ops/triu_native.h>
#include <ATen/ops/zeros.h>
#endif

namespace at {
namespace meta {

TORCH_META_FUNC(tril)(const Tensor& self, int64_t k) {
  set_output_raw_strided(0, self.sizes(), {}, self.options());
}

TORCH_META_FUNC(triu)(const Tensor& self, int64_t k) {
  set_output_raw_strided(0, self.sizes(), {}, self.options());
}

}  // namespace meta

namespace native {
namespace {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ triu/tril ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t>
void apply_triu_tril_single(
    scalar_t* result,
    scalar_t* self,
    bool inplace,
    int64_t k,
    int64_t n,
    int64_t m,
    int64_t res_row_stride,
    int64_t res_col_stride,
    int64_t self_row_stride,
    int64_t self_col_stride,
    bool upper) {
  constexpr int64_t zero = 0;

  if (upper) {
    parallel_for(0, n, 0, [&](int64_t start, int64_t end) {
      for (int64_t i : c10::irange(start, end)) {
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
    parallel_for(0, n, 0, [&](int64_t start, int64_t end) {
      for (int64_t i : c10::irange(start, end)) {
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

template <typename scalar_t>
void apply_triu_tril(const Tensor& result, const Tensor& self, bool inplace, int64_t k, bool upper) {
  auto n = self.size(-2);
  auto m = self.size(-1);
  auto self_data = self.data_ptr<scalar_t>();
  auto self_stride = (self.dim() > 2 && self.stride(-3) > 0) ? self.stride(-3) : 1;
  auto batchsize = batchCountTrilTriu(result);
  auto self_row_stride = self.stride(-2);
  auto self_col_stride = self.stride(-1);

  auto result_data = result.data_ptr<scalar_t>();
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t result_stride, result_row_stride, result_col_stride;
  if (result_data != self_data) {
    result_stride = (result.dim() > 2 && result.stride(-3) > 0) ? result.stride(-3) : 1;
    result_row_stride = result.stride(-2);
    result_col_stride = result.stride(-1);
  } else {
    result_stride = self_stride;
    result_row_stride = self_row_stride;
    result_col_stride = self_col_stride;
  }

  parallel_for(0, batchsize, 0, [&](int64_t start, int64_t end) {
    for (const auto b : c10::irange(start, end)) {
      scalar_t* self_batch = &self_data[b * self_stride];
      scalar_t* result_batch = &result_data[b * result_stride];
      apply_triu_tril_single<scalar_t>(
          result_batch,
          self_batch,
          inplace,
          k,
          n,
          m,
          result_row_stride,
          result_col_stride,
          self_row_stride,
          self_col_stride,
          upper);
    }
  });
}

struct UpperTriangle {
  static constexpr const char* op_name = "triu";
  static constexpr bool upper = true;
};

struct LowerTriangle {
  static constexpr const char *op_name = "tril";
  static constexpr bool upper = false;
};

template <typename Triangle>
void compute_triu_tril(const Tensor& self, int64_t k, const Tensor &result) {
  if (self.numel() == 0) {
    return;
  }

  bool inplace_op = self.is_same(result);

  bool inplace_update = false;
  Tensor self_c;
  std::tie(inplace_update, self_c) = checkTrilTriuBatchContiguous(self, inplace_op);

  Tensor result_c;
  if (inplace_op && !inplace_update) {
    result_c = at::empty_like(result, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  } else {
    result_c = result;
  }

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      ScalarType::BFloat16,
      ScalarType::Half,
      ScalarType::Bool,
      self.scalar_type(),
      Triangle::op_name,
      [&]{
        apply_triu_tril<scalar_t>(result_c, self_c, inplace_op && inplace_update, k, Triangle::upper);
      });

  if (inplace_op && !inplace_update) {
    result.copy_(result_c);
  }
}

}  // namespace

TORCH_IMPL_FUNC(tril_cpu)(const Tensor& self, int64_t k, const Tensor &result) {
  compute_triu_tril<LowerTriangle>(self, k, result);
}

TORCH_IMPL_FUNC(triu_cpu)(const Tensor& self, int64_t k, const Tensor &result) {
  compute_triu_tril<UpperTriangle>(self, k, result);
}

Tensor trace_backward(const Tensor& grad, at::IntArrayRef sizes) {
    return at::native::trace_backward_symint(grad, c10::fromIntArrayRefSlow(sizes));
}

Tensor trace_backward_symint(const Tensor& grad, c10::SymIntArrayRef sizes) {
  if (sizes.size() != 2) {
    throw std::runtime_error("expected matrix input");
  }

  auto grad_input = at::zeros_symint(sizes[0] * sizes[1], grad.options());
  auto indices = at::arange(0, grad_input.numel(), sizes[1] + 1, grad.options().dtype(at::kLong));
  // for composite compliance, use out-of-place variant of
  // `index_fill` if grad tensor is a Tensor Subclass.
  if (isTensorSubclassLike(grad)) {
    grad_input = grad_input.index_fill(0, indices, grad);
  } else {
    grad_input.index_fill_(0, indices, grad);
  }
  return grad_input.view_symint(sizes);
}

}  // namespace native
}  // namespace at
