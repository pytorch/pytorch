#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Config.h>
#include <ATen/Dispatch.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/sparse/ParamUtils.h>
#include <ATen/native/SparseTensorUtils.h>
#include <ATen/Parallel.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/CPUFunctions.h>
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_log_softmax_backward_data_cpu_dispatch.h>
#include <ATen/ops/_log_softmax_cpu_dispatch.h>
#include <ATen/ops/_softmax_backward_data_cpu_dispatch.h>
#include <ATen/ops/_softmax_cpu_dispatch.h>
#include <ATen/ops/_sparse_log_softmax.h>
#include <ATen/ops/_sparse_log_softmax_backward_data_native.h>
#include <ATen/ops/_sparse_log_softmax_native.h>
#include <ATen/ops/_sparse_softmax.h>
#include <ATen/ops/_sparse_softmax_backward_data_native.h>
#include <ATen/ops/_sparse_softmax_native.h>
#endif

#include <map>

namespace at::native {
namespace {

int64_t get_nvalues(const IntArrayRef& sizes, int64_t sparse_dim) {
  /* Return the number of entries in the dense part of a sparse tensor.

     `sizes` is a vector of sparse tensor dimensions.
     `sparse_dim` is the dimension of the sparse part of a sparse tensor.
   */
  return c10::multiply_integers(sizes.begin() + sparse_dim, sizes.end());
}

std::vector<int64_t> get_offsets(const Tensor& indices, const IntArrayRef& sizes, const int64_t dim) {
  /*
    Given the indices of a sparse tensor, return a vector of offsets
    for the entries in the equivalent dense tensor:

      If
        offsets = get_offsets(A._indices(), A.sizes(), -1)
        data = A.to_dense().resize((nnz,))
      then
        data[offsets[n]] == A._values()[n]

    `indices` must be a contiguous 2-d tensor with int64_t entries.
    `sizes` must be a vector with at least ndim entries.

    `dim` is an integer. When >= 0 and < ndim, the indices of all
    entries in the given dimension will be mapped to the index of the
    first entry before computing the offset. Otherwise, the value is
    ignored.

    For example, consider a sparse tensor

      11 ** ** 14 15
      ** 22 ** 24 **

    with

      indices = [[0, 0, 0, 1, 1],
                 [0, 3, 4, 1, 3]]

    then

      get_offsets(indices, (2, 5), -1) -> [0, 3, 4, 6, 8]
      get_offsets(indices, (2, 5), 0) -> [0, 3, 4, 1, 3]
      get_offsets(indices, (2, 5), 1) -> [0, 0, 0, 5, 5]

  */
  auto ndim = indices.size(0);
  auto nnz = indices.size(1);
  std::vector<int64_t> offsets(nnz);
  std::vector<int64_t> strides(ndim, 1);
  auto indices_accessor = indices.accessor<int64_t, 2>();

  if (ndim > 1) {
    for (int64_t i=ndim - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * sizes[i + 1];
    }
  }

  for (const auto i : c10::irange(nnz)) {
    int64_t acc = 0;
    for (const auto j : c10::irange(ndim)) {
      auto indices_row = indices_accessor[j];
      auto stride = strides[j];
      if (j != dim) {
        acc += stride * indices_row[i];
      }
    }
    offsets[i] = acc;
  }

  return offsets;
}

std::vector<std::vector<int64_t>> get_pools(const Tensor& indices, const IntArrayRef& sizes, const int64_t dim) {
  /*
    Return pools of indices that align with the given dimension.

    Parameters:
      `indices` - sparse tensor indices
      `sizes`   - sparse tensor dimensions
      `dim`     - given dimension

    Returns:
      `pools`   - a ragged array of indices

    A pool is defined as a list of indices (of sparse tensor values)
    that participate in the same softmax computation:

    - pools[i] intersection with pools[j] is empty iff i != j
    - union of all pools is set(range(nnz))
    - X.values[k], k in pools[i], does not affect the result of softmax(X)[n], n in pools[j], iff i != j

  */
  std::vector<std::vector<int64_t>> pools;

  auto ndim = indices.size(0);
  auto nnz = indices.size(1);
  std::vector<int64_t> strides(ndim, 1);
  auto indices_accessor = indices.accessor<int64_t, 2>();

  if (ndim > 1) {
    for (int64_t i=ndim - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * (i + 1 == dim? 1 : sizes[i + 1]);
    }
  }

  for (const auto i : c10::irange(nnz)) {
    int64_t pool_index = 0;
    for (const auto j : c10::irange(ndim)) {
      if (j != dim) {
        const auto indices_row = indices_accessor[j];
        const auto stride = strides[j];
        pool_index += stride * indices_row[i];
      }
    }
    if(static_cast<int64_t>(pools.size()) <= pool_index){
      pools.resize(pool_index + 1);
    }
    pools.at(pool_index).push_back(i);
  }

  return pools;
}

template <typename scalar_t, bool LogSoftMax>
void cpu_sparse_coo_softmax(Tensor output, const Tensor& input, const int64_t dim) {
  /*
    See test/test_sparse.py:test_softmax:sparse_softmax for the Python
    prototype of the sparse softmax algorithm that this implementation
    is based on.

    Derivation of the sparse softmax algorithm with an example
    ----------------------------------------------------------

    Consider the following 2-D sparse tensor with 0-D dense part as an
    example, denote it by X:

      11 ** ** 14 15
      ** 22 ** 24 **

    where `**` represent unspecified entries. The COO sparse tensor
    representation of X is:

      indices = [[0, 1, 0, 1, 0],
                 [0, 1, 3, 3, 4]]
      values = [11, 22, 14, 24, 15]

    that after coalescing becomes

      indices = [[0, 0, 0, 1, 1],
                 [0, 3, 4, 1, 3]]
      values = [11, 14, 15, 22, 24]

    The softmax of X along the given dimension d is defined as

      S_d[i, j] = exp(X[i, j]) / sum(exp(X[I_d[k]]), k=0..X.shape[d]-1)

    where the index tuple I_d[k] is defined as

      I_0[k] = k, j
      I_1[k] = i, k

    For sparse tensors, the unspecified entries are skipped in the
    softmax sum of exponents so that the result will be sparse tensor
    with the same indices as the input. Mathematically, this
    corresponds to the case where the unspecified entries are
    interpreted as negative infinities rather than zeros.

    To minimize the defects from numerical evaluation of exponents
    with very large or small arguments, the softmax implementation
    uses the following a numerically stable definition:

      S_d[i, j] = exp(X[i, j] - maxX_d) / sum(exp(X[I_d[k]] - maxX_d), k=0...X.shape[d]-1)

    where

      maxX_d = max(X[I_d[k]], k=0...X.shape[d]-1)

    is the maximum tensor along the direction d (it has dimensionality
    `maxX_d.ndim = X.ndim - 1`).

    For the example sparse tensor X, we have:

      S_0._indices() == S_1._indices() == X._indices()

      maxX_0 = [11, 22, -inf, 24, 15]
      maxX_1 = [15, 24]

      S_0._values() = [exp(11 - maxX_0[0]) / exp(11 - maxX_0[0]),
                       exp(14 - maxX_0[3]) / (exp(14 - maxX_0[3]) + exp(24 - maxX_0[3])),
                       exp(15 - maxX_0[4]) / exp(15 - maxX_0[4]),
                       exp(22 - maxX_0[1]) / exp(22 - maxX_0[1]),
                       exp(24 - maxX_0[3]) / (exp(14 - maxX_0[3]) + exp(24 - maxX_0[3]))]
                    = [1, exp(-10)/(exp(-10) + 1), 1, 1, 1/(exp(-10) + 1)]

      (note that `maxX_0[2] == -inf` not used to obtain S_0)

      S_1._values() = [exp(11 - maxX_1[0]) / (exp(11 - maxX_1[0]) + exp(14 - maxX_1[0]) + exp(15 - maxX_1[0])),
                       exp(14 - maxX_1[0]) / (exp(11 - maxX_1[0]) + exp(14 - maxX_1[0]) + exp(15 - maxX_1[0])),
                       exp(15 - maxX_1[0]) / (exp(11 - maxX_1[0]) + exp(14 - maxX_1[0]) + exp(15 - maxX_1[0])),
                       exp(22 - maxX_1[1]) / (exp(22 - maxX_1[1]) + exp(24 - maxX_1[1])),
                       exp(24 - maxX_1[1]) / (exp(22 - maxX_1[1]) + exp(24 - maxX_1[1]))]
                    = [exp(-4) / (exp(-4) + exp(-1) + 1),
                       exp(-1) / (exp(-4) + exp(-1) + 1),
                       1 / (exp(-4) + exp(-1) + 1),
                       exp(-2) / (exp(-2) + 1),
                       1 / (exp(-2) + 1)]

    To obtain the above via the for-loop over
    `nnz(=len(X._values()))`, we introduce the indices mapping `pool`
    as follows:

      indices = X._indices()
      for i in range(nnz):
          for j in range(nnz):
              if indices[d, i] == indices[d, j]:
                  assert pool_d[i] == pool_d[j]
              else:
                  assert pool_d[i] != pool_d[j]

    that is, the entries with values indices i and j are in the same
    pool iff their locations in the grid of tensor indices align with
    the direction along which the softmax is calculated. The `pool`
    mapping maps the X._values() indices to the corresponding pool
    index.

    To save memory and processor resources, we pre-compute the entries
    of maxX tensor and the sums of exponents as follows:

      mx_d = [max(values[i] for i in range(nnz) if pool_0[i] == k) for k in pool_d]
      exp_sum_d = [sum(exp(values[i] - mx_d[k]) for i in range(nnz) if pool_d[i] == k) for k in pool_d]

    For example, if

      pool_0 = [0, 1, 2, 3, 1]
      pool_1 = [0, 0, 0, 1, 1]

    then

      mx_0 = [11, 24, 15, 22]
      mx_1 = [15, 24]
      exp_sum_0 = [1, (exp(-10) + 1), 1, 1]
      exp_sum_1 = [(exp(-4) + exp(-1) + 1), (exp(-2) + 1)]

    and

      S_0._values() = [exp(11 - mx_0[pool_0[0]]) / exp_sum_0[pool_0[0]]
                       exp(14 - mx_0[pool_0[1]]) / exp_sum_0[pool_0[1]]
                       exp(15 - mx_0[pool_0[2]]) / exp_sum_0[pool_0[2]]
                       exp(22 - mx_0[pool_0[3]]) / exp_sum_0[pool_0[3]]
                       exp(24 - mx_0[pool_0[4]]) / exp_sum_0[pool_0[4]]

    or in general,

      S_d._values() = [exp(values[i] - mx_d[pool_d[i]]) / exp_sum_d[pool_d[i] for i in range(nnz)]

    The above algorithm can be easily extended for cases with
    non-scalar dense part of the sparse tensor where all scalar
    operations become element-wise tensor operations.

    The implementation below has more optimizations such as that
    collect pool indices for enabling concurrency, minimize the calls
    to exp functions as well as reuse of softmax implementation for
    log_softmax.
  */
  auto sparse_dim = input.sparse_dim();
  auto indices = input._indices().contiguous();
  auto values = input._values().contiguous();
  auto out_values = output._values();
  auto out_indices = output._indices();
  out_values.resize_as_(values);
  out_indices.resize_as_(indices);
  out_indices.copy_(indices);

  if (dim >= sparse_dim) {
    if (LogSoftMax) {
      auto new_values =
          at::cpu::_log_softmax(values, dim - sparse_dim + 1, false);
      out_values.set_(new_values);
    } else {
      auto new_values = at::cpu::_softmax(values, dim - sparse_dim + 1, false);
      out_values.set_(new_values);
    }
    return;
  }

  auto nnz = values.size(0);
  auto sizes = input.sizes();
  auto nvalues = get_nvalues(sizes, sparse_dim);

  /* Prepare accessors */
  auto values_2 = values.view({nnz, nvalues});
  auto values_accessor = values_2.accessor<scalar_t, 2>();

  auto out_values_2 = out_values.view({nnz, nvalues});
  auto out_values_accessor = out_values_2.accessor<scalar_t, 2>();

  /* Compute independent pools of indices */
  auto pools = get_pools(indices, sizes, dim);

  int64_t grain_size = 1;
  parallel_for(0, pools.size(), grain_size, [&](int64_t begin, int64_t end) {
      for (const auto p : c10::irange(begin, end)) {
        auto pool_indices = pools[p];

        // Skip empty pools
        if (pool_indices.empty())
          continue;

        /* Prepare scratch space */
        std::vector<scalar_t> mx_row(nvalues, -std::numeric_limits<scalar_t>::infinity());
        std::vector<scalar_t> exp_sums_row(nvalues, 0);

        /* Compute mx */
        for (int64_t i : pool_indices) {
          auto values_row = values_accessor[i];
          for (const auto j : c10::irange(nvalues)) {
            mx_row[j] = std::max(mx_row[j], values_row[j]);
          }
        }

        /* Apply exp to (v - mx) and sum the results */
        for (int64_t i : pool_indices) {
          auto values_row = values_accessor[i];
          auto out_values_row = out_values_accessor[i];
          for (const auto j : c10::irange(nvalues)) {
            auto v = std::exp(values_row[j] - mx_row[j]);
            if (!LogSoftMax) {
              out_values_row[j] = v;
            }
            exp_sums_row[j] += v;
          }
        }

        for (const auto j : c10::irange(nvalues)) {
          if (LogSoftMax) {
            mx_row[j] += std::log(exp_sums_row[j]);
          } else {
            exp_sums_row[j] = 1.0 / exp_sums_row[j];
          }
        }

        /* Normalize with the sum of exponents */
        for (int64_t i : pool_indices) {
          auto values_row = values_accessor[i];
          auto out_values_row = out_values_accessor[i];
          for (const auto j : c10::irange(nvalues)) {
            if (LogSoftMax) {
              out_values_row[j] = values_row[j] - mx_row[j];
            } else {
              out_values_row[j] *= exp_sums_row[j];
            }
          }
        }
      }
    });
}

template <typename scalar_t, bool LogSoftMax>
void cpu_sparse_coo_softmax_backward(const Tensor& grad_input, const Tensor& grad, const Tensor& output, const int64_t dim, ScalarType input_dtype) {
  /*

    If LogSoftMax == false, then

      gI_i = sum_j d<output_j>/d<input_i> * grad_j = sum_j output_i * (1[i==j] - output_j) * grad_j
           = output_i * (grad_i - sum_j output_j * grad_j)

    else

      gI_i = (1-exp(output_i)) * grad_i - sum_{j} 1[i!=j] * exp(output_i) * grad_j
           = grad_i - exp(output_i) * sum_j grad_j.

    where

      i, j in range(shape[dim])
      x_i = x[..., i_dim, ...]
      output.sparse_dim() == grad.sparse_dim()
  */
  auto sparse_dim = output.sparse_dim();
  auto sizes = output.sizes().vec();
  auto grad_indices = grad._indices().contiguous();
  auto grad_values = grad._values().contiguous();
  auto out_indices = output._indices().contiguous();
  auto out_values = output._values().contiguous();
  auto values = grad_input._values();
  auto indices = grad_input._indices();
  auto out_nnz = out_values.size(0);
  auto grad_nnz = grad_values.size(0);

  values.resize_as_(out_values);
  values.zero_();
  indices.resize_as_(out_indices);
  indices.copy_(out_indices);

  auto out_offsets = get_offsets(out_indices, sizes, -1);
  auto grad_offsets = get_offsets(grad_indices, sizes, -1);

  if (dim >= sparse_dim) {
    if (out_offsets == grad_offsets) {
      if (LogSoftMax) {
        auto r = at::cpu::_log_softmax_backward_data(
            grad_values, out_values, dim - sparse_dim + 1, input_dtype);
        values.set_(r);
      } else {
        auto r = at::cpu::_softmax_backward_data(grad_values, out_values, dim - sparse_dim + 1, input_dtype);
        values.set_(r);
      }
    } else {
      for (const auto i : c10::irange(out_nnz)) {
        auto low = std::lower_bound(grad_offsets.begin(), grad_offsets.end(), out_offsets[i]);
        auto j = low - grad_offsets.begin();
        if (j < grad_nnz && out_offsets[i] == grad_offsets[j]) {
          if (LogSoftMax) {
            auto r = at::cpu::_log_softmax_backward_data(
                grad_values[j], out_values[i], dim - sparse_dim, input_dtype);
            values[i].copy_(r);
          } else {
            auto r = at::cpu::_softmax_backward_data(grad_values[j], out_values[i], dim - sparse_dim, input_dtype);
            values[i].copy_(r);
          }
        }
      }
    }
    return;
  }

  auto nnz = values.size(0);
  auto nvalues = get_nvalues(sizes, sparse_dim);

  auto values_2 = values.view({nnz, nvalues});
  auto values_accessor = values_2.accessor<scalar_t, 2>();

  auto out_values_2 = out_values.view({out_nnz, nvalues});
  auto out_values_accessor = out_values_2.accessor<scalar_t, 2>();

  auto grad_values_2 = grad_values.view({grad_nnz, nvalues});
  auto grad_values_accessor = grad_values_2.accessor<scalar_t, 2>();

  /* Compute independent pools of indices */
  auto pools = get_pools(out_indices, sizes, dim);

  int64_t grain_size = 1;
  parallel_for(0, pools.size(), grain_size, [&](int64_t begin, int64_t end) {
      for (const auto p : c10::irange(begin, end)) {
        auto pool_indices = pools[p];

        // Skip empty pools
        if (pool_indices.empty())
          continue;

        std::vector<scalar_t> tmp_row(nvalues, 0);

        /* Compute tmp = - sum_j output_j * grad_j */
        for (int64_t i : pool_indices) {
          auto out_values_row = out_values_accessor[i];
          auto low = std::lower_bound(grad_offsets.begin(), grad_offsets.end(), out_offsets[i]);
          auto j = low - grad_offsets.begin();

          if (j < grad_nnz && (out_offsets[i] == grad_offsets[j])) {
            auto grad_values_row = grad_values_accessor[j];
            for (const auto k : c10::irange(nvalues)) {
              if (LogSoftMax) {
                tmp_row[k] -= grad_values_row[k];
              } else {
                tmp_row[k] -= out_values_row[k] * grad_values_row[k];
              }
            }
          }
        }

        /* Compute grad_input = output * (grad + tmp)*/
        for (int64_t i : pool_indices) {
          auto out_values_row = out_values_accessor[i];
          auto values_row = values_accessor[i];
          auto low = std::lower_bound(grad_offsets.begin(), grad_offsets.end(), out_offsets[i]);
          auto j = low - grad_offsets.begin();

          if (j < grad_nnz && (out_offsets[i] == grad_offsets[j])) {
            auto grad_values_row = grad_values_accessor[j];
            for (const auto k : c10::irange(nvalues)) {
              if (LogSoftMax) {
                values_row[k] = grad_values_row[k] + std::exp(out_values_row[k]) * tmp_row[k];
              } else {
                values_row[k] = out_values_row[k] * (grad_values_row[k] + tmp_row[k]);
              }
            }
          } else {
            for (const auto k : c10::irange(nvalues)) {
              if (LogSoftMax) {
                values_row[k] = std::exp(out_values_row[k]) * tmp_row[k];
              } else {
                values_row[k] = out_values_row[k] * (tmp_row[k]);
              }
            }
          }
        }
      }
    });
}

} // anonymous namespace

Tensor softmax_sparse_cpu(
    const Tensor& input_,
    const int64_t dim_,
    const bool half_to_float) {
  Tensor input, output;
  int64_t dim;
  std::tie(input, output, dim) = softmax_sparse_input_preprocessing(
      input_, dim_, half_to_float, "softmax");
  if (input.numel() == 0) {
    return output;
  }
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "softmax", [&] {
    cpu_sparse_coo_softmax<scalar_t, false>(output, input, dim);
  });
  return output;
}

Tensor log_softmax_sparse_cpu(
    const Tensor& input_,
    const int64_t dim_,
    const bool half_to_float) {
  Tensor input, output;
  int64_t dim;
  std::tie(input, output, dim) = softmax_sparse_input_preprocessing(
      input_, dim_, half_to_float, "log_softmax");
  if (input.numel() == 0) {
    return output;
  }
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "log_softmax", [&] {
    cpu_sparse_coo_softmax<scalar_t, true>(output, input, dim);
  });
  return output;
}

Tensor softmax_backward_sparse_cpu(
    const Tensor& grad_,
    const Tensor& output_,
    int64_t dim_,
    const Tensor& input_) {
  Tensor grad_input, grad, output;
  int64_t dim;
  std::tie(grad_input, grad, output, dim) =
      softmax_backward_sparse_input_preprocessing(
          grad_, output_, dim_, input_, "softmax_backward");
  if (output.numel() == 0) {
    return grad_input;
  }
  AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), "softmax_backward", [&] {
    cpu_sparse_coo_softmax_backward<scalar_t, false>(
        grad_input, grad, output, dim_, input_.scalar_type());
  });
  return grad_input;
}

Tensor log_softmax_backward_sparse_cpu(
    const Tensor& grad_,
    const Tensor& output_,
    int64_t dim_,
    const Tensor& input_) {
  Tensor grad_input, grad, output;
  int64_t dim;
  std::tie(grad_input, grad, output, dim) =
      softmax_backward_sparse_input_preprocessing(
          grad_, output_, dim_, input_, "log_softmax_backward");
  if (output.numel() == 0) {
    return grad_input;
  }
  AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), "log_softmax_backward", [&] {
    cpu_sparse_coo_softmax_backward<scalar_t, true>(
        grad_input, grad, output, dim_, input_.scalar_type());
  });
  return grad_input;
}

Tensor _sparse_softmax(const Tensor& input_, const int64_t dim_, std::optional<ScalarType> dtype) {
  auto result = [&]() {
    NoNamesGuard guard;
    if (input_.is_cuda() && input_.scalar_type() == ScalarType::Half && dtype == ScalarType::Float){
        return at::_sparse_softmax(input_, dim_, true);
    } else {
        Tensor converted = dtype.has_value() ? input_.toType(dtype.value()) : input_;
        return at::_sparse_softmax(converted, dim_, false);
    }
  }();
  namedinference::propagate_names(result, input_);
  return result;
}

Tensor _sparse_softmax(const Tensor& self, Dimname dim, std::optional<ScalarType> dtype) {
  return at::_sparse_softmax(self, dimname_to_position(self, dim), dtype);
}

Tensor _sparse_log_softmax(const Tensor& input_, const int64_t dim_, std::optional<ScalarType> dtype) {
  auto result = [&]() {
    NoNamesGuard guard;
    if (input_.is_cuda() && input_.scalar_type() == ScalarType::Half && dtype == ScalarType::Float){
        return at::_sparse_log_softmax(input_, dim_, true);
    } else {
        Tensor converted = dtype.has_value() ? input_.toType(dtype.value()) : input_;
        return at::_sparse_log_softmax(converted, dim_, false);
    }
  }();
  namedinference::propagate_names(result, input_);
  return result;
}

Tensor _sparse_log_softmax(const Tensor& self, Dimname dim, std::optional<ScalarType> dtype) {
  return at::_sparse_log_softmax(self, dimname_to_position(self, dim), dtype);
}

} // namespace at::native
