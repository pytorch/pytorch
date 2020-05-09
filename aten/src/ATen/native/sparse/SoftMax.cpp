#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/Parallel.h>
#include <ATen/NamedTensorUtils.h>
#include <map>

namespace at {
namespace native {
namespace {

template <typename iscalar_t>
iscalar_t get_nvalues(const std::vector<iscalar_t>& sizes, const int64_t sparse_dim) {
  /* Return the number of entries in the dense part of a sparse tensor.

     `sizes` is a vector of sparse tensor dimensions.
     `sparse_dim` is the dimension of the sparse part of a sparse tensor.
   */
  iscalar_t nvalues = 1;
  for (auto it = sizes.begin() + sparse_dim; it != sizes.end(); ++it) {
    nvalues *= *it;
  }
  return nvalues;
}

template <typename iscalar_t>
std::vector<iscalar_t> get_offsets(const Tensor& indices, const std::vector<iscalar_t>& sizes, const int64_t dim) {
  /*
    Given the indices of a sparse tensor, return a vector of offsets
    for the entries in the equivalent dense tensor:

      If
        offsets = get_offsets(A._indices(), A.sizes(), -1)
        data = A.to_dense().resize((nnz,))
      then
        data[offsets[n]] == A._values()[n]

    `indices` must be a contiguous 2-d tensor with iscalar_t entries.
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

    This function together with `get_dense_offsets` defined below are
    used to compute the indices mapping required in the sparse softmax
    algorithm (see below).
  */
  auto ndim = indices.size(0);
  auto nnz = indices.size(1);
  std::vector<iscalar_t> offsets(nnz, 0);
  std::vector<iscalar_t> strides(ndim, 1);
  iscalar_t* indices_data_base = indices.data_ptr<iscalar_t>();

  if (ndim > 1) {
    for (int64_t i=ndim - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * sizes[i + 1];
    }
  }

  for (int64_t j=0; j < ndim; j++) {
    if (j != dim) {
      for (int64_t i=0; i < nnz; i++) {
        offsets[i] += strides[j] * indices_data_base[j * nnz + i];
      }
    }
  }

  return offsets;
}

template <typename iscalar_t>
std::vector<iscalar_t> get_dense_offsets(iscalar_t &mx, const std::vector<iscalar_t>& offsets) {
  /*
    Return the dense set of offsets:

      If
        pool = get_dense_offsets(mx, offsets)
      then
        pool.size() == offsets.size()
        pool[i] == pool[j] iff offsets[i] == offsets[j]
        set(pool) == set(range(mx))

    and the size of the pool via changing `mx` argument in-place.

    This function remaps a list of integers to a dense list of
    integers. For example,

      get_dense_offsets([0, 3, 4, 1, 3]) -> [0, 1, 2, 3, 1]
  */
  auto n = offsets.size();
  std::vector<iscalar_t> pool(n, 0);
  std::map<iscalar_t, iscalar_t> i2p;
  for (int64_t i=0; i < n; i++) {
    auto c = offsets[i];
    auto it = i2p.find(c);
    iscalar_t p = i2p.size();
    if (it == i2p.end()) {
      i2p.emplace(std::make_pair(c, p));
    } else {
      p = it->second;
    }
    pool[i] = p;
  }
  mx = i2p.size();
  return pool;
}

template <typename scalar_t, typename iscalar_t, bool LogSoftMax>
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
    minimize the calls to exp functions as well as reuse of softmax
    implementation for log_softmax.
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
      auto new_values = log_softmax_cpu(values, dim - sparse_dim + 1, false);
      out_values.copy_(new_values);
    } else {
      auto new_values = softmax_cpu(values, dim - sparse_dim + 1, false);
      out_values.copy_(new_values);
    }
    return;
  }

  auto nnz = values.size(0);
  scalar_t* values_data_base = values.data_ptr<scalar_t>();
  iscalar_t* indices_data_base = indices.data_ptr<iscalar_t>();
  scalar_t* out_values_data_base = out_values.data_ptr<scalar_t>();
  auto sizes = input.sizes().vec();
  auto nvalues = get_nvalues(sizes, sparse_dim);

  /* Compute pool indices */
  std::vector<iscalar_t> offsets = get_offsets<iscalar_t>(indices, sizes, dim);
  iscalar_t mx_p;
  std::vector<iscalar_t> pool = get_dense_offsets<iscalar_t>(mx_p, offsets);

  /* Compute mx - a max tensor along sparse dimension */
  std::vector<int64_t> mx_sizes(sizes.begin() + sparse_dim - 1, sizes.end());
  mx_sizes[0] = mx_p;
  at::Tensor mx = at::empty(mx_sizes, values.options());
  scalar_t* mx_data_base = mx.data_ptr<scalar_t>();
  {
    auto ninf = -std::numeric_limits<scalar_t>::infinity();
    for (int64_t j=0; j < nvalues * mx_p; j++) {
      mx_data_base[j] = ninf;
    }
  }

  for (int64_t i=0; i < nnz; i++) {
    auto p = pool[i];
    scalar_t* mx_data = mx_data_base + p * nvalues;
    scalar_t* values_data = values_data_base + i * nvalues;
    for (int64_t j=0; j < nvalues; j++) {
      mx_data[j] = std::max(mx_data[j], values_data[j]);
    }
  }

  /* apply exp to (v - mx) and sum the results */
  at::Tensor exp_sums = at::zeros_like(mx);
  scalar_t* exp_sums_data_base = exp_sums.data_ptr<scalar_t>();
  for (int64_t i=0; i < nnz; i++) {
    auto p = pool[i];
    scalar_t* mx_data = mx_data_base + p * nvalues;
    scalar_t* values_data = values_data_base + i * nvalues;
    scalar_t* out_values_data = out_values_data_base + i * nvalues;
    scalar_t* exp_sums_data = exp_sums_data_base + p * nvalues;
    for (int64_t j=0; j < nvalues; j++) {
      auto v = std::exp(values_data[j] - mx_data[j]);
      if (!LogSoftMax) {
        out_values_data[j] = v;
      }
      exp_sums_data[j] += v;
    }
  }

  for (int64_t j=0; j < nvalues * mx_p; j++) {
    if (LogSoftMax) {
      mx_data_base[j] += std::log(exp_sums_data_base[j]);
    } else {
      exp_sums_data_base[j] = 1.0 / exp_sums_data_base[j];
    }
  }

  /* normalize with the sum of exponents */
  for (int64_t i=0; i < nnz; i++) {
    auto p = pool[i];
    scalar_t* values_data = values_data_base + i * nvalues;
    scalar_t* out_values_data = out_values_data_base + i * nvalues;
    scalar_t* exp_sums_data = exp_sums_data_base + p * nvalues;
    scalar_t* mx_data = mx_data_base + p * nvalues;
    for (int64_t j=0; j < nvalues; j++) {
      if (LogSoftMax) {
        out_values_data[j] = values_data[j] - mx_data[j];
      } else {
        out_values_data[j] *= exp_sums_data[j];
      }
    }
  }

}

template <typename scalar_t, typename iscalar_t, bool LogSoftMax>
void cpu_sparse_coo_softmax_backward(Tensor& grad_input, const Tensor& grad, const Tensor& output, const int64_t dim) {
  /*
    gI_i = sum_j d<output_i>/d<input_j> * grad_j = sum_j output_i * (1[i==j] - output_j) * grad_j
         = output_i * (grad_i - sum_j output_j * grad_j)

    i, j in range(shape[dim])
    x_i = x[..., i_dim, ...]

    assuming output.sparse_dim() == grad.sparse_dim(), TODO: impl check
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
    for(int64_t i=0; i<out_nnz; i++) {
      Tensor unused;
      auto low = std::lower_bound(grad_offsets.begin(), grad_offsets.end(), out_offsets[i]);
      auto j = low - grad_offsets.begin();
      if (j < grad_nnz && out_offsets[i] == grad_offsets[j]) {
        if (LogSoftMax) {
          auto r = log_softmax_backward_cpu(grad_values[j], out_values[i], dim - sparse_dim, unused);
          values[i].copy_(r);
        } else {
          auto r = softmax_backward_cpu(grad_values[j], out_values[i], dim - sparse_dim, unused);
          values[i].copy_(r);
        }
      }
    }
    return;
  }

  std::vector<iscalar_t> offsets = get_offsets<iscalar_t>(out_indices, sizes, dim);
  iscalar_t mx_p;
  std::vector<iscalar_t> pool = get_dense_offsets<iscalar_t>(mx_p, offsets);
  std::vector<int64_t> tmp_sizes(sizes.begin() + sparse_dim - 1, sizes.end());
  tmp_sizes[0] = mx_p;
  at::Tensor tmp = at::empty(tmp_sizes, out_values.options());
  tmp.zero_();
  scalar_t* tmp_data_base = tmp.data_ptr<scalar_t>();
  scalar_t* out_values_data_base = out_values.data_ptr<scalar_t>();
  iscalar_t* out_indices_data_base = out_indices.data_ptr<iscalar_t>();
  scalar_t* grad_values_data_base = grad_values.data_ptr<scalar_t>();
  iscalar_t* grad_indices_data_base = grad_indices.data_ptr<iscalar_t>();
  scalar_t* values_data_base = values.data_ptr<scalar_t>();
  iscalar_t* indices_data_base = indices.data_ptr<iscalar_t>();

  auto nvalues = get_nvalues(sizes, sparse_dim);

  /* Compute tmp = - sum_j output_j * grad_j */
  for (int64_t i=0; i<out_nnz; i++) {
    auto p = pool[i];
    auto low = std::lower_bound(grad_offsets.begin(), grad_offsets.end(), out_offsets[i]);
    auto j = low - grad_offsets.begin();
    scalar_t* tmp_data = tmp_data_base + p * nvalues;
    scalar_t* out_values_data = out_values_data_base + i * nvalues;
    if (j < grad_nnz && (out_offsets[i] == grad_offsets[j])) {
      scalar_t* grad_values_data = grad_values_data_base + j * nvalues;
      for (int64_t k=0; k<nvalues; k++) {
        if (LogSoftMax) {
          tmp_data[k] -= grad_values_data[k];
        } else {
          tmp_data[k] -= out_values_data[k] * grad_values_data[k];
        }
      }
    }
  }

  /* Compute grad_input = output * (grad + tmp)*/
  for (int64_t i=0; i<out_nnz; i++) {
    auto p = pool[i];
    auto low = std::lower_bound(grad_offsets.begin(), grad_offsets.end(), out_offsets[i]);
    auto j = low - grad_offsets.begin();
    scalar_t* tmp_data = tmp_data_base + p * nvalues;
    scalar_t* out_values_data = out_values_data_base + i * nvalues;
    scalar_t* values_data = values_data_base + i * nvalues;
    if (j < grad_nnz && (out_offsets[i] == grad_offsets[j])) {
      scalar_t* grad_values_data = grad_values_data_base + j * nvalues;
      for (int64_t k=0; k<nvalues; k++) {
        if (LogSoftMax) {
          values_data[k] = grad_values_data[k] + std::exp(out_values_data[k]) * tmp_data[k];
        } else {
          values_data[k] = out_values_data[k] * (grad_values_data[k] + tmp_data[k]);
        }
      }
    } else {
      for (int64_t k=0; k<nvalues; k++) {
        if (LogSoftMax) {
          values_data[k] = std::exp(out_values_data[k]) * tmp_data[k];
        } else {
          values_data[k] = out_values_data[k] * (tmp_data[k]);
        }
      }
    }
  }

}

} // namespace

Tensor softmax_sparse_cpu(const Tensor& input_, const int64_t dim_, const bool half_to_float) {
  AT_ASSERT(input_.is_sparse());
  AT_ASSERTM(!half_to_float, "softmax with half to float conversion is not supported on CPU");
  auto input = input_.coalesce();
  Tensor output = at::native::empty_like(input);
  if (input.numel() == 0) {
    return output;
  }
  TORCH_CHECK(dim_ >= 0 && dim_ < input.dim(),
              "dim must be non-negative and less than input dimensions");
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "softmax", [&] {
      // assuming that the type of input._indices() entries is int64_t
      cpu_sparse_coo_softmax<scalar_t, int64_t, false>(output, input, dim_);
  });
  return output;
}

Tensor log_softmax_sparse_cpu(const Tensor& input_, const int64_t dim_, const bool half_to_float) {
  AT_ASSERT(input_.is_sparse());
  AT_ASSERTM(!half_to_float, "log_softmax with half to float conversion is not supported on CPU");
  auto input = input_.coalesce();
  Tensor output = at::native::empty_like(input);
  if (input.numel() == 0) {
    return output;
  }
  TORCH_CHECK(dim_ >= 0 && dim_ < input.dim(),
              "dim must be non-negative and less than input dimensions");
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "log_softmax", [&] {
      // assuming that the type of input._indices() entries is int64_t
      cpu_sparse_coo_softmax<scalar_t, int64_t, true>(output, input, dim_);
  });
  return output;
}

Tensor softmax_backward_sparse_cpu(
    const Tensor& grad_,
    const Tensor& output_,
    int64_t dim_,
    const Tensor& input_) {
  TensorArg grad_arg{grad_, "grad", 1}, output_arg{output_, "output", 2};
  checkSameSize("softmax_backward", grad_arg, output_arg);

  int64_t dim = maybe_wrap_dim(dim_, grad_.dim());

  auto grad = grad_.coalesce();
  auto output = output_.coalesce();

  Tensor grad_input = at::native::empty_like(output);
  if (output.numel() == 0) {
    return grad_input;
  }
  TORCH_CHECK(
      dim >= 0 && dim < grad.dim(),
      "dim must be non-negative and less than input dimensions");
  TORCH_CHECK(
              grad.sparse_dim() == output.sparse_dim(),
      "grad and output sparse dimensions must be equal");
  AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), "softmax_backward", [&] {
      // assuming that the type of input._indices() entries is int64_t
      cpu_sparse_coo_softmax_backward<scalar_t, int64_t, false>(grad_input, grad, output, dim);
  });
  return grad_input;
}

Tensor log_softmax_backward_sparse_cpu(
    const Tensor& grad_,
    const Tensor& output_,
    int64_t dim_,
    const Tensor& input_) {
  TensorArg grad_arg{grad_, "grad", 1}, output_arg{output_, "output", 2};
  checkSameSize("log_softmax_backward", grad_arg, output_arg);

  int64_t dim = maybe_wrap_dim(dim_, grad_.dim());

  auto grad = grad_.coalesce();
  auto output = output_.coalesce();

  Tensor grad_input = at::native::empty_like(output);
  if (output.numel() == 0) {
    return grad_input;
  }
  TORCH_CHECK(
      dim >= 0 && dim < grad.dim(),
      "dim must be non-negative and less than input dimensions");
  TORCH_CHECK(
              grad.sparse_dim() == output.sparse_dim(),
      "grad and output sparse dimensions must be equal");
  AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), "softmax_backward", [&] {
      // assuming that the type of input._indices() entries is int64_t
      cpu_sparse_coo_softmax_backward<scalar_t, int64_t, true>(grad_input, grad, output, dim);
  });
  return grad_input;
}

Tensor _sparse_softmax(const Tensor& input_, const int64_t dim_) {
  auto result = [&]() {
    NoNamesGuard guard;
    return at::_sparse_softmax(input_, dim_, false);
  }();
  namedinference::propagate_names(result, input_);
  return result;
}

Tensor _sparse_softmax(const Tensor& input_, const int64_t dim_, c10::optional<ScalarType> dtype) {
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

Tensor _sparse_softmax(const Tensor& self, Dimname dim, optional<ScalarType> dtype) {
  return at::_sparse_softmax(self, dimname_to_position(self, dim), dtype);
}

Tensor _sparse_log_softmax(const Tensor& input_, const int64_t dim_) {
  auto result = [&]() {
    NoNamesGuard guard;
    return at::_sparse_log_softmax(input_, dim_, false);
  }();
  namedinference::propagate_names(result, input_);
  return result;
}

Tensor _sparse_log_softmax(const Tensor& input_, const int64_t dim_, c10::optional<ScalarType> dtype) {
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
  
Tensor _sparse_log_softmax(const Tensor& self, Dimname dim, optional<ScalarType> dtype) {
  return at::_sparse_log_softmax(self, dimname_to_position(self, dim), dtype);
}

}
}
