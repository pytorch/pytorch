#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/Parallel.h>
#include <map>

namespace at {
namespace native {
namespace {

template <typename iscalar_t>
iscalar_t get_nvalues(const std::vector<iscalar_t>& sizes, const int64_t sparse_dim) {
  iscalar_t nvalues = 1;
  for (auto it = sizes.begin() + sparse_dim; it != sizes.end(); ++it) {
    nvalues *= *it;
  }
  return nvalues;
}

template <typename iscalar_t>
std::vector<iscalar_t> get_offsets(const Tensor& indices, const std::vector<iscalar_t>& sizes, const int64_t dim) {
  /*
    Given the indices of a sparse tensor return a vector of offsets
    for the entries in the corresponding dense tensor:

      If
        offsets = get_offsets(A._indices(), A.sizes(), -1)
        data = A.to_dense().resize((nnz,))
      then
        data[offsets[n]] == A._values()[n]

    `indices` must be a contiguous 2-d tensor with iscalar_t entries.
    `sizes` must be a vector with at least ndim entries.

    `dim` is an integer. When >= 0 and < ndim, all entries in the
    given dimension will be mapped to the first entry. Otherwise, the
    value is ignored.

    The items in the returned vector are sorted.

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
    Return dense set of offsets.

    If
      pool = get_dense_offsets(mx, offsets)
    then
      pool.size() == offsets.size()
      pool[i] == pool[j] iff offsets[i] == offsets[j]
      set(pool) == set(range(mx))
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
  /* Compute pool indices */
  auto sizes = input.sizes().vec();

  std::vector<iscalar_t> offsets = get_offsets<iscalar_t>(indices, sizes, dim);
  iscalar_t mx_p;
  std::vector<iscalar_t> pool = get_dense_offsets<iscalar_t>(mx_p, offsets);

  /* Compute mx */
  std::vector<int64_t> mx_sizes(sizes.begin() + sparse_dim - 1, sizes.end());
  mx_sizes[0] = mx_p;
  at::Tensor mx = at::empty(mx_sizes, values.options());
  scalar_t* mx_data_base = mx.data_ptr<scalar_t>();

  int64_t nvalues = 1;
  for (int64_t i = 1; i < mx.dim(); ++i)
    nvalues *= mx.size(i);

  auto ninf = -std::numeric_limits<scalar_t>::infinity();
  for (int64_t j=0; j < nvalues * mx_p; j++) {
    mx_data_base[j] = ninf;
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
        auto r = softmax_backward_cpu(grad_values[j], out_values[i], dim - sparse_dim, unused);
        values[i].copy_(r);
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
        tmp_data[k] -= out_values_data[k] * grad_values_data[k];
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
        values_data[k] = out_values_data[k] * (grad_values_data[k] + tmp_data[k]);
      }
    } else {
      for (int64_t k=0; k<nvalues; k++) {
        values_data[k] = out_values_data[k] * (tmp_data[k]);
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
  /*
    Return D(output)/D(input) @ grad.

    input is not used.
   */
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

}
}
