#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/Parallel.h>
#include <map>

namespace at {
namespace native {
namespace {

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

  std::vector<iscalar_t> pool(nnz, 0);
  {
    std::vector<iscalar_t> strides(sparse_dim, 1);
    if (sparse_dim > 1) {
      for (int64_t i=sparse_dim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * sizes[i + 1];
      }
    }

    for (int64_t j=0; j < sparse_dim; j++) {
      if (j != dim) {
        for (int64_t i=0; i < nnz; i++) {
          pool[i] += strides[j] * indices_data_base[j * nnz + i];
        }
      }
    }
  }

  iscalar_t mx_p = -1;
  {
    std::map<iscalar_t, iscalar_t> i2p;
    for (int64_t i=0; i < nnz; i++) {
      auto c = pool[i];
      auto it = i2p.find(c);
      iscalar_t p = i2p.size();
      if (it == i2p.end()) {
        i2p.emplace(std::make_pair(c, p));
        mx_p = p;
      } else {
        p = it->second;
      }
      pool[i] = p;
    }
  }

  /* Compute mx */
  std::vector<int64_t> mx_sizes(sizes.begin() + sparse_dim - 1, sizes.end());
  mx_sizes[0] = mx_p + 1;
  at::Tensor mx = at::empty(mx_sizes, values.options());
  scalar_t* mx_data_base = mx.data_ptr<scalar_t>();

  int64_t nvalues = 1;
  for (int64_t i = 1; i < mx.dim(); ++i)
    nvalues *= mx.size(i);

  auto ninf = -std::numeric_limits<scalar_t>::infinity();
  for (int64_t j=0; j < nvalues * (mx_p + 1); j++) {
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

  if (LogSoftMax) {
    for (int64_t j=0; j < nvalues * (mx_p + 1); j++) {
      mx_data_base[j] += std::log(exp_sums_data_base[j]);
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
        out_values_data[j] /= exp_sums_data[j];
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

}
}
