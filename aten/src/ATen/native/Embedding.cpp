#include "ATen/ATen.h"
#include "ATen/TensorUtils.h"
#include "ATen/NativeFunctions.h"

#include <cstring>
#include <memory>
#include <sstream>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif


namespace at { namespace native {

Tensor embedding(const Tensor & weight, const Tensor & indices,
                 int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
  auto indices_arg = TensorArg(indices, "indices", 1);
  checkScalarType("embedding", indices_arg, kLong);
  checkContiguous("embedding", indices_arg);

  // TODO: use tensor.index() after improving perf
  if (indices.dim() == 1) {
    return weight.index_select(0, indices);
  }

  auto size = std::vector<int64_t>(indices.sizes());
  for (auto d : weight.sizes().slice(1)) {
    size.push_back(d);
  }
  return weight.index_select(0, indices.view(-1)).view(size);
}

Tensor embedding_backward(
    const Tensor & grad, const Tensor & indices, int64_t num_weights,
    int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
  if (sparse) {
    return at::embedding_sparse_backward(
        grad, indices, num_weights, padding_idx, scale_grad_by_freq);
  } else {
    return at::embedding_dense_backward(
        grad, indices, num_weights, padding_idx, scale_grad_by_freq);
  }
}

Tensor embedding_sparse_backward(
    const Tensor & grad_, const Tensor & indices_, int64_t num_weights,
    int64_t padding_idx, bool scale_grad_by_freq) {

  auto indices_arg = TensorArg(indices_, "indices", 2);
  checkScalarType("embedding_backward", indices_arg, kLong);
  checkContiguous("embedding_backward", indices_arg);

  // TODO: implement scale_grad_by_freq
  if (scale_grad_by_freq) {
    AT_ERROR(
        "embedding_backward: scale_grad_by_freq not supported with sparse gradients");
  }

  Tensor indices = indices_;
  Tensor grad = grad_;
  if (padding_idx != -1) {
    auto c = indices != padding_idx;
    indices = indices.index(c);
    grad = grad.index(c);
  }

  int64_t num_features = grad_.size(-1);
  auto weight_size = std::array<int64_t, 2>{{ num_weights, num_features }};
  auto& dense_type = grad.type();
  auto& sparse_type = dense_type.toBackend(grad.is_cuda() ? kSparseCUDA : kSparseCPU);

  // check if all our grad come from padding_idx
  if (grad.numel() == 0) {
    return sparse_type._sparse_coo_tensor_unsafe(indices_.type().tensor(),
                                         dense_type.tensor(), weight_size);
  }

  auto index = indices.view({1, -1});
  auto values = grad.contiguous().view({-1, num_features});
  return sparse_type._sparse_coo_tensor_unsafe(index, values, weight_size);
}

Tensor embedding_backward_cpu(
    const Tensor & grad_, const Tensor & indices, int64_t num_weights,
    int64_t padding_idx, bool scale_grad_by_freq) {

  auto indices_arg = TensorArg(indices, "indices", 2);
  checkScalarType("embedding_backward", indices_arg, kLong);
  checkContiguous("embedding_backward", indices_arg);

  auto indices_data = indices.data<int64_t>();
  int64_t numel = indices.numel();

  std::unique_ptr<int64_t[]> counts;
  if (scale_grad_by_freq) {
    counts.reset(new int64_t[num_weights]);
    for (int i = 0; i < numel; i++) {
      counts[indices_data[i]] = 0;
    }
    for (int i = 0; i < numel; i++) {
      counts[indices_data[i]]++;
    }
  }

  auto grad = grad_.contiguous().view({numel, grad_.size(-1)});
  auto grad_weight = at::zeros(grad_.type(), {num_weights, grad_.size(-1)});

#ifdef _OPENMP
  if (numel > 1000) {
    // The strategy is to parallelize over sections of the vocabulary, so that
    // thread 1 handles updates to gradWeight[0..nVocab/nThreads]. Every thread
    // has to traverse the entire input, but the dominating factor is the axpy
    // BLAS call.
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      int nthreads = omp_get_num_threads();
      int64_t start = tid * (num_weights/nthreads + 1);
      int64_t end = start + (num_weights/nthreads + 1);
      for (int64_t i = 0; i < numel; i++) {
        if (indices_data[i] != padding_idx) {
          int64_t k = indices_data[i];
          if (k >= start && k < end) {
            double scale = 1.0;
            if (scale_grad_by_freq) {
              scale /= counts[k];
            }
            grad_weight[k].add_(grad[i], scale);
          }
        }
      }
    }
    return grad_weight;
  }
#endif

  for (int64_t i = 0; i < numel; i++) {
    if (indices_data[i] != padding_idx) {
      int64_t k = indices_data[i];
      double scale = 1.0;
      if (scale_grad_by_freq) {
        scale /= counts[k];
      }
      grad_weight[k].add_(grad[i], scale);
    }
  }

  return grad_weight;
}

Tensor & embedding_renorm_cpu_(
    Tensor & self, const Tensor & indices, double max_norm, double norm_type) {
  auto self_arg = TensorArg(self, "self", 1);
  auto indices_arg = TensorArg(indices, "indices", 2);
  checkContiguous("embedding_renorm_", self_arg);
  checkDim("embedding_renorm_", self_arg, 2);
  checkContiguous("embedding_renorm_", indices_arg);
  checkScalarType("embedding_renorm_", indices_arg, kLong);

  auto num_indices = indices.numel();
  auto data_ptr = indices.data<int64_t>();
  auto sorted_indices = std::vector<int64_t>(data_ptr, data_ptr + num_indices);
  std::sort(sorted_indices.begin(), sorted_indices.end(), std::less<int64_t>());

  #pragma omp parallel for if(num_indices > 1000)
  for (int64_t i = 0; i < num_indices; i++) {
    if (i > 0 && sorted_indices[i] == sorted_indices[i - 1]) {
      continue;
    }
    auto row = self[sorted_indices[i]];
    auto norm = row.norm(norm_type).toCDouble();
    if (norm > max_norm) {
      auto scale = max_norm / (norm + 1e-7);
      row *= scale;
    }
  }

  return self;
}

}}  // namespace at::native
