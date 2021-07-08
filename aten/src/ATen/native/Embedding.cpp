#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/BinaryOps.h>

#include <c10/util/irange.h>

#include <cstring>
#include <memory>
#include <sstream>
#include <vector>


namespace at { namespace native {

Tensor embedding(const Tensor & weight, const Tensor & indices,
                 int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
  TORCH_CHECK(weight.dim() == 2,  "'weight' must be 2-D");
  auto indices_arg = TensorArg(indices, "indices", 1);
  checkScalarTypes("embedding", indices_arg, {kLong, kInt});

  // TODO: use tensor.index() after improving perf
  if (indices.dim() == 1) {
    return weight.index_select(0, indices);
  }

  auto size = indices.sizes().vec();
  for (auto d : weight.sizes().slice(1)) {
    size.push_back(d);
  }

  return weight.index_select(0, indices.reshape(-1)).view(size);
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
  checkScalarTypes("embedding_backward", indices_arg, {kLong, kInt});

  // TODO: implement scale_grad_by_freq
  if (scale_grad_by_freq) {
    AT_ERROR(
        "embedding_backward: scale_grad_by_freq not supported with sparse gradients");
  }

  Tensor indices = indices_;
  Tensor grad = grad_;
  if (padding_idx != -1) {
    torch::List<c10::optional<Tensor>> c({indices != padding_idx});
    indices = indices.index(c);
    grad = grad.index(c);
  }

  int64_t num_features = grad_.size(-1);
  auto weight_size = std::array<int64_t, 2>{{ num_weights, num_features }};
  auto dense_options = grad.options();

  // check if all our grad come from padding_idx
  if (grad.numel() == 0) {
    return at::_sparse_coo_tensor_unsafe(at::empty({1, 0}, indices_.options().dtype(kLong)),
                                         at::empty({0, num_features}, dense_options),
                                         weight_size);
  }

  auto index = indices.reshape({1, -1});
  auto values = grad.reshape({-1, num_features});
  return at::_sparse_coo_tensor_unsafe(index.to(kLong), values, weight_size);
}

Tensor embedding_dense_backward_cpu(
    const Tensor & grad_, const Tensor & indices, int64_t num_weights,
    int64_t padding_idx, bool scale_grad_by_freq) {

  auto indices_arg = TensorArg(indices, "indices", 2);
  checkScalarTypes("embedding_backward", indices_arg, {kLong, kInt});

  auto grad_weight = at::zeros({num_weights, grad_.size(-1)}, grad_.options());
  auto indices_contig = indices.contiguous();
  int64_t numel = indices.numel();
  auto grad = grad_.contiguous().view({numel, grad_.size(-1)});

  auto add_iter = TensorIteratorConfig()
    .add_output(grad_weight)
    .add_input(grad_weight)
    .add_input(grad)
    .resize_outputs(false)
    .declare_static_shape(grad.sizes(), /*squash_dims=*/0)
    .build();

  const auto gW_data = reinterpret_cast<char*>(grad_weight.data_ptr());
  const auto gO_data = reinterpret_cast<char*>(grad.data_ptr());
  const auto gW_stride = grad_weight.strides()[0] * grad_weight.element_size();
  const auto gO_stride = grad.strides()[0] * grad.element_size();

  AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "embedding_dense_backward_cpu", [&] () {
    auto indices_data = indices_contig.data_ptr<index_t>();

    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    std::unique_ptr<index_t[]> counts;
    if (scale_grad_by_freq) {
      counts.reset(new index_t[num_weights]);
      for (const auto i : c10::irange(numel)) {
        counts[indices_data[i]] = 0;
      }
      for (const auto i : c10::irange(numel)) {
        counts[indices_data[i]]++;
      }
    }

    auto parallel_section = [&](index_t start, index_t end) {
      TensorIterator iter(add_iter);
      for (int64_t i = 0; i < numel; i++) {
        if (indices_data[i] != padding_idx) {
          index_t k = indices_data[i];
          if (k >= start && k < end) {
            double scale = 1.0;
            if (scale_grad_by_freq) {
              // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
              scale /= counts[k];
            }

            // grad_weight[k].add_(grad[i], scale);
            iter.unsafe_replace_operand(0, gW_data + k * gW_stride);
            iter.unsafe_replace_operand(1, gW_data + k * gW_stride);
            iter.unsafe_replace_operand(2, gO_data + i * gO_stride);
            add_stub(kCPU, iter, scale);
          }
        }
      }
    };

    at::parallel_for(0, num_weights, 1000, parallel_section);

  });

  return grad_weight;
}

Tensor & embedding_renorm_cpu_(
    Tensor & self, const Tensor & indices, double max_norm, double norm_type) {
  auto self_arg = TensorArg(self, "self", 1);
  auto indices_arg = TensorArg(indices, "indices", 2);
  checkDim("embedding_renorm_", self_arg, 2);
  checkScalarTypes("embedding_renorm_", indices_arg, {kLong, kInt});

  auto indices_contig = indices.contiguous();
  auto num_indices = indices.numel();

  AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "embedding_renorm_cpu_", [&]() {
    auto data_ptr = indices_contig.data_ptr<index_t>();
    auto sorted_indices = std::vector<index_t>(data_ptr, data_ptr + num_indices);
    std::sort(sorted_indices.begin(), sorted_indices.end());

    // Note that we cannot use at::parallel_for here because we perform operations on
    // Tensor inside the loop. See github.com/pytorch/pytorch/issues/28370 for more details.
    for (auto i = 0; i < num_indices; i++) {
      if (i > 0 && sorted_indices[i] == sorted_indices[i - 1]) {
        continue;
      }
      auto row = self[sorted_indices[i]];
      auto norm = row.norm(norm_type).item<double>();
      if (norm > max_norm) {
        auto scale = max_norm / (norm + 1e-7);
        row *= scale;
      }
    }
  });

  return self;
}


}}  // namespace at::native
