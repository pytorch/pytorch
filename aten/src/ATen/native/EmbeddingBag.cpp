#include "ATen/ATen.h"
#include "ATen/TensorUtils.h"
#include "ATen/NativeFunctions.h"

#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include <TH/THBlas.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace at {
namespace native {

static void make_offset2bag(const Tensor &offsets, const Tensor &indices,
                            Tensor &offset2bag) {
  offset2bag.index_fill_(0, offsets, 1); // offset2bag = [1 0 1 0 1]
  offset2bag[0] = 0;                     // offset2bag = [0 0 1 0 1]
  offset2bag = offset2bag.cumsum(0);     // offset2bag = [0 0 1 1 2]
}

static void make_bag_size(const Tensor &offsets, const Tensor &indices,
                          const int64_t mode, Tensor &bag_size) {
  if (mode == 1) { // MODE_MEAN
    if (offsets.sizes()[0] != 1) {
      bag_size.slice(0, 0, bag_size.sizes()[0] - 1, 1) =
          offsets.slice(0, 1, offsets.sizes()[0], 1) -
          offsets.slice(0, 0, offsets.sizes()[0] - 1, 1);
      bag_size[-1] = indices.sizes()[0] - offsets[-1];
    }
  }
}

static Tensor apply_bag_size(const Tensor &offsets, const Tensor &indices,
                             const int64_t mode, Tensor &output,
                             const Tensor &bag_size) {
  if (mode == 1) { // MODE_MEAN
    if (offsets.sizes()[0] == 1) {
      auto bag_size_ = indices.sizes()[0];
      output /= bag_size_;
    } else {
      auto bag_size_ =
          bag_size.toType(output.type()).unsqueeze(1).expand_as(output);
      output /= bag_size_;
    }
  }
  return output;
}

static Tensor apply_bag_size_backward(const Tensor &offsets,
                                      const Tensor &indices, const int64_t mode,
                                      Tensor &output, const Tensor &offset2bag,
                                      const Tensor &bag_size) {
  if (mode == 1) { // MODE_MEAN
    if (offsets.sizes()[0] == 1) {
      auto bag_size_ = indices.sizes()[0];
      output /= bag_size_;
    } else {
      auto bag_size_ = bag_size.toType(output.type())
                           .unsqueeze(1)
                           .index_select(0, offset2bag);
      output /= bag_size_;
    }
  }
  return output;
}

std::tuple<Tensor, Tensor, Tensor>
embedding_bag_cpu(const Tensor &weight, const Tensor &indices__,
                  const Tensor &offsets__, const bool scale_grad_by_freq,
                  const int64_t mode, bool sparse) {
  auto indices_arg = TensorArg(indices__, "indices__", 1);
  checkScalarType("embedding_bag", indices_arg, kLong);
  auto offsets_arg = TensorArg(offsets__, "offsets__", 1);
  checkScalarType("embedding_bag", offsets_arg, kLong);
  Tensor indices = indices__.contiguous();
  Tensor offsets = offsets__.contiguous();

  auto bag_size = indices.type().zeros(offsets.sizes());
  auto offset2bag =
      indices__.type().zeros({indices.sizes()[0]}); // offset2bag = [0 0 0 0 0]
  make_offset2bag(offsets, indices, offset2bag);
  auto output = weight.type().zeros({offsets.sizes()[0], weight.sizes()[1]});
  auto index_output = weight.index_select(0, indices);
  output.index_add_(0, offset2bag, index_output);
  make_bag_size(offsets, indices, mode, bag_size);
  auto ret = apply_bag_size(offsets, indices, mode, output, bag_size);
  return std::tuple<Tensor, Tensor, Tensor>(ret, offset2bag, bag_size);
}

Tensor embedding_bag_backward(const Tensor &grad_, const Tensor &indices__,
                              const Tensor &offsets__,
                              const Tensor &offset2bag__,
                              const Tensor &bag_size_, int64_t num_weights,
                              bool scale_grad_by_freq, int64_t mode,
                              bool sparse) {
  auto indices_arg = TensorArg(indices__, "indices__", 1);
  checkScalarType("embedding_bag", indices_arg, kLong);
  auto offsets_arg = TensorArg(offsets__, "offsets__", 1);
  checkScalarType("embedding_bag", offsets_arg, kLong);
  auto offset2bag_arg = TensorArg(offset2bag__, "offset2bag__", 1);
  checkScalarType("embedding_bag", offset2bag_arg, kLong);
  checkContiguous("embedding_bag", offset2bag_arg);
  Tensor indices = indices__.contiguous();
  Tensor offsets = offsets__.contiguous();

  if (sparse) {
    return at::embedding_bag_sparse_backward(
        grad_, indices, offsets, offset2bag__, bag_size_, num_weights,
        scale_grad_by_freq, mode);
  } else {
    return at::embedding_bag_dense_backward(
        grad_, indices, offsets, offset2bag__, bag_size_, num_weights,
        scale_grad_by_freq, mode);
  }
}

Tensor embedding_bag_backward_cpu(const Tensor &grad_, const Tensor &indices__,
                                  const Tensor &offsets__,
                                  const Tensor &offset2bag__,
                                  const Tensor &bag_size_, int64_t num_weights,
                                  bool scale_grad_by_freq, int64_t mode) {
  auto grad = grad_.contiguous();
  auto indices_arg = TensorArg(indices__, "indices__", 1);
  checkScalarType("embedding_bag", indices_arg, kLong);
  auto offsets_arg = TensorArg(offsets__, "offsets__", 1);
  checkScalarType("embedding_bag", offsets_arg, kLong);
  auto offset2bag_arg = TensorArg(offset2bag__, "offset2bag__", 1);
  checkScalarType("embedding_bag", offset2bag_arg, kLong);
  checkContiguous("embedding_bag", offset2bag_arg);
  Tensor indices_ = indices__.contiguous();
  Tensor offsets_ = offsets__.contiguous();

  Tensor &offset2bag_ = const_cast<Tensor &>(offset2bag__);

  auto ind_sort_ = indices_.sort();
  auto indices = std::get<0>(ind_sort_);
  auto ind_sort = std::get<1>(ind_sort_);
  auto offset2bag = offset2bag_.index_select(0, ind_sort);

  auto indices_data = indices.data<int64_t>();
  auto offsets_data = offsets_.data<int64_t>();
  auto offset2bag_data = offset2bag.data<int64_t>();
  int64_t numel = indices.numel();

  std::vector<int64_t> counts(num_weights);
  for (int i = 0; i < numel; i++) {
    counts[indices_data[i]] = 0;
  }
  for (int i = 0; i < numel; i++) {
    counts[indices_data[i]]++;
  }

  std::vector<int64_t> counts_uniq;
  counts_uniq.reserve(num_weights);
  int64_t o = 0;
  for (int64_t i = 0; i < numel; i += counts[indices_data[i]]) {
    counts_uniq.push_back(counts[indices_data[i]]);
    if (o > 0) {
      counts_uniq[o] += counts_uniq[o - 1];
    }
    o++;
  }

  auto index_grad_weight =
      grad.type().zeros({num_weights, grad.sizes()[1]}).contiguous();

#pragma omp parallel for if (numel > 1000)
  for (int64_t i = 0; i < (int64_t)counts_uniq.size(); i++) {
    int64_t start = i == 0 ? 0 : counts_uniq[i - 1];
    int64_t index = indices_data[start];
    for (int64_t j = start; j < counts_uniq[i]; j++) {
      int64_t source = offset2bag_data[j];
      double scale = 1.0;
      if (scale_grad_by_freq) {
        scale /= counts[indices_data[i]];
      }
      if (mode == 1) { // MODE_MEAN
        if (offsets_.sizes()[0] == 1) {
          auto bag_size = indices.sizes()[0];
          scale /= bag_size;
        } else {
          if (source == offsets_.sizes()[0] - 1) {
            scale /= indices.sizes()[0] - offsets_data[offsets_.sizes()[0] - 1];
          } else {
            scale /= offsets_data[source + 1] - offsets_data[source];
          }
        }
      }
      int64_t ddim = grad.sizes()[1];
      if (grad.type().scalarType() == kFloat) {
        auto igwd = index_grad_weight.data<float>();
        auto gd = grad.data<float>();
        THFloatBlas_axpy(ddim, (float)scale, gd + ddim * source, 1,
                         igwd + ddim * index, 1);
      } else if (grad.type().scalarType() == kDouble) {
        auto igwd = index_grad_weight.data<double>();
        auto gd = grad.data<double>();
        THDoubleBlas_axpy(ddim, (double)scale, gd + ddim * source, 1,
                          igwd + ddim * index, 1);
      } else {
        index_grad_weight[index].add_(grad[source], scale);
      }
    }
  }

  return index_grad_weight;
}
Tensor embedding_bag_sparse_backward(
    const Tensor &grad_, const Tensor &indices__, const Tensor &offsets__,
    const Tensor &offset2bag__, const Tensor &bag_size_, int64_t num_weights,
    bool scale_grad_by_freq, int64_t mode) {
  auto indices_arg = TensorArg(indices__, "indices__", 1);
  checkScalarType("embedding_bag", indices_arg, kLong);
  auto offsets_arg = TensorArg(offsets__, "offsets__", 1);
  checkScalarType("embedding_bag", offsets_arg, kLong);
  auto offset2bag_arg = TensorArg(offset2bag__, "offset2bag__", 1);
  checkScalarType("embedding_bag", offset2bag_arg, kLong);
  Tensor indices = indices__.contiguous();
  Tensor offsets = offsets__.contiguous();
  Tensor offset2bag = offset2bag__.contiguous();

  Tensor grad = grad_;
  Tensor index_grad = grad_.index_select(0, offset2bag);
  index_grad = apply_bag_size_backward(offsets, indices, mode, index_grad,
                                       offset2bag, bag_size_);
  return native::embedding_backward(index_grad, indices, num_weights, -1,
                                    scale_grad_by_freq, true);
}
}
} // namespace at::native
