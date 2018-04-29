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

namespace {
  const int MODE_SUM = 0;
  const int MODE_MEAN = 1;
  const int MODE_MAX = 2;
}

namespace at {
namespace native {

static void make_offset2bag(const Tensor &offsets, const Tensor &indices,
                            Tensor &offset2bag) {
  offset2bag.index_fill_(0, offsets, 1); // offset2bag = [1 0 1 0 1]
  offset2bag[0] = 0;                     // offset2bag = [0 0 1 0 1]
  offset2bag = offset2bag.cumsum(0);     // offset2bag = [0 0 1 1 2]
}

template<typename T>
static void axpy(int64_t n, T a, T *x, int64_t incx, T *y, int64_t incy);
template<>
void axpy<float>(int64_t n, float a, float *x, int64_t incx,
                 float *y, int64_t incy) {
  THFloatBlas_axpy(n, a, x, incx, y, incy);
}
template<>
void axpy<double>(int64_t n, double a, double *x, int64_t incx,
                  double *y, int64_t incy) {
  THDoubleBlas_axpy(n, a, x, incx, y, incy);
}

// This function combines index_select (using select_indices as the index) and
// index_add (using add_indices as the index), without creating an intermediary
// tensor to hold the selected embeddings
template<typename T>
static void index_select_add(const Tensor &select_indices,
                             const Tensor &add_indices,
                             const Tensor &src,
                             Tensor &output) {
  auto add_indices_data = add_indices.data<int64_t>();
  auto select_indices_data = select_indices.data<int64_t>();
  auto src_data = src.data<T>();
  auto output_data = output.data<T>();
  auto numel = add_indices.numel();
  int64_t ddim = src.size(1);
  for (int64_t i = 0; i < numel; i++) {
    axpy<T>(ddim, 1, src_data + ddim * select_indices_data[i], 1,
            output_data + ddim * add_indices_data[i], 1);
  }
}

static void make_bag_size(const Tensor &offsets, const Tensor &indices,
                          const int64_t mode, Tensor &bag_size) {
  if (mode == 1) { // MODE_MEAN
    if (offsets.size(0) != 1) {
      bag_size.slice(0, 0, bag_size.size(0) - 1, 1) =
          offsets.slice(0, 1, offsets.size(0), 1) -
          offsets.slice(0, 0, offsets.size(0) - 1, 1);
      bag_size[-1] = indices.size(0) - offsets[-1];
    }
  }
}

static Tensor apply_bag_size(const Tensor &offsets, const Tensor &indices,
                             const int64_t mode, Tensor &output,
                             const Tensor &bag_size) {
  if (mode == 1) { // MODE_MEAN
    if (offsets.size(0) == 1) {
      auto bag_size_ = indices.size(0);
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
    if (offsets.size(0) == 1) {
      auto bag_size_ = indices.size(0);
      output /= bag_size_;
    } else {
      auto inv_bag_size_ = (1 / bag_size.toType(output.type()))
                             .unsqueeze(1)
                             .index_select(0, offset2bag);
      output *= inv_bag_size_;
    }
  }
  return output;
}


template <typename scalar_t>
std::tuple<Tensor, Tensor, Tensor, Tensor> embedding_bag_cpu_max(
  const Tensor& weight, const Tensor &indices, const Tensor& offset2bag, const Tensor& output, const Tensor& bag_size, const Tensor& offsets) {
    
    auto max_indices = at::zeros(indices.type(), {offsets.size(0), weight.size(1)});

    int64_t numel = indices.numel();
    int64_t dims = weight.size(1);
    auto indices_data = indices.data<int64_t>();
    auto offset2bag_data = offset2bag.data<int64_t>();

    auto max_indices_data = max_indices.data<int64_t>();
    auto max_indices_stride = max_indices.stride(0);

    auto weight_data = weight.data<scalar_t>();
    auto output_data = output.data<scalar_t>();  
    auto weight_stride = weight.stride(0); 
    auto output_stride = output.stride(0);   

    for (int i = 0; i < numel; i++) {
      auto bag = offset2bag_data[i];
      auto word_idx = indices_data[i];


      for (int dim = 0; dim < dims; dim++) {
        auto& current_item = output_data[output_stride * bag + dim];
        auto weight_item = weight_data[weight_stride * word_idx + dim];
        
        bool is_first_for_bag = (i == 0) || offset2bag_data[i - 1] != bag;

        if (is_first_for_bag || weight_item > current_item) {
          current_item = weight_item;
          max_indices_data[max_indices_stride * bag + dim] = word_idx;
        }
      }
    }

    return std::tuple<Tensor, Tensor, Tensor, Tensor>(output, offset2bag, bag_size, max_indices);
}

std::tuple<Tensor, Tensor, Tensor, Tensor>
embedding_bag_cpu(const Tensor &weight, const Tensor &indices__,
                  const Tensor &offsets__, const bool scale_grad_by_freq,
                  const int64_t mode, bool sparse) {
  auto indices_arg = TensorArg(indices__, "indices__", 1);
  checkScalarType("embedding_bag", indices_arg, kLong);
  auto offsets_arg = TensorArg(offsets__, "offsets__", 1);
  checkScalarType("embedding_bag", offsets_arg, kLong);
  Tensor indices = indices__.contiguous();
  Tensor offsets = offsets__.contiguous();
  auto weight_arg = TensorArg(weight, "weight", 1);
  checkScalarTypes("embedding_bag", weight_arg, {kFloat, kDouble});

  auto bag_size = at::zeros(indices.type(), offsets.sizes());
  auto offset2bag =
      at::zeros(indices__.type(), {indices.size(0)}); // offset2bag = [0 0 0 0 0]
  make_offset2bag(offsets, indices, offset2bag);
  auto output = at::zeros(weight.type(), {offsets.size(0), weight.size(1)});

  if (mode == MODE_MEAN || mode == MODE_SUM) {
    if (weight.type().scalarType() == kFloat) {
      index_select_add<float>(indices, offset2bag, weight, output);
    } else if (weight.type().scalarType() == kDouble) {
      index_select_add<double>(indices, offset2bag, weight, output);
    }
    make_bag_size(offsets, indices, mode, bag_size);
    auto ret = apply_bag_size(offsets, indices, mode, output, bag_size);
    return std::tuple<Tensor, Tensor, Tensor, Tensor>(ret, offset2bag, bag_size, bag_size);
  } else { // MODE_MAX
    return AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      weight.type(), "embedding_bag_cpu_max", [&]() {
        return embedding_bag_cpu_max<scalar_t>(weight, indices, offset2bag, output, bag_size, offsets);
      }
    );
  }
}

Tensor embedding_bag_backward(const Tensor &grad_, const Tensor &indices__,
                              const Tensor &offsets__,
                              const Tensor &offset2bag__,
                              const Tensor &bag_size_, 
                              const Tensor &max_indices_,
                              int64_t num_weights,
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
        grad_, indices, offsets, offset2bag__, bag_size_, max_indices_, num_weights,
        scale_grad_by_freq, mode);
  }
}

Tensor embedding_bag_backward_cpu(const Tensor &grad_, const Tensor &indices__,
                                  const Tensor &offsets__,
                                  const Tensor &offset2bag__,
                                  const Tensor &bag_size_,
                                  const Tensor& max_indices_, int64_t num_weights,
                                  bool scale_grad_by_freq, int64_t mode) {
  auto grad = grad_.contiguous();
  auto grad_arg = TensorArg(grad, "grad_", 1);
  checkScalarTypes("embedding_bag", grad_arg, {kFloat, kDouble});
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

  auto index_grad_weight =
      at::zeros(grad.type(), {num_weights, grad.size(1)}).contiguous();

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

  if (mode == MODE_MEAN || mode == MODE_SUM) {
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
            if (offsets_.size(0) == 1) {
              auto bag_size = indices.size(0);
              scale /= bag_size;
            } else {
              if (source == offsets_.size(0) - 1) {
                scale /= indices.size(0) - offsets_data[offsets_.size(0) - 1];
              } else {
                scale /= offsets_data[source + 1] - offsets_data[source];
              }
            }
          }
          int64_t ddim = grad.size(1);
          if (grad.type().scalarType() == kFloat) {
            auto igwd = index_grad_weight.data<float>();
            auto gd = grad.data<float>();
            axpy<float>(ddim, (float)scale, gd + ddim * source, 1,
                        igwd + ddim * index, 1);
          } else if (grad.type().scalarType() == kDouble) {
            auto igwd = index_grad_weight.data<double>();
            auto gd = grad.data<double>();
            axpy<double>(ddim, (double)scale, gd + ddim * source, 1,
                         igwd + ddim * index, 1);
          }
        }
      } 
  } else if (mode == MODE_MAX) {
    for (int64_t dim = 0; dim < grad.size(1); dim++) {
      index_grad_weight.select(1, dim).index_add_(0, max_indices_.select(1, dim), grad_.select(1, dim));
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
