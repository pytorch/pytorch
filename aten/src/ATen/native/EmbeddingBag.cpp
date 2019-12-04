#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/TensorUtils.h>

#include <TH/THBlasUtils.h>

#include <caffe2/perfkernels/embedding_lookup_idx.h>

#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
#include <algorithm>


namespace {
  const int MODE_SUM = 0;
  const int MODE_MEAN = 1;
  const int MODE_MAX = 2;
}

namespace at {
namespace native {

static void make_offset2bag(const Tensor &offsets, const Tensor &indices, Tensor& offset2bag) {
  offset2bag.index_add_(
      0, offsets, at::ones_like(offsets, LEGACY_CONTIGUOUS_MEMORY_FORMAT)); // offset2bag = [1 0 1 0 1]
  offset2bag[0] -= 1;                     // offset2bag = [0 0 1 0 1]
  offset2bag = offset2bag.cumsum(0);     // offset2bag = [0 0 1 1 2]
}

namespace {

bool isFastPathIndexSelect(const Tensor& src, Tensor& output) {
  return src.scalar_type() == kFloat && src.stride(1) == 1 && output.stride(1) == 1;
}

bool isFastPathIndexSelectScale(const Tensor& src, const Tensor& scale, Tensor& output) {
  return src.scalar_type() == kFloat && src.stride(1) == 1 && output.stride(1) == 1 && scale.stride(0) == 1;
}

// This function combines index_select (using select_indices as the index) and
// index_add (using add_indices as the index), without creating an intermediary
// tensor to hold the selected embeddings
template<typename T>
void index_select_add(const Tensor &select_indices,
                             const Tensor &add_indices,
                             const Tensor &src,
                             Tensor &output,
                             const Tensor& /*offsets*/) {
  AT_ASSERT(select_indices.numel() == add_indices.numel());
  auto add_indices_data = add_indices.data_ptr<int64_t>();
  auto select_indices_data = select_indices.data_ptr<int64_t>();
  auto src_data = src.data_ptr<T>();
  auto output_data = output.data_ptr<T>();
  auto numel = add_indices.numel();
  int64_t ddim = src.size(1);
  auto src_stride0 = src.stride(0);
  auto src_stride1 = src.stride(1);
  auto output_stride0 = output.stride(0);
  auto output_stride1 = output.stride(1);

  for (int64_t i = 0; i < numel; i++) {
    THBlas_axpy<T>(ddim, 1,
            src_data + src_stride0 * select_indices_data[i], src_stride1,
            output_data + output_stride0 * add_indices_data[i], output_stride1);
  }
}

template<>
void index_select_add<float>(const Tensor &select_indices,
                             const Tensor &add_indices,
                             const Tensor &src,
                             Tensor &output,
                             const Tensor& offsets) {
  int64_t ddim = src.size(1);
  auto src_data = src.data_ptr<float>();
  auto select_indices_data = select_indices.data_ptr<int64_t>();
  auto output_data = output.data_ptr<float>();

  if (isFastPathIndexSelect(src, output)) {
    caffe2::EmbeddingLookupIdx(
      /*block_size=*/ddim,
      /*output_size=*/offsets.numel(),
      /*index_size=*/select_indices.numel(),
      /*data_size=*/src.size(0),
      /*input=*/src_data,
      /*indices=*/select_indices_data,
      /*offsets=*/offsets.data_ptr<int64_t>(),
      /*weights=*/nullptr,
      /*scale_bias=*/nullptr,
      /*normalize_by_lengths=*/false,
      /*out=*/output_data
    );
  } else {
    AT_ASSERT(select_indices.numel() == add_indices.numel());
    auto add_indices_data = add_indices.data_ptr<int64_t>();
    auto src_stride0 = src.stride(0);
    auto src_stride1 = src.stride(1);
    auto output_stride0 = output.stride(0);
    auto output_stride1 = output.stride(1);
    auto numel = add_indices.numel();
    for (int64_t i = 0; i < numel; i++) {
      THBlas_axpy<float>(ddim, 1,
              src_data + src_stride0 * select_indices_data[i], src_stride1,
              output_data + output_stride0 * add_indices_data[i], output_stride1);
    }
  }
}

// This function fuses the following three fns:
// index_select (using select_indices as the index)
// mul (scaling by per_sample_weights)
// index_add (using add_indices as the index)
template<typename T>
static void index_select_scale_add(const Tensor &select_indices,
                                   const Tensor &add_indices,
                                   const Tensor &scale,
                                   const Tensor &src,
                                   Tensor &output,
                                   const Tensor& /*offsets*/) {
  AT_ASSERT(select_indices.numel() == add_indices.numel());
  auto add_indices_data = add_indices.data_ptr<int64_t>();
  auto select_indices_data = select_indices.data_ptr<int64_t>();
  auto src_data = src.data_ptr<T>();
  auto output_data = output.data_ptr<T>();
  auto numel = add_indices.numel();
  int64_t ddim = src.size(1);
  auto src_stride0 = src.stride(0);
  auto src_stride1 = src.stride(1);
  auto output_stride0 = output.stride(0);
  auto output_stride1 = output.stride(1);

  auto* scale_data = scale.data_ptr<T>();
  auto scale_stride = scale.stride(0);

  for (int64_t i = 0; i < numel; i++) {
    auto* src_base = src_data + src_stride0 * select_indices_data[i];
    auto* output_base = output_data + output_stride0 * add_indices_data[i];
    auto scale = scale_data[i * scale_stride];
    for (int64_t j = 0; j < ddim; j++) {
      output_base[j * output_stride1] += src_base[j * src_stride1] * scale;
    }
  }
}

template<>
void index_select_scale_add<float>(const Tensor &select_indices,
                                          const Tensor &add_indices,
                                          const Tensor &scale,
                                          const Tensor &src,
                                          Tensor &output,
                                          const Tensor& offsets) {
  int64_t ddim = src.size(1);
  auto* scale_data = scale.data_ptr<float>();
  auto select_indices_data = select_indices.data_ptr<int64_t>();
  auto src_data = src.data_ptr<float>();
  auto output_data = output.data_ptr<float>();

  if (isFastPathIndexSelectScale(src, scale, output)) {
    caffe2::EmbeddingLookupIdx(
      /*block_size=*/ddim,
      /*output_size=*/offsets.numel(),
      /*index_size=*/select_indices.numel(),
      /*data_size=*/src.size(0),
      /*input=*/src_data,
      /*indices=*/select_indices_data,
      /*offsets=*/offsets.data_ptr<int64_t>(),
      /*weights=*/scale_data,
      /*scale_bias=*/nullptr,
      /*normalize_by_lengths=*/false,
      /*out=*/output_data
    );
  } else {
    AT_ASSERT(select_indices.numel() == add_indices.numel());
    auto add_indices_data = add_indices.data_ptr<int64_t>();
    auto src_stride0 = src.stride(0);
    auto src_stride1 = src.stride(1);
    auto output_stride0 = output.stride(0);
    auto output_stride1 = output.stride(1);
    auto scale_stride = scale.stride(0);
    auto numel = add_indices.numel();


    for (int64_t i = 0; i < numel; i++) {
      auto* src_base = src_data + src_stride0 * select_indices_data[i];
      auto* output_base = output_data + output_stride0 * add_indices_data[i];
      auto scale = scale_data[i * scale_stride];
      for (int64_t j = 0; j < ddim; j++) {
        output_base[j * output_stride1] += src_base[j * src_stride1] * scale;
      }
    }
  }
}

}  // namespace

static void make_bag_size(const Tensor &offsets, const Tensor &indices,
                          const int64_t mode, Tensor &bag_size) {
  if (mode == MODE_MEAN || mode == MODE_MAX) {
    // Compute this for MODE_MEAN and MODE_MAX (latter needed for backwards)
    if (offsets.size(0) != 1) {
      bag_size.slice(0, 0, bag_size.size(0) - 1, 1) =
          offsets.slice(0, 1, offsets.size(0), 1) -
          offsets.slice(0, 0, offsets.size(0) - 1, 1);
    }
    bag_size[-1] = indices.size(0) - offsets[-1];
  }
}

static Tensor apply_bag_size(const Tensor &offsets, const Tensor &indices,
                             const int64_t mode, Tensor &output,
                             const Tensor &bag_size) {
  if (mode == MODE_MEAN) {
    // Avoid dividing by 0 for empty bags.
    // Instead we want empty bags to return all 0s
    if (offsets.size(0) == 1) {
      auto bag_size_ = std::max(indices.size(0), static_cast<int64_t>(1));
      output /= bag_size_;
    } else {
      auto bag_size_ = at::max(bag_size, at::ones_like(bag_size, LEGACY_CONTIGUOUS_MEMORY_FORMAT))
                           .to(output.options())
                           .unsqueeze(1)
                           .expand_as(output);
      output /= bag_size_;
    }
  }
  return output;
}

static Tensor apply_bag_size_backward(const Tensor &offsets,
                                      const Tensor &indices, const int64_t mode,
                                      Tensor &output, const Tensor &offset2bag,
                                      const Tensor &bag_size) {
  if (mode == MODE_MEAN) {
    if (offsets.size(0) == 1) {
      auto bag_size_ = indices.size(0);
      output /= bag_size_;
    } else {
      auto inv_bag_size_ = (1 / bag_size.to(output.options()))
                             .unsqueeze(1)
                             .index_select(0, offset2bag);
      output *= inv_bag_size_;
    }
  }
  return output;
}


template <typename scalar_t>
std::tuple<Tensor, Tensor, Tensor, Tensor> embedding_bag_cpu_max(
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offset2bag,
    const Tensor& output,
    const Tensor& bag_size,
    const Tensor& offsets) {

    auto max_indices = at::zeros({offsets.size(0), weight.size(1)}, indices.options());

    int64_t numel = indices.numel();
    int64_t dims = weight.size(1);
    auto indices_data = indices.data_ptr<int64_t>();
    auto offset2bag_data = offset2bag.data_ptr<int64_t>();

    auto max_indices_data = max_indices.data_ptr<int64_t>();
    auto max_indices_stride = max_indices.stride(0);

    auto weight_data = weight.data_ptr<scalar_t>();
    auto output_data = output.data_ptr<scalar_t>();
    auto weight_stride0 = weight.stride(0);
    auto weight_stride1 = weight.stride(1);
    auto output_stride = output.stride(0);

    for (int i = 0; i < numel; i++) {
      auto bag = offset2bag_data[i];
      auto word_idx = indices_data[i];

      for (int dim = 0; dim < dims; dim++) {
        auto& current_item = output_data[output_stride * bag + dim];
        auto weight_item = weight_data[weight_stride0 * word_idx + dim * weight_stride1];
        bool is_first_for_bag = (i == 0) || offset2bag_data[i - 1] != bag;

        if (is_first_for_bag || weight_item > current_item) {
          current_item = weight_item;
          max_indices_data[max_indices_stride * bag + dim] = word_idx;
        }
      }
    }

    return std::tuple<Tensor, Tensor, Tensor, Tensor>(output, offset2bag, bag_size, max_indices);
}

// embedding_bag wrapper to enforce contiguity in tensors other than `weight`.
// This is created to save extra `.contiguous()` call in backward.
// See NOTE [ embedding_bag Native Functions ] in native_functions.yaml for details
std::tuple<Tensor, Tensor, Tensor, Tensor>
embedding_bag(const Tensor &weight, const Tensor &indices,
              const Tensor &offsets, const bool scale_grad_by_freq,
              const int64_t mode, bool sparse,
              const Tensor &per_sample_weights) {
  return at::_embedding_bag(weight, indices.contiguous(), offsets.contiguous(),
                            scale_grad_by_freq, mode, sparse, per_sample_weights);
  };

// Assumes all input tensors except for `weight` are contiguous.
// See NOTE [ embedding_bag Native Functions ] in native_functions.yaml for details
std::tuple<Tensor, Tensor, Tensor, Tensor>
_embedding_bag_cpu(const Tensor &weight, const Tensor &indices,
                  const Tensor &offsets, const bool scale_grad_by_freq,
                  const int64_t mode, bool sparse,
                  const Tensor &per_sample_weights) {
  auto indices_arg = TensorArg(indices, "indices", 1);
  checkScalarType("embedding_bag", indices_arg, kLong);
  auto offsets_arg = TensorArg(offsets, "offsets", 1);
  checkScalarType("embedding_bag", offsets_arg, kLong);
  auto weight_arg = TensorArg(weight, "weight", 1);
  checkScalarTypes("embedding_bag", weight_arg, {kFloat, kDouble});

  if (per_sample_weights.defined()) {
    TORCH_CHECK(mode == MODE_SUM,
        "embedding_bag: per_sample_weights only supported with mode='sum'");
    auto per_input_weights_arg = TensorArg(
        per_sample_weights,"per_sample_weights", 1);
    checkSameType("embedding_bag", weight_arg, per_input_weights_arg);
    AT_ASSERT(per_sample_weights.dim() == 1);
    AT_ASSERT(per_sample_weights.numel() == indices.numel());
  }

  auto bag_size = at::zeros(offsets.sizes(), indices.options());
  make_bag_size(offsets, indices, mode, bag_size);

  auto output = at::zeros({offsets.size(0), weight.size(1)}, weight.options());

  // To save compute, if we are going to go down the fast path case for the 'sum'
  // mode, we skip calculating offset2bag, since it is not going to be used.
  auto fast_path_sum = [&weight, &per_sample_weights, &output]() {
    if (per_sample_weights.defined()) {
      return isFastPathIndexSelectScale(weight, per_sample_weights, output);
    } else {
      return isFastPathIndexSelect(weight, output);
    }
  };

  // Use an empty 0-element tensor as a sentinel that we have skipped the
  // creation of offset2bag because autograd chokes when trying to use an
  // undefined tensor as an input to a backward op.
  Tensor offset2bag = at::empty({0}, offsets.options());
  if (mode == MODE_MEAN || mode == MODE_MAX || !fast_path_sum()) {
    // If the last entries are empty, that the last offsets are irrelevant as they
    // won't change anything in the assignment of ID -> bag, but index_add would
    // throw out of bounds error. So to keep it simple we just add one more
    // entry to the end then get rid of it after make_offset2bag.
    offset2bag = at::zeros(
       {indices.sizes()[0] + 1}, indices.options()); // offset2bag = [0 0 0 0 0]

    make_offset2bag(offsets, indices, offset2bag);

    offset2bag.resize_({indices.sizes()[0]});
  }

  if (mode == MODE_MEAN || mode == MODE_SUM) {
    AT_DISPATCH_FLOATING_TYPES(weight.scalar_type(), "embedding_bag_cpu", [&]() {
      if (per_sample_weights.defined()) {
        AT_ASSERT(mode == MODE_SUM);
        index_select_scale_add<scalar_t>(
            indices, offset2bag, per_sample_weights, weight, output, offsets);
      } else {
        index_select_add<scalar_t>(indices, offset2bag, weight, output, offsets);
      }
    });
    auto ret = apply_bag_size(offsets, indices, mode, output, bag_size);
    return std::tuple<Tensor, Tensor, Tensor, Tensor>(ret, offset2bag, bag_size, bag_size);
  } else { // MODE_MAX
    at::optional<Tensor> maybe_per_sample_weights;
    if (per_sample_weights.defined()) {
      maybe_per_sample_weights = per_sample_weights;
    }
    return AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      weight.scalar_type(), "embedding_bag_cpu_max", [&]() {
        return embedding_bag_cpu_max<scalar_t>(
            weight, indices, offset2bag, output, bag_size, offsets);
      }
    );
  }
}

// Assumes all input tensors are contiguous.
// See NOTE [ embedding_bag Native Functions ] in native_functions.yaml for details
Tensor _embedding_bag_backward(const Tensor &grad, const Tensor &indices,
                              const Tensor &offsets,
                              const Tensor &offset2bag,
                              const Tensor &bag_size_,
                              const Tensor &max_indices_,
                              int64_t num_weights,
                              bool scale_grad_by_freq, int64_t mode,
                              bool sparse,
                              const Tensor& per_sample_weights) {
  auto indices_arg = TensorArg(indices, "indices", 1);
  checkScalarType("embedding_bag", indices_arg, kLong);
  checkContiguous("embedding_bag", indices_arg);
  auto offsets_arg = TensorArg(offsets, "offsets", 1);
  checkScalarType("embedding_bag", offsets_arg, kLong);
  checkContiguous("embedding_bag", offsets_arg);

  Tensor offset2bag_;
  if (indices.numel() != 0 && offset2bag.numel() == 0) {
    offset2bag_ = at::zeros(
       {indices.sizes()[0] + 1}, indices.options()); // offset2bag = [0 0 0 0 0]

    make_offset2bag(offsets, indices, offset2bag_);

    offset2bag_.resize_({indices.sizes()[0]});
  } else {
    auto offset2bag_arg = TensorArg(offset2bag, "offset2bag", 1);
    checkScalarType("embedding_bag", offset2bag_arg, kLong);
    checkContiguous("embedding_bag", offset2bag_arg);
    offset2bag_ = offset2bag;
  }

  if (sparse) {
    return at::_embedding_bag_sparse_backward(
        grad, indices, offsets, offset2bag_, bag_size_, num_weights,
        scale_grad_by_freq, mode, per_sample_weights);
  } else {
    return at::_embedding_bag_dense_backward(
        grad, indices, offsets, offset2bag_, bag_size_, max_indices_, num_weights,
        scale_grad_by_freq, mode, per_sample_weights);
  }
}

static Tensor _embedding_bag_dense_backward_cpu_max(
    const Tensor& grad,
    const Tensor& bag_size,
    const Tensor& max_indices,
    int64_t num_weights) {
  AT_ASSERT(max_indices.defined());
  auto index_grad_weight =
      at::zeros({num_weights, grad.size(1)}, grad.options());
  auto nonempty_max_indices = max_indices.index_select(0, bag_size.nonzero().view(-1));
  auto nonempty_grad = grad.index_select(0, bag_size.nonzero().view(-1));

  for (int64_t dim = 0; dim < grad.size(1); dim++) {
    index_grad_weight.select(1, dim).index_add_(
      0, nonempty_max_indices.select(1, dim), nonempty_grad.select(1, dim));
  }
  return index_grad_weight;
}

static std::vector<int64_t> compute_counts(
    int64_t num_weights,
    int64_t* indices_data,
    int64_t indices_length) {
  std::vector<int64_t> counts(num_weights, 0);
  for (int i = 0; i < indices_length; i++) {
    counts[indices_data[i]]++;
  }
  return counts;
}

// counts_uniq stores the index of the NEXT unique element
// of the (sorted) indices vector.
//
// For example:
// indices: [0, 0, 0, 1, 3, 3, 4]
// counts: [3, 1, 0, 2, 1, 0]
// counts_uniq: [3, 4, 6, 7]
//
// The unique indices can be found at index 0, 3, 4, 6.
static std::vector<int64_t> compute_counts_uniq(
    int64_t num_weights,
    int64_t* indices_data,
    int64_t indices_length,
    const std::vector<int64_t>& counts) {
  std::vector<int64_t> counts_uniq;
  counts_uniq.reserve(num_weights);
  int64_t o = 0;
  for (int64_t i = 0; i < indices_length; i += counts[indices_data[i]]) {
    counts_uniq.push_back(counts[indices_data[i]]);
    if (o > 0) {
      counts_uniq[o] += counts_uniq[o - 1];
    }
    o++;
  }
  return counts_uniq;
}

template <typename scalar_t>
void _embedding_bag_dense_backward_cpu_sum_mean(
    const Tensor& grad,
    const Tensor& indices_,
    const Tensor& offsets_,
    const Tensor& offset2bag__,
    int64_t num_weights,
    bool scale_grad_by_freq,
    int64_t mode,
    const Tensor& per_sample_weights_,
    Tensor& index_grad_weight) {

  Tensor &offset2bag_ = const_cast<Tensor &>(offset2bag__);

  auto ind_sort_ = indices_.sort();
  auto indices = std::get<0>(ind_sort_);
  auto ind_sort = std::get<1>(ind_sort_);
  auto offset2bag = offset2bag_.index_select(0, ind_sort);

  optional<Tensor> per_sample_weights;
  scalar_t* per_sample_weights_data;
  optional<int64_t> per_sample_weights_stride;
  if (per_sample_weights_.defined()) {
    per_sample_weights = per_sample_weights_.index_select(0, ind_sort);
    per_sample_weights_data = per_sample_weights->data_ptr<scalar_t>();
    per_sample_weights_stride = per_sample_weights->stride(0);
  }

  auto indices_data = indices.data_ptr<int64_t>();
  auto offsets_data = offsets_.data_ptr<int64_t>();
  auto offset2bag_data = offset2bag.data_ptr<int64_t>();
  int64_t numel = indices.numel();

  auto counts = compute_counts(num_weights, indices_data, numel);
  auto next_unique_index_idx =
      compute_counts_uniq(num_weights, indices_data, numel, counts);

  auto loop = [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; i++) {
      int64_t start = i == 0 ? 0 : next_unique_index_idx[i - 1];
      int64_t index = indices_data[start];
      for (int64_t j = start; j < next_unique_index_idx[i]; j++) {
        int64_t source = offset2bag_data[j];
        double scale = 1.0;
        if (per_sample_weights) {
          AT_ASSERT(mode == MODE_SUM);
          scale = per_sample_weights_data[*per_sample_weights_stride * j];
        }
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
        auto igwd = index_grad_weight.data_ptr<scalar_t>();
        auto gd = grad.data_ptr<scalar_t>();
        THBlas_axpy<scalar_t>(ddim, (scalar_t)scale, gd + ddim * source, 1,
                    igwd + ddim * index, 1);
      }
    }
  };
  if (numel > 1000) {
    at::parallel_for(0, (int64_t)next_unique_index_idx.size(), 0, loop);
  } else {
    loop(0, (int64_t)next_unique_index_idx.size());
  }
}

Tensor _embedding_bag_dense_backward_cpu(const Tensor &grad_, const Tensor &indices_,
                                  const Tensor &offsets_,
                                  const Tensor &offset2bag__,
                                  const Tensor &bag_size_,
                                  const Tensor& max_indices_, int64_t num_weights,
                                  bool scale_grad_by_freq, int64_t mode,
                                  const Tensor& per_sample_weights_) {
  // indices_, offsets_ and offset2bag__ are assumed having correct dtypes and
  // contiguous here due to the checks in _embedding_bag_backward above.
  // Also see NOTE [ embedding_bag Native Functions ] in native_functions.yaml
  // for more details.
  auto grad = grad_.contiguous();
  auto grad_arg = TensorArg(grad, "grad_", 1);
  checkScalarTypes("embedding_bag", grad_arg, {kFloat, kDouble});

  if (mode == MODE_MAX) {
    return _embedding_bag_dense_backward_cpu_max(
        grad_, bag_size_, max_indices_, num_weights);
  }
  AT_ASSERT(mode == MODE_MEAN || mode == MODE_SUM);

  auto index_grad_weight =
      at::zeros({num_weights, grad.size(1)}, grad.options());

  AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), "embedding_bag_backward", [&] {
      _embedding_bag_dense_backward_cpu_sum_mean<scalar_t>(
          grad, indices_, offsets_, offset2bag__, num_weights,
          scale_grad_by_freq, mode, per_sample_weights_, index_grad_weight);
  });
  return index_grad_weight;
}

template<typename scalar_t>
Tensor _embedding_bag_per_sample_weights_backward_cpu_template(
    const Tensor& grad,
    const Tensor& weight,  // NB: embedding table, not per_sample_weights
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& offset2bag,
    int64_t mode) {
  TORCH_CHECK(
      mode == MODE_SUM,
      "embedding_bag_backward: per_sample_weights only supported for mode='sum'");

  AT_ASSERT(grad.dim() == 2);
  auto embedding_features = grad.size(1);

  AT_ASSERT(indices.dim() == 1);
  auto num_samples = indices.size(0);

  AT_ASSERT(weight.dim() == 2);
  AT_ASSERT(weight.size(1) == embedding_features);

  auto output = at::zeros({num_samples}, grad.options());

  auto indices_arg = TensorArg(indices, "indices", 1);
  checkScalarType("embedding_bag", indices_arg, kLong);
  checkContiguous("embedding_bag", indices_arg);

  Tensor offset2bag_;
  if (indices.numel() != 0 && offset2bag.numel() == 0) {
    offset2bag_ = at::zeros(
       {indices.sizes()[0] + 1}, indices.options()); // offset2bag = [0 0 0 0 0]

    make_offset2bag(offsets, indices, offset2bag_);

    offset2bag_.resize_({indices.sizes()[0]});
  } else {
    auto offset2bag_arg = TensorArg(offset2bag, "offset2bag", 1);
    checkScalarType("embedding_bag", offset2bag_arg, kLong);
    checkContiguous("embedding_bag", offset2bag_arg);
    offset2bag_ = offset2bag;
  }

  auto grad_data = grad.data_ptr<scalar_t>();
  auto grad_stride0 = grad.stride(0);
  auto grad_stride1 = grad.stride(1);

  auto weight_data = weight.data_ptr<scalar_t>();
  auto weight_stride0 = weight.stride(0);
  auto weight_stride1 = weight.stride(1);

  auto indices_data = indices.data_ptr<int64_t>();

  // The following are contiguous
  auto output_data = output.data_ptr<scalar_t>();
  auto offset2bag_data = offset2bag_.data_ptr<int64_t>();

  // XXX: 64 was arbitrarily chosen. There is probably a sweet spot for this number.
  parallel_for(0, num_samples, 64, [&](int64_t begin, int64_t end) {
    for (int64_t sample_idx = begin; sample_idx < end; sample_idx++) {
      auto bag_idx = offset2bag_data[sample_idx];
      auto embedding_idx = indices_data[sample_idx];

      output_data[sample_idx] = THBlas_dot<scalar_t>(
          embedding_features,
          grad_data + grad_stride0 * bag_idx, grad_stride1,
          weight_data + weight_stride0 * embedding_idx, weight_stride1);
    }
  });
  return output;
}

Tensor _embedding_bag_per_sample_weights_backward_cpu(
    const Tensor& grad,
    const Tensor& weight,  // NB: embedding table, not per_sample_weights
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& offset2bag,
    int64_t mode) {
  return AT_DISPATCH_FLOATING_TYPES(
    grad.scalar_type(), "_embedding_bag_per_sample_weights_backward_cpu", [&]() {
      return _embedding_bag_per_sample_weights_backward_cpu_template<scalar_t>(
          grad, weight, indices, offsets, offset2bag, mode);
    }
  );
}

Tensor _embedding_bag_sparse_backward(
    const Tensor &grad_, const Tensor &indices, const Tensor &offsets,
    const Tensor &offset2bag, const Tensor &bag_size_, int64_t num_weights,
    bool scale_grad_by_freq, int64_t mode, const Tensor& per_sample_weights) {
  // indices, offsets and offset2bag are assumed having correct dtypes and
  // contiguous here due to the checks in _embedding_bag_backward above.
  // Also see NOTE [ embedding_bag Native Functions ] in native_functions.yaml
  // for more details.

  Tensor grad = grad_;
  Tensor index_grad = grad_.index_select(0, offset2bag);
  index_grad = apply_bag_size_backward(offsets, indices, mode, index_grad,
                                       offset2bag, bag_size_);
  if (per_sample_weights.defined()) {
    AT_ASSERT(mode == MODE_SUM);
    index_grad.mul_(per_sample_weights.unsqueeze(1));
  }
  return native::embedding_backward(index_grad, indices, num_weights, -1,
                                    scale_grad_by_freq, true);
}
}
} // namespace at::native
