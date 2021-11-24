#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/EmbeddingBag.h>
#include <ATen/native/quantized/cpu/quant_utils.h>
#include <torch/library.h>
#include <c10/util/irange.h>

#include <ATen/native/CPUBlas.h>

#include <cstring>
#include <memory>
#include <vector>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <tuple>
#include <vector>

namespace at {
namespace native {


struct TensorQuantizationFloatParams {
  double scale;
  float zero_point;
  int precision;
};

namespace {
  const int MODE_SUM = 0;
  const int MODE_MEAN = 1;
  const int MODE_MAX = 2;
}

TensorQuantizationFloatParams calc_per_channel_affine_float_qparams(
    float min_val,
    float max_val,
    int32_t qmin,
    int32_t qmax) {

  TORCH_CHECK(
      min_val <= max_val,
      "In ChooseQuantizationParams, min should be less than or equal to max");

  float min_val_neg = std::fmin(min_val, 0.0);
  float min_val_pos = std::fmax(max_val, 0.0);

  float scale = (max_val - min_val) / float(qmax - qmin);
  if(scale <= std::numeric_limits<float>().epsilon()){
    scale = 1.0;
  }
  float zero_point = -1.0 * min_val / scale;

  TensorQuantizationFloatParams result;
  result.scale = scale;
  result.zero_point = zero_point;
  return result;
}

static Tensor make_bag_size(
    const Tensor& offsets,
    const Tensor& indices,
    const int64_t mode,
    const bool include_last_offset,
    const bool requires_grad) {
  Tensor bag_size = at::empty(offsets.sizes(), offsets.options());
  make_bag_size_out(bag_size, offsets, indices, mode, include_last_offset, requires_grad);
  return bag_size;
}

static Tensor make_max_indices(
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& bag_size,
    const int64_t mode,
    bool include_last_offset) {
  Tensor max_indices = at::empty(bag_size.sizes(), offsets.options());
  make_max_indices_out(max_indices, weight, indices, offsets, bag_size, mode, include_last_offset);
  return max_indices;
}

static Tensor make_offset2bag(
    Tensor& output,
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    const int64_t mode,
    const c10::optional<Tensor>& per_sample_weights,
    const int64_t padding_idx) {
  Tensor offset2bag = at::empty({0}, offsets.options());
  make_offset2bag_out(offset2bag, output, weight, indices, offsets, mode, per_sample_weights, padding_idx);
  return offset2bag;
}

static Tensor apply_bag_size(
    const int64_t mode,
    Tensor &output,
    const Tensor &bag_size) {
  if (mode == MODE_MEAN) {
    auto bag_size_ = at::max(bag_size, at::ones_like(bag_size, LEGACY_CONTIGUOUS_MEMORY_FORMAT))
                         .to(output.options())
                         .unsqueeze(1)
                         .expand_as(output);
    output /= bag_size_;
  }
  return output;
}

std::pair<Tensor, Tensor> promoteIndicesAndOffsets(
    const Tensor& indices,
    const Tensor& offsets) {
  const auto commonType =
      promoteTypes(offsets.scalar_type(), indices.scalar_type());
  return {
      indices.scalar_type() == commonType ? indices
                                          : indices.toType(commonType),
      offsets.scalar_type() == commonType ? offsets
                                          : offsets.toType(commonType)};
}

/* Structure copied from caffe2::EmbeddingLookupGenericSlow */
template<typename data_t, typename index_t>
bool FusedFakeQuantEmbeddingLookup(
    const int64_t block_size,
    const int64_t output_size,
    const int64_t index_size,
    const int64_t data_size,
    const data_t* input,
    const index_t* indices,
    const index_t* lengths,
    const float* weights, // optional, can be null for sum reducer
    const float* scale_bias, // optional scale & bias params for uint8 input
    bool normalize_by_lengths,
    data_t* out) {

  /* pass quant_min and quant_max as parameters */
  const int64_t quant_min = 0;
  const int64_t quant_max = 255;

  int64_t current = 0;
  for (const auto m : c10::irange(output_size)) {
    memset(out, 0, sizeof(data_t) * block_size);
    if (current + lengths[m] > index_size) {
      return false;
    }
    for (int i = 0; i < lengths[m]; ++i) {
      int64_t idx = indices[current];
      if (idx < 0 || idx >= data_size) {
        return false;
      }
#ifdef __GNUC__
      if (current + 1 < index_size) {
        __builtin_prefetch(input + block_size * indices[current + 1], 0, 1);
      }
#endif // __GNUC__

      float w = 1.f, b = 0.f;
      if (weights) {
        w = weights[current];
      }
      if (scale_bias) {
        b = w * scale_bias[2 * indices[current] + 1];
        w = w * scale_bias[2 * indices[current]];
      }

      /* Inefficient hack, create tensor for row */
      at::Tensor w_row = at::empty({1, block_size},at::kFloat);
      std::memcpy(w_row.data_ptr(), input + (block_size * indices[current]), block_size * sizeof(data_t));

      at::Tensor w_min, w_max;
      TensorQuantizationFloatParams w_qparams{};
      std::tie(w_min, w_max) = at::_aminmax(w_row, 1);
      float* w_min_data = w_min.data_ptr<float>();
      float* w_max_data = w_max.data_ptr<float>();

      at::Tensor scales = at::empty_like(w_min);
      at::Tensor zero_points = at::empty_like(w_min, w_min.options().dtype(at::kFloat));

      w_qparams = calc_per_channel_affine_float_qparams(
        w_min_data[0],
        w_max_data[0],
        quant_min,
        quant_max);
      scales[0] = w_qparams.scale;
      zero_points[0] = w_qparams.zero_point;

      at::Tensor fake_quant_w = at::fake_quantize_per_channel_affine(w_row, scales, zero_points, 0, quant_min, quant_max);
      float* fake_quant_w_data = fake_quant_w.data_ptr<float>();

      for (const auto j : c10::irange(block_size)) {
        //out[j] += w * input[block_size * indices[current] + j] + b;
        out[j] += w * input[j] + b;
        //out[j] = i;
      }

      ++current;
    }
    if (normalize_by_lengths && lengths[m]) {
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      float scale = 1.f / lengths[m];
      for (const auto j : c10::irange(block_size)) {
        out[j] *= scale;
      }
    }
    out += block_size;
  }
  return current == index_size;
}

template<typename data_t, typename index_t>
void fused_index_select_add(const Tensor &select_indices,
                             const Tensor &add_indices,
                             const Tensor &src,
                             Tensor &output,
                             const Tensor& offsets,
                             bool include_last_offset,
                             Tensor &bag_size,
                             index_t padding_idx) {
  int64_t ddim = src.sizes()[1];
  auto* select_indices_data = select_indices.data_ptr<index_t>();
  auto* output_data = output.data_ptr<float>();

  /* TODO: add if/else for non-fast-path ones */
  auto src_contig = src.contiguous();
  auto* src_data = src_contig.data_ptr<float>();
  int64_t output_size = offsets.numel() - 1;
  index_t* offsets_data = offsets.data_ptr<index_t>();
  std::vector<index_t> offsets_include_last;

  if (include_last_offset) {
    output_size = offsets.numel() - 1;
  } else {
    output_size = offsets.numel();
    offsets_include_last.resize(offsets.numel() + 1);
    if (offsets.numel() > 0) {
      std::memcpy(
          offsets_include_last.data(),
          offsets.data_ptr<index_t>(),
          sizeof(index_t) * offsets.numel());
    }
    offsets_include_last[offsets.numel()] = select_indices.numel();
    offsets_data = offsets_include_last.data();
  }

  at::parallel_for(
      0, output_size, 1, [&](index_t start_idx, index_t end_idx) {
        FusedFakeQuantEmbeddingLookup<data_t, index_t>(
            /*block_size=*/ddim,
            /*output_size=*/end_idx - start_idx,
            /*index_size=*/offsets_data[end_idx] - offsets_data[start_idx],
            /*data_size=*/src.sizes()[0],
            /*input=*/src_data,
            /*indices=*/select_indices_data + offsets_data[start_idx],
            /*offsets=*/offsets_data + start_idx,
            /*weights=*/nullptr,
            /*scale_bias=*/nullptr,
            /*normalize_by_lengths=*/false,
            /*out=*/output_data + start_idx * ddim);
      });
}

void _fused_embedding_bag_cpu_impl_out(Tensor& output, Tensor& offset2bag,
                            Tensor& bag_size, Tensor& max_indices,
                            const Tensor &weight, const Tensor &indices,
                            const Tensor &offsets, const int64_t mode,
                            const c10::optional<Tensor>& per_sample_weights,
                            bool include_last_offset, int64_t padding_idx) {
  if (mode == MODE_MEAN || mode == MODE_SUM) {
      TORCH_CHECK(weight.scalar_type() == at::kFloat, "only weight dtype of kFloat is currently supported for fused FakeQuant EmbeddingBag")
      AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "embedding_bag_no_grad_cpu_out",
        [&indices, &offset2bag, &per_sample_weights, &weight, &output, &offsets, &include_last_offset, &mode, &bag_size, &padding_idx]() {
        TORCH_CHECK(!(per_sample_weights.has_value() && per_sample_weights.value().defined()),
                    "per_sample_weights are not currently supported for fused FakeQuant EmbeddingBag");
        fused_index_select_add<float, index_t>(indices, offset2bag, weight, output, offsets, include_last_offset, bag_size, padding_idx);
      });
    apply_bag_size(mode, output, bag_size);
    if (mode == MODE_SUM) {
      // make bag_size output deterministic
      at::native::zero_(bag_size);
    }
     max_indices.copy_(bag_size);
  } else { // MODE_MAX
    TORCH_CHECK(mode == MODE_MEAN || mode == MODE_SUM,
                "Only MODE_MEAN and MODE_SUM are currently supported for fused FakeQuant embeddingbag");
  /*
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      weight.scalar_type(), "embedding_bag_cpu_max_out", [&]() {
        embedding_bag_cpu_max_out<scalar_t>(
          max_indices, weight, indices, offset2bag, output, include_last_offset, bag_size, padding_idx);
      }
    );
  */
  }
}

// Assumes all input tensors except for `weight` are contiguous.
// See NOTE [ embedding_bag Native Functions ] in native_functions.yaml for details
std::tuple<Tensor, Tensor, Tensor, Tensor> _fused_embedding_bag_cpu_impl(
    const Tensor& weight,
    const Tensor& indices_,
    const Tensor& offsets_,
    const int64_t mode,
    const Tensor& per_sample_weights,
    bool include_last_offset,
    int64_t padding_idx,
    bool requires_grad) {
  Tensor indices, offsets;
  std::tie(indices, offsets) = promoteIndicesAndOffsets(indices_, offsets_);
  check_arguments(weight, indices, offsets, mode, per_sample_weights, include_last_offset);

  Tensor output = at::empty(
      {include_last_offset ? offsets.sizes()[0] - 1 : offsets.sizes()[0],
       weight.sizes()[1]},
      weight.options());

  Tensor offset2bag = make_offset2bag(output, weight, indices, offsets, mode, per_sample_weights, padding_idx);

  Tensor bag_size = make_bag_size(offsets, indices, mode, include_last_offset, requires_grad);

  Tensor max_indices = make_max_indices(weight, indices, offsets, bag_size, mode, include_last_offset);

  _fused_embedding_bag_cpu_impl_out(output, offset2bag,
                          bag_size, max_indices,
                          weight, indices, offsets,
                          mode, per_sample_weights,
                          include_last_offset, padding_idx);

  return std::make_tuple(std::move(output), std::move(offset2bag), std::move(bag_size), std::move(max_indices));
}

std::tuple<Tensor, Tensor, Tensor, Tensor> _fused_fake_quant_embedding_bag_cpu(
                  const Tensor &weight, const Tensor &indices,
                  const Tensor &offsets,
                  int64_t quant_min, int64_t quant_max,
                  const bool scale_grad_by_freq,
                  const int64_t mode, bool sparse,
                  const c10::optional<Tensor>& per_sample_weights_opt,
                  bool include_last_offset,
                  int64_t padding_idx)
{  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> per_sample_weights_maybe_owned = at::borrow_from_optional_tensor(per_sample_weights_opt);
  const Tensor& per_sample_weights = *per_sample_weights_maybe_owned;
  std::ignore = scale_grad_by_freq;
  std::ignore = sparse;

  return _fused_embedding_bag_cpu_impl(
      weight,
      indices,
      offsets,
      mode,
      per_sample_weights,
      include_last_offset,
      padding_idx,
      /*requires_grad=*/false);
}


TORCH_LIBRARY_IMPL(quantized, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::fused_fake_quant_embedding_bag"),
      TORCH_FN(_fused_fake_quant_embedding_bag_cpu));
}

} // namespace native
} // namespace at


/*

#include <ATen/native/EmbeddingBag.h>
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/quantized/cpu/quant_utils.h>

#include <c10/util/irange.h>


namespace at {
namespace native {

namespace {
std::tuple<Tensor, Tensor, Tensor, Tensor> _fused_fake_quant_embedding_bag_cpu(const Tensor &weight, const Tensor &indices,
                  const Tensor &offsets,
                  int64_t quant_min, int64_t quant_max,
                  const bool scale_grad_by_freq,
                  const int64_t mode, bool sparse, const c10::optional<Tensor>& per_sample_weights_opt, bool include_last_offset,
                  int64_t padding_idx)
{
  Tensor a,b,c,d;

  return std::make_tuple(std::move(a), std::move(b), std::move(c), std::move(d));
}


TORCH_LIBRARY_IMPL(quantized, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::fused_fake_quant_embedding_bag"),
      TORCH_FN(_fused_fake_quant_embedding_bag_cpu));
}

} // namespace
} // namespace native
} // namespace at
*/
