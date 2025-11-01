#include <ATen/native/mps/kernels/EmbeddingBag.h>
#include <c10/metal/utils.h>
#include <metal_array>
#include <metal_stdlib>

using namespace metal;
using namespace c10::metal;

template <EmbeddingBagMode M, typename T>
struct ReductionOpInit {
  inline opmath_t<T> operator()() {
    return 0;
  }
};

template <typename T>
struct ReductionOpInit<EmbeddingBagMode::MAX, T> {
  inline opmath_t<T> operator()() {
    return static_cast<opmath_t<T>>(-INFINITY);
  }
};

template <EmbeddingBagMode M, typename T>
struct ReductionOp {
  inline opmath_t<T> operator()(
      opmath_t<T> weight_val,
      opmath_t<T> out_val,
      bool /*is_first*/) {
    return weight_val + out_val;
  }
};

template <typename T>
struct ReductionOp<EmbeddingBagMode::MAX, T> {
  inline opmath_t<T> operator()(
      opmath_t<T> weight_val,
      opmath_t<T> out_val,
      bool is_first) {
    return (is_first || weight_val > out_val) ? weight_val : out_val;
  }
};

template <EmbeddingBagMode M, typename T>
struct MaybeApplyPerSampleWeight {
  inline opmath_t<T> operator()(
      opmath_t<T> weight_val,
      uint32_t /*per_sample_weights_index*/,
      constant T* /*per_sample_weights*/,
      uint32_t /*per_sample_weights_stride*/) {
    return weight_val;
  }
};

template <typename T>
struct MaybeApplyPerSampleWeight<EmbeddingBagMode::SUM, T> {
  inline opmath_t<T> operator()(
      opmath_t<T> weight_val,
      uint32_t per_sample_weights_index,
      constant T* per_sample_weights,
      uint32_t per_sample_weights_stride) {
    if (per_sample_weights_stride) {
      T per_sample_weight = per_sample_weights
          [per_sample_weights_stride * per_sample_weights_index];
      return static_cast<opmath_t<T>>(per_sample_weight) * weight_val;
    } else {
      return weight_val;
    }
  }
};

template <EmbeddingBagMode M, typename T, typename I>
struct MaybeCalcMaxIndex {
  inline void operator()(
      opmath_t<T> /*weight_val*/,
      opmath_t<T> /*out_val*/,
      bool /*is_first*/,
      thread I& /*max_idx*/,
      I /*weight_idx*/,
      bool /*pad*/) {}
};

template <typename T, typename I>
struct MaybeCalcMaxIndex<EmbeddingBagMode::MAX, T, I> {
  inline void operator()(
      opmath_t<T> weight_val,
      opmath_t<T> out_val,
      bool is_first,
      thread I& max_idx,
      I weight_idx,
      bool pad) {
    max_idx = !pad && (is_first || weight_val > out_val) ? weight_idx : max_idx;
  }
};

template <EmbeddingBagMode M, typename T>
struct ReductionOpFinal {
  inline T operator()(opmath_t<T> val, uint32_t) {
    return static_cast<T>(val);
  }
};

template <typename T>
struct ReductionOpFinal<EmbeddingBagMode::MEAN, T> {
  inline T operator()(opmath_t<T> val, uint32_t count) {
    auto out = val / count;
    return static_cast<T>((count == 0) ? 0 : out);
  }
};

template <typename T>
struct ReductionOpFinal<EmbeddingBagMode::MAX, T> {
  inline T operator()(opmath_t<T> val, uint32_t count) {
    return static_cast<T>((count == 0) ? 0 : val);
  }
};

template <EmbeddingBagMode M, typename I>
struct MaybeWriteMaxIndex {
  inline void operator()(
      device I*,
      const constant ::c10::metal::array<uint32_t, 2>&,
      uint32_t,
      uint32_t,
      I) {}
};

template <typename I>
struct MaybeWriteMaxIndex<EmbeddingBagMode::MAX, I> {
  inline void operator()(
      device I* max_indices,
      const constant ::c10::metal::array<uint32_t, 2>& max_indices_strides,
      uint32_t bag_idx,
      uint32_t feature_idx,
      I max_idx) {
    max_indices
        [bag_idx * max_indices_strides[0] +
         feature_idx * max_indices_strides[1]] = max_idx;
  }
};

template <EmbeddingBagMode M, typename T, typename I>
void embedding_bag_impl(
    constant T* weight,
    constant I* indices,
    constant I* offsets,
    constant T* per_sample_weights,
    device T* output,
    device I* offset2bag,
    device I* bag_size,
    device I* max_indices,
    constant EmbeddingBagParams<uint32_t>& params,
    uint tid) {
  auto num_indices = params.num_indices;
  auto num_bags = params.num_bags;
  auto feature_size = params.feature_size;
  auto padding_idx = params.padding_idx;
  auto per_sample_weights_stride = params.per_sample_weights_stride;
  constant auto& output_strides = params.output_strides;
  constant auto& weight_strides = params.weight_strides;
  constant auto& max_indices_strides = params.max_indices_strides;

  auto bag_idx = tid / feature_size;
  auto feature_idx = tid % feature_size;

  uint32_t offsets_end = min(bag_idx + 1, num_bags - 1);
  bool is_last_bag = bag_idx + 1 == num_bags;
  uint32_t indices_start = static_cast<uint32_t>(offsets[bag_idx]);
  uint32_t indices_end = is_last_bag * (num_indices) +
      (!is_last_bag) * (static_cast<uint32_t>(offsets[offsets_end]));

  auto out_val = ReductionOpInit<M, T>()();

  uint32_t bag_size_ = 0;
  I max_idx = 0;

  for (uint32_t indices_idx = indices_start; indices_idx < indices_end;
       indices_idx++) {
    I weight_idx = indices[indices_idx];
    bool pad = (weight_idx == padding_idx);
    auto weight_val = static_cast<opmath_t<T>>(
        weight
            [static_cast<uint32_t>(weight_idx) * weight_strides[0] +
             feature_idx * weight_strides[1]]);

    weight_val = MaybeApplyPerSampleWeight<M, T>()(
        weight_val, indices_idx, per_sample_weights, per_sample_weights_stride);

    auto new_out_val = ReductionOp<M, T>()(weight_val, out_val, bag_size_ == 0);

    MaybeCalcMaxIndex<M, T, I>()(
        weight_val, out_val, bag_size_ == 0, max_idx, weight_idx, pad);

    out_val = pad ? out_val : new_out_val;
    offset2bag[indices_idx] = bag_idx;
    bag_size_ += static_cast<uint32_t>(!pad);
  }

  output[bag_idx * output_strides[0] + feature_idx * output_strides[1]] =
      ReductionOpFinal<M, T>()(out_val, bag_size_);

  bag_size[bag_idx] = bag_size_;

  MaybeWriteMaxIndex<M, I>()(
      max_indices, max_indices_strides, bag_idx, feature_idx, max_idx);
}

#define DISPATCH_IMPL(MODE)        \
  return embedding_bag_impl<MODE>( \
      weight,                      \
      indices,                     \
      offsets,                     \
      per_sample_weights,          \
      output,                      \
      offset2bag,                  \
      bag_size,                    \
      max_indices,                 \
      params,                      \
      tid)

template <typename T, typename I>
kernel void embedding_bag(
    constant T* weight [[buffer(0)]],
    constant I* indices [[buffer(1)]],
    constant I* offsets [[buffer(2)]],
    constant T* per_sample_weights [[buffer(3)]],
    device T* output [[buffer(4)]],
    device I* offset2bag [[buffer(5)]],
    device I* bag_size [[buffer(6)]],
    device I* max_indices [[buffer(7)]],
    constant EmbeddingBagParams<uint32_t>& params [[buffer(8)]],
    uint tid [[thread_position_in_grid]]) {
  switch (params.mode) {
    case EmbeddingBagMode::SUM:
      DISPATCH_IMPL(EmbeddingBagMode::SUM);
    case EmbeddingBagMode::MEAN:
      DISPATCH_IMPL(EmbeddingBagMode::MEAN);
    case EmbeddingBagMode::MAX:
      DISPATCH_IMPL(EmbeddingBagMode::MAX);
  }
}

#define REGISTER_EMBEDDING_BAG_OP(T, I)                             \
  template [[host_name("embedding_bag_" #T "_" #I)]]                \
  kernel void embedding_bag<T, I>(                                  \
      constant T * weight [[buffer(0)]],                            \
      constant I * indices [[buffer(1)]],                           \
      constant I * offsets [[buffer(2)]],                           \
      constant T * per_sample_weights [[buffer(3)]],                \
      device T * output [[buffer(4)]],                              \
      device I * offset2bag [[buffer(5)]],                          \
      device I * bag_size [[buffer(6)]],                            \
      device I * max_indices [[buffer(7)]],                         \
      constant EmbeddingBagParams<uint32_t> & params [[buffer(8)]], \
      uint tid [[thread_position_in_grid]]);

REGISTER_EMBEDDING_BAG_OP(float, int);
REGISTER_EMBEDDING_BAG_OP(float, long);
REGISTER_EMBEDDING_BAG_OP(half, int);
REGISTER_EMBEDDING_BAG_OP(half, long);
REGISTER_EMBEDDING_BAG_OP(bfloat, int);
REGISTER_EMBEDDING_BAG_OP(bfloat, long);
