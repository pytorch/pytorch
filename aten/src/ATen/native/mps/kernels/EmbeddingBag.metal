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
      T weight_val,
      opmath_t<T> out_val,
      uint32_t per_sample_weights_index,
      constant T* per_sample_weights,
      uint32_t per_sample_weights_strides);
};

template <typename T>
struct ReductionOp<EmbeddingBagMode::SUM, T> {
  inline opmath_t<T> operator()(
      T weight_val,
      opmath_t<T> out_val,
      uint32_t per_sample_weights_index,
      constant T* per_sample_weights,
      uint32_t per_sample_weights_strides) {
    if (per_sample_weights_strides) {
      T per_sample_weight = per_sample_weights
          [per_sample_weights_strides * per_sample_weights_index];
      return static_cast<opmath_t<T>>(per_sample_weight) *
          static_cast<opmath_t<T>>(weight_val) +
          out_val;
    } else {
      return static_cast<opmath_t<T>>(weight_val) + out_val;
    }
  }
};

template <typename T>
struct ReductionOp<EmbeddingBagMode::MEAN, T> {
  inline opmath_t<T> operator()(
      T weight_val,
      opmath_t<T> out_val,
      uint32_t,
      constant T*,
      uint32_t) {
    return static_cast<opmath_t<T>>(weight_val) + out_val;
  }
};

template <typename T>
struct ReductionOp<EmbeddingBagMode::MAX, T> {
  inline opmath_t<T> operator()(
      T weight_val,
      opmath_t<T> out_val,
      uint32_t,
      constant T*,
      uint32_t) {
    return max(static_cast<opmath_t<T>>(weight_val), out_val);
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
  auto per_sample_weights_strides = params.per_sample_weights_strides;
  constant auto& output_strides = params.output_strides;
  constant auto& weight_strides = params.weight_strides;
  constant auto& max_indices_strides = params.max_indices_strides;

  auto bag_idx = tid / feature_size;
  auto feature_idx = tid % feature_size;

  output += bag_idx * output_strides[0] + feature_idx * output_strides[1];

  uint32_t offsets_end = min(bag_idx + 1, num_bags - 1);
  bool is_last_bag = bag_idx + 1 == num_bags;
  uint32_t indices_start = static_cast<uint32_t>(offsets[bag_idx]);
  uint32_t indices_end = is_last_bag * (num_indices) +
      (!is_last_bag) * (static_cast<uint32_t>(offsets[offsets_end]));

  auto out_val = ReductionOpInit<M, T>()();

  uint32_t bag_size_ = 0;

  for (uint32_t indices_idx = indices_start; indices_idx < indices_end;
       indices_idx++) {
    I weight_idx = indices[indices_idx];
    bool pad = (weight_idx == padding_idx);
    T weight_val = weight
        [static_cast<uint32_t>(weight_idx) * weight_strides[0] +
         feature_idx * weight_strides[1]];

    bag_size_ += static_cast<uint32_t>(!pad);

    auto tmp_val = ReductionOp<M, T>()(
        weight_val,
        out_val,
        indices_idx,
        per_sample_weights,
        per_sample_weights_strides);

    out_val = pad ? out_val : tmp_val;
  }

  *output = ReductionOpFinal<M, T>()(out_val, bag_size_);
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
