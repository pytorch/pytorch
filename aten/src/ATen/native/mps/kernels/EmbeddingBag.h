#pragma once
#include <c10/metal/common.h>

#ifdef __METAL__
enum class EmbeddingBagMode { SUM = 0, MEAN, MAX };
#else
#include <ATen/native/EmbeddingBag.h>
using at::native::EmbeddingBagMode;
#endif

template <typename idx_type_t = uint32_t>
struct EmbeddingBagParams {
  ::c10::metal::array<idx_type_t, 2> weight_strides;
  ::c10::metal::array<idx_type_t, 2> output_strides;
  ::c10::metal::array<idx_type_t, 2> max_indices_strides;

  bool use_per_sample_weights;
  idx_type_t per_sample_weights_stride;

  idx_type_t num_indices;
  idx_type_t num_bags;
  idx_type_t feature_size;
  idx_type_t num_weights;

  EmbeddingBagMode mode;
  int64_t padding_idx;
};

template <typename idx_type_t = uint32_t>
struct EmbeddingBagBackwardParams {
  ::c10::metal::array<idx_type_t, 2> weight_grad_strides;
  ::c10::metal::array<idx_type_t, 2> output_grad_strides;
  ::c10::metal::array<idx_type_t, 2> max_indices_strides;
  bool use_per_sample_weights;
  idx_type_t per_sample_weights_stride;
  idx_type_t feature_size;
  EmbeddingBagMode mode;
  int64_t padding_idx;
};

template <typename idx_type_t = uint32_t>
struct EmbeddingBagPerSampleWeightsBackwardParams {
  ::c10::metal::array<idx_type_t, 2> output_grad_strides;
  ::c10::metal::array<idx_type_t, 2> weight_strides;
  idx_type_t per_sample_weights_grad_stride;
  idx_type_t feature_size;
  int64_t padding_idx;
};
