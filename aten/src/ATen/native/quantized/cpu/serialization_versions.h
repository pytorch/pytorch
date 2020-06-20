#pragma once

#include <ATen/ATen.h>
#include <ATen/core/List.h>

#include <tuple>


/* Convoution prepacked parameters serialization.
 *
 * Files that need to be updated if version changes:
 * - ATen/native/quantized/cpu/fbgemm_utils.cpp
 * - jit/passes/onnx/unpack_quantized_weights.cpp
 *
 * Version 1
 * - Fields:
 *  1. weight
 *  2. bias
 *  3. stride x kSpatialDim
 *  4. padding x kSpatialDim
 *  5. dilation x kSpatialDim
 *  6. groups
 * - If the `groups` field has a tensor of size 1, it is the first version
 *
 * Version 2
 * - Fields:
 *  1. weight
 *  2. bias
 *  3. stride x kSpatialDim, padding x kSpatialDim, dilation x kSpatialDim
 *  4. Unused
 *  5. Unused
 *  6. version, groups
 * - Entry (3) is a list of tensors
 * - Entry (6) is a tensor of size 2
 *
 * Version 3
 * - Fields:
 *  1. weight
 *  2. bias
 *  3. stride x kSpatialDim, padding x kSpatialDim, dilation x kSpatialDim,
       output_padding x kSpatialDim
 *  4. Unused
 *  5. Unused
 *  6. version, groups, transpose
 * - Entry (3) is a list of tensors
 * - Entry (6) is a tensor of size 2
 */
constexpr int64_t kConvPackedParamsSerializationVersion = 2;
using ConvPackedParamsSerializationType = std::tuple<
  at::Tensor /*weight*/,
  c10::optional<at::Tensor> /*bias*/,
  // these are meant to be torch::List<int64_t> but
  // it's not supported by onnx, so we'll use Tensor as
  // a workaround
  torch::List<at::Tensor> /*stride, padding, dilation, output_padding*/,
  torch::List<at::Tensor> /*Unused*/,
  torch::List<at::Tensor> /*Unused*/,
  at::Tensor /*version, groups, transpose*/
>;
