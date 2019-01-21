#pragma once

#include <ATen/core/Tensor.h>
#include <c10/util/Array.h>

namespace caffe2 {
namespace ops {

struct SparseLengthsSum final {
  static constexpr const char* name = "sparse_lengths_sum";

  using Signature = void(
      const at::Tensor& data,
      const at::Tensor& indices,
      const at::Tensor& lengths,
      const at::Tensor& output);

  static constexpr size_t num_dispatch_args() {return 3;}

  static constexpr size_t num_outputs() {return 1;}

  static constexpr c10::guts::array<const char*, 4> parameter_names = {
      {"data", "indices", "lengths", "output"}};
};

} // namespace ops
} // namespace caffe2
