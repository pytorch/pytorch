#pragma once

#include <ATen/core/Tensor.h>
#include <c10/util/Array.h>

namespace caffe2 {
namespace ops {

struct Relu final {
  static constexpr const char* name = "relu";

  using Signature =
      void(const at::Tensor& input, const at::Tensor& output);

  static constexpr size_t num_dispatch_args() {return 1;}

  static constexpr size_t num_outputs() {return 1;}

  static constexpr c10::guts::array<const char*, 2> parameter_names = {
      {"input", "output"}};
};

} // namespace ops
} // namespace caffe2
