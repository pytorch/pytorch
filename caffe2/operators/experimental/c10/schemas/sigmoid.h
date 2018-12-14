#pragma once

#include <c10/core/Tensor.h>
#include <c10/util/Array.h>

namespace caffe2 {
namespace ops {

struct Sigmoid final {
  static constexpr const char* name = "sigmoid";

  using Signature =
      void(const C10Tensor& input, const C10Tensor& output);

  static constexpr size_t num_dispatch_args() {return 1;}

  static constexpr size_t num_outputs() {return 1;}

  static constexpr c10::guts::array<const char*, 2> parameter_names = {
      {"input", "output"}};
};

} // namespace ops
} // namespace caffe2
