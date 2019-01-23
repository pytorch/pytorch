#pragma once

#include <ATen/core/Tensor.h>
#include <c10/util/Array.h>
#include "caffe2/core/context_base.h"
#include <ATen/core/ivalue.h>
#include <ATen/core/blob.h>

namespace caffe2 {
namespace ops {

struct ExpandDims final {
  struct State {
    std::vector<int64_t> dims;
    bool initialized = false;
  };

  static constexpr const char* name = "expand_dims";

  using Signature = void(
      const at::Tensor& input,
      const at::Tensor& output,
      ArrayRef<int64_t> dims,
      intrusive_ptr<Blob> state);

  static constexpr size_t num_dispatch_args() {return 1;}

  static constexpr size_t num_outputs() {return 1;}

  static constexpr c10::guts::array<const char*, 4> parameter_names = {
      {"input", "output", "dims", "state"}};
};

} // namespace ops
} // namespace caffe2
