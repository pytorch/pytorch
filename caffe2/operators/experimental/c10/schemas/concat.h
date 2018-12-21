#pragma once

#include <c10/core/dispatch/DeviceId.h>
#include "caffe2/core/tensor.h"
#include <c10/util/Array.h>
#include <c10/util/ArrayRef.h>

namespace caffe2 {
namespace ops {

struct Concat final {
  static constexpr const char* name = "concat";

  using Signature = void(
      at::ArrayRef<const Tensor*> inputs,
      Tensor* output,
      Tensor* split_info,
      int add,
      int add_axis,
      BaseContext* context);

  static constexpr c10::guts::array<const char*, 6> parameter_names = {
      {"inputs", "output", "split_info_output", "add", "add_axis", "context"}};

  static c10::DeviceTypeId dispatch_key(
      at::ArrayRef<const Tensor*> inputs,
      Tensor* output,
      Tensor* split_info,
      int add,
      int add_axis,
      BaseContext* context) {
    return c10::DeviceTypeId::CPU;
  }
};

} // namespace ops
} // namespace caffe2
