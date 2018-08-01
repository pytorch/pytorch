#pragma once

#include "caffe2/core/dispatch/DeviceId.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/Array.h"
#include "caffe2/utils/ArrayRef.h"

namespace caffe2 {
namespace ops {

struct Concat final {
  static constexpr const char* name = "concat";

  using Signature = void(
      c10::ArrayRef<const Tensor<CPUContext>*> inputs,
      Tensor<CPUContext>* output,
      Tensor<CPUContext>* split_info,
      int add,
      int add_axis,
      CPUContext* context);

  static constexpr c10::guts::array<const char*, 6> parameter_names = {
      {"inputs", "output", "split_info_output", "add", "add_axis", "context"}};

  static c10::DeviceTypeId dispatch_key(
      c10::ArrayRef<const Tensor<CPUContext>*> inputs,
      Tensor<CPUContext>* output,
      Tensor<CPUContext>* split_info,
      int add,
      int add_axis,
      CPUContext* context) {
    return c10::DeviceTypeId::CPU;
  }
};

} // namespace ops
} // namespace caffe2
