#pragma once

/*
 * This op is only for testing the c10 dispatcher and might not support all
 * parameter combinations or backends the corresponding caffe2 op supports.
 * Please ignore this.
 * TODO Remove this comment once this is more final
 */

#include "caffe2/core/dispatch/DeviceId.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/Array.h"
#include <ATen/core/ArrayRef.h>

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
