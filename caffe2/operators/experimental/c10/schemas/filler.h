#pragma once

#include "caffe2/core/dispatch/DeviceId.h"
#include "caffe2/core/tensor.h"
#include <c10/util/Array.h>
#include <ATen/core/ArrayRef.h>

namespace caffe2 {
namespace ops {

// GivenTensorFill
// GivenTensorInt64Fill
// GivenTensorIntFill

template <class T>
struct GivenTensorFill final {
  static constexpr const char* name = "given_tensor_fill";

  using Signature = void(
      at::ArrayRef<const Tensor*> inputs,
      Tensor* output,
      const std::vector<int64_t>& shape,
      const std::vector<int>& extra_shape,
      bool input_as_shape,
      const Tensor& values,
      BaseContext* context);

  static constexpr c10::guts::array<const char*, 7> parameter_names = {
      {"inputs",
       "output",
       "shape",
       "extra_shape",
       "input_as_shape",
       "values",
       "context"}};

  static c10::DeviceTypeId dispatch_key(
      at::ArrayRef<const Tensor*> inputs,
      Tensor* output,
      const std::vector<int64_t>& shape,
      const std::vector<int>& extra_shape,
      bool input_as_shape,
      const Tensor& values,
      BaseContext* context) {
    return c10::DeviceTypeId::CPU;
  }
};

struct ConstantFill final {
  union Value {
    float as_float;
    int32_t as_int32;
    int64_t as_int64;
    bool as_bool;
  };
  static constexpr const char* name = "constant_fill";

  using Signature = void(
      at::ArrayRef<const Tensor*> inputs,
      Tensor* output,
      const std::vector<int64_t>& shape,
      const std::vector<int>& extra_shape,
      bool input_as_shape,
      int dtype,
      Value value,
      BaseContext* context);

  static constexpr c10::guts::array<const char*, 8> parameter_names = {
      {"inputs",
       "output",
       "shape",
       "extra_shape",
       "input_as_shape",
       "dtype",
       "value",
       "context"}};

  static c10::DeviceTypeId dispatch_key(
      at::ArrayRef<const Tensor*> inputs,
      Tensor* output,
      const std::vector<int64_t>& shape,
      const std::vector<int>& extra_shape,
      bool input_as_shape,
      int dtype,
      Value value,
      BaseContext* context) {
    return c10::DeviceTypeId::CPU;
  }
};

struct UniformFill final {
  static constexpr const char* name = "uniform_fill";

  using Signature = void(
      at::ArrayRef<const Tensor*> inputs,
      Tensor* output,
      const std::vector<int64_t>& shape,
      const std::vector<int>& extra_shape,
      bool input_as_shape,
      float min,
      float max,
      BaseContext* context);

  static constexpr c10::guts::array<const char*, 8> parameter_names = {
      {"inputs",
       "output",
       "shape",
       "extra_shape",
       "input_as_shape",
       "min",
       "max",
       "context"}};

  static c10::DeviceTypeId dispatch_key(
      at::ArrayRef<const Tensor*> inputs,
      Tensor* output,
      const std::vector<int64_t>& shape,
      const std::vector<int>& extra_shape,
      bool input_as_shape,
      float min,
      float max,
      BaseContext* context) {
    return c10::DeviceTypeId::CPU;
  }
};

} // namespace ops
} // namespace caffe2
