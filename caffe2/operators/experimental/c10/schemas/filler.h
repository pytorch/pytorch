#pragma once

#include <c10/core/dispatch/DeviceId.h>
#include <c10/core/Tensor.h>
#include <c10/util/Array.h>
#include <c10/util/ArrayRef.h>
#include "caffe2/core/context_base.h"

namespace caffe2 {
namespace ops {

// GivenTensorFill
// GivenTensorInt64Fill
// GivenTensorIntFill

template <class T>
struct GivenTensorFill final {
  static constexpr const char* name = "given_tensor_fill";

  using Signature = void(
      at::ArrayRef<C10Tensor> inputs,
      const C10Tensor& output,
      const std::vector<int64_t>& shape,
      const std::vector<int>& extra_shape,
      bool input_as_shape,
      const C10Tensor& values,
      BaseContext* context);

  static constexpr c10::guts::array<const char*, 7> parameter_names = {
      {"inputs",
       "output",
       "shape",
       "extra_shape",
       "input_as_shape",
       "values",
       "context"}};

   static constexpr size_t num_outputs() {return 1;}

   static c10::DeviceTypeId dispatch_key(
      at::ArrayRef<C10Tensor> inputs,
      const C10Tensor& output,
      const std::vector<int64_t>& shape,
      const std::vector<int>& extra_shape,
      bool input_as_shape,
      const C10Tensor& values,
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
      at::ArrayRef<C10Tensor> inputs,
      const C10Tensor& output,
      const std::vector<int64_t>& shape,
      const std::vector<int>& extra_shape,
      bool input_as_shape,
      int dtype,
      Value value,
      BaseContext* context);

  static constexpr size_t num_outputs() {return 1;}

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
      at::ArrayRef<C10Tensor> inputs,
      const C10Tensor& output,
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
      at::ArrayRef<C10Tensor> inputs,
      const C10Tensor& output,
      const std::vector<int64_t>& shape,
      const std::vector<int>& extra_shape,
      bool input_as_shape,
      float min,
      float max,
      BaseContext* context);

  static constexpr size_t num_outputs() {return 1;}

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
      at::ArrayRef<C10Tensor> inputs,
      const C10Tensor& output,
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
