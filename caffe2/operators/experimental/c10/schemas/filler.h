#pragma once

#include <ATen/core/dispatch/OpSchema.h>
#include <ATen/core/dispatch/DeviceId.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/ivalue.h>
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
      ArrayRef<at::Tensor> inputs,
      const at::Tensor& output,
      ArrayRef<int64_t> shape,
      ArrayRef<int64_t> extra_shape,
      bool input_as_shape,
      const at::Tensor& values);

  static constexpr c10::guts::array<const char*, 6> parameter_names = {
      {"inputs",
       "output",
       "shape",
       "extra_shape",
       "input_as_shape",
       "values"}};

   static constexpr size_t num_outputs() {return 1;}

   static c10::DeviceTypeId dispatch_key(
      const Stack* stack) {
    return c10::DeviceTypeId::CPU;
  }
};

struct ConstantFill final {
  static constexpr const char* name = "constant_fill";

  using Signature = void(
      ArrayRef<at::Tensor> inputs,
      const at::Tensor& output,
      ArrayRef<int64_t> shape,
      ArrayRef<int64_t> extra_shape,
      bool input_as_shape,
      int dtype,
      IValue value);

  static constexpr size_t num_outputs() {return 1;}

  static constexpr c10::guts::array<const char*, 7> parameter_names = {
      {"inputs",
       "output",
       "shape",
       "extra_shape",
       "input_as_shape",
       "dtype",
       "value"}};

  static c10::DeviceTypeId dispatch_key(
      const Stack* stack) {
    return c10::DeviceTypeId::CPU;
  }
};

struct UniformFill final {
  static constexpr const char* name = "uniform_fill";

  using Signature = void(
      ArrayRef<at::Tensor> inputs,
      const at::Tensor& output,
      ArrayRef<int64_t> shape,
      ArrayRef<int64_t> extra_shape,
      bool input_as_shape,
      float min,
      float max);

  static constexpr size_t num_outputs() {return 1;}

  static constexpr c10::guts::array<const char*, 7> parameter_names = {
      {"inputs",
       "output",
       "shape",
       "extra_shape",
       "input_as_shape",
       "min",
       "max"}};

  static c10::DeviceTypeId dispatch_key(
      const Stack* stack) {
    return c10::DeviceTypeId::CPU;
  }
};

} // namespace ops
} // namespace caffe2
