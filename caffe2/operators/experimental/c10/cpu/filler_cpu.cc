#include "caffe2/core/dispatch/KernelRegistration.h"
#include "caffe2/operators/experimental/c10/schemas/filler.h"
#include "caffe2/utils/math.h"

using caffe2::CPUContext;
using caffe2::Tensor;
using caffe2::TensorCPU;
using std::vector;

namespace caffe2 {
namespace {
void filler_init(
    at::ArrayRef<const Tensor*> inputs,
    Tensor* output,
    const std::vector<int64_t>& shape,
    const std::vector<int>& extra_shape,
    bool input_as_shape) {
  if (inputs.size()) {
    auto real_shape = vector<int64_t>{};
    if (input_as_shape) {
      // Shape input must be in CPU context
      auto& input = *inputs[0];
      CAFFE_ENFORCE_EQ(
          input.dim(),
          1,
          "When input_as_shape is true, the input must be a 1D tensor of "
          "data type int64_t");
      auto* shape_data = input.template data<int64_t>();
      real_shape.insert(
          real_shape.end(), shape_data, shape_data + input.dim32(0));
    } else {
      auto& input = *inputs[0];
      real_shape.insert(
          real_shape.end(), input.sizes().begin(), input.sizes().end());
    }
    real_shape.insert(real_shape.end(), extra_shape.begin(), extra_shape.end());
    output->Resize(real_shape);
  } else {
    output->Resize(shape);
  }
}

template <class Type, class Context>
void given_tensor_fill_op_cpu_impl(
    at::ArrayRef<const Tensor*> inputs,
    Tensor* output,
    const std::vector<int64_t>& shape,
    const std::vector<int>& extra_shape,
    bool input_as_shape,
    const Tensor& values,
    BaseContext* context) {
  filler_init(inputs, output, shape, extra_shape, input_as_shape);

  // TODO T might not be the correct type to call, since float allows others.

  DCHECK_EQ(output->numel(), values.numel())
      << "output size: " << output->numel()
      << " given size: " << values.numel();
  auto* data = output->template mutable_data<Type>();
  const Type* values_data = values.template data<Type>();
  if (output->numel()) {
    context->CopySameDevice(output->numel(), values_data, data);
  }
}

void constant_fill_op_cpu_impl(
    at::ArrayRef<const Tensor*> inputs,
    Tensor* output,
    const std::vector<int64_t>& shape,
    const std::vector<int>& extra_shape,
    bool input_as_shape,
    int dtype,
    caffe2::ops::ConstantFill::Value value,
    BaseContext* context) {
  filler_init(inputs, output, shape, extra_shape, input_as_shape);

  if (output->numel()) {
    if (dtype == caffe2::TensorProto_DataType_FLOAT) {
      caffe2::math::Set<float, CPUContext>(
          output->numel(),
          value.as_float,
          output->template mutable_data<float>(),
          static_cast<CPUContext*>(context));
    } else if (dtype == caffe2::TensorProto_DataType_INT32) {
      caffe2::math::Set<int32_t, CPUContext>(
          output->numel(),
          value.as_int32,
          output->template mutable_data<int32_t>(),
          static_cast<CPUContext*>(context));
    } else if (dtype == caffe2::TensorProto_DataType_INT64) {
      caffe2::math::Set<int64_t, CPUContext>(
          output->numel(),
          value.as_int64,
          output->template mutable_data<int64_t>(),
          static_cast<CPUContext*>(context));
    } else if (dtype == caffe2::TensorProto_DataType_BOOL) {
      caffe2::math::Set<bool, CPUContext>(
          output->numel(),
          value.as_bool,
          output->template mutable_data<bool>(),
          static_cast<CPUContext*>(context));
    } else {
      throw std::logic_error(
          "Unimplemented data type for ConstantFill: " +
          c10::guts::to_string(dtype));
    }
  }
}

void uniform_fill_op_cpu_impl(
    at::ArrayRef<const Tensor*> inputs,
    Tensor* output,
    const std::vector<int64_t>& shape,
    const std::vector<int>& extra_shape,
    bool input_as_shape,
    float min,
    float max,
    BaseContext* context) {
  filler_init(inputs, output, shape, extra_shape, input_as_shape);

  if (inputs.size() == 3) {
    CAFFE_ENFORCE_EQ(1, inputs[1]->numel(), "min blob must be scalar");
    CAFFE_ENFORCE_EQ(1, inputs[2]->numel(), "max blob must be scalar");
    min = *inputs[1]->template data<float>();
    max = *inputs[2]->template data<float>();
    if (min > max) {
      auto shape = output->sizes().vec();
      shape[0] = 0;
      output->Resize(shape);
      output->template mutable_data<float>();
      return;
    }
  }
  caffe2::math::RandUniform<float, CPUContext>(
      output->numel(),
      min,
      max,
      output->template mutable_data<float>(),
      static_cast<CPUContext*>(context));
}
} // namespace
} // namespace caffe2

namespace c10 {
C10_REGISTER_KERNEL(caffe2::ops::ConstantFill)
    .kernel(&caffe2::constant_fill_op_cpu_impl)
    .dispatchKey(c10::DeviceTypeId::CPU);

C10_REGISTER_KERNEL(caffe2::ops::UniformFill)
    .kernel(&caffe2::uniform_fill_op_cpu_impl)
    .dispatchKey(c10::DeviceTypeId::CPU);

C10_REGISTER_KERNEL(caffe2::ops::GivenTensorFill<float>)
    .kernel(&caffe2::given_tensor_fill_op_cpu_impl<float, caffe2::CPUContext>)
    .dispatchKey(c10::DeviceTypeId::CPU);

C10_REGISTER_KERNEL(caffe2::ops::GivenTensorFill<int>)
    .kernel(&caffe2::given_tensor_fill_op_cpu_impl<int, caffe2::CPUContext>)
    .dispatchKey(c10::DeviceTypeId::CPU);

C10_REGISTER_KERNEL(caffe2::ops::GivenTensorFill<int64_t>)
    .kernel(&caffe2::given_tensor_fill_op_cpu_impl<int64_t, caffe2::CPUContext>)
    .dispatchKey(c10::DeviceTypeId::CPU);
} // namespace c10
