#include "caffe2/core/dispatch/KernelRegistration.h"
#include "caffe2/operators/experimental/c10/schemas/concat.h"
#include "caffe2/utils/math.h"

using caffe2::BaseContext;
using caffe2::CPUContext;
using caffe2::Tensor;
using caffe2::TensorCPU;
using std::vector;

namespace caffe2 {
namespace {
template <class DataType, class Context>
void concat_op_cpu_impl(
    at::ArrayRef<const Tensor*> inputs,
    Tensor* output,
    Tensor* split,
    int axis,
    int add_axis,
    BaseContext* context) {
  split->Resize(vector<int64_t>(1, inputs.size()));
  int* axis_data = split->template mutable_data<int>();
  int adj_size = inputs[0]->dim() + (add_axis ? 1 : 0);
  int canonical_axis = caffe2::canonical_axis_index_(axis, adj_size);
  CAFFE_ENFORCE_LT(canonical_axis, adj_size, "Axis not in input ndim range.");
  for (int i = 1; i < inputs.size(); ++i) {
    CAFFE_ENFORCE(
        inputs[i]->dtype() == inputs[0]->dtype(),
        "All inputs must have the same type, expected: ",
        inputs[0]->dtype().name(),
        " but got: ",
        inputs[i]->dtype().name(),
        " for input: ",
        i);
  }

  int before = 1, after = 1;
  vector<int64_t> output_dims(inputs[0]->sizes().vec());
  for (int i = 0; i < inputs[0]->dim(); ++i) {
    if (i == canonical_axis && !add_axis) {
      continue;
    }
    int dim = inputs[0]->dim32(i);
    if (i < canonical_axis) {
      before *= dim;
    } else { // i > canonical_axis || i == canonical_axis && add_axis
      after *= dim;
    }
    // check the input dims are compatible.
    for (int j = 1; j < inputs.size(); ++j) {
      int dim_j = inputs[j]->dim32(i);
      CAFFE_ENFORCE(
          dim == dim_j,
          "Expect dimension = ",
          dim,
          " got ",
          dim_j,
          " at axis = ",
          i,
          " for input: ",
          j,
          ". The input tensors can only have different dimensions "
          "when arg 'add_axis' = 0 and along the axis = ",
          canonical_axis,
          " <",
          inputs[0]->sizes(),
          "> vs <",
          inputs[j]->sizes(),
          ">.");
    }
  }

  int output_channels = 0;
  for (int i = 0; i < inputs.size(); ++i) {
    axis_data[i] = add_axis ? 1 : inputs[i]->dim32(canonical_axis);
    output_channels += axis_data[i];
  }
  if (add_axis) {
    output_dims.insert(output_dims.begin() + canonical_axis, output_channels);
  } else {
    output_dims[canonical_axis] = output_channels;
  }
  output->Resize(output_dims);
  size_t output_offset = 0;
  for (int i = 0; i < inputs.size(); ++i) {
    auto& input = *inputs[i];
    auto axis_dim = add_axis ? 1 : input.dim32(canonical_axis);
    caffe2::math::CopyMatrix<Context>(
        input.itemsize(),
        before,
        axis_dim * after,
        input.raw_data(),
        axis_dim * after,
        static_cast<char*>(output->raw_mutable_data(inputs[0]->dtype())) +
            output_offset,
        output_channels * after,
        static_cast<Context*>(context),
        inputs[0]->dtype().copy());
    output_offset += axis_dim * after * input.itemsize();
  }
}
} // namespace
} // namespace caffe2

namespace c10 {
C10_REGISTER_KERNEL(caffe2::ops::Concat)
    .kernel(&caffe2::concat_op_cpu_impl<float, CPUContext>)
    .dispatchKey(c10::DeviceTypeId::CPU);
} // namespace c10
