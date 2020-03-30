#include <ATen/core/op_registration/op_registration.h>
#include "caffe2/core/export_c10_op_to_caffe2.h"
#include "caffe2/core/tensor.h"
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
    c10::List<at::Tensor> inputs,
    const at::Tensor& output_,
    const at::Tensor& split_,
    int64_t axis,
    int64_t add_axis) {
  Tensor output(output_);
  Tensor split(split_);
  CPUContext context;

  split.Resize(vector<int64_t>(1, inputs.size()));
  int* axis_data = split.template mutable_data<int>();
  int adj_size = Tensor(inputs[0]).dim() + (add_axis ? 1 : 0);
  int canonical_axis = caffe2::canonical_axis_index_(axis, adj_size);
  CAFFE_ENFORCE_LT(canonical_axis, adj_size, "Axis not in input ndim range.");
  for (size_t i = 1; i < inputs.size(); ++i) {
    CAFFE_ENFORCE(
        Tensor(inputs[i]).dtype() == Tensor(inputs[0]).dtype(),
        "All inputs must have the same type, expected: ",
        Tensor(inputs[0]).dtype().name(),
        " but got: ",
        Tensor(inputs[i]).dtype().name(),
        " for input: ",
        i);
  }

  int before = 1, after = 1;
  vector<int64_t> output_dims(Tensor(inputs[0]).sizes().vec());
  for (int i = 0; i < Tensor(inputs[0]).dim(); ++i) {
    if (i == canonical_axis && !add_axis) {
      continue;
    }
    int dim = Tensor(inputs[0]).dim32(i);
    if (i < canonical_axis) {
      before *= dim;
    } else { // i > canonical_axis || i == canonical_axis && add_axis
      after *= dim;
    }
    // check the input dims are compatible.
    for (size_t j = 1; j < inputs.size(); ++j) {
      int dim_j = Tensor(inputs[j]).dim32(i);
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
          Tensor(inputs[0]).sizes(),
          "> vs <",
          Tensor(inputs[j]).sizes(),
          ">.");
    }
  }

  int output_channels = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    axis_data[i] = add_axis ? 1 : Tensor(inputs[i]).dim32(canonical_axis);
    output_channels += axis_data[i];
  }
  if (add_axis) {
    output_dims.insert(output_dims.begin() + canonical_axis, output_channels);
  } else {
    output_dims[canonical_axis] = output_channels;
  }
  output.Resize(output_dims);
  size_t output_offset = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    Tensor input(inputs[i]);
    auto axis_dim = add_axis ? 1 : input.dim32(canonical_axis);
    caffe2::math::CopyMatrix<Context>(
        input.itemsize(),
        before,
        axis_dim * after,
        input.raw_data(),
        axis_dim * after,
        static_cast<char*>(output.raw_mutable_data(Tensor(inputs[0]).dtype())) +
            output_offset,
        output_channels * after,
        static_cast<Context*>(&context),
        Tensor(inputs[0]).dtype().copy());
    output_offset += axis_dim * after * input.itemsize();
  }
}

static auto registry = c10::RegisterOperators().op(
    "_c10_experimental::Concat",
    c10::RegisterOperators::options()
      .kernel<
        decltype(concat_op_cpu_impl<float, CPUContext>),
        &concat_op_cpu_impl<float, CPUContext>>(DispatchKey::CPUTensorId));

} // namespace

C10_EXPORT_C10_OP_TO_CAFFE2_CPU(
    "_c10_experimental::Concat",
    C10Concat_DontUseThisOpYet)

} // namespace caffe2
