#include "caffe2/operators/pad_new_op.h"

#include <algorithm>

namespace caffe2 {

using std::max;
using std::min;

template <>
bool PadOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  const int n_dim = X.ndim();
  CAFFE_ENFORCE(
      pads_.size() == 2 * n_dim,
      "The length of pads must be equal to twice of number of dimensions of X");
  const vector<TIndex> x_dims = X.dims();
  vector<TIndex> y_dims(n_dim, 0);
  for (int i = 0; i < n_dim; ++i) {
    y_dims[i] = X.dim32(i) + pads_[i] + pads_[n_dim + i];
  }
  Y->Resize(y_dims);
  vector<int> x_size_from(n_dim, 0);
  vector<int> y_size_from(n_dim, 0);
  for (int i = 0; i < n_dim; ++i) {
    x_size_from[i] = X.size_from_dim(i + 1);
    y_size_from[i] = Y->size_from_dim(i + 1);
  }
  vector<int> pos(n_dim, 0);
  const float* Xdata = X.data<float>();
  float* Ydata = Y->mutable_data<float>();
  // The main loop
  switch (mode_) {
    case PadMode::CONSTANT:
      math::Set<float, CPUContext>(Y->size(), value_, Ydata, &context_);
      for (int i = 0; i < X.size(); i += x_dims[n_dim - 1]) {
        // convert flat index i of X into multi-dimensions index of X
        int temp = i;
        for (int j = 0; j < n_dim; ++j) {
          pos[j] = temp / x_size_from[j];
          temp %= x_size_from[j];
        }
        // convert multi-dimensions index of X into multi-dimensions index of Y
        for (int j = 0; j < n_dim; ++j) {
          pos[j] += pads_[j];
        }
        // convert multi-dimensions index of Y into flat index k of Y
        int k = 0;
        for (int j = 0; j < n_dim; ++j) {
          k += y_size_from[j] * pos[j];
        }
        // copy over
        auto p = Xdata + i;
        std::copy(p, p + x_dims[n_dim - 1], Ydata + k);
      }
      break;
    case PadMode::EDGE:
      break;
    case PadMode::REFLECT:
      break;
  }
  return true;
}

template <>
bool PadGradientOp<float, CPUContext>::RunOnDevice() {
  auto& dY = Input(0);
  auto* dX = Output(0);
  const int n_dim = dY.ndim();
  const vector<TIndex> dY_dims = dY.dims();
  vector<TIndex> dX_dims(n_dim, 0);
  for (int i = 0; i < n_dim; ++i) {
    dX_dims[i] = dY.dim32(i) - pads_[i] - pads_[n_dim + i];
  }
  dX->Resize(dX_dims);
  vector<int> dY_size_from(n_dim, 0);
  vector<int> dX_size_from(n_dim, 0);
  for (int i = 0; i < n_dim; ++i) {
    dY_size_from[i] = dY.size_from_dim(i + 1);
    dX_size_from[i] = dX->size_from_dim(i + 1);
  }
  vector<int> pos(n_dim, 0);
  const float* dYdata = dY.data<float>();
  float* dXdata = dX->mutable_data<float>();
  math::Set<float, CPUContext>(dX->size(), 0, dXdata, &context_);
  switch (mode_) {
    case PadMode::CONSTANT:
      for (int i = 0; i < dX->size(); i += dX_dims[n_dim - 1]) {
        int temp = i;
        for (int j = 0; j < n_dim; ++j) {
          pos[j] = temp / dX_size_from[j];
          temp %= dX_size_from[j];
        }
        for (int j = 0; j < n_dim; ++j) {
          pos[j] += pads_[j];
        }
        int k = 0;
        for (int j = 0; j < n_dim; ++j) {
          k += dY_size_from[j] * pos[j];
        }
        auto p = dYdata + k;
        std::copy(p, p + dX_dims[n_dim - 1], dXdata + i);
      }
      break;
    case PadMode::REFLECT:
      break;
    case PadMode::EDGE:
      break;
  }
  return true;
}

REGISTER_CPU_OPERATOR(Pad, PadOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(PadGradient, PadGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(Pad)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Pads a tensor according to the pads you specify.
  )DOC")
    .Input(0, "X", "Input Tensor.")
    .Output(0, "Y", "Tensor after padding.")
    .Arg("mode", "Three modes: constant(default), reflect, edge.")
    .Arg(
        "pads",
        "Required: list of ints indicating the number of padding elements to add at the beginning and end of each axis.")
    .Arg("value", "One float, indicates the value to be filled, default is 0.")
    .InheritOnnxSchema("Pad");

OPERATOR_SCHEMA(PadGradient).NumInputs(1).NumOutputs(1);

class GetPadGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "Pad", "", vector<string>{GO(0)}, vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(Pad, GetPadGradient);

} // namespace caffe2
