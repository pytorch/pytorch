#include "integral_image_op.h"
#include "caffe2/utils/eigen_utils.h"

namespace caffe2 {

namespace {
template <typename T>
using EigenMatrixMapRowMajor = Eigen::Map<
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

template <typename T>
using ConstEigenMatrixMapRowMajor = Eigen::Map<
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
} // namespace

template <>
bool IntegralImageOp<float, CPUContext>::RunOnDevice() {
  const auto& X = Input(0);

  CAFFE_ENFORCE_EQ(X.dim(), 4, "Only supports 4D tensors for the moment");

  vector<int64_t> out_shape(X.sizes().vec());
  out_shape[2] += 1; // H + 1 output size
  out_shape[3] += 1; // W + 1 output size
  auto* Y = Output(0, out_shape, at::dtype<float>());
  const int ind = X.dim32(0);
  const int chans = X.dim32(1);
  const int rows_in = X.dim32(2);
  const int cols_in = X.dim32(3);
  const int rows_out = Y->dim32(2);
  const int cols_out = Y->dim32(3);

  const float* input_data = X.template data<float>();
  float* output_data = Y->template mutable_data<float>();

  const int row_out_pass_size = ind * chans * rows_out;
  const int row_in_pass_size = ind * chans * rows_in;
  EigenMatrixMapRowMajor<float> Y_arr(output_data, row_out_pass_size, cols_out);
  ConstEigenMatrixMapRowMajor<float> X_arr(
      input_data, row_in_pass_size, cols_in);

  // Row Pass
  for (int i = 0; i < row_out_pass_size; i++) {
    int row = i % rows_out;
    int diff = i / rows_out + 1;
    Y_arr(i, 0) = 0.;
    if (row == 0) {
      for (int j = 1; j < cols_out; ++j) {
        Y_arr(i, j) = 0.;
      }
    } else {
      for (int j = 1; j < cols_out; ++j) {
        Y_arr(i, j) = Y_arr(i, j - 1) + X_arr(i - diff, j - 1);
      }
    }
  }

  // Col Pass
  const int col_out_pass_size = X.dim32(0) * chans * cols_out;
  for (int i = 0; i < col_out_pass_size; i++) {
    int col = i % cols_out;
    int row = i / cols_out;
    for (int j = row * rows_out + 1; j < (row + 1) * rows_out; ++j) {
      Y_arr(j, col) += Y_arr(j - 1, col);
    }
  }
  return true;
}

template <>
bool IntegralImageGradientOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0); // Original input to "forward" op
  auto& dY = Input(1); // Gradient of net w.r.t. output of "forward" op
  // (aka "gradOutput")
  auto* dX = Output(
      0, X.sizes(), at::dtype<float>()); // Gradient of net w.r.t. input to
                                         // "forward" op (aka "gradInput")

  const int ind = X.dim32(0);
  const int chans = X.dim32(1);
  const int rows_in = dY.dim32(2);
  const int cols_in = dY.dim32(3);
  const int rows_out = dX->dim32(2);
  const int cols_out = dX->dim32(3);

  const float* input_data = dY.template data<float>();
  float* output_data = dX->template mutable_data<float>();

  const int row_out_pass_size = ind * chans * rows_out;
  const int row_in_pass_size = ind * chans * rows_in;
  EigenMatrixMapRowMajor<float> dX_arr(
      output_data, row_out_pass_size, cols_out);
  ConstEigenMatrixMapRowMajor<float> dY_arr(
      input_data, row_in_pass_size, cols_in);
  Eigen::MatrixXf tmp(row_in_pass_size, cols_out);

  // Row Pass dY(N, C, H+1, W+1) => tmp(N, C, H+1, W)
  for (int i = 0; i < row_in_pass_size; i++) {
    tmp(i, 0) = dY_arr(i, 0);
    for (int j = 1; j < cols_out; ++j) {
      tmp(i, j) = tmp(i, j - 1) + dY_arr(i, j);
    }
  }

  // Col Pass tmp(N, C, H+1, W)=>dX(N, C, H, W)
  const int col_out_pass_size = X.dim32(0) * chans * cols_out;
  for (int i = 0; i < col_out_pass_size; i++) {
    int col = i % cols_out;
    int row_out_start = (i / cols_out) * rows_out;
    int row_in_start = (i / cols_out) * rows_in;
    dX_arr(row_out_start, col) = tmp(row_in_start, col);
    for (int j = 1; j < rows_out; ++j) {
      dX_arr(row_out_start + j, col) =
          dX_arr(row_out_start + j - 1, col) + tmp(row_in_start + j, col);
    }
  }
  return true;
}

REGISTER_CPU_OPERATOR(IntegralImage, IntegralImageOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    IntegralImageGradient,
    IntegralImageGradientOp<float, CPUContext>);

// Input: X; Output: Y
OPERATOR_SCHEMA(IntegralImage)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Computes an integral image, which contains the sum of pixel values within
an image vertically and horizontally. This integral image can then be used
with other detection and tracking techniques.
)DOC")
    .Input(0, "X", "Images tensor of the form (N, C, H, W)")
    .Output(0, "Y", "Integrated image of the form (N, C, H+1, W+1)");

// Input: X, dY (aka "gradOutput"); Output: dX (aka "gradInput")
OPERATOR_SCHEMA(IntegralImageGradient).NumInputs(2).NumOutputs(1);

class GetIntegralImageGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "IntegralImageGradient",
        "",
        vector<string>{I(0), GO(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(IntegralImage, GetIntegralImageGradient);

} // namespace caffe2
