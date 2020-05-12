#include <caffe2/core/operator.h>
#include <caffe2/core/tensor_int8.h>
#include "caffe2/quantization/server/fbgemm_pack_blob.h"
#include "fake_nnpi_ops_utils.h"

#include <cmath>

namespace caffe2 {

class Int8FCFakeAcc32NNPIOp final : public Operator<CPUContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CPUContext);
  Int8FCFakeAcc32NNPIOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws),
        axis_(this->template GetSingleArgument<int32_t>("axis", 1)),
        axis_w_(this->template GetSingleArgument<int32_t>("axis_w", 1)),
        o_scale_(this->template GetSingleArgument<float>("Y_scale", 0)),
        o_zero_point_(
            this->template GetSingleArgument<int32_t>("Y_zero_point", 0)) {}
  ~Int8FCFakeAcc32NNPIOp() override {}
  bool RunOnDevice() override {
    // Qint8 input
    const auto& X = this->Input<int8::Int8TensorCPU>(0);
    // Qint8 weight
    const auto& W = this->Input<Int8FCDNNLowPPackedWeightBlob>(1);
    // Qint32 bias
    const auto& B = this->Input<int8::Int8TensorCPU>(2);
    const auto& X_tensor = X.t;
    const auto& W_tensor = W.original_tensor;
    const float x_scale = X.scale;
    const int32_t x_zero_point = X.zero_point;
    const float w_scale = W.qparams[0].scale;
    const int32_t w_zero_point = W.qparams[0].zero_point;
    const auto canonical_axis = X_tensor.canonical_axis_index(axis_);
    const auto M = X_tensor.size_to_dim(canonical_axis);
    const auto K = X_tensor.size_from_dim(canonical_axis);
    const auto canonical_axis_w = W_tensor.canonical_axis_index(axis_w_);
    const int N = W_tensor.size_to_dim(canonical_axis_w);
    const uint8_t* X_data = X_tensor.template data<uint8_t>();
    const int8_t* W_data = W_tensor.template data<int8_t>();
    const int32_t* B_data = B.t.template data<int32_t>();

    auto* Y = this->Output<int8::Int8TensorCPU>(0);
    Y->scale = o_scale_;
    Y->zero_point = o_zero_point_;
    ReinitializeTensor(&Y->t, {M, N}, at::dtype<uint8_t>().device(CPU));
    uint8_t* Y_data = Y->t.template mutable_data<uint8_t>();

    fake_nnpi::matmul_u8i8u8acc32_ref(
        M,
        N,
        K,
        K,
        K,
        N,
        X_data,
        x_zero_point,
        W_data,
        w_zero_point,
        B_data,
        Y_data,
        x_scale * w_scale / o_scale_,
        o_zero_point_,
        false /* fuse_relu */);
    return true;
  }

 private:
  const int32_t axis_;
  const int32_t axis_w_;
  const float o_scale_;
  const int32_t o_zero_point_;
};

REGISTER_CPU_OPERATOR(Int8FCFakeAcc32NNPI, Int8FCFakeAcc32NNPIOp);
OPERATOR_SCHEMA(Int8FCFakeAcc32NNPI).NumInputs(3).NumOutputs(1);
} // namespace caffe2
