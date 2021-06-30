#include "quant_decode_op.h"
// NOLINTNEXTLINE(modernize-deprecated-headers)
#include <stdint.h>
#include "caffe2/core/tensor.h"
#include <c10/util/typeid.h>

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(QuantDecode, QuantDecodeOp<QuantDecodeRunTy::RUN_ALWAYS>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_GRADIENT_OPERATOR(QuantDecodeGradient, QuantDecodeGradientOp);
#ifdef CAFFE2_USE_MPSCNN
REGISTER_CPU_OPERATOR(
    MPSCNNQuantDecode,
    QuantDecodeOp<QuantDecodeRunTy::RUN_ONCE>);
#endif

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(QuantDecode)
    .NumInputsOutputs([](int in, int out) { return in > 1 && out + 1 == in; })
    .SetDoc(R"DOC(
Decode inputs using codebook. This is a general LUT operator that returns
tensors with values from codebook (input 0) based on given indices in
codes (input 1 ~ n).


Example:


Input:
  codebook = [1.5, 2.5, 3.5]
  codes_0 = [0, 1, 1, 2]
  codes_1 = [2, 0, 0]


Output:
  decoded_0 = [1.5, 2.5, 2.5, 3.5]
  decoded_1 = [3.5, 1.5, 1.5]
)DOC")
    .Input(0, "codebook", "Codebook in 1d tensor (float)")
    .Input(1, "codes_0", "Encoded codes 0 (uint8/uint16/int32)")
    .Input(2, "codes_1", "Encoded codes 1 if existed (uint8/uint16/int32)")
    .Input(3, "codes_n", "Encoded codes n if existed (uint8/uint16/int32)")
    .Output(0, "decoded_0", "Decoded tensor for codes_0 (float)")
    .Output(1, "decoded_1", "Decoded tensor for codes_1 (float)")
    .Output(2, "decoded_n", "Decoded tensor for codes_n (float)");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
GRADIENT_OPERATOR_SCHEMA(QuantDecodeGradient)
    .NumInputs([](int in) { return in >= 3 && in % 2 == 1; })
    .NumOutputs(1);

class GetQuantDecodeGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    CAFFE_ENFORCE_EQ(Def().input_size(), Def().output_size() + 1);
    vector<string> gradient_op_inputs;
    for (int i = 0; i < Def().input_size(); i++) {
      // NOLINTNEXTLINE(performance-inefficient-vector-operation)
      gradient_op_inputs.push_back(I(i));
    }
    for (int i = 0; i < Def().output_size(); i++) {
      gradient_op_inputs.push_back(GO(i));
    }
    return SingleGradientDef(
        "QuantDecodeGradient", "", gradient_op_inputs, vector<string>{GI(0)});
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(QuantDecode, GetQuantDecodeGradient);

} // namespace caffe2
