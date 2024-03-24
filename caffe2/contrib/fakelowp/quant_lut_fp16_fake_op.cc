#include "caffe2/contrib/fakelowp/quant_lut_fp16_fake_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(TanhQuantFakeFp16NNPI, TanhInt8QuantizeNNPIOp);

OPERATOR_SCHEMA(TanhQuantFakeFp16NNPI)
    .Arg("Y_scale", "Output tensor quantization scale")
    .Arg("Y_zero_point", "Output tensor quantization offset")
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Apply TanH and convert the result to Int8.
<details>
</details>
)DOC")
    .Input(0, "X", "Float Tensor X.")
    .Output(0, "Y", "Int8 Tensor Y.");

} // namespace caffe2
