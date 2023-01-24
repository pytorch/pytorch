#include "caffe2/operators/quantized/int8_add_op.h"

#include <climits>

#include "caffe2/operators/utility_ops.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(Int8Add, int8::Int8AddOp<int8::Activation::NONE>);
REGISTER_CPU_OPERATOR(Int8AddRelu, int8::Int8AddOp<int8::Activation::RELU>);

REGISTER_CPU_OPERATOR(Int8Sum, int8::Int8AddOp<int8::Activation::NONE>);
REGISTER_CPU_OPERATOR(Int8SumRelu, int8::Int8AddOp<int8::Activation::RELU>);

OPERATOR_SCHEMA(Int8Add)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{0, 0}, {1, 0}})
    .Arg("Y_scale", "Output tensor quantization scale")
    .Arg("Y_zero_point", "Output tensor quantization offset")
    .SetDoc(R"DOC(
    Performs element-wise binary Add (with no broadcast support).
)DOC")
    .Input(
        0,
        "A",
        "First operand, should share the type with the second operand.")
    .Input(1, "B", "Second operand. It should be of the same size as A.")
    .Output(0, "C", "Result, has same dimensions and type as A");

OPERATOR_SCHEMA(Int8AddRelu)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{0, 0}, {1, 0}})
    .Arg("Y_scale", "Output tensor quantization scale")
    .Arg("Y_zero_point", "Output tensor quantization offset")
    .SetDoc(R"DOC(
    Performs element-wise binary Add (with no broadcast support). "
    "Output will go through rectified linear "
    "function, where y = max(0, x).
)DOC")
    .Input(
        0,
        "A",
        "First operand, should share the type with the second operand.")
    .Input(1, "B", "Second operand. It should be of the same size as A.")
    .Output(0, "C", "Result, has same dimensions and type as A");

/*
 * These ops are defined as alias of Int8Add/Int8AddRelu for compatibility
 * with current production models. In the future these ops will be changed
 * to an equivalent of Sum op, which does reduction of a single argument.
 * We deliberately omit schema for Int8Sum/Int8SumRelu so they can
 * temporary use either legacy or the new semantics depending on the engine.
 */
OPERATOR_SCHEMA(Int8Sum)
    .NumInputs(1, std::numeric_limits<int>::max())
    .NumOutputs(1)
    .AllowInplace({{0, 0}, {1, 0}})
    .CostInferenceFunction(CostInferenceForSum)
    .IdenticalTypeAndShapeOfInput(0)
    .Arg("Y_scale", "Output tensor quantization scale")
    .Arg("Y_zero_point", "Output tensor quantization offset");

OPERATOR_SCHEMA(Int8SumRelu)
    .NumInputs(1, std::numeric_limits<int>::max())
    .NumOutputs(1)
    .AllowInplace({{0, 0}, {1, 0}})
    .CostInferenceFunction(CostInferenceForSum)
    .IdenticalTypeAndShapeOfInput(0)
    .Arg("Y_scale", "Output tensor quantization scale")
    .Arg("Y_zero_point", "Output tensor quantization offset");

} // namespace caffe2
