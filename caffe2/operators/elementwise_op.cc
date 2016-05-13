#include "caffe2/operators/elementwise_op.h"

namespace caffe2 {
namespace {

REGISTER_CPU_OPERATOR(Add, AddOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(Sub, SubOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(Mul, MulOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(Div, DivOp<float, CPUContext>);

OPERATOR_SCHEMA(Add).NumInputs(2).NumOutputs(1).AllowInplace({{0, 0}, {1, 0}});
OPERATOR_SCHEMA(Sub).NumInputs(2).NumOutputs(1).AllowInplace({{0, 0}, {1, 0}});
OPERATOR_SCHEMA(Mul).NumInputs(2).NumOutputs(1).AllowInplace({{0, 0}, {1, 0}});
OPERATOR_SCHEMA(Div).NumInputs(2).NumOutputs(1).AllowInplace({{0, 0}, {1, 0}});

GRADIENT_NOT_IMPLEMENTED_YET(Add);
GRADIENT_NOT_IMPLEMENTED_YET(Sub);
GRADIENT_NOT_IMPLEMENTED_YET(Mul);
GRADIENT_NOT_IMPLEMENTED_YET(Div);

}  // namespace
}  // namespace caffe2
