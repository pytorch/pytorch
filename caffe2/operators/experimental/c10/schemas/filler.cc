#include "caffe2/operators/experimental/c10/schemas/filler.h"
#include <ATen/core/dispatch/OpSchemaRegistration.h>
#include "caffe2/core/operator_c10wrapper.h"
#include "caffe2/utils/cast.h"

using caffe2::CPUContext;
using c10::C10Tensor;
using c10::ivalue::IntList;
using c10::intrusive_ptr;

namespace caffe2 {
namespace ops {
// TODO Parse schema strings instead of creating FunctionSchema manually
C10_DEFINE_OP_SCHEMA(
    ConstantFill,
    FunctionSchema(
        "_c10_experimental::ConstantFill",
        "",
        (std::vector<c10::Argument>{
            c10::Argument("inputs", ListType::ofTensors()),
            c10::Argument("output"),
            c10::Argument("shape", ListType::ofInts()),
            c10::Argument("extra_shape", ListType::ofInts()),
            c10::Argument("input_as_shape", BoolType::get()),
            c10::Argument("dtype", IntType::get()),
            c10::Argument("value", NumberType::get())}),
        (std::vector<c10::Argument>{})));
C10_DEFINE_OP_SCHEMA(
    UniformFill,
    FunctionSchema(
        "_c10_experimental::ConstantFill",
        "",
        (std::vector<c10::Argument>{
            c10::Argument("inputs", ListType::ofTensors()),
            c10::Argument("output"),
            c10::Argument("shape", ListType::ofInts()),
            c10::Argument("extra_shape", ListType::ofInts()),
            c10::Argument("input_as_shape", BoolType::get()),
            c10::Argument("min", FloatType::get()),
            c10::Argument("max", FloatType::get())}),
        (std::vector<c10::Argument>{})));
C10_DEFINE_OP_SCHEMA(
    GivenTensorFill,
    FunctionSchema(
        "_c10_experimental::ConstantFill",
        "",
        (std::vector<c10::Argument>{
            c10::Argument("inputs", ListType::ofTensors()),
            c10::Argument("output"),
            c10::Argument("shape", ListType::ofInts()),
            c10::Argument("extra_shape", ListType::ofInts()),
            c10::Argument("input_as_shape", BoolType::get()),
            c10::Argument("values"),
        }),
        (std::vector<c10::Argument>{})));
C10_DEFINE_OP_SCHEMA(
    GivenTensorIntFill,
    FunctionSchema(
        "_c10_experimental::ConstantFill",
        "",
        (std::vector<c10::Argument>{
            c10::Argument("inputs", ListType::ofTensors()),
            c10::Argument("output"),
            c10::Argument("shape", ListType::ofInts()),
            c10::Argument("extra_shape", ListType::ofInts()),
            c10::Argument("input_as_shape", BoolType::get()),
            c10::Argument("values"),
        }),
        (std::vector<c10::Argument>{})));
C10_DEFINE_OP_SCHEMA(
    GivenTensorInt64Fill,
    FunctionSchema(
        "_c10_experimental::ConstantFill",
        "",
        (std::vector<c10::Argument>{
            c10::Argument("inputs", ListType::ofTensors()),
            c10::Argument("output"),
            c10::Argument("shape", ListType::ofInts()),
            c10::Argument("extra_shape", ListType::ofInts()),
            c10::Argument("input_as_shape", BoolType::get()),
            c10::Argument("values"),
        }),
        (std::vector<c10::Argument>{})));
}
}

namespace caffe2 {
REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_CPU(
    ops::ConstantFill(),
    C10ConstantFill_DontUseThisOpYet)
REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_CPU(
    ops::UniformFill(),
    C10UniformFill_DontUseThisOpYet)

REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_CPU(
    ops::GivenTensorFill(),
    C10GivenTensorFill_DontUseThisOpYet)
REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_CPU(
    ops::GivenTensorIntFill(),
    C10GivenTensorIntFill_DontUseThisOpYet)
REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_CPU(
    ops::GivenTensorInt64Fill(),
    C10GivenTensorInt64Fill_DontUseThisOpYet)
} // namespace caffe2
