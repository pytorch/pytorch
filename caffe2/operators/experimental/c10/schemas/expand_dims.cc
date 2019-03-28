#include "caffe2/operators/experimental/c10/schemas/expand_dims.h"
#include <ATen/core/dispatch/OpSchemaRegistration.h>
#include "caffe2/core/operator_c10wrapper.h"

using caffe2::CPUContext;
using c10::intrusive_ptr;
using c10::ivalue::IntList;

namespace caffe2 {
namespace ops {
// TODO Parse schema string instead of creating FunctionSchema manually
C10_DEFINE_OP_SCHEMA(
    ExpandDims,
    FunctionSchema(
        "_c10_experimental::ExpandDims",
        "",
        (std::vector<c10::Argument>{c10::Argument("input"),
                                    c10::Argument("output"),
                                    c10::Argument("dims", ListType::ofInts())}),
        (std::vector<c10::Argument>{})));
}
}

namespace caffe2 {
REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_CPU(
    "_c10_experimental::ExpandDims",
    C10ExpandDims_DontUseThisOpYet)
}
