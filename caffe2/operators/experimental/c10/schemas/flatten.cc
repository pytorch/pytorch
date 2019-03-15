#include "caffe2/operators/experimental/c10/schemas/flatten.h"
#include <ATen/core/dispatch/OpSchemaRegistration.h>
#include "caffe2/core/operator_c10wrapper.h"

using caffe2::CPUContext;

namespace caffe2 {
namespace ops {
// TODO Parse schema string instead of creating FunctionSchema manually
C10_DEFINE_OP_SCHEMA(Flatten, FunctionSchema(
    "_c10_experimental::Flatten",
    (std::vector<c10::Argument>{
      c10::Argument("input"),
      c10::Argument("output"),
      c10::Argument("axis", IntType::get())
    }), (std::vector<c10::Argument>{
    })
));
}
}

namespace caffe2 {
REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_CPU(
    ops::Flatten(),
    C10Flatten_DontUseThisOpYet)
}
