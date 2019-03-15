#include "caffe2/operators/experimental/c10/schemas/concat.h"
#include <ATen/core/dispatch/OpSchemaRegistration.h>
#include "caffe2/core/operator_c10wrapper.h"

using caffe2::CPUContext;

namespace caffe2 {
namespace ops {
// TODO Parse schema string instead of creating FunctionSchema manually
C10_DEFINE_OP_SCHEMA(Concat, FunctionSchema(
    "_c10_experimental::Concat",
    (std::vector<c10::Argument>{
      c10::Argument("inputs", ListType::ofTensors()),
      c10::Argument("output"),
      c10::Argument("split_info", FloatType::get()),
      c10::Argument("add", IntType::get()),
      c10::Argument("add_axis", IntType::get())
    }), (std::vector<c10::Argument>{
    })
));
}
}

namespace caffe2 {
REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH(
    ops::Concat(),
    C10Concat_DontUseThisOpYet)
}
