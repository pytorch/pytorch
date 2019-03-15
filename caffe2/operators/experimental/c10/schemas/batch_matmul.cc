#include "caffe2/operators/experimental/c10/schemas/batch_matmul.h"
#include <ATen/core/dispatch/OpSchemaRegistration.h>
#include "caffe2/core/operator_c10wrapper.h"

using caffe2::CPUContext;

namespace caffe2 {
namespace ops {
// TODO Parse schema string instead of creating FunctionSchema manually
C10_DEFINE_OP_SCHEMA(BatchMatmul, FunctionSchema(
    "_c10_experimental::BatchMatmul",
    (std::vector<c10::Argument>{
      c10::Argument("A"),
      c10::Argument("B"),
      c10::Argument("output"),
      c10::Argument("trans_a", IntType::get()),
      c10::Argument("trans_b", IntType::get()),
      c10::Argument("broadcast", IntType::get())
    }), (std::vector<c10::Argument>{
    })
));
}
}

namespace caffe2 {

REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH(
    ops::BatchMatmul(),
    C10BatchMatMul_DontUseThisOpYet)
}
