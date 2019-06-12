#include "caffe2/operators/experimental/c10/schemas/cast.h"
#include "caffe2/core/dispatch/OpSchemaRegistration.h"
#include "caffe2/core/operator_c10wrapper.h"
#include "caffe2/utils/cast.h"

using caffe2::CPUContext;
using caffe2::Tensor;

C10_DEFINE_OP_SCHEMA(caffe2::ops::Cast);

namespace {

struct ToParameter final {
  using type = caffe2::TensorProto_DataType;
  static caffe2::TensorProto_DataType parse(
      const caffe2::ArgumentHelper& helper) {
    return caffe2::cast::GetCastDataType(helper, "to");
  }
};
} // namespace

namespace caffe2 {
REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_WITH_PARAMETERS(
    ops::Cast,
    void,
    C10Cast_DontUseThisOpYet,
    ToParameter)
}
