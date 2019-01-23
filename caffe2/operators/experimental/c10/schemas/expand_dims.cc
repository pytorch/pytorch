#include "caffe2/operators/experimental/c10/schemas/expand_dims.h"
#include <ATen/core/dispatch/OpSchemaRegistration.h>
#include "caffe2/core/operator_c10wrapper.h"

using caffe2::CPUContext;
using c10::intrusive_ptr;
using c10::ivalue::IntList;

C10_DEFINE_OP_SCHEMA(caffe2::ops::ExpandDims);

namespace {
struct DimsParameter final {
  using type = intrusive_ptr<IntList>;
  static intrusive_ptr<IntList> parse(const caffe2::ArgumentHelper& helper) {
    return IntList::create(helper.GetRepeatedArgument<int64_t>("dims"));
  }
};
} // namespace

namespace caffe2 {

CAFFE_KNOWN_TYPE(ops::ExpandDims::State);

REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_WITH_PARAMETERS(
    ops::ExpandDims,
    ops::ExpandDims::State,
    C10ExpandDims_DontUseThisOpYet,
    DimsParameter)
}
