#include "caffe2/core/operator_c10wrapper.h"

namespace caffe2 {

C10_DEFINE_REGISTRY(
    C10OperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);
}
