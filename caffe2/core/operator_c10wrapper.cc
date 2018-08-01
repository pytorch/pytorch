#include "caffe2/core/operator_c10wrapper.h"

namespace caffe2 {

CAFFE_DEFINE_REGISTRY(
    C10OperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);
}
