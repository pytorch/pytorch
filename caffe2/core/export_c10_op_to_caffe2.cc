#include "caffe2/core/export_c10_op_to_caffe2.h"

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_DEFINE_REGISTRY(
    C10OperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);
}
