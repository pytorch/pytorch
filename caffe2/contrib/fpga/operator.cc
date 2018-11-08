#include "operator.h"
#include "caffe2/core/operator.h"
#include "caffe2/proto/caffe2_pb.h"

namespace caffe2 {

C10_DEFINE_REGISTRY(
    OpenCLOperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);

} // namespace caffe2
