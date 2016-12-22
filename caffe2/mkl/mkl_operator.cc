#include "caffe2/core/operator.h"
#include "caffe2/proto/caffe2.pb.h"

namespace caffe2 {

CAFFE_DEFINE_REGISTRY(
    MKLOperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);
CAFFE_REGISTER_DEVICE_TYPE(DeviceType::MKLDNN, MKLOperatorRegistry);

} // namespace caffe2
