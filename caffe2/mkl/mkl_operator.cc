#include "caffe2/core/operator.h"
#include "caffe2/proto/caffe2.pb.h"

CAFFE2_DEFINE_bool(
    caffe2_mkl_memonger_in_use,
    false,
    "Turn on if memonger is used to force reallocate intermediate "
    "and output buffers within each op");

namespace caffe2 {

CAFFE_DEFINE_REGISTRY(
    MKLOperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);
CAFFE_REGISTER_DEVICE_TYPE(DeviceType::MKLDNN, MKLOperatorRegistry);

} // namespace caffe2
