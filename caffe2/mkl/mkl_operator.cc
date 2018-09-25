#include "caffe2/core/operator.h"
#include "caffe2/proto/caffe2_pb.h"

CAFFE2_DEFINE_bool(
    caffe2_mkl_memonger_in_use,
    false,
    "Turn on if memonger is used to force reallocate intermediate "
    "and output buffers within each op");

C10_DEFINE_REGISTRY(
    MKLOperatorRegistry,
    caffe2::OperatorBase,
    const caffe2::OperatorDef&,
    caffe2::Workspace*);

namespace caffe2 {

CAFFE_REGISTER_DEVICE_TYPE(DeviceType::MKLDNN, MKLOperatorRegistry);

} // namespace caffe2
