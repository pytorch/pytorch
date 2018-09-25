#include "operator.h"

C10_DEFINE_REGISTRY(GLOperatorRegistry, caffe2::OperatorBase, const caffe2::OperatorDef &,
                      caffe2::Workspace *);

namespace caffe2 {

CAFFE_REGISTER_DEVICE_TYPE(DeviceType::OPENGL, GLOperatorRegistry);

} // namespace caffe2
