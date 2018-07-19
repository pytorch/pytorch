#include "operator.h"

namespace caffe2 {

CAFFE_DEFINE_REGISTRY(GLOperatorRegistry, OperatorBase, const OperatorDef &,
                      Workspace *);
CAFFE_REGISTER_DEVICE_TYPE(DeviceType::OPENGL, GLOperatorRegistry);

} // namespace caffe2
