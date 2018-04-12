#ifndef CAFFE2_OPT_MOBILE_H_
#define CAFFE2_OPT_MOBILE_H_

#include "caffe2/core/common.h"
#include "caffe2/proto/caffe2.pb.h"

namespace caffe2 {
namespace opt {

caffe2::NetDef addNNPACK(caffe2::NetDef net, bool low_memory = false);
caffe2::NetDef fuseNNPACKConvRelu(caffe2::NetDef net);

} // namespace opt
} // namespace caffe2

#endif // CAFFE2_OPT_MOBILE_H_
