#ifndef CAFFE2_CORE_MEMONGER_H_
#define CAFFE2_CORE_MEMONGER_H_

#include "caffe2/core/common.h"
#include "caffe2/core/workspace.h"
#include "caffe2/proto/caffe2.pb.h"

namespace caffe2 {
namespace memonger {

NetDef optimize_inference_net(
    const NetDef& net,
    const std::set<string>& static_blobs);
}
}

#endif
