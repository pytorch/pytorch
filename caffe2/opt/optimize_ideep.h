#pragma once

#include "caffe2/core/common.h"
#include "caffe2/proto/caffe2.pb.h"

namespace caffe2 {
namespace opt {

caffe2::NetDef OptimizeForIdeep(caffe2::NetDef net);
}
} // namespace caffe2
