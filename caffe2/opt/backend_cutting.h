#pragma once


#include "caffe2/core/common.h"
#include "caffe2/proto/caffe2_pb.h"

#include <functional>

namespace caffe2 {
namespace opt {

CAFFE2_API caffe2::NetDef OptimizeForBackend(
    caffe2::NetDef& net,
    std::function<bool(const caffe2::OperatorDef&)> supports,
    std::function<caffe2::NetDef(const caffe2::NetDef&)> transform_func,
    bool debug = false);
}
} // namespace caffe2
