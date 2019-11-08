#pragma once

#include <string>
#include <unordered_set>
#include <vector>

#include <caffe2/core/net.h>
#include <caffe2/core/workspace.h>
#include <caffe2/proto/caffe2_pb.h>

namespace caffe2 {
namespace opt {

// Transform normal fp32 operators to fakefp16 operators.
void fakeFp16Transform(NetDef* net);

} // namespace opt
} // namespace caffe2
