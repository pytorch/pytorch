#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include <caffe2/core/net.h>
#include <caffe2/core/workspace.h>
#include <caffe2/proto/caffe2_pb.h>

namespace caffe2 {
namespace opt {

// Mapping from fp32 ops to fakefp16 ops
CAFFE2_API std::unordered_map<std::string, std::string> getFakeFp16OpMapping(
    bool use_fp16_acc = false,
    bool use_nnpi = false);

// Transform normal fp32 operators to fakefp16 operators.
void fakeFp16Transform(NetDef* net);

} // namespace opt
} // namespace caffe2
