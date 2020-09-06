#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "caffe2/core/net.h"
#include "caffe2/core/workspace.h"
#include "caffe2/opt/shape_info.h"
#include "caffe2/proto/caffe2_pb.h"

namespace caffe2 {
namespace opt {

// Add Tile ops for some input tensors
void inBatchBroadcast(
    NetDef* net,
    const std::unordered_set<std::string>& to_broadcast_blobs,
    int32_t batch_size,
    ShapeInfoMap& shape_hints);

} // namespace opt
} // namespace caffe2
