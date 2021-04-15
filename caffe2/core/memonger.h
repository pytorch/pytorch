#ifndef CAFFE2_CORE_MEMONGER_H_
#define CAFFE2_CORE_MEMONGER_H_

#include <unordered_set>

#include "caffe2/core/common.h"
#include "caffe2/core/workspace.h"
#include "caffe2/proto/caffe2_pb.h"

namespace caffe2 {

// op schema check
TORCH_API void run_schema_check(const NetDef& net);

namespace memonger {

TORCH_API NetDef optimize_inference_net(
    const NetDef& net,
    const std::set<string>& static_blobs);

TORCH_API NetDef compute_blob_recycling_for_dag(
    const NetDef& net,
    const std::vector<string>& heads,
    const std::vector<int>& op_indices,
    const std::unordered_set<string>& shareable_blob_names,
    const string& namescope,
    const std::unordered_set<string>& dont_share_blob_names,
    const std::unordered_map<string, vector<int>>& blob_shapes);

} // memonger
} // caffe2

#endif
