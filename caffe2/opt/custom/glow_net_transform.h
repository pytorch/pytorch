#pragma once

#include <string>
#include <unordered_set>
#include <vector>

#include <caffe2/core/net.h>
#include <caffe2/core/workspace.h>
#include <caffe2/proto/caffe2_pb.h>

C10_DECLARE_string(onnxifi_blacklist);
C10_DECLARE_string(onnxifi_blacklist_ops);

namespace caffe2 {
namespace glow {

// Onnxifi transformation on the net and workspace.  We also
// needed the input data/shape to populate the shape. In addition, we take a \p
// blacklist to control and mask what ops we want to consider in onnxifi
// process. We can also set whether to use ONNX proto or C2 proto through
// ONNXIFI interface.
void onnxifi(
    NetDef* net,
    Workspace* ws,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::vector<std::string>& weight_names,
    const std::unordered_set<int>& blacklist,
    const std::unordered_map<std::string, TensorShape>& shape_hints,
    bool use_onnx,
    size_t max_batch_size = 0,
    size_t max_seq_size = 0);

std::unordered_set<int> ParseNetPositionList(const std::string& str);
std::unordered_set<std::string> ParseBlackListOps(const std::string& str);

} // namespace glow
} // namespace caffe2
