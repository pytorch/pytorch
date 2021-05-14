#pragma once

#include <string>
#include <unordered_set>
#include <vector>

#include <caffe2/core/net.h>
#include <caffe2/core/workspace.h>
#include <caffe2/opt/shape_info.h>
#include <caffe2/proto/caffe2_pb.h>

C10_DECLARE_string(onnxifi_blacklist);
C10_DECLARE_string(onnxifi_blacklist_ops);

namespace caffe2 {
namespace glow {
/// Onnxifi transformation on the net and workspace.  We also
/// needed the input data/shape to populate the shape. In addition, we take a \p
/// blocklist to control and mask what ops we want to consider in onnxifi
/// process. We can also set whether to use ONNX proto or C2 proto through
/// ONNXIFI interface.
void onnxifi(
    NetDef* net,
    Workspace* ws,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::vector<std::string>& weight_names,
    const std::unordered_set<int>& blocklist,
    const ShapeInfoMap& shape_hints_max_bs,
    bool use_onnx,
    size_t max_batch_size = 0,
    size_t max_seq_size = 0,
    bool load_model_by_blob = false,
    bool predictor_net_ssa_rewritten = false,
    const std::unordered_map<int, ShapeInfoMap> &shape_hints_per_bs = {},
    const c10::optional<std::string> &blacklist_ops = c10::nullopt,
    const c10::optional<size_t> &min_ops = c10::nullopt,
    const std::unordered_set<std::string> &blocklist_blobs = {});

std::unordered_set<int> ParseNetPositionList(const std::string& str);
std::unordered_set<std::string> ParseBlockListOps(const std::string& str);

} // namespace glow
} // namespace caffe2
