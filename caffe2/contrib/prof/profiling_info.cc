// Implements ProfilingInfo class.
#include "caffe2/contrib/prof/profiling_info.h"

namespace caffe2 {
namespace contrib {
namespace prof {

bool ProfilingInfo::Init(const NetDef& netDef, const ProfDAGProtos& profile) {
  blobMap_.clear();
  operatorMap_.clear();
  bool success = true;
  int opIdx = 0;
  for (const auto& op : netDef.op()) {
    if (!addOperatorAnnotation(profile, opIdx, op.name())) {
      success = false;
    }
    int blobIdx = 0;
    for (const auto& output : op.output()) {
      if (!addDataAnnotation(profile, opIdx, blobIdx, output)) {
        success = false;
      }
      ++blobIdx;
    }
    ++opIdx;
  }
  return success;
}

bool ProfilingInfo::addOperatorAnnotation(
    const ProfDAGProtos& profile,
    int idx,
    const string& opName) {
  if (idx >= profile.stats_size()) {
    LOG(ERROR) << __func__ << ": indexing " << idx << " within "
               << profile.stats_size();
    return false;
  }
  const auto& op_node = profile.stats(idx);
  if (op_node.name() != opName) {
    LOG(ERROR) << "Unmatched name in ProfDAGProtos and NetDef. Respectively: "
               << op_node.name() << ", " << opName;
    return false;
  }
  return operatorMap_.emplace(idx, ProfilingOperatorAnnotation(op_node)).second;
}

bool ProfilingInfo::addDataAnnotation(
    const ProfDAGProtos& profile,
    int opIdx,
    int blobIdx,
    const string& output_name) {
  if (opIdx >= profile.stats_size() ||
      blobIdx >= profile.stats(opIdx).output_profile_size()) {
    LOG(ERROR) << __func__ << ": indexing " << opIdx << " within "
               << profile.stats_size() << ", and " << blobIdx << "within"
               << profile.stats(opIdx).output_profile_size();
    return false;
  }
  const auto& data_node = profile.stats(opIdx).output_profile(blobIdx);
  if (output_name != data_node.name()) {
    LOG(ERROR) << "Unmatched name in ProfDAGProtos and NetDef. Respectively: "
               << data_node.name() << ", " << output_name;
    return false;
  }
  return blobMap_.emplace(output_name, ProfilingDataAnnotation(data_node))
      .second;
}

} // namespace prof
} // namespace contrib
} // namespace caffe2
