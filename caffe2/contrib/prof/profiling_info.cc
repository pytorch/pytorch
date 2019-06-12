// Implements ProfilingInfo class.
#include "caffe2/contrib/prof/profiling_info.h"

namespace caffe2 {
namespace contrib {
namespace prof {

bool ProfilingInfo::Init(const NetDef& netDef) {
  bool success = true;
  int opIdx = 0;
  name_ = netDef.name();
  blobMap_.clear();
  operatorMap_.clear();
  for (const auto& op : netDef.op()) {
    bool inserted =
        operatorMap_.emplace(opIdx, ProfilingOperatorAnnotation()).second;
    if (!inserted) {
      success = false;
    }
    for (const auto& output : op.output()) {
      inserted = blobMap_.emplace(output, ProfilingDataAnnotation()).second;
      if (!inserted) {
        success = false;
      }
    }
    ++opIdx;
  }
  return success;
}

bool ProfilingInfo::Restore(
    const NetDef& netDef,
    const ProfDAGProtos& profile) {
  if (netDef.name() != profile.net_name()) {
    // If profile is not in the old format then it should contain the same name.
    return false;
  }
  name_ = netDef.name();
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

bool ProfilingInfo::GetOperatorAndDataStats(
    const NetDef& netDef,
    bool oldFormat,
    ProfDAGProtos* serialized) const {
  if (!oldFormat) {
    serialized->set_net_name(name_);
  }
  bool success = true;
  int opIdx = 0;
  for (const auto& op : netDef.op()) {
    auto opIt = operatorMap_.find(opIdx);
    if (opIt == operatorMap_.end()) {
      success = false;
      continue;
    }
    auto* stats = serialized->add_stats();
    // Set required fields for compatibility for both formats.
    stats->set_mean(opIt->second.getExecutionTimeMs().getMean());
    stats->set_stddev(opIt->second.getExecutionTimeMs().getStddev());
    if (oldFormat) {
      stats->set_name(getNameInOldFormat(opIdx, op.type()));
    } else {
      stats->set_name(op.name());
      *stats->mutable_execution_time() =
          opIt->second.getExecutionTimeMs().ToProto();
      for (const auto& output : op.output()) {
        auto blobIt = blobMap_.find(output);
        if (blobIt == blobMap_.end()) {
          success = false;
          continue;
        }
        auto* output_profile = stats->add_output_profile();
        output_profile->set_name(output);
        *output_profile->mutable_bytes_used() =
            blobIt->second.getUsedBytes().ToProto();
      }
    }
    ++opIdx;
  }
  return success;
}

bool ProfilingInfo::GetOperatorTypeStats(
    const NetDef& netDef,
    ProfDAGProtos* serialized) const {
  bool success = true;
  std::unordered_map<string /* type */, TwoNumberStats> typeStats;
  int opIdx = 0;
  for (const auto& op : netDef.op()) {
    auto opIt = operatorMap_.find(opIdx);
    if (opIt == operatorMap_.end()) {
      success = false;
      continue;
    }
    auto stats_it =
        typeStats.emplace(op.type(), opIt->second.getExecutionTimeMs());
    if (!stats_it.second) { // existing type
      stats_it.first->second.Merge(opIt->second.getExecutionTimeMs());
    }
    ++opIdx;
  }

  for (const auto& type_and_stat : typeStats) {
    auto* stat = serialized->add_stats();
    stat->set_name(type_and_stat.first);
    stat->set_mean(type_and_stat.second.getMean());
    stat->set_stddev(type_and_stat.second.getStddev());
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
