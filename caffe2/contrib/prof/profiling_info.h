// Defines the class for storing profiling information for a neural net.
#pragma once

#include <unordered_map>

#include "caffe2/contrib/prof/profiling_annotations.h"

namespace caffe2 {
namespace contrib {
namespace prof {

// Holds the profiling data for the execution of a net.
class ProfilingInfo {
 public:
  using BlobMapType = std::unordered_map<std::string, ProfilingDataAnnotation>;
  using OperatorMapType = std::unordered_map<int, ProfilingOperatorAnnotation>;
  // Generates a fresh data structured to be initialized. Use Init() below to
  // restore a previous profile.
  ProfilingInfo() {}
  // Uses ProfDAGProtos for existing profile, and NetDef for graph definition,
  // and restores the state. Returns whether profile and net_def were
  // consistent. This is defined over the "indices" of operators and outputs:
  // the indices within the net_def should exist with the profile, and the names
  // should match. Errors are handled with best effort: matching state is
  // populated.
  bool Init(const NetDef& net_def, const ProfDAGProtos& profile);
  // Accessors.
  const BlobMapType& getBlobMap() {
    return blobMap_;
  }
  const OperatorMapType& getOperatorMap() {
    return operatorMap_;
  }

 private:
  bool addOperatorAnnotation(
      const ProfDAGProtos& profile,
      int idx,
      const string& opName);
  bool addDataAnnotation(
      const ProfDAGProtos& profile,
      int opIdx,
      int blobIdx,
      const string& output_name);

  // Maps blob name to its node int he dataFlowGraph_.
  BlobMapType blobMap_;
  // Maps a node index to its NodeRef.
  OperatorMapType operatorMap_;
};

} // namespace prof
} // namespace contrib
} // namespace caffe2
