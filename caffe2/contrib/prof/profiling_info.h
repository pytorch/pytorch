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
  // Generates a fresh data structure to be initialized. Use Init() or Restore()
  // below to initialize the data structure.
  ProfilingInfo() {}
  // Generates 0-initialized stats for node and blob profiles from a given
  // NetDef. Use this to start profiling.
  bool Init(const NetDef& netDef);
  // Uses ProfDAGProtos for existing profile, and NetDef for graph definition,
  // and restores the state. Returns whether profile and net_def were
  // consistent. This is defined over the "indices" of operators and outputs:
  // the indices within the net_def should exist with the profile, and the names
  // should match. Errors are handled with best effort: matching state is
  // populated.
  bool Restore(const NetDef& netDef, const ProfDAGProtos& profile);
  // Appends ProfDAGProtos from the internal representation representing each
  // operator and blob in the graph as separate entities. Returns false if
  // NetDef is inconsistent with the original. It uses the old format when
  // oldFormat is set, which looks like: net__opidx__optype.
  bool GetOperatorAndDataStats(
      const NetDef& net_def,
      bool oldFormat,
      ProfDAGProtos* serialized) const;
  // Appends ProfDAGProtos using each op type as a 'stats' element. Returns
  // false if NetDef is inconsistent with the original. This function is only
  // used for oldFormat because the information is redundant. The user of the
  // library can iterate over the map and recreate if needed.
  bool GetOperatorTypeStats(const NetDef& net_def, ProfDAGProtos* serialized)
      const;

  // Accessors.
  const BlobMapType& getBlobMap() {
    return blobMap_;
  }
  const OperatorMapType& getOperatorMap() {
    return operatorMap_;
  }
  BlobMapType* getMutableBlobMap() {
    return &blobMap_;
  }
  OperatorMapType* getMutableOperatorMap() {
    return &operatorMap_;
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
  string getNameInOldFormat(int idx, const string& opType) const {
    return name_ + "___" + to_string(idx) + "___" + opType;
  }

  // Maps blob name to its node int he dataFlowGraph_.
  BlobMapType blobMap_;
  // Maps a node index to its NodeRef.
  OperatorMapType operatorMap_;
  // Net for which this profile was collected (NetDef.name).
  string name_;
};

} // namespace prof
} // namespace contrib
} // namespace caffe2
