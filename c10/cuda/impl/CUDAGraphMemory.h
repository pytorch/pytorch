#pragma once

#include <c10/cuda/CUDAGraphMemory.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/flat_hash_map.h>

#include <optional>
#include <vector>

namespace c10::cuda::CUDAGraphMemory {

struct AllocationContext {
  bool is_capturing{false};
  std::optional<CaptureId_t> tracked_capture_id;
  cudaStream_t block_reuse_stream;
};

// Tracks conditional capture relationships and the capture that allocated each
// block. Block lifecycle and CUDA-DAG policy are added in later changes.
class CaptureTracker {
 public:
  void captureBegin();
  void captureBegin(const CaptureRegistration& registration);
  void captureEnd();
  size_t captureEnd(CaptureId_t capture_id);
  bool hasActiveCaptures() const {
    return active_captures_ != 0;
  }
  AllocationContext allocationContext(cudaStream_t request_stream) const;
  void recordAllocation(const void* block, const AllocationContext& context);
  void recordFree(const void* block, cudaStream_t free_stream);

 private:
  struct CaptureTreeNode {
    MempoolId_t mempool_id;
    cudaStream_t primary_stream;
    cudaStream_t block_reuse_stream;
    std::optional<CaptureId_t> parent_capture_id;
    CaptureId_t root_capture_id;
    bool is_active;
    size_t invalid_capture_free_count{0};
  };

  // Walk from the allocation capture to the root. Allow only the same capture
  // or an ancestor capture to free the block.
  bool isFreeInAllocationCaptureOrAncestor(
      CaptureId_t allocation_capture_id,
      CaptureId_t free_capture_id) const;
  int eraseCaptureTree(CaptureId_t root_capture_id);

  int active_captures_{0};
  ska::flat_hash_map<CaptureId_t, CaptureTreeNode> capture_tree_;
  ska::flat_hash_map<const void*, CaptureId_t> block_allocation_captures_;
};

struct CaptureDAGInfo {
  cudaGraph_t graph{};
  CaptureId_t capture_id{0};
  const cudaGraphNode_t* terminals{nullptr};
  size_t num_terminals{0};
  cudaStreamCaptureStatus status{cudaStreamCaptureStatusNone};
};

class CaptureDAGQuery {
 public:
  virtual ~CaptureDAGQuery() = default;

  virtual CaptureDAGInfo captureInfo(cudaStream_t stream) const = 0;
  virtual std::vector<cudaGraphNode_t> dependencies(
      cudaGraphNode_t node) const = 0;
};

class CaptureDAG {
 public:
  class FreeMarkerState {
   public:
    FreeMarkerState() = default;

   private:
    bool valid_{true};
    CaptureId_t capture_id_{0};
    cudaGraph_t graph_{nullptr};
    ska::flat_hash_set<cudaGraphNode_t> markers_;

    friend class CaptureDAG;
  };

  class TraversalState {
   public:
    TraversalState() = default;

   private:
    bool initialized_{false};
    CaptureId_t capture_id_{0};
    cudaGraph_t graph_{nullptr};
    ska::flat_hash_set<cudaGraphNode_t> visited_;

    friend class CaptureDAG;
  };

  CaptureDAG();
  explicit CaptureDAG(const CaptureDAGQuery& query);
  CaptureDAG(CaptureDAGQuery&&) = delete;
  CaptureDAG(const CaptureDAGQuery&&) = delete;

  CaptureDAGInfo captureInfo(cudaStream_t stream) const;
  bool recordFreeMarkersForStream(cudaStream_t stream, FreeMarkerState& state)
      const;
  std::vector<cudaGraphNode_t> takeFreeMarkers(FreeMarkerState&& state) const;
  void updateVisited(const CaptureDAGInfo& info, TraversalState& state) const;
  bool areMarkersReachable(
      ArrayRef<cudaGraphNode_t> markers,
      const TraversalState& state) const;

 private:
  const CaptureDAGQuery& query_;
};

} // namespace c10::cuda::CUDAGraphMemory
