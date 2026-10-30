#pragma once

#include <c10/cuda/CUDAGraphMemory.h>
#include <c10/util/flat_hash_map.h>

#include <optional>

namespace c10::cuda::CUDAGraphMemory {

struct AllocationContext {
  bool is_capturing{false};
  std::optional<CaptureId_t> tracked_capture_id;
};

// Tracks active conditional captures and the capture responsible for each
// allocation. Surviving allocations move to the parent when a child ends.
class CaptureTracker {
 public:
  void captureBegin(const CaptureRegistration& registration);
  size_t captureEnd(CaptureId_t capture_id);
  bool hasActiveCaptures() const {
    return active_captures_ != 0;
  }
  AllocationContext allocationContext(cudaStream_t request_stream) const;
  cudaStream_t reuseStream(cudaStream_t stream) const;
  void recordAllocation(const void* block, const AllocationContext& context);
  void recordFree(const void* block, cudaStream_t free_stream);

 private:
  struct CaptureTreeNode {
    std::optional<CaptureId_t> parent_capture_id;
    CaptureId_t root_capture_id{};
    // Used only to remove the alias when this capture ends.
    cudaStream_t primary_stream{};
    size_t invalid_capture_free_count{0};
  };

  // Within one capture tree, allow only the allocation capture or an ancestor
  // capture to free the block. Unrelated capture trees are independent.
  bool isFreeInAllocationCaptureOrAncestor(
      CaptureId_t allocation_capture_id,
      CaptureId_t free_capture_id) const;
  int eraseCaptureTree(CaptureId_t root_capture_id);

  int active_captures_{0};
  ska::flat_hash_map<CaptureId_t, CaptureTreeNode> capture_tree_;
  ska::flat_hash_map<cudaStream_t, cudaStream_t> reuse_stream_aliases_;
  ska::flat_hash_map<const void*, CaptureId_t> block_allocation_captures_;
};

} // namespace c10::cuda::CUDAGraphMemory
