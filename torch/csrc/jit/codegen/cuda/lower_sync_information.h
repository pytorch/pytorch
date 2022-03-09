#pragma once

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/parallel_type_bitmap.h>

#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class SyncMap {
 public:
  std::string toString() const;

  //! Validates all tensors are consistently parallelized. Basically,
  //! when a producer axis is threaded, either with threadIdx or
  //! blockIdx, there must be a mapped consumer axis with the
  //! same ParallelType with some exceptions.
  //!
  //! This function assumes Loop and Parallel ComputeAtMaps are already
  //! built as they are used to validate consistency.
  //!
  //! Fills needs_raw_sync with output TVs if they need a raw sync if on smem or
  //! gmem. The second entry in this map is the parallel dimensions being
  //! communicated across.
  void build(Fusion* fusion);

  ParallelTypeBitmap needsRawSync(TensorView* tv) const {
    auto it = needs_raw_sync_.find(tv);
    if (it != needs_raw_sync_.end()) {
      return it->second;
    }
    return ParallelTypeBitmap();
  }

 private:
  std::unordered_map<TensorView*, ParallelTypeBitmap> needs_raw_sync_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
