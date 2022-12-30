#pragma once

#include <ir_all_nodes.h>
#include <parallel_type_bitmap.h>

#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class TORCH_CUDA_CU_API SyncMap {
 public:
  //! Validates all tensors are consistently parallelized. Basically,
  //! when a producer axis is threaded, either with threadIdx or
  //! blockIdx, there must be a mapped consumer axis with the
  //! same ParallelType with some exceptions.
  //!
  //! ComputeAtMap is already built as they are used to validate consistency.
  //!
  //! Fills needs_raw_sync with output TVs if they need a raw sync if on smem or
  //! gmem. The second entry in this map is the parallel dimensions being
  //! communicated across.
  SyncMap(Fusion* fusion);

  std::string toString() const;

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
