#pragma once

#include <c10/macros/Export.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

struct VectorizedSetInfo {
  //! Producer of a vectorized set
  TensorView* producer_tv = nullptr;
  //! Consumer of a vectorized set
  TensorView* consumer_tv = nullptr;
  //! Number of elements to vectorize
  int word_size = -1;
  //! Vectorized domain
  IterDomain* vectorized_leaf_id = nullptr;
  //! Right-most root dependent domain of the leaf domain
  IterDomain* vectorized_root_id = nullptr;
  //! All of the dependent root domains that are contiguously merged
  std::unordered_set<IterDomain*> contig_root_ids;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
