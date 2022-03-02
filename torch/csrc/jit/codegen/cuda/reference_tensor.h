#pragma once

#include <c10/macros/Export.h>

#include <torch/csrc/jit/codegen/cuda/ir_interface_nodes.h>

#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

struct ReferenceTensor {
  TensorDomain* domain = nullptr;

  // Map from concrete iteration domains in ComputeAtMaps to iter domains
  // including those used to construct domain.
  std::unordered_map<IterDomain*, IterDomain*> concrete_to_id;
  // Map from reference iteration domains to concrete iteration domains.
  std::unordered_map<IterDomain*, IterDomain*> id_to_concrete;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
