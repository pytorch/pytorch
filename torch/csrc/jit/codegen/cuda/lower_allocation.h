#pragma once

#include <c10/macros/Export.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>

#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! Buffer allocation information to store in GPU lower to avoid
//!  logic duplication
struct LocalAllocationInfo {
  kir::Allocate* alloc_expr = nullptr;
  std::vector<IterDomain*> alloc_domains;
  bool has_halo = false;
};

using LocalAllocationInfoMap = std::
    unordered_map<const kir::Allocate*, std::unique_ptr<LocalAllocationInfo>>;

//! Insert buffer allocations
std::vector<Expr*> insertAllocations(const std::vector<Expr*>& exprs);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
