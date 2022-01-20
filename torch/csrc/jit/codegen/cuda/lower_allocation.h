#pragma once

#include <torch/csrc/Export.h>

#include <torch/csrc/jit/codegen/cuda/dispatch.h>
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
  std::vector<kir::IterDomain*> alloc_domains;
  bool has_halo = false;
};

using LocalAllocationInfoMap =
    std::unordered_map<kir::Allocate*, std::unique_ptr<LocalAllocationInfo>>;

//! Insert buffer allocations
std::vector<kir::Expr*> insertAllocations(const std::vector<kir::Expr*>& exprs);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
