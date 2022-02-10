#pragma once

#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

struct WarpPaddedParallelInfo {
  bool is_tidx_padded = false;
  bool is_tidx_single_warp = false;
  bool has_warp_reduction = false;
};

std::vector<kir::Expr*> fuseWarpReduce(const std::vector<kir::Expr*> exprs);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
