#pragma once

#include <kernel_ir.h>

namespace nvfuser {

struct WarpPaddedParallelInfo {
  bool is_tidx_padded = false;
  bool is_tidx_single_warp = false;
  bool has_warp_reduction = false;
};

std::vector<Expr*> fuseWarpReduce(const std::vector<Expr*> exprs);

} // namespace nvfuser
