#pragma once
// Detail transform bodies for the pass registry.
//
// Declares the detail bodies (defined in passes.cpp / kernel_ir.cpp /
// tiled_ir.cpp) that the registry in pass_manager.cpp binds. Public API
// in passes.h / kernel_ir.h / tiled_ir.h keeps its signatures; those
// public functions are thin shims dispatching through pipeline::.
//
// Layering (no header cycle): pass_manager.h does NOT include this header;
// only .cpp files include it.

#include "analyses.h"
#include "ir.h"
#include "kernel_ir.h"
#include "tiled_ir.h"

namespace torch::inductor::loops_ir {
namespace detail {

// Each body verifies before/after, genuinely reads its descriptor's
// `required` analyses from ctx (shared verbatim algorithms in analyses.h),
// and never dispatches through the public shims (no recursive dispatch).
Program viewInlineImpl(const Program& program, analyses::AnalysisContext& ctx);
Program unitRangeRemovalImpl(const Program& program, analyses::AnalysisContext& ctx);
Program indexSimplifyImpl(const Program& program, analyses::AnalysisContext& ctx);
Program localCseDceImpl(const Program& program, analyses::AnalysisContext& ctx);
Program deadPruneImpl(const Program& program, analyses::AnalysisContext& ctx);
KernelProgram lowerToKernelsImpl(const Program& program, analyses::AnalysisContext& ctx);
KernelProgram fuseKernelsImpl(const KernelProgram& program, analyses::AnalysisContext& ctx);
// Tiling body (defined in tiled_ir.cpp). tiled_ir.h includes ir.h only,
// so including it here creates no cycle.
TiledProgram lowerToTiledImpl(const KernelProgram& program, const TileOptions& options, analyses::AnalysisContext& ctx);

} // namespace detail
} // namespace torch::inductor::loops_ir
