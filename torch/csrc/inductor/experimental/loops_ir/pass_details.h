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
// Compose read-only view mappings into consumer accesses without copying data.
// Preserve storage offsets and aliases that remain observable at outputs.
Program viewInlineImpl(const Program& program, analyses::AnalysisContext& ctx);

// Remove extent-one pointwise loop dimensions and substitute their sole index.
// Reduction dimensions are left unchanged.
Program unitRangeRemovalImpl(const Program& program, analyses::AnalysisContext& ctx);

// Simplify integer address arithmetic with checked constant folding.
// Floating-point computation and evaluation order are left unchanged.
Program indexSimplifyImpl(const Program& program, analyses::AnalysisContext& ctx);

// Reuse identical SSA values within a compute body and remove unused values.
// Stores and accumulates seed liveness; expressions do not cross extern calls.
Program localCseDceImpl(const Program& program, analyses::AnalysisContext& ctx);

// Prune computes and view aliases not needed by outputs or retained operations.
// Keep extern calls conservatively because they may have side effects.
Program deadPruneImpl(const Program& program, analyses::AnalysisContext& ctx);

// Lower pointwise LoopsIR computes into KernelIR functions and dependencies.
// Native calls remain externs; reduction kernels are not supported here.
KernelProgram lowerToKernelsImpl(const Program& program, analyses::AnalysisContext& ctx);

// Inline eligible single-use pointwise producers into their consumer kernels.
// Require compatible iteration/storage maps; externs split fusion windows.
KernelProgram fuseKernelsImpl(const KernelProgram& program, analyses::AnalysisContext& ctx);

// Map pointwise KernelIR regions onto tiled block/lane execution for codegen.
// Validate tiling/codegen flags against the program's numerical policy.
TiledProgram lowerToTiledImpl(const KernelProgram& program, const TileOptions& options, analyses::AnalysisContext& ctx);

} // namespace detail
} // namespace torch::inductor::loops_ir
