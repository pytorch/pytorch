#pragma once
// Typed compiler passes: Pass<InputIR, OutputIR> = descriptor + callable.
//
// A pass is a descriptor (name/version, stage pair, analysis sets,
// numeric policy, capability note) bound to a callable with a fixed
// signature for its stage pair. Registration requires a non-empty
// executor; duplicates and unknown names throw. Adding a transform means
// registering one Pass and listing it in the pipeline order below.

#include <cstdint>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "analyses.h"
#include "ir.h"
#include "kernel_ir.h"
#include "tiled_ir.h"

namespace torch::inductor::loops_ir {
namespace pipeline {

// One version per pass; bump only on semantic (dump-changing) edits.
inline constexpr const char* kViewInlineVersion = "1.0.0";
inline constexpr const char* kUnitRangeRemovalVersion = "1.0.0";
inline constexpr const char* kIndexSimplifyVersion = "1.0.0";
inline constexpr const char* kLocalCseDceVersion = "1.0.0";
inline constexpr const char* kDeadPruneVersion = "1.0.0";
inline constexpr const char* kLowerToKernelsVersion = "1.0.0";
inline constexpr const char* kFuseKernelsVersion = "1.0.0";
inline constexpr const char* kLowerToTiledVersion = "1.0.0";

// Full pipeline order, then the legacy 5 micro-pass names (order and
// content identical to the pre-formal pipeline).
inline const std::vector<std::string>& canonicalOrder() {
  static const std::vector<std::string> kOrder = {
      "view_inline",
      "unit_range_removal",
      "index_simplify",
      "local_cse_dce",
      "dead_prune",
      "lower_to_kernels",
      "fuse_kernels",
      "lower_to_tiled",
  };
  return kOrder;
}

inline const std::vector<std::string>& legacyPassNames() {
  static const std::vector<std::string> kLegacy = {
      "view_inline",
      "unit_range_removal",
      "index_simplify",
      "local_cse_dce",
      "dead_prune",
  };
  return kLegacy;
}

// ---------------------------------------------------------------------------
// PassDescriptor: metadata for one Pass<InputIR, OutputIR>.
// ---------------------------------------------------------------------------

struct PassDescriptor {
  std::string name;
  std::string version;
  analyses::IRKind inputIr = analyses::IRKind::Loops;
  analyses::IRKind outputIr = analyses::IRKind::Loops;
  // Analyses the callable genuinely reads from the AnalysisContext.
  std::vector<analyses::AnalysisKind> required;
  // Required-or-known kinds provably untouched by the transform.
  std::vector<analyses::AnalysisKind> preserved;
  // Caches dropped (kind-epoch bumps) after the pass.
  std::vector<analyses::AnalysisKind> invalidated;
  // All passes preserve FP order/rounding under both numerics policies.
  bool strictSafe = true;
  // Only lower_to_tiled consults TileOptions flags; others ignore them.
  bool consultsTileOptions = false;
  std::string capability;
};

// Honesty rules (checked by tests): `required` holds only kinds the
// callable genuinely reads; `invalidated` holds every kind the transform
// may change; invalidated ∩ preserved is empty.
inline std::vector<PassDescriptor> allDescriptors() {
  using analyses::AnalysisKind;
  using analyses::IRKind;
  return {
      {"view_inline",
       kViewInlineVersion,
       IRKind::Loops,
       IRKind::Loops,
       {AnalysisKind::AliasStorageRoots},
       {AnalysisKind::AliasStorageRoots, AnalysisKind::Effects},
       {AnalysisKind::IndexIntervals, AnalysisKind::Liveness},
       true,
       false,
       "view_read only; alias offsets composed with checked arithmetic; "
       "unknown views fail loudly"},
      {"unit_range_removal",
       kUnitRangeRemovalVersion,
       IRKind::Loops,
       IRKind::Loops,
       {},
       {AnalysisKind::AliasStorageRoots, AnalysisKind::Effects},
       {AnalysisKind::IndexIntervals, AnalysisKind::Liveness},
       true,
       false,
       "pointwise ranges only; extent classification reads the compute's "
       "own ranges (pure local); reduction ranges unchanged"},
      {"index_simplify",
       kIndexSimplifyVersion,
       IRKind::Loops,
       IRKind::Loops,
       {},
       {AnalysisKind::AliasStorageRoots, AnalysisKind::Effects},
       {AnalysisKind::IndexIntervals, AnalysisKind::Liveness},
       true,
       false,
       "integer-only identities + checked constant folding; no FP folding"},
      {"local_cse_dce",
       kLocalCseDceVersion,
       IRKind::Loops,
       IRKind::Loops,
       {},
       {AnalysisKind::AliasStorageRoots, AnalysisKind::IndexIntervals, AnalysisKind::Effects},
       {AnalysisKind::Liveness},
       true,
       false,
       "per-body typed SSA; structural index equality; bit-exact constant "
       "match; DCE seeds from stores/accumulates"},
      {"dead_prune",
       kDeadPruneVersion,
       IRKind::Loops,
       IRKind::Loops,
       {AnalysisKind::Liveness, AnalysisKind::Effects},
       {AnalysisKind::IndexIntervals, AnalysisKind::Effects},
       {AnalysisKind::Liveness, AnalysisKind::AliasStorageRoots},
       true,
       false,
       "extern calls always kept (conservative unknown effects); pruned "
       "views drop alias facts"},
      {"lower_to_kernels",
       kLowerToKernelsVersion,
       IRKind::Loops,
       IRKind::Kernel,
       {AnalysisKind::Effects},
       {AnalysisKind::Effects},
       {AnalysisKind::AliasStorageRoots,
        AnalysisKind::IndexIntervals,
        AnalysisKind::Liveness,
        AnalysisKind::Dependencies},
       true,
       false,
       "pointwise-only computes become single-region functions; reductions "
       "must already be eager externs; dependencies rebuilt"},
      {"fuse_kernels",
       kFuseKernelsVersion,
       IRKind::Kernel,
       IRKind::Kernel,
       {AnalysisKind::Dependencies,
        AnalysisKind::Effects,
        AnalysisKind::AliasStorageRoots,
        AnalysisKind::DenseInjective},
       {AnalysisKind::Effects},
       {AnalysisKind::AliasStorageRoots, AnalysisKind::Dependencies, AnalysisKind::DenseInjective},
       true,
       false,
       "adjacent single-use pointwise chains with proven dense/injective "
       "stores and identical iteration maps; idempotent fixed point"},
      {"lower_to_tiled",
       kLowerToTiledVersion,
       IRKind::Kernel,
       IRKind::Tiled,
       {AnalysisKind::Effects},
       {AnalysisKind::Effects},
       {AnalysisKind::AliasStorageRoots, AnalysisKind::Dependencies, AnalysisKind::DenseInjective},
       true,
       true,
       "one tiled function per pointwise region; externs stay eager; "
       "strict rejects enable_fp_fusion / disable_ftz=False"},
  };
}

// ---------------------------------------------------------------------------
// Pass<InputIR, OutputIR>: descriptor + bound callable.
// ---------------------------------------------------------------------------

// Every callable takes TileOptions explicitly (pure-local passes ignore
// it); there are no hidden globals.
template <typename InputIR, typename OutputIR>
struct Pass {
  PassDescriptor descriptor;
  std::function<OutputIR(const InputIR&, const TileOptions&, analyses::AnalysisContext&)> run;
};

using LoopsPass = Pass<Program, Program>;
using LowerPass = Pass<Program, KernelProgram>;
using FusePass = Pass<KernelProgram, KernelProgram>;
using TilePass = Pass<KernelProgram, TiledProgram>;

// Registry entries: exactly the supported stage pairs.
using AnyPass = std::variant<LoopsPass, LowerPass, FusePass, TilePass>;

// C++ type -> IRKind, used at registration to check the descriptor.
template <typename IR>
analyses::IRKind irKindOf();
template <>
inline analyses::IRKind irKindOf<Program>() {
  return analyses::IRKind::Loops;
}
template <>
inline analyses::IRKind irKindOf<KernelProgram>() {
  return analyses::IRKind::Kernel;
}
template <>
inline analyses::IRKind irKindOf<TiledProgram>() {
  return analyses::IRKind::Tiled;
}

// ---------------------------------------------------------------------------
// PassTraceEntry / PipelineResult, bound read-only to Python.
// ---------------------------------------------------------------------------

struct PassTraceEntry {
  std::string name;
  std::string version;
  std::string inputIr;
  std::string outputIr;
  std::string numerics;
  std::string inputFingerprint;
  std::string outputFingerprint;
  std::vector<std::string> analysesComputed;
  std::vector<std::string> analysesInvalidated;
  std::string dump; // empty unless capture_dumps
};

struct PipelineResult {
  // One entry per pipeline stage, in execution order.
  std::vector<PassTraceEntry> trace;
  Program optimizedProgram = Program("pipeline_empty");
  KernelProgram kernelBeforeFusion = KernelProgram("pipeline_empty");
  KernelProgram kernelProgram = KernelProgram("pipeline_empty");
  TiledProgram tiledProgram = TiledProgram("strict_eager");
};

// ---------------------------------------------------------------------------
// Registry (defined in pass_manager.cpp).
// ---------------------------------------------------------------------------

// All 8 pass names in pipeline order.
std::vector<std::string> pipelinePassNames();

// Legacy 5 micro-pass names.
std::vector<std::string> legacyPassNamesList();

// Fetch one descriptor by name; throws on unknown names.
PassDescriptor passDescriptor(const std::string& name);

// All 8 descriptors in pipeline order.
std::vector<PassDescriptor> passDescriptors();

// Bounded test helper for registration validation (duplicate, missing
// executor, unknown, stage mismatch). Returns true; throws on failure.
bool registrySelfTest();

// Register a pass. Requires a non-empty executor whose stage pair agrees
// with the descriptor; rejects duplicates. All 8 builtins register at
// static initialization.
void registerPass(AnyPass pass);

// Single-pass runners (verify before/after, typed registry lookup,
// invalidations). Throw on unknown names or stage mismatch.
Program runTypedLoopsPass(const std::string& name, const Program& program, analyses::AnalysisContext& context);
KernelProgram runTypedKernelPass(const std::string& name,
                                 const KernelProgram& program,
                                 analyses::AnalysisContext& context);
KernelProgram runLowerToKernels(const Program& program, analyses::AnalysisContext& context);
TiledProgram runLowerToTiled(const KernelProgram& program,
                             const TileOptions& options,
                             analyses::AnalysisContext& context);

// Full 8-stage pipeline with per-pass trace entries. captureDumps controls
// whether trace dumps carry text (fingerprints are always recorded).
PipelineResult runPipeline(const Program& program,
                           const TileOptions& options = TileOptions(),
                           bool captureDumps = true);

} // namespace pipeline
} // namespace torch::inductor::loops_ir
