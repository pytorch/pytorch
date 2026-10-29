// Pass registry: typed Pass entries, generic driver, 8-stage pipeline.

#include "pass_manager.h"

#include <map>
#include <set>
#include <sstream>

#include "pass_details.h"

namespace torch::inductor::loops_ir {
namespace pipeline {
namespace {

using analyses::AnalysisContext;
using analyses::AnalysisKind;

// The registry: name -> typed Pass. Built once at static initialization
// by registering each builtin's descriptor bound to its detail callable.
// No string-based transform dispatch exists: runners retrieve the typed
// entry and invoke its callable through the generic driver below.
const std::map<std::string, AnyPass>& registry() {
  static const std::map<std::string, AnyPass> kRegistry = []() {
    std::map<std::string, AnyPass> out;
    auto add = [&](AnyPass pass) {
      const PassDescriptor* desc = std::visit(
          [](const auto& p) -> const PassDescriptor* { return &p.descriptor; },
          pass);
      if (desc->name.empty()) {
        throw std::invalid_argument("pass registration: empty pass name");
      }
      bool hasExecutor = (bool)std::visit(
          [](const auto& p) -> bool { return (bool)p.run; }, pass);
      if (!hasExecutor) {
        throw std::invalid_argument(
            "pass registration '" + desc->name + "': missing executor");
      }
      if (!out.emplace(desc->name, std::move(pass)).second) {
        throw std::invalid_argument(
            "duplicate pass registration '" + desc->name + "'");
      }
    };
    for (const auto& d : allDescriptors()) {
      if (d.name == "view_inline") {
        add(LoopsPass{
            d, [](const Program& p, const TileOptions&, AnalysisContext& c) {
              return detail::viewInlineImpl(p, c);
            }});
      } else if (d.name == "unit_range_removal") {
        add(LoopsPass{
            d, [](const Program& p, const TileOptions&, AnalysisContext& c) {
              return detail::unitRangeRemovalImpl(p, c);
            }});
      } else if (d.name == "index_simplify") {
        add(LoopsPass{
            d, [](const Program& p, const TileOptions&, AnalysisContext& c) {
              return detail::indexSimplifyImpl(p, c);
            }});
      } else if (d.name == "local_cse_dce") {
        add(LoopsPass{
            d, [](const Program& p, const TileOptions&, AnalysisContext& c) {
              return detail::localCseDceImpl(p, c);
            }});
      } else if (d.name == "dead_prune") {
        add(LoopsPass{
            d, [](const Program& p, const TileOptions&, AnalysisContext& c) {
              return detail::deadPruneImpl(p, c);
            }});
      } else if (d.name == "lower_to_kernels") {
        add(LowerPass{
            d, [](const Program& p, const TileOptions&, AnalysisContext& c) {
              return detail::lowerToKernelsImpl(p, c);
            }});
      } else if (d.name == "fuse_kernels") {
        add(FusePass{
            d,
            [](const KernelProgram& p, const TileOptions&, AnalysisContext& c) {
              return detail::fuseKernelsImpl(p, c);
            }});
      } else if (d.name == "lower_to_tiled") {
        add(TilePass{
            d,
            [](const KernelProgram& p,
               const TileOptions& o,
               AnalysisContext& c) {
              return detail::lowerToTiledImpl(p, o, c);
            }});
      } else {
        throw std::invalid_argument(
            "pass '" + d.name + "' has no bound detail callable");
      }
    }
    return out;
  }();
  return kRegistry;
}

// Fixed kind order for deterministic trace output.
const std::vector<AnalysisKind>& kindOrder() {
  static const std::vector<AnalysisKind> kKinds = {
      AnalysisKind::AliasStorageRoots,
      AnalysisKind::IndexIntervals,
      AnalysisKind::Liveness,
      AnalysisKind::Dependencies,
      AnalysisKind::Effects,
      AnalysisKind::DenseInjective,
  };
  return kKinds;
}

std::vector<std::string> kindNames(const std::vector<AnalysisKind>& kinds) {
  std::vector<std::string> out;
  for (AnalysisKind k : kinds) {
    out.push_back(analyses::toString(k));
  }
  return out;
}

// Kinds genuinely (re)computed during a pass (counter deltas); cache hits
// are reuse, not recomputation.
std::vector<std::string> computedKindNames(
    const std::map<AnalysisKind, uint64_t>& before,
    const std::map<AnalysisKind, uint64_t>& after) {
  std::vector<std::string> out;
  for (AnalysisKind k : kindOrder()) {
    uint64_t b = 0, a = 0;
    auto bit = before.find(k);
    if (bit != before.end()) {
      b = bit->second;
    }
    auto ait = after.find(k);
    if (ait != after.end()) {
      a = ait->second;
    }
    if (a > b) {
      out.push_back(analyses::toString(k));
    }
  }
  return out;
}

// Generic driver shared by every stage pair: verify input -> fingerprint
// -> snapshot counters -> invoke the registered callable -> verify output
// -> trace entry (computed kinds from real counter deltas) -> invalidate.
template <typename InputIR, typename OutputIR>
OutputIR drive(
    const PassDescriptor& desc,
    const InputIR& input,
    const std::function<
        OutputIR(const InputIR&, const TileOptions&, AnalysisContext&)>& run,
    const TileOptions& options,
    AnalysisContext& ctx,
    bool captureDumps,
    PassTraceEntry& entry) {
  input.verify();
  std::string inputFp = analyses::fingerprintOf(input);
  auto before = ctx.snapshotComputationCounts();
  OutputIR out = run(input, options, ctx);
  out.verify();
  auto after = ctx.snapshotComputationCounts();
  entry.name = desc.name;
  entry.version = desc.version;
  entry.inputIr = analyses::toString(desc.inputIr);
  entry.outputIr = analyses::toString(desc.outputIr);
  entry.numerics = input.numerics();
  entry.inputFingerprint = inputFp;
  entry.outputFingerprint = analyses::fingerprintOf(out);
  entry.analysesComputed = computedKindNames(before, after);
  entry.analysesInvalidated = kindNames(desc.invalidated);
  entry.dump = captureDumps ? out.dump() : "";
  ctx.invalidate(desc.invalidated);
  return out;
}

// Retrieve a registered pass and check it holds the expected stage pair.
// The variant alternative IS the stage check: no name comparisons.
template <typename Expected>
const Expected& getTyped(const std::string& name) {
  auto it = registry().find(name);
  if (it == registry().end()) {
    throw std::invalid_argument("unknown pass '" + name + "'");
  }
  const Expected* typed = std::get_if<Expected>(&it->second);
  if (typed == nullptr) {
    const PassDescriptor* desc = std::visit(
        [](const auto& p) -> const PassDescriptor* { return &p.descriptor; },
        it->second);
    throw std::invalid_argument(
        "pass '" + name + "' expects " + analyses::toString(desc->inputIr) +
        "->" + analyses::toString(desc->outputIr) + " IR (stage mismatch)");
  }
  return *typed;
}

} // namespace

std::vector<std::string> pipelinePassNames() {
  return canonicalOrder();
}

std::vector<std::string> legacyPassNamesList() {
  return legacyPassNames();
}

PassDescriptor passDescriptor(const std::string& name) {
  auto it = registry().find(name);
  if (it == registry().end()) {
    throw std::invalid_argument("unknown pass '" + name + "'");
  }
  return *std::visit(
      [](const auto& p) -> const PassDescriptor* { return &p.descriptor; },
      it->second);
}

std::vector<PassDescriptor> passDescriptors() {
  return allDescriptors();
}

// Bounded test helper: exercises registration validation paths in C++
// (duplicate builtin, missing executor, unknown lookup, stage mismatch).
// Throws std::logic_error if any check misbehaves; returns true otherwise.
// No dummy passes are registered (the builtin registry is unchanged).
bool registrySelfTest() {
  auto expectInvalid = [](const std::string& what, std::function<void()> fn) {
    try {
      fn();
    } catch (const std::invalid_argument&) {
      return;
    }
    throw std::logic_error(
        "registrySelfTest: '" + what + "' did not throw invalid_argument");
  };
  // Duplicate builtin registration.
  expectInvalid("duplicate", []() {
    registerPass(
        LoopsPass{
            passDescriptor("view_inline"),
            [](const Program& p, const TileOptions&, AnalysisContext& c) {
              return detail::viewInlineImpl(p, c);
            }});
  });
  // Missing executor.
  expectInvalid("missing executor", []() {
    registerPass(LoopsPass{passDescriptor("dead_prune"), nullptr});
  });
  // Descriptor/callable stage disagreement.
  expectInvalid("stage disagreement", []() {
    PassDescriptor desc = passDescriptor("fuse_kernels");
    registerPass(
        LoopsPass{
            desc, [](const Program& p, const TileOptions&, AnalysisContext& c) {
              return detail::deadPruneImpl(p, c);
            }});
  });
  // Unknown lookup.
  expectInvalid("unknown", []() { passDescriptor("no_such_pass"); });
  // Stage mismatch on typed runners.
  AnalysisContext ctx;
  Program loopsProg("self_test_probe");
  expectInvalid("loops runner on kernel pass", [&]() {
    runTypedLoopsPass("fuse_kernels", loopsProg, ctx);
  });
  KernelProgram kernelProg("self_test_probe");
  expectInvalid("kernel runner on loops pass", [&]() {
    runTypedKernelPass("view_inline", kernelProg, ctx);
  });
  if (pipelinePassNames().size() != 8) {
    throw std::logic_error("registrySelfTest: expected 8 passes");
  }
  return true;
}

void registerPass(AnyPass pass) {
  const PassDescriptor* desc = std::visit(
      [](const auto& p) -> const PassDescriptor* { return &p.descriptor; },
      pass);
  if (desc->name.empty()) {
    throw std::invalid_argument("pass registration: empty pass name");
  }
  bool hasExecutor =
      (bool)std::visit([](const auto& p) -> bool { return (bool)p.run; }, pass);
  if (!hasExecutor) {
    throw std::invalid_argument(
        "pass registration '" + desc->name + "': missing executor");
  }
  // The stage pair is fixed by the variant alternative; the descriptor
  // must agree with it.
  auto checkStages = [&](analyses::IRKind in, analyses::IRKind out) {
    if (desc->inputIr != in || desc->outputIr != out) {
      throw std::invalid_argument(
          "pass registration '" + desc->name +
          "': descriptor stage disagrees with callable stage");
    }
  };
  if (std::holds_alternative<LoopsPass>(pass)) {
    checkStages(irKindOf<Program>(), irKindOf<Program>());
  } else if (std::holds_alternative<LowerPass>(pass)) {
    checkStages(irKindOf<Program>(), irKindOf<KernelProgram>());
  } else if (std::holds_alternative<FusePass>(pass)) {
    checkStages(irKindOf<KernelProgram>(), irKindOf<KernelProgram>());
  } else {
    checkStages(irKindOf<KernelProgram>(), irKindOf<TiledProgram>());
  }
  if (registry().count(desc->name) != 0) {
    throw std::invalid_argument(
        "duplicate pass registration '" + desc->name + "'");
  }
  // The builtin registry is fixed at static initialization; dynamic
  // extension registers through this API in a mutable overlay is not
  // supported (no metadata-only or callback-based passes).
  throw std::invalid_argument(
      "pass registration '" + desc->name +
      "': dynamic extension is unsupported (all passes are builtins)");
}

Program runTypedLoopsPass(
    const std::string& name,
    const Program& program,
    AnalysisContext& ctx) {
  const LoopsPass& pass = getTyped<LoopsPass>(name);
  PassTraceEntry entry;
  return drive<Program, Program>(
      pass.descriptor, program, pass.run, TileOptions(), ctx, false, entry);
}

KernelProgram runTypedKernelPass(
    const std::string& name,
    const KernelProgram& program,
    AnalysisContext& ctx) {
  const FusePass& pass = getTyped<FusePass>(name);
  PassTraceEntry entry;
  return drive<KernelProgram, KernelProgram>(
      pass.descriptor, program, pass.run, TileOptions(), ctx, false, entry);
}

KernelProgram runLowerToKernels(const Program& program, AnalysisContext& ctx) {
  const LowerPass& pass = getTyped<LowerPass>("lower_to_kernels");
  PassTraceEntry entry;
  return drive<Program, KernelProgram>(
      pass.descriptor, program, pass.run, TileOptions(), ctx, false, entry);
}

TiledProgram runLowerToTiled(
    const KernelProgram& program,
    const TileOptions& options,
    AnalysisContext& ctx) {
  const TilePass& pass = getTyped<TilePass>("lower_to_tiled");
  PassTraceEntry entry;
  return drive<KernelProgram, TiledProgram>(
      pass.descriptor, program, pass.run, options, ctx, false, entry);
}

PipelineResult runPipeline(
    const Program& program,
    const TileOptions& options,
    bool captureDumps) {
  program.verify();
  PipelineResult result;
  AnalysisContext ctx;
  Program cur = program;
  for (const std::string& name : legacyPassNames()) {
    const LoopsPass& pass = getTyped<LoopsPass>(name);
    PassTraceEntry entry;
    cur = drive<Program, Program>(
        pass.descriptor,
        cur,
        pass.run,
        TileOptions(),
        ctx,
        captureDumps,
        entry);
    result.trace.push_back(std::move(entry));
  }
  result.optimizedProgram = cur;
  {
    const LowerPass& lower = getTyped<LowerPass>("lower_to_kernels");
    PassTraceEntry entry;
    result.kernelBeforeFusion = drive<Program, KernelProgram>(
        lower.descriptor,
        cur,
        lower.run,
        TileOptions(),
        ctx,
        captureDumps,
        entry);
    result.trace.push_back(std::move(entry));
    const FusePass& fuse = getTyped<FusePass>("fuse_kernels");
    PassTraceEntry fuseEntry;
    result.kernelProgram = drive<KernelProgram, KernelProgram>(
        fuse.descriptor,
        result.kernelBeforeFusion,
        fuse.run,
        TileOptions(),
        ctx,
        captureDumps,
        fuseEntry);
    result.trace.push_back(std::move(fuseEntry));
  }
  {
    const TilePass& tile = getTyped<TilePass>("lower_to_tiled");
    PassTraceEntry entry;
    result.tiledProgram = drive<KernelProgram, TiledProgram>(
        tile.descriptor,
        result.kernelProgram,
        tile.run,
        options,
        ctx,
        captureDumps,
        entry);
    result.trace.push_back(std::move(entry));
  }
  return result;
}

} // namespace pipeline
} // namespace torch::inductor::loops_ir
