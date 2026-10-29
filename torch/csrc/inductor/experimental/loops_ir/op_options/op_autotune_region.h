#pragma once
// Auxiliary compile-time region extraction helper for fair fused-region
// autotuning (NOT a registered pass; no GraphLowering/TritonTemplate, no
// runtime IR interpretation).
//
// Owns ONLY this header (+ op_autotune_region.cpp anchor). Reads base IR
// headers and op_options passes; never edits them.
//
// Purpose: given a verified post-fuse_kernels KernelProgram and one
// FusionVariant, extract a typed AutotuneRegion that owns:
//   * baseline: a KernelProgram containing EXACTLY the variant's consumed
//     nodes (original Compute/ExternCall objects, original order), with
//     external storage roots as program inputs and the FINAL variant output
//     (not the anchor output) as the single program output. Prologue
//     intermediates are allocated/computed by their original kernels, the
//     anchor runs as the original native ExternCall, epilogue kernels run
//     verbatim. No synthetic baseline function is ever built.
//   * frontier: the ordered/deduplicated external ROOT set (storage roots
//     read by consumed nodes but produced outside them), each with flat
//     geometry (elements + core dtype). Both baseline and any fused
//     candidate must allocate the SAME frontier backing tensors; native
//     views are reconstructed from binding offsets/strides.
//   * safe slot mappings: per-signature-input and per-extra frontier indices
//     (-1 = prologue-computed slot, fail closed otherwise).
//
// Link safety: this header uses ONLY inline KernelProgram builders/getters
// (addInput/addView/addFunction/addExternCall/addOutput + const accessors),
// ir.cpp-linked Program::verify helpers (via fused_region validators), and
// op_options units already in the isolated extension. It NEVER calls
// KernelProgram::verify()/buildDependencies()/buffers()/dump() (those live
// in kernel_ir.cpp, which the isolated op_options extension does NOT link).
// The owner (backend/selector, which links kernel_ir.cpp) must run
//   baseline.buildDependencies(); baseline.verify();
// before executing the baseline. Standalone CPU tests link kernel_ir.cpp
// explicitly and do run them to prove executability.
//
// CPU safety: C++ Buffer/Binding carries no device. The caller passes the
// explicit cpu_roots set (root names whose frozen ConstantBinding metadata
// is CPU, including hidden IR inputs). Extraction records has_cpu_frontier
// / cuda_generated_safe; requireAutotuneRegionCudaSafe() vetoes generated
// CUDA use. Native fallback stays available/untouched; no CPU pointer may
// reach generated candidates. Never assume whole-op CUDA => all-CUDA.
//
// Fail-closed: any unsupported/malformed shape throws OptionRejected
// (Alias/Context/Signature/Fusion). No partial region is ever returned.

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include "../kernel_ir.h"
#include "op_decode.h"
#include "op_fusion_regions.h"
#include "program_facts.h"

namespace torch::inductor::loops_ir::op_options {

// One external storage root the region reads but does not produce.
struct FrontierRoot {
  std::string root;
  int64_t elements = 0;
  std::string dtype = "f32"; // core dtype spelling behind the root
  bool is_cpu = false;
};

// Bounded typed autotune region. See header comment for ownership.
struct AutotuneRegion {
  explicit AutotuneRegion(OpSignature sig) : signature(std::move(sig)) {}
  KernelProgram baseline;
  std::vector<FrontierRoot> frontier; // first-use order, deduped by root
  // Per signature input slot: frontier index, or -1 when prologue-computed.
  std::vector<int64_t> input_frontier;
  // Per extra binding (parallel to extra_bindings): frontier index (>= 0).
  std::vector<int64_t> extra_frontier;
  OpSignature signature; // variant signature (final output layout)
  std::vector<std::optional<TensorBinding>> input_bindings;
  std::vector<TensorBinding> extra_bindings;
  TensorBinding output_binding; // final variant output (not anchor output)
  TensorLayout output_layout; // signature.output().layout
  std::vector<size_t> consumed_nodes; // ascending, includes anchor
  size_t placement = 0; // == consumed_nodes.back()
  std::string produces; // == output_binding.root
  std::vector<std::string> eliminated_buffers;
  bool has_cpu_frontier = false;
  bool cuda_generated_safe = true; // false iff any frontier root is CPU
  std::string canonical() const {
    std::ostringstream os;
    os << "autotune_region{consumed[";
    for (size_t n : consumed_nodes) {
      os << n << ",";
    }
    os << "] placement=" << placement << " produces=" << produces << " frontier[";
    for (const auto& f : frontier) {
      os << f.root << ":" << f.elements << ":" << f.dtype << (f.is_cpu ? ":cpu" : ":cuda") << ";";
    }
    os << "] inputs[";
    for (int64_t i : input_frontier) {
      os << i << ",";
    }
    os << "] extras[";
    for (int64_t i : extra_frontier) {
      os << i << ",";
    }
    os << "] out=" << output_binding.root << "@" << output_binding.root_offset << ":" << output_layout.canonical()
       << " cuda_safe=" << cuda_generated_safe << "}";
    return os.str();
  }
};

// Header-only version (no .cpp link needed by the isolated extension).

// True when no frontier root is CPU: generated CUDA use is not vetoed.
// Native fallback is always available regardless.
inline bool autotuneRegionCudaSafe(const AutotuneRegion& region) {
  return region.cuda_generated_safe && !region.has_cpu_frontier;
}

// Throws OptionRejected(Context) when a CPU frontier root would reach a
// generated kernel. Native fallback stays untouched.
inline void requireAutotuneRegionCudaSafe(const AutotuneRegion& region) {
  if (!autotuneRegionCudaSafe(region)) {
    std::string roots;
    for (const auto& f : region.frontier) {
      if (f.is_cpu) {
        roots += (roots.empty() ? "" : ",") + f.root;
      }
    }
    reject(RejectReason::Context,
           "autotune region has CPU frontier root(s) [" + roots + "]; generated CUDA use vetoed");
  }
}

// Header-only version (no .cpp link needed by the isolated extension).
inline std::string autotuneRegionVersion() {
  return "autotune_region_v1";
}

namespace detail_autotune {

inline std::vector<std::string> consumedReads(const KernelProgram::Node& node) {
  if (const auto* kf = std::get_if<KernelFunction>(&node)) {
    std::vector<std::string> out;
    for (const Compute& c : kf->regions()) {
      for (const std::string& in : c.inputs()) {
        out.push_back(in);
      }
    }
    return out;
  }
  return std::get<ExternCall>(node).inputs();
}

inline std::vector<Buffer> consumedResults(const KernelProgram::Node& node) {
  if (const auto* kf = std::get_if<KernelFunction>(&node)) {
    std::vector<Buffer> out;
    for (const Compute& c : kf->regions()) {
      for (const Buffer& b : c.results()) {
        out.push_back(b);
      }
    }
    return out;
  }
  return std::get<ExternCall>(node).results();
}

// Bounds/ownership check for one binding against facts + logical extent.
inline void checkBindingRoot(const ProgramFacts& facts,
                             const TensorBinding& b,
                             int64_t extent,
                             const std::string& ctx) {
  if (b.ir_name.empty() || b.root.empty()) {
    reject(RejectReason::Signature, ctx + ": binding has an empty ir_name or root");
  }
  const RootRef* rr = nullptr;
  try {
    rr = &facts.resolve(b.ir_name);
  } catch (const std::invalid_argument& e) {
    reject(RejectReason::Signature, ctx + ": dangling ir_name '" + b.ir_name + "' (" + e.what() + ")");
  }
  if (rr->root != b.root) {
    reject(RejectReason::Signature,
           ctx + ": ir_name '" + b.ir_name + "' resolves to root '" + rr->root + "' != binding root '" + b.root + "'");
  }
  auto itE = facts.root_elements.find(b.root);
  if (itE == facts.root_elements.end()) {
    reject(RejectReason::Signature, ctx + ": unknown storage root '" + b.root + "'");
  }
  if (b.root_elements != itE->second) {
    reject(RejectReason::Signature,
           ctx + ": root_elements " + std::to_string(b.root_elements) + " != program root '" + b.root + "' elements " +
               std::to_string(itE->second));
  }
  const bool isNodeResult = facts.producer.count(b.root) > 0;
  if (b.node_result != isNodeResult) {
    reject(RejectReason::Signature,
           ctx + ": node_result flag disagrees with the program for root '" + b.root + "'");
  }
  if (b.root_offset < 0) {
    reject(RejectReason::Signature, ctx + ": negative root_offset for root '" + b.root + "'");
  }
  if (b.root_offset < rr->offset) {
    reject(RejectReason::Signature,
           ctx + ": root_offset " + std::to_string(b.root_offset) + " below view base " + std::to_string(rr->offset) +
               " for '" + b.ir_name + "'");
  }
  if (extent < 0) {
    reject(RejectReason::Signature, ctx + ": negative extent for root '" + b.root + "'");
  }
  if (b.root_offset + extent > b.root_elements) {
    reject(RejectReason::Signature,
           ctx + ": store range [" + std::to_string(b.root_offset) + "," +
               std::to_string(b.root_offset + extent) + ") exceeds root '" + b.root + "' elements " +
               std::to_string(b.root_elements));
  }
}

inline int64_t frontierIndex(const std::vector<FrontierRoot>& frontier, const std::string& root) {
  for (size_t i = 0; i < frontier.size(); ++i) {
    if (frontier[i].root == root) {
      return static_cast<int64_t>(i);
    }
  }
  return -1;
}

} // namespace detail_autotune

// Extract one variant's fair autotune region. Throws OptionRejected on any
// unsupported/malformed shape (shared claims, dangling names, bounds,
// ownership, ordering, multi-anchor, reduction kernels in consumed set).
// cpu_roots: root names whose frozen ConstantBinding is CPU (may be empty).
// shared_nodes: FusedOpRegions::shared_nodes; any consumed node in it fails
// closed (Alias) since selection must keep consumed sets disjoint.
inline AutotuneRegion extractAutotuneRegion(const KernelProgram& program,
                                            const FusionVariant& variant,
                                            const std::set<std::string>& cpu_roots = {},
                                            const std::vector<size_t>& shared_nodes = {}) {
  using namespace detail_autotune;
  if (variant.consumed_nodes.empty()) {
    reject(RejectReason::Signature, "autotune region: variant has no consumed nodes");
  }
  for (size_t i = 1; i < variant.consumed_nodes.size(); ++i) {
    if (!(variant.consumed_nodes[i - 1] < variant.consumed_nodes[i])) {
      reject(RejectReason::Signature, "autotune region: consumed_nodes must be strictly ascending");
    }
  }
  for (size_t n : variant.consumed_nodes) {
    if (n >= program.nodes().size()) {
      reject(RejectReason::Signature,
             "autotune region: consumed node " + std::to_string(n) + " out of range (" +
                 std::to_string(program.nodes().size()) + " nodes)");
    }
  }
  if (variant.placement != variant.consumed_nodes.back()) {
    reject(RejectReason::Signature, "autotune region: placement must equal max consumed index");
  }
  if (!shared_nodes.empty()) {
    std::set<size_t> shared(shared_nodes.begin(), shared_nodes.end());
    for (size_t n : variant.consumed_nodes) {
      if (shared.count(n)) {
        reject(RejectReason::Alias,
               "autotune region: consumed node " + std::to_string(n) +
                   " is shared with another op; selection must keep consumed sets disjoint");
      }
    }
  }

  ProgramFacts facts;
  try {
    facts = analyzeProgram(program);
  } catch (const std::invalid_argument& e) {
    reject(RejectReason::Signature, std::string("autotune region: program facts failed: ") + e.what());
  }

  // Exactly one extern anchor; every consumed kernel is single-region
  // pointwise (no generated reductions inside the baseline).
  size_t anchors = 0;
  for (size_t n : variant.consumed_nodes) {
    const auto& node = program.nodes()[n];
    if (std::holds_alternative<ExternCall>(node)) {
      ++anchors;
      continue;
    }
    const KernelFunction& kf = std::get<KernelFunction>(node);
    if (kf.regions().size() != 1) {
      reject(RejectReason::Fusion,
             "autotune region: consumed kernel node " + std::to_string(n) + " has more than one region");
    }
    const Compute& c = kf.regions().front();
    if (!c.reductionRanges().empty() || !c.accumulators().empty() || !c.reductionBody().empty()) {
      reject(RejectReason::Fusion,
             "autotune region: consumed kernel node " + std::to_string(n) + " is a reduction (unsupported)");
    }
    if (c.results().size() != 1) {
      reject(RejectReason::Fusion,
             "autotune region: consumed kernel node " + std::to_string(n) + " must define exactly one result");
    }
  }
  if (anchors != 1) {
    reject(RejectReason::Fusion,
           "autotune region: consumed set must contain exactly one extern anchor (found " + std::to_string(anchors) +
               ")");
  }

  // Internal roots: produced by consumed nodes (ownership).
  std::set<std::string> internal;
  std::set<size_t> consumedSet(variant.consumed_nodes.begin(), variant.consumed_nodes.end());
  for (size_t n : variant.consumed_nodes) {
    for (const Buffer& b : consumedResults(program.nodes()[n])) {
      auto pit = facts.producer.find(b.name);
      if (pit == facts.producer.end() || pit->second != n) {
        reject(RejectReason::Signature,
               "autotune region: result '" + b.name + "' ownership disagrees with the program facts");
      }
      if (!internal.insert(b.name).second) {
        reject(RejectReason::Signature, "autotune region: duplicate result '" + b.name + "' in consumed set");
      }
      auto itE = facts.root_elements.find(b.name);
      if (itE == facts.root_elements.end() || itE->second != b.elements) {
        reject(RejectReason::Signature,
               "autotune region: result '" + b.name + "' geometry disagrees with the program facts");
      }
    }
  }

  // Frontier: external roots read by consumed nodes, first-use order.
  std::vector<FrontierRoot> frontier;
  for (size_t n : variant.consumed_nodes) {
    for (const std::string& name : consumedReads(program.nodes()[n])) {
      const RootRef* rr = nullptr;
      try {
        rr = &facts.resolve(name);
      } catch (const std::invalid_argument& e) {
        reject(RejectReason::Signature,
               "autotune region: consumed node " + std::to_string(n) + " reads dangling '" + name + "' (" + e.what() +
                   ")");
      }
      const std::string& root = rr->root;
      if (internal.count(root)) {
        auto pit = facts.producer.find(root);
        if (pit == facts.producer.end() || !consumedSet.count(pit->second) || pit->second >= n) {
          reject(RejectReason::Signature,
                 "autotune region: internal root '" + root + "' read before it is produced");
        }
        continue;
      }
      if (frontierIndex(frontier, root) >= 0) {
        continue;
      }
      auto itE = facts.root_elements.find(root);
      auto itD = facts.root_dtype.find(root);
      if (itE == facts.root_elements.end() || itD == facts.root_dtype.end()) {
        reject(RejectReason::Signature, "autotune region: frontier root '" + root + "' has no program geometry");
      }
      FrontierRoot fr;
      fr.root = root;
      fr.elements = itE->second;
      fr.dtype = itD->second;
      // CPU veto covers aliased/view roots too: any name (buffer or view)
      // resolving to this storage root that the caller flags as CPU taints
      // the whole root. Never assume whole-op CUDA => all-CUDA.
      fr.is_cpu = cpu_roots.count(root) > 0;
      if (!fr.is_cpu) {
        for (const auto& kv : facts.roots) {
          if (kv.second.root == root && cpu_roots.count(kv.first)) {
            fr.is_cpu = true;
            break;
          }
        }
      }
      frontier.push_back(std::move(fr));
    }
  }

  // Signature/extra slot validation + frontier mapping.
  const OpSignature& sig = variant.signature;
  if (variant.input_bindings.size() != sig.inputs().size()) {
    reject(RejectReason::Signature, "autotune region: input_bindings size disagrees with the signature");
  }
  if (variant.extra_bindings.size() != variant.fused_loops.extra_inputs.size()) {
    reject(RejectReason::Signature, "autotune region: extra_bindings size disagrees with fused_loops.extra_inputs");
  }
  AutotuneRegion out(variant.signature);
  out.input_bindings = variant.input_bindings;
  out.extra_bindings = variant.extra_bindings;
  out.output_binding = variant.output_binding;
  out.output_layout = sig.output().layout;
  out.consumed_nodes = variant.consumed_nodes;
  out.placement = variant.placement;
  out.produces = variant.produces;
  out.eliminated_buffers = variant.eliminated_buffers;

  out.input_frontier.assign(sig.inputs().size(), -1);
  for (size_t k = 0; k < sig.inputs().size(); ++k) {
    const auto& slot = variant.input_bindings[k];
    if (!slot) {
      out.input_frontier[k] = -1; // prologue-computed, not a kernel argument
      continue;
    }
    checkBindingRoot(facts, *slot, sig.inputs()[k].layout.extent(), "autotune region input[" + std::to_string(k) + "]");
    if (internal.count(slot->root)) {
      reject(RejectReason::Fusion,
             "autotune region: input slot " + std::to_string(k) + " reads internal root '" + slot->root +
                 "' but is not prologue-computed");
    }
    const int64_t fi = frontierIndex(frontier, slot->root);
    if (fi < 0) {
      reject(RejectReason::Signature,
             "autotune region: input slot " + std::to_string(k) + " root '" + slot->root +
                 "' is never read by the consumed nodes");
    }
    out.input_frontier[k] = fi;
  }
  out.extra_frontier.assign(variant.extra_bindings.size(), -1);
  for (size_t i = 0; i < variant.extra_bindings.size(); ++i) {
    const TensorBinding& b = variant.extra_bindings[i];
    const TensorLayout& lay = variant.fused_loops.extra_inputs[i].layout;
    checkBindingRoot(facts, b, lay.extent(), "autotune region extra[" + std::to_string(i) + "] '" + b.ir_name + "'");
    if (internal.count(b.root)) {
      reject(RejectReason::Fusion,
             "autotune region: extra '" + b.ir_name + "' reads internal root '" + b.root + "' (unsupported)");
    }
    const int64_t fi = frontierIndex(frontier, b.root);
    if (fi < 0) {
      reject(RejectReason::Signature,
             "autotune region: extra '" + b.ir_name + "' root '" + b.root + "' is never read by the consumed nodes");
    }
    // Extra bindings carry the flat root geometry; the frontier entry must
    // agree on the backing allocation.
    if (frontier[static_cast<size_t>(fi)].elements != b.root_elements) {
      reject(RejectReason::Signature,
             "autotune region: extra '" + b.ir_name + "' root_elements disagree with the frontier");
    }
    out.extra_frontier[i] = fi;
  }
  checkBindingRoot(facts, variant.output_binding, out.output_layout.extent(), "autotune region output");
  if (!internal.count(variant.output_binding.root)) {
    reject(RejectReason::Alias,
           "autotune region: output root '" + variant.output_binding.root + "' is not produced by the consumed set");
  }
  if (variant.produces != variant.output_binding.root) {
    reject(RejectReason::Signature, "autotune region: produces disagrees with the output binding root");
  }
  for (const std::string& e : variant.eliminated_buffers) {
    if (!internal.count(e)) {
      reject(RejectReason::Signature,
             "autotune region: eliminated buffer '" + e + "' is not produced by the consumed set");
    }
    if (e == variant.output_binding.root) {
      reject(RejectReason::Signature,
             "autotune region: eliminated buffer '" + e + "' is the final output");
    }
  }

  out.frontier = frontier;
  out.has_cpu_frontier = false;
  for (const auto& f : frontier) {
    out.has_cpu_frontier = out.has_cpu_frontier || f.is_cpu;
  }
  out.cuda_generated_safe = !out.has_cpu_frontier;

  // Baseline: exactly the consumed nodes in original order (verbatim copies;
  // prologue intermediates materialized by their own kernels, anchor as the
  // original ExternCall, epilogue kernels verbatim). Final output is the
  // variant output, never the anchor output.
  KernelProgram base(program.name() + "__autotune_region", program.numerics());
  for (const FrontierRoot& f : frontier) {
    base.addInput(Buffer(f.root, f.dtype, f.elements));
  }
  for (const Descriptor& d : program.inputDescriptors()) {
    if (frontierIndex(frontier, d.buffer) >= 0) {
      base.addInputDescriptor(d);
    }
  }
  // Retain relevant views (read closure) without dangling: include every
  // view read by consumed nodes plus transitive view sources, in original
  // order; each source must be a frontier input, an internal result, or an
  // earlier retained view.
  {
    std::set<std::string> readNames;
    for (size_t n : variant.consumed_nodes) {
      for (const std::string& r : consumedReads(program.nodes()[n])) {
        readNames.insert(r);
      }
    }
    for (const auto& slot : variant.input_bindings) {
      if (slot) {
        readNames.insert(slot->ir_name);
      }
    }
    for (const auto& b : variant.extra_bindings) {
      readNames.insert(b.ir_name);
    }
    std::map<std::string, const View*> byName;
    for (const View& v : program.views()) {
      byName[v.name] = &v;
    }
    std::set<std::string> needed;
    for (const std::string& r : readNames) {
      if (byName.count(r)) {
        needed.insert(r);
      }
    }
    bool grew = true;
    while (grew) {
      grew = false;
      std::set<std::string> add;
      for (const std::string& nname : needed) {
        auto it = byName.find(nname);
        if (it == byName.end()) {
          continue;
        }
        if (byName.count(it->second->source)) {
          add.insert(it->second->source);
        }
      }
      for (const std::string& a : add) {
        grew = needed.insert(a).second || grew;
      }
    }
    std::set<std::string> available;
    for (const auto& f : frontier) {
      available.insert(f.root);
    }
    for (const auto& r : internal) {
      available.insert(r);
    }
    // Fixed-point emission in original order: a view is emitted once its
    // source is available (frontier/internal/earlier view). Any view left
    // unemitted has a dangling source inside the region: fail closed.
    std::set<std::string> emitted;
    bool progressed = true;
    while (progressed) {
      progressed = false;
      for (const View& v : program.views()) {
        if (!needed.count(v.name) || emitted.count(v.name)) {
          continue;
        }
        if (!available.count(v.source)) {
          continue;
        }
        base.addView(v);
        available.insert(v.name);
        emitted.insert(v.name);
        progressed = true;
      }
    }
    for (const std::string& nname : needed) {
      if (!emitted.count(nname)) {
        reject(RejectReason::Signature,
               "autotune region: view '" + nname + "' source is dangling in the region");
      }
    }
  }
  for (size_t n : variant.consumed_nodes) {
    const auto& node = program.nodes()[n];
    if (const auto* kf = std::get_if<KernelFunction>(&node)) {
      base.addFunction(*kf);
    } else {
      base.addExternCall(std::get<ExternCall>(node));
    }
  }
  base.addOutput(Descriptor("autotune_out", variant.output_binding.root, out.output_layout.sizes,
                            out.output_layout.strides, variant.output_binding.root_offset));
  out.baseline = std::move(base);
  return out;
}

} // namespace torch::inductor::loops_ir::op_options
