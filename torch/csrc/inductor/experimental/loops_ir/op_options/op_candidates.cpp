#include "op_candidates.h"

#include <algorithm>

namespace torch::inductor::loops_ir::op_options {

namespace {

bool isBaselineOption(const OptionDescriptor& opt, size_t variant) {
  return variant == 0 && opt.fence == NumericsFence::StrictEager && opt.strategy == CodegenStrategy::NativeAten;
}

// Generated kernels write the output binding in place, so require a proof
// from physical bindings: a fresh node-result root, disjoint by root from
// every read binding, with the store range inside the root allocation.
// Native options return a fresh eager tensor and are exempt.
std::optional<std::string> generatedOutputVeto(const FusionVariant& v) {
  const TensorBinding& out = v.output_binding;
  if (!out.node_result) {
    return "output root '" + out.root + "' is not a fresh node result (aliasing not provable)";
  }
  for (const auto& in : v.input_bindings) {
    if (in && in->root == out.root) {
      return "output root '" + out.root + "' is also read as an operand";
    }
  }
  for (const auto& e : v.extra_bindings) {
    if (e.root == out.root) {
      return "output root '" + out.root + "' is also read as an extra input";
    }
  }
  const int64_t store = v.fused_loops.epilogue ? v.fused_loops.epilogue->store_offset : 0;
  if (out.root_offset + store + v.signature.output().layout.extent() > out.root_elements) {
    return "output store range exceeds root '" + out.root + "'";
  }
  return std::nullopt;
}

void enumerateVariant(const OptionRegistry& reg,
                      const CodegenContext& ctx,
                      size_t vi,
                      const FusionVariant& v,
                      OpCandidates& oc) {
  for (const std::string& id : reg.optionsFor(v.signature.kind())) {
    if (!v.acceptedBy(id)) {
      continue; // contract rejection already recorded by form_fused_op_regions
    }
    const OptionDescriptor& opt = reg.get(id);
    if (opt.strategy != CodegenStrategy::NativeAten) {
      if (auto veto = generatedOutputVeto(v)) {
        oc.rejections.push_back(CandidateRejection{vi, id, RejectReason::Alias, *veto});
        continue;
      }
    }
    std::vector<OptionConfig> configs;
    try {
      configs = reg.candidateConfigs(id, v.signature, ctx);
    } catch (const OptionRejected& e) {
      oc.rejections.push_back(CandidateRejection{vi, id, e.reason(), e.what()});
      continue;
    }
    for (const OptionConfig& c : configs) {
      try {
        const ValidatedSelection sel = reg.validate(id, v.signature, c, v.fused_loops, ctx);
        Candidate cand;
        cand.kind = opt.strategy == CodegenStrategy::NativeAten ? CandidateKind::NativeOption
                                                                : CandidateKind::GeneratedOption;
        cand.variant = vi;
        cand.option_id = id;
        cand.option_version = opt.version;
        cand.config = c;
        cand.cache_key = sel.cache_key;
        cand.baseline = isBaselineOption(opt, vi);
        oc.candidates.push_back(std::move(cand));
      } catch (const OptionRejected& e) {
        oc.rejections.push_back(CandidateRejection{vi, id, e.reason(), e.what()});
      }
    }
  }
}

void placeBaselineFirst(OpCandidates& oc) {
  auto it = std::find_if(oc.candidates.begin(), oc.candidates.end(), [](const Candidate& c) { return c.baseline; });
  if (it != oc.candidates.end()) {
    std::rotate(oc.candidates.begin(), it, it + 1);
    return;
  }
  Candidate original;
  original.kind = CandidateKind::OriginalExtern;
  original.baseline = true;
  oc.candidates.insert(oc.candidates.begin(), original);
}

} // namespace

const char* candidateKindName(CandidateKind kind) {
  switch (kind) {
    case CandidateKind::OriginalExtern:
      return "original_extern";
    case CandidateKind::NativeOption:
      return "native_option";
    case CandidateKind::GeneratedOption:
      return "generated_option";
  }
  return "?";
}

CandidatePlan enumerateOpCandidates(const FusedOpRegions& regions,
                                    const DecodedOps& decoded,
                                    const OptionRegistry& registry,
                                    const CodegenContext& context) {
  context.validate();
  if (decoded.numerics != numericsPolicyName(context.numerics)) {
    reject(RejectReason::Context,
           "context numerics '" + std::string(numericsPolicyName(context.numerics)) + "' != program numerics '" +
               decoded.numerics + "'");
  }
  CandidatePlan plan;
  plan.numerics = context.numerics;
  plan.context = context.canonical();
  plan.shared_nodes = regions.shared_nodes;
  for (const OpRegionSet& set : regions.ops) {
    OpCandidates oc(decoded.ops.at(set.decoded_index).fallback);
    oc.decoded_index = set.decoded_index;
    oc.node_index = set.node_index;
    oc.variants = set.variants;
    for (size_t vi = 0; vi < oc.variants.size(); ++vi) {
      enumerateVariant(registry, context, vi, oc.variants[vi], oc);
    }
    placeBaselineFirst(oc);
    plan.ops.push_back(std::move(oc));
  }
  return plan;
}

} // namespace torch::inductor::loops_ir::op_options
