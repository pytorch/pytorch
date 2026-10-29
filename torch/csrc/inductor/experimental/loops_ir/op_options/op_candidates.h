#pragma once
// Pass: enumerate_op_candidates (FusedOpRegions -> CandidatePlan).
//
// For each fused region set, enumerates every (variant, option, config)
// that passes registry eligibility under the frozen context and validates
// with that variant's fused loops. Configs come only from each option's
// bounded candidate manifest. Candidate 0 is always the baseline: the
// eager-identical native option on the unfused variant when eligible,
// otherwise the original ExternCall itself. No candidate is selected here;
// selection belongs to a later measured (serialized) autotune step.

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "op_fusion_regions.h"

namespace torch::inductor::loops_ir::op_options {

enum class CandidateKind { OriginalExtern, NativeOption, GeneratedOption };
const char* candidateKindName(CandidateKind kind);

struct Candidate {
  CandidateKind kind = CandidateKind::OriginalExtern;
  size_t variant = 0;
  std::string option_id; // empty for OriginalExtern
  int64_t option_version = 0;
  std::optional<OptionConfig> config;
  std::string cache_key; // ValidatedSelection key (empty for OriginalExtern)
  bool baseline = false;
};

struct CandidateRejection {
  size_t variant = 0;
  std::string option_id;
  RejectReason reason = RejectReason::Config;
  std::string detail;
};

struct OpCandidates {
  explicit OpCandidates(ExternCall call) : fallback(std::move(call)) {}
  size_t decoded_index = 0;
  size_t node_index = 0;
  ExternCall fallback;
  std::vector<FusionVariant> variants;
  std::vector<Candidate> candidates; // candidates[0] is the baseline
  std::vector<CandidateRejection> rejections;
};

struct CandidatePlan {
  NumericsPolicy numerics = NumericsPolicy::StrictEager;
  std::string context; // CodegenContext::canonical()
  std::vector<OpCandidates> ops;
  std::vector<size_t> shared_nodes;
};

CandidatePlan enumerateOpCandidates(const FusedOpRegions& regions,
                                    const DecodedOps& decoded,
                                    const OptionRegistry& registry,
                                    const CodegenContext& context);

} // namespace torch::inductor::loops_ir::op_options
