#pragma once
// Pass: form_fused_op_regions (KernelProgram + DecodedOps -> FusedOpRegions,
// read-only; the input program is never modified).
//
// For every Decoded op it finds structurally fusible neighbours:
//   prologue[k]: the single-region pointwise KernelFunction producing the
//     storage root of operand k, used only by this op (counted through every
//     alias/view), not a program output, read through exactly one operand;
//   epilogue: the single-region pointwise KernelFunction that is the only
//     user of the op result (through every alias/view), not an output.
// Only KernelFunctions may lie between fused nodes (no extern is crossed:
// unknown/RNG/mutating externs stay barriers). Domains must equal the
// operand/output logical sizes and stores must be exact affine maps.
// Each structural combination becomes a FusionVariant that is then checked
// against every implementing option's SDK fusion contract (and the native
// plan pattern); variants no option accepts are dropped and recorded.
// Variant 0 is always the unfused op. Regions of different ops may claim the
// same node (e.g. mm -> relu -> mm); shared_nodes lists them and selection
// must pick disjoint consumed_nodes.

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "op_decode.h"
#include "option_registry.h"

namespace torch::inductor::loops_ir::op_options {

enum class FusionVetoKind {
  NotPointwiseKernel,
  MultipleRegions,
  ReductionRegion,
  ViewRead,
  UnusedInput,
  MultipleUses,
  EscapesOutput,
  CrossesExtern,
  OperandRepeated,
  DomainMismatch,
  NonAffineStore,
  NoOptionContract,
};
const char* fusionVetoKindName(FusionVetoKind kind);

struct FusionVeto {
  FusionVetoKind kind = FusionVetoKind::NotPointwiseKernel;
  std::string role; // "prologue[k]", "epilogue" or "variant"
  std::optional<size_t> node;
  std::string detail;
};

struct OptionContractResult {
  std::string option_id;
  bool accepted = false;
  std::optional<RejectReason> reason;
  std::string detail;
};

struct FusionVariant {
  explicit FusionVariant(OpSignature sig) : signature(std::move(sig)) {}
  std::vector<int64_t> prologue_operands;
  bool epilogue = false;
  OpSignature signature; // output replaced by the epilogue result when fused
  FusedLoops fused_loops;
  // Per signature input slot; nullopt when a prologue computes the operand.
  std::vector<std::optional<TensorBinding>> input_bindings;
  std::vector<TensorBinding> extra_bindings; // parallel to fused_loops.extra_inputs
  TensorBinding output_binding;
  std::vector<size_t> consumed_nodes; // ascending, includes the op node
  size_t placement = 0; // node index the fused op replaces in program order
  std::string produces; // buffer written by the fused op
  std::vector<std::string> eliminated_buffers; // no longer materialized
  std::vector<OptionContractResult> contracts;
  bool acceptedBy(const std::string& option_id) const;
};

struct OpRegionSet {
  size_t decoded_index = 0;
  size_t node_index = 0;
  std::vector<FusionVariant> variants;
  std::vector<FusionVeto> vetoes;
};

struct FusedOpRegions {
  std::vector<OpRegionSet> ops;
  std::vector<size_t> shared_nodes;
};

FusedOpRegions formFusedOpRegions(const KernelProgram& program,
                                  const DecodedOps& decoded,
                                  const OptionRegistry& registry);

} // namespace torch::inductor::loops_ir::op_options
