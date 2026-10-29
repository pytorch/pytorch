// Reduction implementation options for sum/mean over explicit dims:
// eager-identical native ATen delegation and a generated single-axis Triton
// row reduction with prologue/epilogue fusion. Var/std/softmax kinds are
// SDK-reserved and intentionally have no option here.

#include <algorithm>
#include <set>

#include "option_registry.h"

namespace torch::inductor::loops_ir::op_options {

namespace {

constexpr int64_t kReductionTileElementBudget = 8192;
constexpr int64_t kMaxBlockR = 4096;

int64_t nextPow2(int64_t v) {
  int64_t p = 1;
  while (p < v) {
    p *= 2;
  }
  return p;
}

int64_t nonTrivialDims(const TensorLayout& layout) {
  return std::count_if(layout.sizes.begin(), layout.sizes.end(), [](int64_t s) { return s > 1; });
}

OptionDescriptor nativeReduction() {
  OptionDescriptor o;
  o.id = "native_aten_reduction";
  o.version = 1;
  o.op_kinds = {OpKind::SumDim, OpKind::MeanDim};
  o.fence = NumericsFence::StrictEager;
  o.strategy = CodegenStrategy::NativeAten;
  o.config_kind = ConfigKind::Native;
  o.effects = "pure: one eager reduction call; fresh eager-allocated output";
  o.fusion_contract = [](const OpSignature&) { return FusionContract{}; };
  o.check_eligible = [](const OpSignature& sig, const CodegenContext&) {
    requireCommonTensorAbi(sig, sig.device());
    const TensorLayout& out = sig.output().layout;
    // Eager TensorIterator picks the result strides; only accept signatures
    // whose layout is unique (contiguous input or <= 1 non-trivial dim).
    if (!out.isContiguous() || (nonTrivialDims(out) > 1 && !sig.inputs()[0].layout.isContiguous())) {
      reject(RejectReason::Layout, "eager reduction result layout is not statically determined for this input");
    }
  };
  o.candidates = [](const OpSignature&, const CodegenContext&) { return std::vector<OptionConfig>{NativeConfig{}}; };
  o.native_plan = [](const OpSignature& sig, const ValidatedFusion&) {
    NativeCallPlan p;
    p.target = sig.kind() == OpKind::SumDim ? NativeTarget::AtenSumDimIntList : NativeTarget::AtenMeanDim;
    p.operand_slots = {0};
    p.dims = sig.reduction().dims;
    p.keepdim = sig.reduction().keepdim;
    return p;
  };
  return o;
}

int64_t reducedExtent(const OpSignature& sig) {
  return sig.inputs()[0].layout.sizes[static_cast<size_t>(sig.reduction().dims[0])];
}

int64_t outputRows(const OpSignature& sig) {
  return sig.output().layout.numel();
}

void checkReductionConfig(const OpSignature&, const OptionConfig& config, const CodegenContext&) {
  const auto& c = std::get<ReductionConfig>(config);
  if (c.block_x * c.block_r > kReductionTileElementBudget) {
    reject(RejectReason::Config, "block_x*block_r exceeds the register tile budget");
  }
}

void addReductionCandidates(std::vector<OptionConfig>& out,
                            std::set<std::string>& seen,
                            const std::vector<int64_t>& blockXs,
                            int64_t blockR,
                            int64_t rows) {
  for (int64_t bx : blockXs) {
    if (bx > std::max<int64_t>(1, nextPow2(rows)) || bx * blockR > kReductionTileElementBudget) {
      continue;
    }
    for (int64_t warps : {4, 8}) {
      ReductionConfig c;
      c.block_x = bx;
      c.block_r = blockR;
      c.num_warps = warps;
      if (seen.insert(c.canonical()).second) {
        out.emplace_back(c);
      }
    }
  }
}

std::vector<OptionConfig> reductionCandidates(const OpSignature& sig, const CodegenContext&) {
  const int64_t r = reducedExtent(sig);
  const int64_t rows = outputRows(sig);
  std::vector<OptionConfig> out;
  std::set<std::string> seen;
  const int64_t single = std::max<int64_t>(16, nextPow2(r));
  if (single <= kMaxBlockR) {
    addReductionCandidates(out, seen, {1, 2, 8, 32}, single, rows);
  }
  for (int64_t br : {512, 1024, 2048}) {
    if (br < r) {
      addReductionCandidates(out, seen, {1, 4}, br, rows);
    }
  }
  return out;
}

FusionContract tritonReductionContract(const OpSignature&) {
  FusionContract c;
  c.prologue_operands = {0};
  c.epilogue = true;
  c.allowed_ops = {RegionOpKind::LoadSelf,
                   RegionOpKind::LoadExtra,
                   RegionOpKind::Constant,
                   RegionOpKind::Add,
                   RegionOpKind::Mul,
                   RegionOpKind::Relu};
  c.max_extra_inputs = 4;
  c.allow_broadcast = true;
  return c;
}

OptionDescriptor tritonReduction() {
  OptionDescriptor o;
  o.id = "triton_row_reduction";
  o.version = 1;
  o.op_kinds = {OpKind::SumDim, OpKind::MeanDim};
  // Blocked tree summation differs from the eager reduction order.
  o.fence = NumericsFence::InductorOnly;
  o.strategy = CodegenStrategy::TritonReduction;
  o.config_kind = ConfigKind::Reduction;
  o.effects = "pure: reads input/extra inputs, writes every output element exactly once";
  o.fusion_contract = tritonReductionContract;
  o.check_eligible = [](const OpSignature& sig, const CodegenContext&) {
    requireCommonTensorAbi(sig, DeviceKind::Cuda);
    if (sig.reduction().dims.size() != 1) {
      reject(RejectReason::Rank, "triton_row_reduction reduces exactly one dim");
    }
    if (!sig.output().layout.isNonOverlapping()) {
      reject(RejectReason::Layout, "output layout overlaps (stores would not be injective)");
    }
  };
  o.candidates = reductionCandidates;
  o.check_config = checkReductionConfig;
  return o;
}

} // namespace

void registerReductionOptions(OptionRegistry& registry) {
  registry.add(nativeReduction());
  registry.add(tritonReduction());
}

} // namespace torch::inductor::loops_ir::op_options
