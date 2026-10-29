// GEMM implementation options: eager-identical native ATen delegation,
// native mm + bias epilogue as one aten.addmm call, and a generated tiled
// Triton GEMM with prologue/epilogue fusion.

#include <algorithm>
#include <set>

#include "option_registry.h"

namespace torch::inductor::loops_ir::op_options {

namespace {

constexpr int64_t kGemmTileElementBudget = 128 * 256;

// Stock Inductor CUDA mm manifest (block_m, block_n, block_k, stages, warps).
struct GemmManifestEntry {
  int64_t block_m, block_n, block_k, num_stages, num_warps;
};
const std::vector<GemmManifestEntry>& gemmManifest() {
  static const std::vector<GemmManifestEntry> m{
      {32, 32, 16, 1, 2},    {32, 32, 128, 2, 4},  {32, 64, 32, 5, 8},   {64, 32, 32, 5, 8},
      {64, 32, 128, 5, 4},   {64, 64, 16, 2, 4},   {64, 64, 32, 2, 4},   {64, 64, 64, 3, 8},
      {64, 64, 128, 5, 4},   {64, 128, 32, 3, 4},  {64, 128, 32, 4, 8},  {64, 128, 64, 3, 4},
      {64, 128, 128, 4, 4},  {128, 64, 32, 3, 4},  {128, 64, 32, 4, 8},  {128, 128, 32, 2, 8},
      {128, 128, 32, 3, 4},  {128, 128, 64, 3, 4}, {128, 128, 64, 5, 8}, {128, 128, 128, 4, 8},
      {128, 256, 64, 4, 8},
  };
  return m;
}

int64_t clampBlock(int64_t block, int64_t dim) {
  int64_t p = 16;
  while (p < dim && p < block) {
    p *= 2;
  }
  return std::min(block, p);
}

void requireOutputLayout(const OpSignature& sig, bool contiguous) {
  const TensorLayout& out = sig.output().layout;
  if (contiguous && !out.isContiguous()) {
    reject(RejectReason::Layout, "native eager result is contiguous; output layout must be contiguous");
  }
  if (!out.isNonOverlapping()) {
    reject(RejectReason::Layout, "output layout overlaps (stores would not be injective)");
  }
}

NativeTarget gemmTarget(OpKind kind) {
  switch (kind) {
    case OpKind::Mm:
      return NativeTarget::AtenMm;
    case OpKind::Addmm:
      return NativeTarget::AtenAddmm;
    default:
      return NativeTarget::AtenBmm;
  }
}

std::vector<OptionConfig> nativeCandidates(const OpSignature&, const CodegenContext&) {
  return {NativeConfig{}};
}

FusionContract noFusion(const OpSignature&) {
  return FusionContract{};
}

OptionDescriptor nativeGemm() {
  OptionDescriptor o;
  o.id = "native_aten_gemm";
  o.version = 1;
  o.op_kinds = {OpKind::Mm, OpKind::Addmm, OpKind::Bmm};
  o.fence = NumericsFence::StrictEager;
  o.strategy = CodegenStrategy::NativeAten;
  o.config_kind = ConfigKind::Native;
  o.effects = "pure: reads operands, returns a fresh eager-allocated output; no input aliasing";
  o.fusion_contract = noFusion;
  o.check_eligible = [](const OpSignature& sig, const CodegenContext&) {
    requireCommonTensorAbi(sig, sig.device());
    requireOutputLayout(sig, true);
  };
  o.candidates = nativeCandidates;
  o.native_plan = [](const OpSignature& sig, const ValidatedFusion&) {
    NativeCallPlan p;
    p.target = gemmTarget(sig.kind());
    for (size_t i = 0; i < sig.inputs().size(); ++i) {
      p.operand_slots.push_back(static_cast<int64_t>(i));
    }
    p.alpha = sig.matmul().alpha;
    p.beta = sig.matmul().beta;
    return p;
  };
  return o;
}

// Right-aligned broadcast of `t` to `target` as an access map, or an
// impossible map when not broadcastable.
AccessMap broadcastMap(const TensorLayout& t, const std::vector<int64_t>& target) {
  AccessMap m;
  m.coeffs.assign(target.size(), 0);
  if (t.rank() > target.size()) {
    m.offset = -1;
    return m;
  }
  const size_t lead = target.size() - t.rank();
  for (size_t d = 0; d < t.rank(); ++d) {
    if (t.sizes[d] != 1 && t.sizes[d] != target[lead + d]) {
      m.offset = -1;
      return m;
    }
    m.coeffs[lead + d] = (t.sizes[d] > 1 && target[lead + d] > 1) ? t.strides[d] : 0;
  }
  return m;
}

// Epilogue must be exactly out = acc + bias (either operand order).
NativeBiasView matchBiasEpilogue(const OpSignature& sig, const ValidatedFusion& f) {
  if (!f.epilogue) {
    reject(RejectReason::Fusion, "native_aten_mm_bias_epilogue requires a bias epilogue");
  }
  const LoweredRegion& r = *f.epilogue;
  const RegionOp* self = nullptr;
  const RegionOp* bias = nullptr;
  const RegionOp* add = nullptr;
  for (const auto& op : r.ops) {
    if (op.kind == RegionOpKind::LoadSelf && !self) {
      self = &op;
    } else if (op.kind == RegionOpKind::LoadExtra && !bias) {
      bias = &op;
    } else if (op.kind == RegionOpKind::Add && !add) {
      add = &op;
    } else {
      reject(RejectReason::Fusion, "native addmm epilogue must be exactly acc + bias");
    }
  }
  const bool operandsMatch = self && bias && add &&
      ((add->lhs == self->id && add->rhs == bias->id) || (add->lhs == bias->id && add->rhs == self->id));
  if (!operandsMatch || r.store_value != add->id) {
    reject(RejectReason::Fusion, "native addmm epilogue must store acc + bias");
  }
  if (r.store_offset != 0) {
    reject(RejectReason::Fusion, "native addmm returns a fresh tensor; the fused output must start at offset 0");
  }
  NativeBiasView v;
  v.extra = bias->extra;
  v.direct = bias->access == broadcastMap(f.extra_inputs.at(static_cast<size_t>(bias->extra)).layout,
                                          sig.output().layout.sizes);
  v.sizes = sig.output().layout.sizes;
  v.strides = bias->access.coeffs;
  v.offset = bias->access.offset;
  return v;
}

OptionDescriptor nativeMmBiasEpilogue() {
  OptionDescriptor o;
  o.id = "native_aten_mm_bias_epilogue";
  o.version = 1;
  o.op_kinds = {OpKind::Mm};
  // aten.addmm selects a different vendor kernel/epilogue than mm followed by
  // add, so the rewrite is not eager-identical.
  o.fence = NumericsFence::InductorOnly;
  o.strategy = CodegenStrategy::NativeAten;
  o.config_kind = ConfigKind::Native;
  o.effects = "pure: one eager aten.addmm call on (bias view, a, b); fresh output";
  o.fusion_contract = [](const OpSignature&) {
    FusionContract c;
    c.epilogue = true;
    c.allowed_ops = {RegionOpKind::LoadSelf, RegionOpKind::LoadExtra, RegionOpKind::Add};
    c.max_extra_inputs = 1;
    c.allow_broadcast = true;
    return c;
  };
  o.check_eligible = [](const OpSignature& sig, const CodegenContext&) {
    requireCommonTensorAbi(sig, sig.device());
    requireOutputLayout(sig, true);
  };
  o.candidates = nativeCandidates;
  o.native_plan = [](const OpSignature& sig, const ValidatedFusion& f) {
    NativeCallPlan p;
    p.target = NativeTarget::AtenAddmm;
    p.operand_slots = {0, 1};
    p.bias_view = matchBiasEpilogue(sig, f);
    return p;
  };
  return o;
}

void checkGemmConfig(const OpSignature& sig, const OptionConfig& config, const CodegenContext& ctx) {
  const auto& c = std::get<MatmulConfig>(config);
  const int64_t smem = c.num_stages * (c.block_m * c.block_k + c.block_k * c.block_n) * 4;
  if (smem > ctx.max_shared_memory_bytes) {
    reject(RejectReason::Config,
           "pipelined tiles need " + std::to_string(smem) + " bytes of shared memory > device limit " +
               std::to_string(ctx.max_shared_memory_bytes));
  }
  if (c.block_m * c.block_n > kGemmTileElementBudget) {
    reject(RejectReason::Config, "block_m*block_n exceeds the accumulator register budget");
  }
  if (c.matrix_precision == MatmulPrecision::Tf32 && !stockAllowsTf32(sig, ctx)) {
    reject(RejectReason::Config, "matrix_precision=tf32 but stock effective settings require ieee fp32");
  }
}

std::vector<OptionConfig> gemmCandidates(const OpSignature& sig, const CodegenContext& ctx) {
  const GemmDims d = sig.gemmDims();
  const MatmulPrecision precision = stockAllowsTf32(sig, ctx) ? MatmulPrecision::Tf32 : MatmulPrecision::Ieee;
  std::vector<OptionConfig> out;
  std::set<std::string> seen;
  for (const auto& e : gemmManifest()) {
    MatmulConfig c;
    c.block_m = clampBlock(e.block_m, d.m);
    c.block_n = clampBlock(e.block_n, d.n);
    c.block_k = clampBlock(e.block_k, d.k);
    c.group_m = 8;
    c.num_stages = e.num_stages;
    c.num_warps = e.num_warps;
    c.matrix_precision = precision;
    try {
      checkGemmConfig(sig, c, ctx);
    } catch (const OptionRejected&) {
      continue;
    }
    if (seen.insert(c.canonical()).second) {
      out.emplace_back(c);
    }
  }
  return out;
}

FusionContract tritonGemmContract(const OpSignature& sig) {
  FusionContract c;
  c.prologue_operands = {static_cast<int64_t>(sig.gemmOperandA()), static_cast<int64_t>(sig.gemmOperandB())};
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

OptionDescriptor tritonGemm() {
  OptionDescriptor o;
  o.id = "triton_tiled_gemm";
  o.version = 1;
  o.op_kinds = {OpKind::Mm, OpKind::Addmm, OpKind::Bmm};
  // Tile-order K accumulation differs from the vendor kernel; no formal
  // IEEE tree-parity proof exists.
  o.fence = NumericsFence::InductorOnly;
  o.strategy = CodegenStrategy::TritonGemm;
  o.config_kind = ConfigKind::Matmul;
  o.effects = "pure: reads operands/extra inputs, writes every output element exactly once";
  o.fusion_contract = tritonGemmContract;
  o.check_eligible = [](const OpSignature& sig, const CodegenContext& ctx) {
    requireCommonTensorAbi(sig, DeviceKind::Cuda);
    requireOutputLayout(sig, false);
    if (sig.kind() == OpKind::Addmm) {
      (void)addmmEagerPlan(sig, ctx);
    }
  };
  o.candidates = gemmCandidates;
  o.check_config = checkGemmConfig;
  return o;
}

// isInputCompliesAddmmCudaLt for f32, no activation, fresh contiguous
// 2-D result.
bool takesLtBiasPath(const OpSignature& sig, const GemmDims& d) {
  const TensorLayout& bias = sig.inputs()[0].layout;
  size_t nonUnit = 0;
  for (int64_t s : bias.sizes) {
    nonUnit += s != 1 ? 1 : 0;
  }
  const bool squeezesTo1d = bias.rank() == 1 || nonUnit == 1;
  return sig.matmul().beta == 1.0 && bias.isContiguous() && squeezesTo1d && bias.rank() >= 1 &&
      bias.sizes.back() == d.n && d.k > 1 && d.n > 1;
}

} // namespace

const char* addmmPathName(AddmmPath path) {
  switch (path) {
    case AddmmPath::BetaZero:
      return "beta_zero";
    case AddmmPath::EmptyK:
      return "empty_k";
    case AddmmPath::LtBias:
      return "lt_bias";
    case AddmmPath::Gemm:
      return "gemm";
  }
  return "?";
}

std::string AddmmEagerPlan::canonical() const {
  return std::string(addmmPathName(path)) + " read=" + std::to_string(read_operands) +
      " scale_product=" + std::to_string(scale_product) + " bias=" + std::to_string(load_bias) +
      " scale_bias=" + std::to_string(scale_bias) + " plus_zero=" + std::to_string(add_positive_zero);
}

AddmmEagerPlan addmmEagerPlan(const OpSignature& sig, const CodegenContext& ctx) {
  if (ctx.addmm_lt_policy == AddmmLtPolicy::Unknown) {
    reject(RejectReason::Context, "eager addmm vendor dispatch is not modelled for this context");
  }
  const double alpha = sig.matmul().alpha;
  const double beta = sig.matmul().beta;
  const GemmDims d = sig.gemmDims();
  AddmmEagerPlan p;
  if (beta == 0.0) {
    p.path = AddmmPath::BetaZero;
    p.read_operands = alpha != 0.0 && d.k > 0;
  } else if (d.k == 0) {
    p.path = AddmmPath::EmptyK;
    p.read_operands = false;
    p.load_bias = true;
    p.scale_bias = true;
  } else if (ctx.addmm_lt_policy == AddmmLtPolicy::CudaDefault && takesLtBiasPath(sig, d)) {
    p.path = AddmmPath::LtBias;
    p.load_bias = true;
  } else {
    p.path = AddmmPath::Gemm;
    p.read_operands = alpha != 0.0;
    p.load_bias = true;
    p.scale_bias = beta != 1.0;
    p.add_positive_zero = alpha == 0.0 && beta != 1.0;
  }
  p.scale_product = p.read_operands && alpha != 1.0;
  return p;
}

void registerGemmOptions(OptionRegistry& registry) {
  registry.add(nativeGemm());
  registry.add(nativeMmBiasEpilogue());
  registry.add(tritonGemm());
}

} // namespace torch::inductor::loops_ir::op_options
