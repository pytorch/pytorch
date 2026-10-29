#include "option_registry.h"

#include <algorithm>
#include <sstream>

namespace torch::inductor::loops_ir::op_options {

const char* codegenStrategyName(CodegenStrategy strategy) {
  switch (strategy) {
    case CodegenStrategy::NativeAten:
      return "native_aten";
    case CodegenStrategy::TritonGemm:
      return "triton_gemm";
    case CodegenStrategy::TritonReduction:
      return "triton_reduction";
  }
  return "?";
}

const char* numericsFenceName(NumericsFence fence) {
  return fence == NumericsFence::StrictEager ? "strict_eager" : "inductor_only";
}

const char* nativeTargetName(NativeTarget target) {
  switch (target) {
    case NativeTarget::AtenMm:
      return "aten.mm.default";
    case NativeTarget::AtenAddmm:
      return "aten.addmm.default";
    case NativeTarget::AtenBmm:
      return "aten.bmm.default";
    case NativeTarget::AtenSumDimIntList:
      return "aten.sum.dim_IntList";
    case NativeTarget::AtenMeanDim:
      return "aten.mean.dim";
  }
  return "?";
}

std::string NativeCallPlan::canonical() const {
  std::ostringstream os;
  os << nativeTargetName(target) << " slots[";
  for (int64_t s : operand_slots) {
    os << s << ",";
  }
  os << "]";
  if (bias_view) {
    os << " bias_view(x" << bias_view->extra << ",direct=" << bias_view->direct << ",sizes";
    for (int64_t s : bias_view->sizes) {
      os << "," << s;
    }
    os << ",strides";
    for (int64_t s : bias_view->strides) {
      os << "," << s;
    }
    os << ",off" << bias_view->offset << ")";
  }
  os << " alpha=0x" << f64Bits(alpha) << " beta=0x" << f64Bits(beta) << " dims[";
  for (int64_t d : dims) {
    os << d << ",";
  }
  os << "] keepdim=" << keepdim;
  return os.str();
}

bool OptionDescriptor::implements(OpKind kind) const {
  return std::find(op_kinds.begin(), op_kinds.end(), kind) != op_kinds.end();
}

FusionContract OptionDescriptor::fusionFor(const OpSignature& sig) const {
  requireOpKind(*this, sig);
  return fusion_contract(sig);
}

void requireOpKind(const OptionDescriptor& option, const OpSignature& sig) {
  if (!option.implements(sig.kind())) {
    reject(RejectReason::OpKind,
           "option '" + option.id + "' does not implement op kind '" + opKindName(sig.kind()) + "'");
  }
}

void requireCommonTensorAbi(const OpSignature& sig, DeviceKind device) {
  if (sig.device() != device) {
    reject(RejectReason::Device, std::string("option requires device ") + deviceKindName(device));
  }
  std::vector<const TensorOperand*> all;
  for (const auto& in : sig.inputs()) {
    all.push_back(&in);
  }
  all.push_back(&sig.output());
  for (const TensorOperand* t : all) {
    if (t->layout.dtype != StorageDtype::F32) {
      reject(RejectReason::Dtype,
             "operand '" + t->name + "' has dtype " + storageDtypeName(t->layout.dtype) +
                 " (options are f32-only; no storage casts)");
    }
    if (t->layout.shape_kind != ShapeKind::Static) {
      reject(RejectReason::DynamicShape, "operand '" + t->name + "' has a dynamic shape");
    }
  }
}

namespace {

void checkFenceAndEligibility(const OptionDescriptor& opt, const OpSignature& sig, const CodegenContext& ctx) {
  requireOpKind(opt, sig);
  if (opt.fence == NumericsFence::InductorOnly && ctx.numerics == NumericsPolicy::StrictEager) {
    reject(RejectReason::NumericsFence,
           "option '" + opt.id + "' reassociates/rounds differently from eager; not eligible under strict_eager");
  }
  if (ctx.device != sig.device()) {
    reject(RejectReason::Device, "context device does not match signature device");
  }
  opt.check_eligible(sig, ctx);
}

std::string selectionCanonical(const OptionDescriptor& opt, const ValidatedSelection& s) {
  std::ostringstream os;
  os << "option=" << opt.id << "@v" << opt.version << " strategy=" << codegenStrategyName(opt.strategy)
     << " fence=" << numericsFenceName(opt.fence) << "\ncontract=" << opt.fusionFor(s.signature).canonical()
     << "\nsig=" << s.signature.canonical()
     << "\nconfig=" << canonicalConfig(s.config) << "\nfusion=" << s.fusion.canonical()
     << "\ncontext=" << s.context.canonical();
  if (s.native) {
    os << "\nnative=" << s.native->canonical();
  }
  if (s.addmm) {
    os << "\naddmm=" << s.addmm->canonical();
  }
  return os.str();
}

} // namespace

void OptionRegistry::add(OptionDescriptor option) {
  if (option.id.empty() || option.op_kinds.empty() || !option.check_eligible || !option.candidates ||
      !option.fusion_contract) {
    throw std::invalid_argument("OptionRegistry.add: incomplete option descriptor");
  }
  for (OpKind k : option.op_kinds) {
    if (isReservedOpKind(k)) {
      throw std::invalid_argument("OptionRegistry.add: op kind is SDK-reserved");
    }
  }
  if ((option.strategy == CodegenStrategy::NativeAten) != static_cast<bool>(option.native_plan)) {
    throw std::invalid_argument("OptionRegistry.add: native_plan required exactly for native options");
  }
  for (const auto& o : options_) {
    if (o.id == option.id) {
      throw std::invalid_argument("OptionRegistry.add: duplicate option id '" + option.id + "'");
    }
  }
  options_.push_back(std::move(option));
}

const OptionDescriptor& OptionRegistry::get(const std::string& id) const {
  for (const auto& o : options_) {
    if (o.id == id) {
      return o;
    }
  }
  reject(RejectReason::UnknownOption, "no registered option '" + id + "'");
}

std::vector<std::string> OptionRegistry::optionIds() const {
  std::vector<std::string> ids;
  for (const auto& o : options_) {
    ids.push_back(o.id);
  }
  return ids;
}

std::vector<std::string> OptionRegistry::optionsFor(OpKind kind) const {
  std::vector<std::string> ids;
  for (const auto& o : options_) {
    if (o.implements(kind)) {
      ids.push_back(o.id);
    }
  }
  return ids;
}

std::vector<std::string> OptionRegistry::eligibleOptions(const OpSignature& sig, const CodegenContext& ctx) const {
  ctx.validate();
  std::vector<std::string> ids;
  for (const auto& o : options_) {
    if (!o.implements(sig.kind())) {
      continue;
    }
    try {
      checkFenceAndEligibility(o, sig, ctx);
      ids.push_back(o.id);
    } catch (const OptionRejected&) {
    }
  }
  return ids;
}

std::vector<OptionConfig> OptionRegistry::candidateConfigs(const std::string& id,
                                                           const OpSignature& sig,
                                                           const CodegenContext& ctx) const {
  const OptionDescriptor& opt = get(id);
  ctx.validate();
  checkFenceAndEligibility(opt, sig, ctx);
  std::vector<OptionConfig> configs = opt.candidates(sig, ctx);
  for (const auto& c : configs) {
    if (configKind(c) != opt.config_kind) {
      throw std::logic_error("candidate generator for '" + id + "' produced the wrong config kind");
    }
    validateDomain(c);
    if (opt.check_config) {
      opt.check_config(sig, c, ctx);
    }
  }
  return configs;
}

ValidatedSelection OptionRegistry::validate(const std::string& id,
                                            const OpSignature& sig,
                                            const OptionConfig& config,
                                            const FusedLoops& fused,
                                            const CodegenContext& ctx) const {
  const OptionDescriptor& opt = get(id);
  ctx.validate();
  checkFenceAndEligibility(opt, sig, ctx);
  if (configKind(config) != opt.config_kind) {
    reject(RejectReason::Config,
           "option '" + id + "' takes a " + configKindName(opt.config_kind) + " config, got " +
               configKindName(configKind(config)));
  }
  validateDomain(config);
  if (opt.check_config) {
    opt.check_config(sig, config, ctx);
  }
  ValidatedSelection s(sig, config, ctx);
  s.option_id = opt.id;
  s.option_version = opt.version;
  s.strategy = opt.strategy;
  s.fence = opt.fence;
  s.fusion = validateFusion(opt.fusionFor(sig), sig, fused);
  if (opt.native_plan) {
    s.native = opt.native_plan(sig, s.fusion);
  }
  if (opt.strategy == CodegenStrategy::TritonGemm && sig.kind() == OpKind::Addmm) {
    s.addmm = addmmEagerPlan(sig, ctx);
  }
  s.canonical = selectionCanonical(opt, s);
  s.cache_key = hex64(fnv1a64(s.canonical));
  return s;
}

const OptionRegistry& defaultRegistry() {
  static const OptionRegistry registry = [] {
    OptionRegistry r;
    registerGemmOptions(r);
    registerReductionOptions(r);
    return r;
  }();
  return registry;
}

} // namespace torch::inductor::loops_ir::op_options
