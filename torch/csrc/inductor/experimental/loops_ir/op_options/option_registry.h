#pragma once
// Per-operation implementation-option registry. An option declares a stable
// id/version, the op kinds it implements, its numerics fence, eligibility,
// a bounded candidate-configuration generator, its fusion contract and the
// codegen strategy (backend dialect) that renders a validated selection.
// The registry only lists options that have a real codegen strategy.

#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "fused_region.h"
#include "op_config.h"
#include "op_types.h"

namespace torch::inductor::loops_ir::op_options {

// Backend dialect mapped to a Python codegen function by the frontend.
enum class CodegenStrategy { NativeAten, TritonGemm, TritonReduction };
const char* codegenStrategyName(CodegenStrategy strategy);

// StrictEager: valid under both policies (eager-identical). InductorOnly:
// generated reassociation/rounding; never eligible under strict_eager.
enum class NumericsFence { StrictEager, InductorOnly };
const char* numericsFenceName(NumericsFence fence);

enum class NativeTarget { AtenMm, AtenAddmm, AtenBmm, AtenSumDimIntList, AtenMeanDim };
const char* nativeTargetName(NativeTarget target);

// Extra input fed to a native call as addmm self. `direct`: the epilogue
// map is exactly the tensor's own right-aligned broadcast, so the tensor is
// passed unchanged; otherwise a strided [M,N] view (sizes/strides/offset).
struct NativeBiasView {
  int64_t extra = -1;
  bool direct = false;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  int64_t offset = 0;
};

// Exact eager call: target overload plus ordered signature input slots and
// preserved scalar attributes.
struct NativeCallPlan {
  NativeTarget target = NativeTarget::AtenMm;
  std::vector<int64_t> operand_slots;
  std::optional<NativeBiasView> bias_view;
  double alpha = 1.0;
  double beta = 1.0;
  std::vector<int64_t> dims;
  bool keepdim = false;
  std::string canonical() const;
};

// Eager CUDA aten.addmm result formula for one signature under the frozen
// vendor policy (Blas.cpp addmm_out_cuda_impl, fresh contiguous result):
//   BetaZero : beta == +-0  -> alpha*(A@B), or +0 when alpha == 0 or K == 0
//   EmptyK   : K == 0       -> beta*bias (signed zeros kept)
//   LtBias   : cuBLASLt bias epilogue (beta == 1, contiguous bias squeezing
//              to [N], K > 1, N > 1) -> alpha*(A@B) + bias; A/B read even
//              when alpha == 0 (NaN/Inf propagate)
//   Gemm     : bias copied into the result, then gemm -> alpha*(A@B) +
//              beta*bias; when alpha == 0 A/B are never read and the result
//              is bias (beta == 1) or beta*bias + 0 (-0 becomes +0)
enum class AddmmPath { BetaZero, EmptyK, LtBias, Gemm };
const char* addmmPathName(AddmmPath path);

struct AddmmEagerPlan {
  AddmmPath path = AddmmPath::Gemm;
  bool read_operands = true; // false: no A/B loads, prologues or dot
  bool scale_product = false; // multiply the product by alpha
  bool load_bias = false;
  bool scale_bias = false; // multiply the bias by beta
  bool add_positive_zero = false; // Gemm alpha == 0, beta != 1
  std::string canonical() const;
};

// Throws OptionRejected(Context) when the policy is Unknown.
AddmmEagerPlan addmmEagerPlan(const OpSignature& sig, const CodegenContext& ctx);

struct ValidatedSelection {
  ValidatedSelection(OpSignature sig, OptionConfig cfg, CodegenContext ctx)
      : signature(std::move(sig)), config(std::move(cfg)), context(ctx) {}
  std::string option_id;
  int64_t option_version = 0;
  CodegenStrategy strategy = CodegenStrategy::NativeAten;
  NumericsFence fence = NumericsFence::StrictEager;
  OpSignature signature;
  OptionConfig config;
  ValidatedFusion fusion;
  CodegenContext context;
  std::optional<NativeCallPlan> native;
  std::optional<AddmmEagerPlan> addmm; // generated GEMM options on addmm
  std::string canonical;
  std::string cache_key;
};

struct OptionDescriptor {
  std::string id;
  int64_t version = 1;
  std::vector<OpKind> op_kinds;
  NumericsFence fence = NumericsFence::StrictEager;
  CodegenStrategy strategy = CodegenStrategy::NativeAten;
  ConfigKind config_kind = ConfigKind::Native;
  std::string effects;
  std::function<FusionContract(const OpSignature&)> fusion_contract;
  std::function<void(const OpSignature&, const CodegenContext&)> check_eligible;
  std::function<std::vector<OptionConfig>(const OpSignature&, const CodegenContext&)> candidates;
  std::function<void(const OpSignature&, const OptionConfig&, const CodegenContext&)> check_config;
  // Native options only: map the validated fusion to an exact eager call.
  std::function<NativeCallPlan(const OpSignature&, const ValidatedFusion&)> native_plan;

  bool implements(OpKind kind) const;
  FusionContract fusionFor(const OpSignature& sig) const;
};

class OptionRegistry {
 public:
  void add(OptionDescriptor option);
  const OptionDescriptor& get(const std::string& id) const;
  std::vector<std::string> optionIds() const;
  std::vector<std::string> optionsFor(OpKind kind) const;
  // Options passing fence + eligibility for this signature/context.
  std::vector<std::string> eligibleOptions(const OpSignature& sig, const CodegenContext& ctx) const;
  // Bounded, deterministic, every entry passes validate() with no fusion.
  std::vector<OptionConfig> candidateConfigs(const std::string& id,
                                             const OpSignature& sig,
                                             const CodegenContext& ctx) const;
  ValidatedSelection validate(const std::string& id,
                              const OpSignature& sig,
                              const OptionConfig& config,
                              const FusedLoops& fused,
                              const CodegenContext& ctx) const;

 private:
  std::vector<OptionDescriptor> options_;
};

// Shared eligibility building blocks for option factories.
void requireOpKind(const OptionDescriptor& option, const OpSignature& sig);
void requireCommonTensorAbi(const OpSignature& sig, DeviceKind device);

void registerGemmOptions(OptionRegistry& registry);
void registerReductionOptions(OptionRegistry& registry);
const OptionRegistry& defaultRegistry();

} // namespace torch::inductor::loops_ir::op_options
