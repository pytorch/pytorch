#pragma once
// Typed semantic operation signatures and frozen codegen context for the
// per-operation implementation-option registry. C++17, no torch/c10 deps.
// Only depends on the core LoopsIR header for shared dtype/numerics names.

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace torch::inductor::loops_ir::op_options {

// Semantic operation families. VarCorrection/StdCorrection/Softmax/
// LogSoftmax are SDK-reserved: signatures can be described, but no option
// implements them yet and the registry never lists one for them.
enum class OpKind { Mm, Addmm, Bmm, SumDim, MeanDim, VarCorrection, StdCorrection, Softmax, LogSoftmax };
const char* opKindName(OpKind kind);
bool isReservedOpKind(OpKind kind);

enum class StorageDtype { F16, BF16, F32, F64, I32, I64, Bool };
const char* storageDtypeName(StorageDtype dtype);
// Core LoopsIR boundary dtype spelling ("f32", "bf16", ...).
std::string coreDtypeName(StorageDtype dtype);

enum class ShapeKind { Static, Dynamic };
enum class DeviceKind { Cuda, Cpu };
const char* deviceKindName(DeviceKind device);

// Same spellings as core strictNumericsName()/inductorNumericsName().
enum class NumericsPolicy { StrictEager, Inductor };
const char* numericsPolicyName(NumericsPolicy policy);

enum class MatmulPrecision { Ieee, Tf32 };
const char* matmulPrecisionName(MatmulPrecision precision);

// Frozen eager aten.addmm vendor dispatch (aten/src/ATen/native/cuda/Blas.cpp,
// non-ROCm): CudaDefault = cuBLASLt bias path enabled, CudaDisabled =
// DISABLE_ADDMM_CUDA_LT=1, Unknown = semantics not modelled (e.g. ROCm).
enum class AddmmLtPolicy { CudaDefault, CudaDisabled, Unknown };
const char* addmmLtPolicyName(AddmmLtPolicy policy);

enum class RejectReason {
  UnknownOption,
  OpKind,
  Device,
  Dtype,
  DynamicShape,
  Rank,
  Layout,
  NumericsFence,
  Config,
  Fusion,
  Context,
  Signature,
  Alias,
};
const char* rejectReasonName(RejectReason reason);

// Typed rejection. what() is "<reason>: <detail>".
class OptionRejected : public std::invalid_argument {
 public:
  OptionRejected(RejectReason reason, const std::string& detail);
  RejectReason reason() const {
    return reason_;
  }

 private:
  RejectReason reason_;
};

[[noreturn]] void reject(RejectReason reason, const std::string& detail);

// Physical tensor ABI: sizes/strides in elements relative to the data
// pointer handed to the kernel (storage offset already applied).
struct TensorLayout {
  StorageDtype dtype = StorageDtype::F32;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  ShapeKind shape_kind = ShapeKind::Static;

  size_t rank() const {
    return sizes.size();
  }
  int64_t numel() const;
  // 1 + max reachable element offset, or 0 for an empty tensor.
  int64_t extent() const;
  bool isContiguous() const;
  // Sufficient non-overlap proof (sorted-stride nesting); stores need it.
  bool isNonOverlapping() const;
  // Rank/stride/size consistency; never checks option-specific limits.
  void validate(const std::string& ctx) const;
  std::string canonical() const;
};

struct TensorOperand {
  std::string name;
  TensorLayout layout;
};

// alpha/beta keep the exact f64 bits supplied by the frontend.
struct MatmulAttrs {
  double alpha = 1.0;
  double beta = 1.0;
};

// dims are normalized (non-negative, sorted, unique) by the factory.
struct ReductionAttrs {
  std::vector<int64_t> dims;
  bool keepdim = false;
  std::optional<double> correction;
};

using OpAttrs = std::variant<MatmulAttrs, ReductionAttrs>;

struct GemmDims {
  int64_t batch = 1;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
};

// Semantic operation (attributes only, no tunables). Input order:
//   Mm:    {a [M,K], b [K,N]}
//   Addmm: {bias (broadcastable to [M,N]), a [M,K], b [K,N]}
//   Bmm:   {a [B,M,K], b [B,K,N]}
//   reductions: {x}
class OpSignature {
 public:
  static OpSignature Mm(DeviceKind device, TensorOperand a, TensorOperand b, TensorOperand out);
  static OpSignature
  Addmm(DeviceKind device, TensorOperand bias, TensorOperand a, TensorOperand b, TensorOperand out, double alpha, double beta);
  static OpSignature Bmm(DeviceKind device, TensorOperand a, TensorOperand b, TensorOperand out);
  // SumDim/MeanDim/VarCorrection/StdCorrection/Softmax/LogSoftmax.
  static OpSignature Reduction(OpKind kind,
                               DeviceKind device,
                               TensorOperand x,
                               std::vector<int64_t> dims,
                               bool keepdim,
                               std::optional<double> correction,
                               TensorOperand out);

  OpKind kind() const {
    return kind_;
  }
  DeviceKind device() const {
    return device_;
  }
  const std::vector<TensorOperand>& inputs() const {
    return inputs_;
  }
  const TensorOperand& output() const {
    return output_;
  }
  const OpAttrs& attrs() const {
    return attrs_;
  }
  bool isMatmul() const;
  bool isReduction() const;
  const MatmulAttrs& matmul() const;
  const ReductionAttrs& reduction() const;
  // Operand slots of A and B (1/2 for addmm, 0/1 otherwise).
  size_t gemmOperandA() const;
  size_t gemmOperandB() const;
  GemmDims gemmDims() const;
  // Same semantic op writing a different output operand (epilogue fusion):
  // sizes/dtype must match, the name must not collide with an input.
  OpSignature withOutput(TensorOperand out) const;
  // Deterministic text (f64 attributes as exact bits).
  std::string canonical() const;

 private:
  OpSignature() = default;
  OpKind kind_ = OpKind::Mm;
  DeviceKind device_ = DeviceKind::Cuda;
  std::vector<TensorOperand> inputs_;
  TensorOperand output_;
  OpAttrs attrs_;
};

// Numerical/device settings frozen once at compile time by the frontend.
// Codegen reads only this, never process globals.
struct CodegenContext {
  NumericsPolicy numerics = NumericsPolicy::StrictEager;
  DeviceKind device = DeviceKind::Cuda;
  int64_t compute_capability = 0;
  int64_t max_shared_memory_bytes = 0;
  // Stock torch.backends.cuda.matmul.fp32_precision ("tf32" -> Tf32).
  MatmulPrecision stock_fp32_matmul_precision = MatmulPrecision::Ieee;
  bool enable_fp_fusion = false;
  bool disable_ftz = true;
  bool division_rounding = false;
  int64_t device_index = 0;
  AddmmLtPolicy addmm_lt_policy = AddmmLtPolicy::CudaDefault;

  void validate() const;
  std::string canonical() const;
};

// Stock Inductor's effective GEMM TF32 rule (mm_allow_tf32 on CUDA):
// fp32_precision == "tf32" and M >= 16 and min(N, K) >= 512.
bool stockAllowsTf32(const OpSignature& sig, const CodegenContext& ctx);

std::string f64Bits(double value);
uint64_t fnv1a64(const std::string& text);
std::string hex64(uint64_t value);
int64_t checkedMul64(int64_t a, int64_t b, const std::string& ctx);
int64_t checkedAdd64(int64_t a, int64_t b, const std::string& ctx);

} // namespace torch::inductor::loops_ir::op_options
