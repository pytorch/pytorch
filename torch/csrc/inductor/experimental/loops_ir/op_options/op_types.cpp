#include "op_types.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <sstream>

namespace torch::inductor::loops_ir::op_options {

const char* opKindName(OpKind kind) {
  switch (kind) {
    case OpKind::Mm:
      return "mm";
    case OpKind::Addmm:
      return "addmm";
    case OpKind::Bmm:
      return "bmm";
    case OpKind::SumDim:
      return "sum_dim";
    case OpKind::MeanDim:
      return "mean_dim";
    case OpKind::VarCorrection:
      return "var_correction";
    case OpKind::StdCorrection:
      return "std_correction";
    case OpKind::Softmax:
      return "softmax";
    case OpKind::LogSoftmax:
      return "log_softmax";
  }
  return "?";
}

bool isReservedOpKind(OpKind kind) {
  return kind == OpKind::VarCorrection || kind == OpKind::StdCorrection || kind == OpKind::Softmax ||
      kind == OpKind::LogSoftmax;
}

const char* storageDtypeName(StorageDtype dtype) {
  switch (dtype) {
    case StorageDtype::F16:
      return "f16";
    case StorageDtype::BF16:
      return "bf16";
    case StorageDtype::F32:
      return "f32";
    case StorageDtype::F64:
      return "f64";
    case StorageDtype::I32:
      return "i32";
    case StorageDtype::I64:
      return "i64";
    case StorageDtype::Bool:
      return "bool";
  }
  return "?";
}

std::string coreDtypeName(StorageDtype dtype) {
  return storageDtypeName(dtype);
}

const char* deviceKindName(DeviceKind device) {
  switch (device) {
    case DeviceKind::Cuda:
      return "cuda";
    case DeviceKind::Cpu:
      return "cpu";
  }
  return "?";
}

const char* numericsPolicyName(NumericsPolicy policy) {
  switch (policy) {
    case NumericsPolicy::StrictEager:
      return "strict_eager";
    case NumericsPolicy::Inductor:
      return "inductor";
  }
  return "?";
}

const char* matmulPrecisionName(MatmulPrecision precision) {
  switch (precision) {
    case MatmulPrecision::Ieee:
      return "ieee";
    case MatmulPrecision::Tf32:
      return "tf32";
  }
  return "?";
}

const char* addmmLtPolicyName(AddmmLtPolicy policy) {
  switch (policy) {
    case AddmmLtPolicy::CudaDefault:
      return "cuda_default";
    case AddmmLtPolicy::CudaDisabled:
      return "cuda_disabled";
    case AddmmLtPolicy::Unknown:
      return "unknown";
  }
  return "?";
}

const char* rejectReasonName(RejectReason reason) {
  switch (reason) {
    case RejectReason::UnknownOption:
      return "unknown_option";
    case RejectReason::OpKind:
      return "op_kind";
    case RejectReason::Device:
      return "device";
    case RejectReason::Dtype:
      return "dtype";
    case RejectReason::DynamicShape:
      return "dynamic_shape";
    case RejectReason::Rank:
      return "rank";
    case RejectReason::Layout:
      return "layout";
    case RejectReason::NumericsFence:
      return "numerics_fence";
    case RejectReason::Config:
      return "config";
    case RejectReason::Fusion:
      return "fusion";
    case RejectReason::Context:
      return "context";
    case RejectReason::Signature:
      return "signature";
    case RejectReason::Alias:
      return "alias";
  }
  return "?";
}

OptionRejected::OptionRejected(RejectReason reason, const std::string& detail)
    : std::invalid_argument(std::string(rejectReasonName(reason)) + ": " + detail), reason_(reason) {}

void reject(RejectReason reason, const std::string& detail) {
  throw OptionRejected(reason, detail);
}

int64_t checkedMul64(int64_t a, int64_t b, const std::string& ctx) {
  int64_t out = 0;
  if (__builtin_mul_overflow(a, b, &out)) {
    reject(RejectReason::Layout, ctx + ": int64 overflow");
  }
  return out;
}

int64_t checkedAdd64(int64_t a, int64_t b, const std::string& ctx) {
  int64_t out = 0;
  if (__builtin_add_overflow(a, b, &out)) {
    reject(RejectReason::Layout, ctx + ": int64 overflow");
  }
  return out;
}

int64_t TensorLayout::numel() const {
  int64_t n = 1;
  for (int64_t s : sizes) {
    n = checkedMul64(n, s, "numel");
  }
  return n;
}

int64_t TensorLayout::extent() const {
  if (numel() == 0) {
    return 0;
  }
  int64_t hi = 0;
  for (size_t d = 0; d < sizes.size(); ++d) {
    hi = checkedAdd64(hi, checkedMul64(sizes[d] - 1, strides[d], "extent"), "extent");
  }
  return checkedAdd64(hi, 1, "extent");
}

bool TensorLayout::isContiguous() const {
  int64_t expected = 1;
  for (size_t d = sizes.size(); d-- > 0;) {
    if (sizes[d] != 1 && strides[d] != expected) {
      return false;
    }
    expected = checkedMul64(expected, sizes[d], "contiguity");
  }
  return true;
}

bool TensorLayout::isNonOverlapping() const {
  std::vector<size_t> order;
  for (size_t d = 0; d < sizes.size(); ++d) {
    if (sizes[d] > 1) {
      order.push_back(d);
    }
  }
  std::sort(order.begin(), order.end(), [&](size_t x, size_t y) { return strides[x] < strides[y]; });
  int64_t span = 1;
  for (size_t d : order) {
    if (strides[d] < span) {
      return false;
    }
    span = checkedMul64(strides[d], sizes[d], "overlap");
  }
  return true;
}

void TensorLayout::validate(const std::string& ctx) const {
  if (std::string(storageDtypeName(dtype)) == "?" ||
      (shape_kind != ShapeKind::Static && shape_kind != ShapeKind::Dynamic)) {
    reject(RejectReason::Signature, ctx + ": invalid dtype or shape-kind enum value");
  }
  if (sizes.size() != strides.size()) {
    reject(RejectReason::Signature, ctx + ": sizes/strides rank mismatch");
  }
  for (size_t d = 0; d < sizes.size(); ++d) {
    if (sizes[d] < 0) {
      reject(RejectReason::Signature, ctx + ": negative size");
    }
    if (strides[d] < 0) {
      reject(RejectReason::Layout, ctx + ": negative stride (not supported)");
    }
  }
  (void)extent();
}

namespace {

std::string intList(const std::vector<int64_t>& v) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < v.size(); ++i) {
    os << (i ? "," : "") << v[i];
  }
  os << "]";
  return os.str();
}

void requireSameDtype(const std::vector<const TensorOperand*>& ops, const std::string& ctx) {
  for (const TensorOperand* op : ops) {
    if (op->layout.dtype != ops.front()->layout.dtype) {
      reject(RejectReason::Signature, ctx + ": operand dtypes differ ('" + op->name + "')");
    }
  }
}

void requireDevice(DeviceKind device, const std::string& ctx) {
  if (std::string(deviceKindName(device)) == "?") {
    reject(RejectReason::Device, ctx + ": invalid device enum value");
  }
}

void requireNamed(const std::vector<const TensorOperand*>& ops, const std::string& ctx) {
  std::vector<std::string> seen;
  for (const TensorOperand* op : ops) {
    if (op->name.empty()) {
      reject(RejectReason::Signature, ctx + ": empty operand name");
    }
    if (std::find(seen.begin(), seen.end(), op->name) != seen.end()) {
      reject(RejectReason::Signature, ctx + ": duplicate operand name '" + op->name + "'");
    }
    seen.push_back(op->name);
    op->layout.validate(ctx + " operand '" + op->name + "'");
  }
}

void requireSizes(const TensorOperand& op, const std::vector<int64_t>& expected, const std::string& ctx) {
  if (op.layout.sizes != expected) {
    reject(RejectReason::Signature,
           ctx + ": operand '" + op.name + "' sizes " + intList(op.layout.sizes) + " != expected " + intList(expected));
  }
}

void requireRank(const TensorOperand& op, size_t rank, const std::string& ctx) {
  if (op.layout.rank() != rank) {
    reject(RejectReason::Rank, ctx + ": operand '" + op.name + "' must have rank " + std::to_string(rank));
  }
}

std::vector<int64_t> normalizeDims(std::vector<int64_t> dims, size_t rank, const std::string& ctx) {
  const auto r = static_cast<int64_t>(rank);
  for (int64_t& d : dims) {
    if (d < -r || d >= r) {
      reject(RejectReason::Signature, ctx + ": reduction dim out of range");
    }
    if (d < 0) {
      d += r;
    }
  }
  std::sort(dims.begin(), dims.end());
  if (std::adjacent_find(dims.begin(), dims.end()) != dims.end()) {
    reject(RejectReason::Signature, ctx + ": duplicate reduction dim");
  }
  return dims;
}

std::vector<int64_t> reducedSizes(const std::vector<int64_t>& sizes, const std::vector<int64_t>& dims, bool keepdim) {
  std::vector<int64_t> out;
  for (size_t d = 0; d < sizes.size(); ++d) {
    const bool reduced = std::find(dims.begin(), dims.end(), static_cast<int64_t>(d)) != dims.end();
    if (!reduced) {
      out.push_back(sizes[d]);
    } else if (keepdim) {
      out.push_back(1);
    }
  }
  return out;
}

void requireBroadcastable(const TensorOperand& bias, int64_t m, int64_t n, const std::string& ctx) {
  const std::vector<int64_t> target{m, n};
  if (bias.layout.rank() > 2) {
    reject(RejectReason::Rank, ctx + ": bias rank must be <= 2");
  }
  const size_t lead = 2 - bias.layout.rank();
  for (size_t d = 0; d < bias.layout.rank(); ++d) {
    const int64_t s = bias.layout.sizes[d];
    if (s != 1 && s != target[lead + d]) {
      reject(RejectReason::Signature, ctx + ": bias not broadcastable to [M,N]");
    }
  }
}

} // namespace

OpSignature OpSignature::Mm(DeviceKind device, TensorOperand a, TensorOperand b, TensorOperand out) {
  const std::string ctx = "mm signature";
  requireDevice(device, ctx);
  requireNamed({&a, &b, &out}, ctx);
  requireSameDtype({&a, &b, &out}, ctx);
  requireRank(a, 2, ctx);
  requireRank(b, 2, ctx);
  requireSizes(b, {a.layout.sizes[1], b.layout.sizes[1]}, ctx);
  requireSizes(out, {a.layout.sizes[0], b.layout.sizes[1]}, ctx);
  OpSignature s;
  s.kind_ = OpKind::Mm;
  s.device_ = device;
  s.inputs_ = {std::move(a), std::move(b)};
  s.output_ = std::move(out);
  s.attrs_ = MatmulAttrs{};
  return s;
}

OpSignature OpSignature::Addmm(DeviceKind device,
                               TensorOperand bias,
                               TensorOperand a,
                               TensorOperand b,
                               TensorOperand out,
                               double alpha,
                               double beta) {
  const std::string ctx = "addmm signature";
  requireDevice(device, ctx);
  requireNamed({&bias, &a, &b, &out}, ctx);
  requireSameDtype({&bias, &a, &b, &out}, ctx);
  requireRank(a, 2, ctx);
  requireRank(b, 2, ctx);
  requireSizes(b, {a.layout.sizes[1], b.layout.sizes[1]}, ctx);
  requireSizes(out, {a.layout.sizes[0], b.layout.sizes[1]}, ctx);
  requireBroadcastable(bias, a.layout.sizes[0], b.layout.sizes[1], ctx);
  OpSignature s;
  s.kind_ = OpKind::Addmm;
  s.device_ = device;
  s.inputs_ = {std::move(bias), std::move(a), std::move(b)};
  s.output_ = std::move(out);
  s.attrs_ = MatmulAttrs{alpha, beta};
  return s;
}

OpSignature OpSignature::Bmm(DeviceKind device, TensorOperand a, TensorOperand b, TensorOperand out) {
  const std::string ctx = "bmm signature";
  requireDevice(device, ctx);
  requireNamed({&a, &b, &out}, ctx);
  requireSameDtype({&a, &b, &out}, ctx);
  requireRank(a, 3, ctx);
  requireRank(b, 3, ctx);
  const auto& as = a.layout.sizes;
  requireSizes(b, {as[0], as[2], b.layout.sizes[2]}, ctx);
  requireSizes(out, {as[0], as[1], b.layout.sizes[2]}, ctx);
  OpSignature s;
  s.kind_ = OpKind::Bmm;
  s.device_ = device;
  s.inputs_ = {std::move(a), std::move(b)};
  s.output_ = std::move(out);
  s.attrs_ = MatmulAttrs{};
  return s;
}

OpSignature OpSignature::Reduction(OpKind kind,
                                   DeviceKind device,
                                   TensorOperand x,
                                   std::vector<int64_t> dims,
                                   bool keepdim,
                                   std::optional<double> correction,
                                   TensorOperand out) {
  const std::string ctx = std::string(opKindName(kind)) + " signature";
  if (std::string(opKindName(kind)) == "?" || kind == OpKind::Mm || kind == OpKind::Addmm || kind == OpKind::Bmm) {
    reject(RejectReason::OpKind, ctx + ": not a reduction kind");
  }
  requireDevice(device, ctx);
  requireNamed({&x, &out}, ctx);
  requireSameDtype({&x, &out}, ctx);
  if (x.layout.rank() == 0) {
    reject(RejectReason::Rank, ctx + ": rank-0 input not supported");
  }
  if (dims.empty()) {
    reject(RejectReason::Signature, ctx + ": explicit reduction dims required");
  }
  dims = normalizeDims(std::move(dims), x.layout.rank(), ctx);
  const bool needsCorrection = kind == OpKind::VarCorrection || kind == OpKind::StdCorrection;
  if (needsCorrection != correction.has_value()) {
    reject(RejectReason::Signature, ctx + ": correction must be given exactly for var/std");
  }
  const bool softmax = kind == OpKind::Softmax || kind == OpKind::LogSoftmax;
  if (softmax) {
    if (dims.size() != 1 || keepdim) {
      reject(RejectReason::Signature, ctx + ": softmax takes one dim and no keepdim");
    }
    requireSizes(out, x.layout.sizes, ctx);
  } else {
    requireSizes(out, reducedSizes(x.layout.sizes, dims, keepdim), ctx);
  }
  OpSignature s;
  s.kind_ = kind;
  s.device_ = device;
  s.inputs_ = {std::move(x)};
  s.output_ = std::move(out);
  s.attrs_ = ReductionAttrs{std::move(dims), keepdim, correction};
  return s;
}

bool OpSignature::isMatmul() const {
  return std::holds_alternative<MatmulAttrs>(attrs_);
}

bool OpSignature::isReduction() const {
  return std::holds_alternative<ReductionAttrs>(attrs_);
}

const MatmulAttrs& OpSignature::matmul() const {
  if (!isMatmul()) {
    reject(RejectReason::OpKind, "signature is not a matmul");
  }
  return std::get<MatmulAttrs>(attrs_);
}

const ReductionAttrs& OpSignature::reduction() const {
  if (!isReduction()) {
    reject(RejectReason::OpKind, "signature is not a reduction");
  }
  return std::get<ReductionAttrs>(attrs_);
}

size_t OpSignature::gemmOperandA() const {
  (void)matmul();
  return kind_ == OpKind::Addmm ? 1 : 0;
}

size_t OpSignature::gemmOperandB() const {
  return gemmOperandA() + 1;
}

GemmDims OpSignature::gemmDims() const {
  const auto& a = inputs_[gemmOperandA()].layout.sizes;
  const auto& b = inputs_[gemmOperandB()].layout.sizes;
  if (kind_ == OpKind::Bmm) {
    return GemmDims{a[0], a[1], b[2], a[2]};
  }
  return GemmDims{1, a[0], b[1], a[1]};
}

OpSignature OpSignature::withOutput(TensorOperand out) const {
  out.layout.validate("withOutput operand '" + out.name + "'");
  if (out.layout.sizes != output_.layout.sizes || out.layout.dtype != output_.layout.dtype) {
    reject(RejectReason::Signature, "withOutput: output sizes/dtype differ from the op result");
  }
  for (const auto& in : inputs_) {
    if (in.name == out.name) {
      reject(RejectReason::Signature, "withOutput: output name collides with an input");
    }
  }
  OpSignature s = *this;
  s.output_ = std::move(out);
  return s;
}

std::string f64Bits(double value) {
  uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return hex64(bits);
}

uint64_t fnv1a64(const std::string& text) {
  uint64_t h = 1469598103934665603ULL;
  for (unsigned char c : text) {
    h ^= c;
    h *= 1099511628211ULL;
  }
  return h;
}

std::string hex64(uint64_t value) {
  char buf[19];
  std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(value));
  return buf;
}

std::string TensorLayout::canonical() const {
  return std::string(storageDtypeName(dtype)) + intList(sizes) + intList(strides) +
      (shape_kind == ShapeKind::Static ? "static" : "dynamic");
}

std::string OpSignature::canonical() const {
  std::ostringstream os;
  os << opKindName(kind_) << "@" << deviceKindName(device_) << "(";
  for (const auto& in : inputs_) {
    os << in.name << ":" << in.layout.canonical() << ";";
  }
  os << ")->" << output_.name << ":" << output_.layout.canonical();
  if (isMatmul()) {
    os << " alpha=0x" << f64Bits(matmul().alpha) << " beta=0x" << f64Bits(matmul().beta);
  } else {
    const auto& r = reduction();
    os << " dims=" << intList(r.dims) << " keepdim=" << r.keepdim;
    os << " correction=" << (r.correction ? "0x" + f64Bits(*r.correction) : std::string("none"));
  }
  return os.str();
}

void CodegenContext::validate() const {
  if (std::string(numericsPolicyName(numerics)) == "?" || std::string(deviceKindName(device)) == "?" ||
      std::string(matmulPrecisionName(stock_fp32_matmul_precision)) == "?" ||
      std::string(addmmLtPolicyName(addmm_lt_policy)) == "?") {
    reject(RejectReason::Context, "invalid enum value in codegen context");
  }
  if (device_index < 0) {
    reject(RejectReason::Context, "negative device_index");
  }
  if (numerics == NumericsPolicy::StrictEager && (enable_fp_fusion || !disable_ftz)) {
    reject(RejectReason::Context, "strict_eager requires enable_fp_fusion=false and disable_ftz=true");
  }
  if (device == DeviceKind::Cuda && (compute_capability <= 0 || max_shared_memory_bytes <= 0)) {
    reject(RejectReason::Context, "cuda context requires compute_capability and max_shared_memory_bytes");
  }
}

std::string CodegenContext::canonical() const {
  std::ostringstream os;
  os << "numerics=" << numericsPolicyName(numerics) << " device=" << deviceKindName(device)
     << " cc=" << compute_capability << " smem=" << max_shared_memory_bytes
     << " stock_fp32_matmul=" << matmulPrecisionName(stock_fp32_matmul_precision)
     << " fp_fusion=" << enable_fp_fusion << " disable_ftz=" << disable_ftz
     << " division_rounding=" << division_rounding << " device_index=" << device_index
     << " addmm_lt=" << addmmLtPolicyName(addmm_lt_policy);
  return os.str();
}

bool stockAllowsTf32(const OpSignature& sig, const CodegenContext& ctx) {
  if (ctx.device != DeviceKind::Cuda || ctx.stock_fp32_matmul_precision != MatmulPrecision::Tf32) {
    return false;
  }
  const GemmDims d = sig.gemmDims();
  return d.m >= 16 && std::min(d.n, d.k) >= 512;
}

} // namespace torch::inductor::loops_ir::op_options
