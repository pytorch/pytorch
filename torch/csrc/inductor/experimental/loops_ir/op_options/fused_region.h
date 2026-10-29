#pragma once
// Prologue/epilogue fusion over core LoopsIR Compute bodies.
//
// A fused region is an ordinary pointwise core Compute (Load/Constant/
// Binary add|mul/Relu/Store; no reductions, no view reads) whose domain
// equals the fused operand's (prologue) or the op output's (epilogue)
// logical shape. It is verified by the core Program verifier first, then
// checked against the selected option's FusionContract and lowered to the
// flat RegionOp list that codegen consumes. Core Binary has no sub/div:
// x - y is expressible exactly as add(x, mul(y, -1)); division is not
// representable and is therefore not fusible (no local scalar AST).

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "../ir.h"
#include "op_types.h"

namespace torch::inductor::loops_ir::op_options {

struct ExtraInput {
  std::string name;
  TensorLayout layout;
};

// Prologue for input slot `operand`: may load that operand (raw, at its
// layout map, offset 0) plus extra inputs; stores its single result at
// store_offset + operand layout map. The op consumes the stored value in
// place of the raw load. A prologue that never loads the operand computes
// it entirely (graph producer); the operand is then not a kernel argument.
struct PrologueRegion {
  int64_t operand = 0;
  Compute body;
  int64_t store_offset = 0;
};

// Epilogue: loads `op_result` (the op value) at op_result_offset + the
// op_result_layout map (default: signature output layout) plus extra
// inputs; stores its single result at store_offset + the signature output
// layout map.
struct EpilogueRegion {
  std::string op_result;
  Compute body;
  std::optional<TensorLayout> op_result_layout;
  int64_t op_result_offset = 0;
  int64_t store_offset = 0;
};

struct FusedLoops {
  std::vector<PrologueRegion> prologues;
  std::optional<EpilogueRegion> epilogue;
  std::vector<ExtraInput> extra_inputs;
};

enum class RegionOpKind { LoadSelf, LoadExtra, Constant, Add, Mul, Relu };
const char* regionOpKindName(RegionOpKind kind);

// offset + sum(coeffs[d] * coord[d]) over the region domain. Coefficients
// of size-<=1 domain dims are canonicalized to 0.
struct AccessMap {
  int64_t offset = 0;
  std::vector<int64_t> coeffs;
  bool operator==(const AccessMap& o) const {
    return offset == o.offset && coeffs == o.coeffs;
  }
};

struct RegionOp {
  RegionOpKind kind = RegionOpKind::Constant;
  int id = -1;
  int lhs = -1;
  int rhs = -1;
  int64_t extra = -1; // FusedLoops::extra_inputs slot for LoadExtra
  AccessMap access; // LoadExtra only
  double value = 0.0; // Constant only (f32-representable, finite)
};

struct LoweredRegion {
  std::vector<int64_t> domain;
  std::vector<RegionOp> ops;
  int store_value = -1;
  int64_t store_offset = 0;
  bool reads_self = false;
  std::string canonical() const;
};

struct LoweredPrologue {
  int64_t operand = 0;
  LoweredRegion region;
};

struct ValidatedFusion {
  std::vector<LoweredPrologue> prologues;
  std::optional<LoweredRegion> epilogue;
  std::vector<ExtraInput> extra_inputs;
  bool empty() const {
    return prologues.empty() && !epilogue.has_value();
  }
  const LoweredRegion* prologueFor(int64_t operand) const;
  std::string canonical() const;
};

struct FusionContract {
  std::vector<int64_t> prologue_operands;
  bool epilogue = false;
  std::vector<RegionOpKind> allowed_ops;
  int64_t max_extra_inputs = 0;
  // Stride-0 (broadcast) extra-input maps over a non-trivial domain dim.
  bool allow_broadcast = false;
  std::string canonical() const;
};

// Map of the operand/output itself: coeffs = layout strides (size<=1 dims
// canonicalized to 0), offset 0.
AccessMap selfAccessMap(const TensorLayout& layout);

// Exact affine map of a core Index over the given pointwise ranges/domain.
// Throws OptionRejected(Fusion) for floordiv/mod, products of variables or
// variables outside the ranges.
AccessMap affineAccessMap(const Index& idx,
                          const std::vector<Range>& ranges,
                          const std::vector<int64_t>& domain,
                          const std::string& ctx);

// Throws OptionRejected(Fusion|Dtype|...) on any violation; never drops a
// region silently.
ValidatedFusion validateFusion(const FusionContract& contract, const OpSignature& sig, const FusedLoops& fused);

} // namespace torch::inductor::loops_ir::op_options
