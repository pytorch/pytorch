#pragma once
// TiledIR: function-local tiled SSA for strict-FP32 pointwise execution.
//
// Owner: tiled-backend worker. This header builds only on the shared core
// headers plus the agreed Kernel API contract (published by the core worker
// in kernel_ir.h):
//   KernelFunction: name(), regions() -> std::vector<Compute>
//     (executable tiling requires exactly 1 region; reductions live in
//     eager extern calls elsewhere and are rejected here).
//   KernelProgram: name(), inputs(), inputDescriptors(), views(), outputs(),
//     nodes() -> std::vector<std::variant<KernelFunction, ExternCall>>,
//     dependencies(); nodes keep kernel launches and extern calls.
//     Optional buffers() lists every buffer definition when provided.
//   Core LoopIR post-body instructions keep their getters (.kind(), .id(),
//   .buffer(), .offset(), .op(), .lhs/.rhs/.arg/.value, ...) and the Index
//   tree stays shared/const with expr()/ptr() exposed.
//
// KernelProgram/KernelFunction are forward-declared here so this header
// parses standalone; tiled_ir.cpp includes "kernel_ir.h" for the lowering.
//
// Design rules (strict bitwise parity with eager, static FP32 pointwise):
// - Straight-line entry block + return first. No fake CFG/goto/parallel
//   groups. Terminator is currently always "return".
// - Types are explicit: integer arithmetic is int64 ("i64"), data is "f32",
//   masks are "bool". Shapes are explicit: {} scalar or {lanes} vector.
// - FP32 constants carry explicit type; scalar vs broadcast is explicit IR
//   (const_float scalar + broadcast), never implicit.
// - Operand order of floating-point ops is preserved; no reassociation, no
//   contraction/FMA, no FTZ, no algebraic neutral rewrites, no data folding.
// - ReLU stays its own op so the emitter lowers it to compare/select
//   (NaN-payload and signed-zero preserving), never tl.maximum.
// - Casts must be precision-preserving no-ops (same dtype/shape); opaque
//   conversions happen outside generated code.
// - Loads/stores always carry explicit offset + mask operands and reference
//   launch-order parameters. Strides/layout are NOT consulted by the
//   emitter; offsets are already explicit flat trees.
// - No tiling/layout decisions here beyond the caller's TileOptions
//   (default block 256, warps 4, fp_fusion off, ftz disabled). Zero extents
//   lower to grid 0 with no unsafe divide-by-zero and no launch.
// - Numerical policy rides on every function: strict defaults
//   enable_fp_fusion=false, disable_ftz=true, numerics="strict_eager".
//   The verifier rejects strict deviations; inductor policy permits the
//   effective flags carried by TileOptions.

#include <cstdint>
#include <string>
#include <vector>

#include "ir.h"

namespace torch::inductor::loops_ir {

// Forward declarations; full definitions live in kernel_ir.h (core worker).
class KernelFunction;
class KernelProgram;

// ---------------------------------------------------------------------------
// TileOptions
// ---------------------------------------------------------------------------

struct TileOptions {
  TileOptions(int64_t block_size = 256, int64_t num_warps = 4, bool enable_fp_fusion = false, bool disable_ftz = true);
  int64_t block_size = 256;
  int64_t num_warps = 4;
  bool enable_fp_fusion = false;
  bool disable_ftz = true;
};

// ---------------------------------------------------------------------------
// TiledInstruction: one typed SSA value, except store.
//
// op names (exact strings; the Triton emitter switches on these, which is
// more stable than an enum across the C++/Python boundary):
//   "program_id"  : pid scalar.                  defines i64 scalar.
//   "arange"      : lane ids 0..lanes-1.         defines i64 {lanes}.
//   "const_int"   : int_value().                  defines i64 scalar.
//   "const_float" : float_value(), f32-checked.   defines f32 scalar.
//   "broadcast"   : operands {scalar}.            scalar -> {lanes}.
//   "add","mul","floordiv","mod": operands {lhs,rhs}, i64, same shape.
//   "cmp_lt"      : operands {lhs,rhs}, i64 same shape -> bool same shape.
//   "select"      : operands {cond bool, on_true, on_false}; last two share
//                   dtype/shape with the result.
//   "fadd","fmul" : operands {lhs,rhs}, f32, same shape, order preserved.
//   "relu"        : operands {arg}, f32 -> f32 same shape. Emitter lowers to
//                   tl.where(x < 0.0, 0.0, x): NaN (payload) and -0.0 flow
//                   to the false lane, negatives become +0.0.
//   "load"        : operands {offset i64 vec, mask bool vec}, buffer().
//                   defines f32 {lanes}.
//   "store"       : operands {value f32 vec, offset i64 vec, mask bool vec},
//                   buffer(). Defines nothing (id() == -1).
//   "cast"        : operands {arg}. Verifier requires result dtype/shape to
//                   EQUAL the operand's (precision-preserving no-op only).
// dtype(): "i64", "f32", or "bool".
// shape(): {} for scalars, {lanes} for vectors (lanes == block_size).
// Scalar/vector convention for the factories below: ProgramId, ConstInt,
// ConstFloat always define scalars ({}); Arange always defines a vector;
// Broadcast and Load require lanes > 0 and define vectors; IntBinary,
// CmpLt, Select, FloatBinary, Relu, Cast take lanes == 0 for a scalar
// result or lanes > 0 for a vector result (operands must already match
// that shape; the lowering inserts explicit broadcasts, never implicit
// ones). Store defines nothing (id() == -1).
// ---------------------------------------------------------------------------

class TiledInstruction {
 public:
  static TiledInstruction ProgramId(int id);
  static TiledInstruction Arange(int id, int64_t lanes);
  static TiledInstruction ConstInt(int id, int64_t value);
  static TiledInstruction ConstFloat(int id, double value);
  static TiledInstruction Broadcast(int id, int operand, std::string dtype, int64_t lanes);
  static TiledInstruction IntBinary(std::string op, int id, int lhs, int rhs, int64_t lanes);
  static TiledInstruction CmpLt(int id, int lhs, int rhs, int64_t lanes);
  static TiledInstruction Select(int id, int cond, int onTrue, int onFalse, std::string dtype, int64_t lanes);
  static TiledInstruction FloatBinary(std::string op, int id, int lhs, int rhs, int64_t lanes);
  static TiledInstruction Relu(int id, int arg, int64_t lanes);
  static TiledInstruction Load(int id, std::string buffer, int offset, int mask, int64_t lanes);
  static TiledInstruction Store(std::string buffer, int value, int offset, int mask);
  static TiledInstruction Cast(int id, int arg, std::string dtype, int64_t lanes);

  const std::string& op() const {
    return op_;
  }
  int id() const {
    return id_;
  }
  const std::string& dtype() const {
    return dtype_;
  }
  const std::vector<int64_t>& shape() const {
    return shape_;
  }
  const std::vector<int>& operands() const {
    return operands_;
  }
  int64_t int_value() const {
    return int_value_;
  }
  double float_value() const {
    return float_value_;
  }
  const std::string& buffer() const {
    return buffer_;
  }
  bool isVector() const {
    return !shape_.empty();
  }

 private:
  std::string op_;
  int id_ = -1;
  std::string dtype_ = "f32";
  std::vector<int64_t> shape_;
  std::vector<int> operands_;
  int64_t int_value_ = 0;
  double float_value_ = 0.0;
  std::string buffer_;
};

// ---------------------------------------------------------------------------
// TiledBlock: straight-line instructions + terminator ("return").
// ---------------------------------------------------------------------------

class TiledBlock {
 public:
  explicit TiledBlock(std::string name) : name_(std::move(name)) {}

  const std::string& name() const {
    return name_;
  }
  const std::vector<TiledInstruction>& instructions() const {
    return instructions_;
  }
  const std::string& terminator() const {
    return terminator_;
  }

  void addInstruction(const TiledInstruction& ins) {
    instructions_.push_back(ins);
  }
  void setTerminator(const std::string& t);

 private:
  std::string name_;
  std::vector<TiledInstruction> instructions_;
  std::string terminator_ = "return";
};

// ---------------------------------------------------------------------------
// TiledFunction: one launchable kernel.
//
// parameters() are Buffers in EXACT launch order: distinct loaded inputs in
// first-use order, then stored outputs (results) in store order. Output
// pointers are included. The runtime launches
//   jit[(grid_size,)](*buffers, num_warps=..., enable_fp_fusion=False,
//                     enable_reflect_ftz=False)
// (raw-Triton spelling of the disable_ftz=True attribute; raw Triton has no
// `disable_ftz` launch kwarg -- verified: it raises KeyError -- and inductor
// maps compile_meta["disable_ftz"] to options["enable_reflect_ftz"]=False)
// and must skip the launch entirely when grid_size == 0.
// ---------------------------------------------------------------------------

class TiledFunction {
 public:
  TiledFunction(std::string name,
                std::vector<Buffer> parameters,
                int64_t grid_size,
                int64_t block_size,
                int64_t num_warps,
                bool enable_fp_fusion = false,
                bool disable_ftz = true,
                std::string numerics = "strict_eager");

  const std::string& name() const {
    return name_;
  }
  const std::vector<Buffer>& parameters() const {
    return parameters_;
  }
  const std::vector<TiledBlock>& blocks() const {
    return blocks_;
  }
  int64_t grid_size() const {
    return grid_size_;
  }
  int64_t block_size() const {
    return block_size_;
  }
  int64_t num_warps() const {
    return num_warps_;
  }
  bool enable_fp_fusion() const {
    return enable_fp_fusion_;
  }
  bool disable_ftz() const {
    return disable_ftz_;
  }
  const std::string& numerics() const {
    return numerics_;
  }

  void addBlock(const TiledBlock& b) {
    blocks_.push_back(b);
  }

 private:
  std::string name_;
  std::vector<Buffer> parameters_;
  std::vector<TiledBlock> blocks_;
  int64_t grid_size_ = 0;
  int64_t block_size_ = 256;
  int64_t num_warps_ = 4;
  bool enable_fp_fusion_ = false;
  bool disable_ftz_ = true;
  std::string numerics_ = "strict_eager";
};

// ---------------------------------------------------------------------------
// TiledProgram: ordered tiled functions (one per tiled KernelFunction;
// extern calls stay eager outside and produce no function).
// ---------------------------------------------------------------------------

class TiledProgram {
 public:
  explicit TiledProgram(std::string numerics = "strict_eager") : numerics_(std::move(numerics)) {
    if (!isValidNumerics(numerics_)) {
      throw std::invalid_argument("TiledProgram: unsupported numerics '" + numerics_ +
                                  "' (allowed: 'strict_eager', 'inductor')");
    }
  }

  const std::vector<TiledFunction>& functions() const {
    return functions_;
  }
  const std::string& numerics() const {
    return numerics_;
  }
  void addFunction(const TiledFunction& f) {
    if (f.numerics() != numerics_) {
      throw std::invalid_argument("TiledProgram.addFunction: function numerics '" + f.numerics() +
                                  "' does not match program numerics '" + numerics_ + "'");
    }
    functions_.push_back(f);
  }

  // Full verification; throws std::invalid_argument with function context.
  void verify() const;
  // Verifies first, then returns human-readable text (never parsed).
  std::string dump() const;

 private:
  std::vector<TiledFunction> functions_;
  std::string numerics_ = "strict_eager";
};

// Tile one pointwise region per KernelFunction. Skips ExternCall nodes
// (eager elsewhere). Throws on: != 1 region, any reduction ranges /
// accumulators / reduction body (must be extern), ViewRead (run view_inline
// first), Accumulate/ReadAccumulator in post body, unknown buffers, int64
// overflow in tile math, or bad TileOptions. Thin shim through the typed
// registry; see detail::lowerToTiledImpl.
TiledProgram lower_to_tiled(const KernelProgram& program, const TileOptions& options = TileOptions());

} // namespace torch::inductor::loops_ir

// NOTE (cross-worker contract): the core worker's bindings.cpp forward
// declares and calls this global helper after core registration:
//   namespace pybind11 { class module_; }
//   void bind_tiled_ir(pybind11::module_& m);
// It is defined in tiled_bindings.cpp. This header intentionally does not
// include pybind11.
