// TiledIR implementation: typed SSA factories, tile verifier, printer, and
// the strict-FP32 pointwise tiling pass.
//
// Owner: tiled-backend worker. Builds on ir.h + kernel_ir.h only (C++17, no
// torch/c10 deps). See tiled_ir.h for the op contract.
//
// Strictness notes:
// - Floating-point order preserved: LoopIR post-body order is replayed
//   verbatim (constant/load/binary/relu/store); no folding, reassociation,
//   FMA, FTZ, or neutral rewrites. Integer offset trees are re-emitted as
//   written (fresh SSA per Index node, no folding).
// - ReLU stays a distinct op; the Triton emitter lowers it to compare/select.
// - Zero extents lower to grid 0 with an all-false mask and const-0
//   coordinates, so no floordiv/mod by zero is ever emitted and the runtime
//   must skip the launch entirely.

#include "tiled_ir.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <sstream>

#include "kernel_ir.h"

namespace torch::inductor::loops_ir {
namespace {

[[noreturn]] void fail(const std::string& ctx, const std::string& msg) {
  throw std::invalid_argument(ctx + ": " + msg);
}

int64_t checkedAddSub(const std::string& ctx, int64_t a, int64_t b, bool add) {
  __int128 v = add ? (__int128)a + b : (__int128)a - b;
  if (v > std::numeric_limits<int64_t>::max() ||
      v < std::numeric_limits<int64_t>::min()) {
    fail(ctx, "int64 overflow in tile math");
  }
  return static_cast<int64_t>(v);
}

void requireFiniteF32Tiled(const std::string& ctx, double v) {
  if (!std::isfinite(v)) {
    fail(ctx, "const_float is non-finite double");
  }
  float f = static_cast<float>(v);
  if (!std::isfinite(f)) {
    fail(ctx, "const_float overflows f32");
  }
}

bool isVecShape(const std::vector<int64_t>& s, int64_t block) {
  return s.size() == 1 && s[0] == block;
}

bool isScalarShape(const std::vector<int64_t>& s) {
  return s.empty();
}

std::string shapeStr(const std::vector<int64_t>& s) {
  if (s.empty()) {
    return "{}";
  }
  std::string r = "{";
  for (size_t i = 0; i < s.size(); ++i) {
    if (i) {
      r += ",";
    }
    r += std::to_string(s[i]);
  }
  return r + "}";
}

} // namespace

// ---------------------------------------------------------------------------
// TileOptions
// ---------------------------------------------------------------------------

TileOptions::TileOptions(
    int64_t block_size,
    int64_t num_warps,
    bool enable_fp_fusion,
    bool disable_ftz)
    : block_size(block_size),
      num_warps(num_warps),
      enable_fp_fusion(enable_fp_fusion),
      disable_ftz(disable_ftz) {
  if (block_size <= 0) {
    throw std::invalid_argument("TileOptions: block_size must be > 0");
  }
  if (num_warps <= 0) {
    throw std::invalid_argument("TileOptions: num_warps must be > 0");
  }
}

// ---------------------------------------------------------------------------
// TiledInstruction factories
// ---------------------------------------------------------------------------
// NOTE: TiledInstruction has only private data with public static factories;
// each factory validates its own arguments so malformed IR fails fast.

TiledInstruction TiledInstruction::ProgramId(int id) {
  if (id < 0) {
    throw std::invalid_argument("TiledInstruction.program_id: negative id");
  }
  TiledInstruction i;
  i.op_ = "program_id";
  i.id_ = id;
  i.dtype_ = "i64";
  return i;
}

TiledInstruction TiledInstruction::Arange(int id, int64_t lanes) {
  if (id < 0) {
    throw std::invalid_argument("TiledInstruction.arange: negative id");
  }
  if (lanes <= 0) {
    throw std::invalid_argument("TiledInstruction.arange: lanes must be > 0");
  }
  TiledInstruction i;
  i.op_ = "arange";
  i.id_ = id;
  i.dtype_ = "i64";
  i.shape_ = {lanes};
  return i;
}

TiledInstruction TiledInstruction::ConstInt(int id, int64_t value) {
  if (id < 0) {
    throw std::invalid_argument("TiledInstruction.const_int: negative id");
  }
  TiledInstruction i;
  i.op_ = "const_int";
  i.id_ = id;
  i.dtype_ = "i64";
  i.int_value_ = value;
  return i;
}

TiledInstruction TiledInstruction::ConstFloat(int id, double value) {
  if (id < 0) {
    throw std::invalid_argument("TiledInstruction.const_float: negative id");
  }
  requireFiniteF32Tiled(
      "TiledInstruction.const_float v" + std::to_string(id), value);
  TiledInstruction i;
  i.op_ = "const_float";
  i.id_ = id;
  i.dtype_ = "f32";
  i.float_value_ = value;
  return i;
}

TiledInstruction TiledInstruction::Broadcast(
    int id,
    int operand,
    std::string dtype,
    int64_t lanes) {
  if (id < 0) {
    throw std::invalid_argument("TiledInstruction.broadcast: negative id");
  }
  if (operand < 0) {
    throw std::invalid_argument("TiledInstruction.broadcast: negative operand");
  }
  if (dtype != "i64" && dtype != "f32" && dtype != "bool") {
    throw std::invalid_argument(
        "TiledInstruction.broadcast: unsupported dtype '" + dtype + "'");
  }
  if (lanes <= 0) {
    throw std::invalid_argument(
        "TiledInstruction.broadcast: lanes must be > 0");
  }
  TiledInstruction i;
  i.op_ = "broadcast";
  i.id_ = id;
  i.dtype_ = std::move(dtype);
  i.shape_ = {lanes};
  i.operands_ = {operand};
  return i;
}

TiledInstruction TiledInstruction::IntBinary(
    std::string op,
    int id,
    int lhs,
    int rhs,
    int64_t lanes) {
  if (op != "add" && op != "mul" && op != "floordiv" && op != "mod") {
    throw std::invalid_argument(
        "TiledInstruction.int_binary: unsupported op '" + op + "'");
  }
  if (id < 0) {
    throw std::invalid_argument("TiledInstruction.int_binary: negative id");
  }
  if (lhs < 0 || rhs < 0) {
    throw std::invalid_argument(
        "TiledInstruction.int_binary: negative operand");
  }
  if (lanes < 0) {
    throw std::invalid_argument("TiledInstruction.int_binary: negative lanes");
  }
  TiledInstruction i;
  i.op_ = std::move(op);
  i.id_ = id;
  i.dtype_ = "i64";
  if (lanes > 0) {
    i.shape_ = {lanes};
  }
  i.operands_ = {lhs, rhs};
  return i;
}

TiledInstruction TiledInstruction::CmpLt(
    int id,
    int lhs,
    int rhs,
    int64_t lanes) {
  if (id < 0) {
    throw std::invalid_argument("TiledInstruction.cmp_lt: negative id");
  }
  if (lhs < 0 || rhs < 0) {
    throw std::invalid_argument("TiledInstruction.cmp_lt: negative operand");
  }
  if (lanes < 0) {
    throw std::invalid_argument("TiledInstruction.cmp_lt: negative lanes");
  }
  TiledInstruction i;
  i.op_ = "cmp_lt";
  i.id_ = id;
  i.dtype_ = "bool";
  if (lanes > 0) {
    i.shape_ = {lanes};
  }
  i.operands_ = {lhs, rhs};
  return i;
}

TiledInstruction TiledInstruction::Select(
    int id,
    int cond,
    int onTrue,
    int onFalse,
    std::string dtype,
    int64_t lanes) {
  if (id < 0) {
    throw std::invalid_argument("TiledInstruction.select: negative id");
  }
  if (cond < 0 || onTrue < 0 || onFalse < 0) {
    throw std::invalid_argument("TiledInstruction.select: negative operand");
  }
  if (dtype != "i64" && dtype != "f32" && dtype != "bool") {
    throw std::invalid_argument(
        "TiledInstruction.select: unsupported dtype '" + dtype + "'");
  }
  if (lanes < 0) {
    throw std::invalid_argument("TiledInstruction.select: negative lanes");
  }
  TiledInstruction i;
  i.op_ = "select";
  i.id_ = id;
  i.dtype_ = std::move(dtype);
  if (lanes > 0) {
    i.shape_ = {lanes};
  }
  i.operands_ = {cond, onTrue, onFalse};
  return i;
}

TiledInstruction TiledInstruction::FloatBinary(
    std::string op,
    int id,
    int lhs,
    int rhs,
    int64_t lanes) {
  if (op != "fadd" && op != "fmul") {
    throw std::invalid_argument(
        "TiledInstruction.float_binary: unsupported op '" + op + "'");
  }
  if (id < 0) {
    throw std::invalid_argument("TiledInstruction.float_binary: negative id");
  }
  if (lhs < 0 || rhs < 0) {
    throw std::invalid_argument(
        "TiledInstruction.float_binary: negative operand");
  }
  if (lanes < 0) {
    throw std::invalid_argument(
        "TiledInstruction.float_binary: negative lanes");
  }
  TiledInstruction i;
  i.op_ = std::move(op);
  i.id_ = id;
  i.dtype_ = "f32";
  if (lanes > 0) {
    i.shape_ = {lanes};
  }
  i.operands_ = {lhs, rhs};
  return i;
}

TiledInstruction TiledInstruction::Relu(int id, int arg, int64_t lanes) {
  if (id < 0) {
    throw std::invalid_argument("TiledInstruction.relu: negative id");
  }
  if (arg < 0) {
    throw std::invalid_argument("TiledInstruction.relu: negative operand");
  }
  if (lanes < 0) {
    throw std::invalid_argument("TiledInstruction.relu: negative lanes");
  }
  TiledInstruction i;
  i.op_ = "relu";
  i.id_ = id;
  i.dtype_ = "f32";
  if (lanes > 0) {
    i.shape_ = {lanes};
  }
  i.operands_ = {arg};
  return i;
}

TiledInstruction TiledInstruction::Load(
    int id,
    std::string buffer,
    int offset,
    int mask,
    int64_t lanes) {
  if (id < 0) {
    throw std::invalid_argument("TiledInstruction.load: negative id");
  }
  if (buffer.empty()) {
    throw std::invalid_argument("TiledInstruction.load: empty buffer");
  }
  if (offset < 0 || mask < 0) {
    throw std::invalid_argument("TiledInstruction.load: negative operand");
  }
  if (lanes <= 0) {
    throw std::invalid_argument("TiledInstruction.load: lanes must be > 0");
  }
  TiledInstruction i;
  i.op_ = "load";
  i.id_ = id;
  i.dtype_ = "f32";
  i.shape_ = {lanes};
  i.operands_ = {offset, mask};
  i.buffer_ = std::move(buffer);
  return i;
}

TiledInstruction TiledInstruction::Store(
    std::string buffer,
    int value,
    int offset,
    int mask) {
  if (buffer.empty()) {
    throw std::invalid_argument("TiledInstruction.store: empty buffer");
  }
  if (value < 0 || offset < 0 || mask < 0) {
    throw std::invalid_argument("TiledInstruction.store: negative operand");
  }
  TiledInstruction i;
  i.op_ = "store";
  i.id_ = -1;
  i.dtype_ = "f32";
  i.operands_ = {value, offset, mask};
  i.buffer_ = std::move(buffer);
  return i;
}

TiledInstruction TiledInstruction::Cast(
    int id,
    int arg,
    std::string dtype,
    int64_t lanes) {
  if (id < 0) {
    throw std::invalid_argument("TiledInstruction.cast: negative id");
  }
  if (arg < 0) {
    throw std::invalid_argument("TiledInstruction.cast: negative operand");
  }
  if (dtype != "i64" && dtype != "f32" && dtype != "bool") {
    throw std::invalid_argument(
        "TiledInstruction.cast: unsupported dtype '" + dtype + "'");
  }
  if (lanes < 0) {
    throw std::invalid_argument("TiledInstruction.cast: negative lanes");
  }
  TiledInstruction i;
  i.op_ = "cast";
  i.id_ = id;
  i.dtype_ = std::move(dtype);
  if (lanes > 0) {
    i.shape_ = {lanes};
  }
  i.operands_ = {arg};
  return i;
}

// ---------------------------------------------------------------------------
// TiledBlock / TiledFunction
// ---------------------------------------------------------------------------

void TiledBlock::setTerminator(const std::string& t) {
  if (t != "return") {
    throw std::invalid_argument(
        "TiledBlock.terminator: only 'return' supported (got '" + t + "')");
  }
  terminator_ = t;
}

TiledFunction::TiledFunction(
    std::string name,
    std::vector<Buffer> parameters,
    int64_t grid_size,
    int64_t block_size,
    int64_t num_warps,
    bool enable_fp_fusion,
    bool disable_ftz,
    std::string numerics)
    : name_(std::move(name)),
      parameters_(std::move(parameters)),
      grid_size_(grid_size),
      block_size_(block_size),
      num_warps_(num_warps),
      enable_fp_fusion_(enable_fp_fusion),
      disable_ftz_(disable_ftz),
      numerics_(std::move(numerics)) {
  if (!isValidNumerics(numerics_)) {
    throw std::invalid_argument(
        "TiledFunction: unsupported numerics '" + numerics_ +
        "' (allowed: 'strict_eager', 'inductor')");
  }
  if (numerics_ == strictNumericsName()) {
    if (enable_fp_fusion_) {
      throw std::invalid_argument(
          "TiledFunction: enable_fp_fusion must be False under strict_eager");
    }
    if (!disable_ftz_) {
      throw std::invalid_argument(
          "TiledFunction: disable_ftz must be True under strict_eager");
    }
  }
}

// ---------------------------------------------------------------------------
// Tile verifier
// ---------------------------------------------------------------------------

namespace {

struct DefInfo {
  std::string dtype;
  std::vector<int64_t> shape;
};

const DefInfo& lookupDef(
    const std::map<int, DefInfo>& defined,
    int id,
    const std::string& ictx) {
  auto it = defined.find(id);
  if (it == defined.end()) {
    fail(
        ictx,
        "use of undefined SSA v" + std::to_string(id) +
            " (must dominate use in this function)");
  }
  return it->second;
}

void checkShapeMatchesBlock(
    const std::vector<int64_t>& s,
    int64_t block,
    const std::string& ictx,
    const std::string& what) {
  if (!isScalarShape(s) && !isVecShape(s, block)) {
    fail(
        ictx,
        what + " shape " + shapeStr(s) + " must be {} or {" +
            std::to_string(block) + "}");
  }
}

void verifyTiledInstruction(
    const TiledInstruction& ins,
    std::map<int, DefInfo>& defined,
    const std::map<std::string, Buffer>& params,
    int64_t block,
    const std::string& ctx) {
  const std::string ictx = ctx + " " + ins.op() +
      (ins.id() >= 0 ? (" v" + std::to_string(ins.id())) : " store");
  const std::string& op = ins.op();
  if (op != "program_id" && op != "arange" && op != "const_int" &&
      op != "const_float" && op != "broadcast" && op != "add" && op != "mul" &&
      op != "floordiv" && op != "mod" && op != "cmp_lt" && op != "select" &&
      op != "fadd" && op != "fmul" && op != "relu" && op != "load" &&
      op != "store" && op != "cast") {
    fail(ictx, "unknown op '" + op + "'");
  }
  if (ins.dtype() != "i64" && ins.dtype() != "f32" && ins.dtype() != "bool") {
    fail(ictx, "unsupported dtype '" + ins.dtype() + "'");
  }
  checkShapeMatchesBlock(ins.shape(), block, ictx, "result");
  const bool isVec = !ins.shape().empty();

  auto expectOperands = [&](size_t n) {
    if (ins.operands().size() != n) {
      fail(
          ictx,
          "expected " + std::to_string(n) + " operands, got " +
              std::to_string(ins.operands().size()));
    }
  };

  if (op == "program_id") {
    expectOperands(0);
    if (ins.id() < 0) {
      fail(ictx, "negative SSA id");
    }
    if (ins.dtype() != "i64" || isVec) {
      fail(ictx, "program_id must be i64 scalar");
    }
  } else if (op == "arange") {
    expectOperands(0);
    if (ins.id() < 0) {
      fail(ictx, "negative SSA id");
    }
    if (ins.dtype() != "i64" || !isVec) {
      fail(ictx, "arange must be i64 vector");
    }
  } else if (op == "const_int") {
    expectOperands(0);
    if (ins.id() < 0) {
      fail(ictx, "negative SSA id");
    }
    if (ins.dtype() != "i64" || isVec) {
      fail(ictx, "const_int must be i64 scalar");
    }
  } else if (op == "const_float") {
    expectOperands(0);
    if (ins.id() < 0) {
      fail(ictx, "negative SSA id");
    }
    if (ins.dtype() != "f32" || isVec) {
      fail(ictx, "const_float must be f32 scalar");
    }
    requireFiniteF32Tiled(ictx, ins.float_value());
  } else if (op == "broadcast") {
    expectOperands(1);
    if (ins.id() < 0) {
      fail(ictx, "negative SSA id");
    }
    if (!isVec) {
      fail(ictx, "broadcast result must be a vector");
    }
    const DefInfo& src = lookupDef(defined, ins.operands()[0], ictx);
    if (!isScalarShape(src.shape)) {
      fail(ictx, "broadcast operand must be a scalar");
    }
    if (src.dtype != ins.dtype()) {
      fail(ictx, "broadcast dtype mismatch with operand");
    }
  } else if (op == "add" || op == "mul" || op == "floordiv" || op == "mod") {
    expectOperands(2);
    if (ins.id() < 0) {
      fail(ictx, "negative SSA id");
    }
    if (ins.dtype() != "i64") {
      fail(ictx, "'" + op + "' must be i64");
    }
    for (int o : ins.operands()) {
      const DefInfo& d = lookupDef(defined, o, ictx);
      if (d.dtype != "i64" || d.shape != ins.shape()) {
        fail(ictx, "'" + op + "' operand dtype/shape mismatch");
      }
    }
  } else if (op == "cmp_lt") {
    expectOperands(2);
    if (ins.id() < 0) {
      fail(ictx, "negative SSA id");
    }
    if (ins.dtype() != "bool") {
      fail(ictx, "cmp_lt must be bool");
    }
    for (int o : ins.operands()) {
      const DefInfo& d = lookupDef(defined, o, ictx);
      if (d.dtype != "i64" || d.shape != ins.shape()) {
        fail(ictx, "cmp_lt operand must be i64 with result shape");
      }
    }
  } else if (op == "select") {
    expectOperands(3);
    if (ins.id() < 0) {
      fail(ictx, "negative SSA id");
    }
    const DefInfo& c = lookupDef(defined, ins.operands()[0], ictx);
    const DefInfo& t = lookupDef(defined, ins.operands()[1], ictx);
    const DefInfo& f = lookupDef(defined, ins.operands()[2], ictx);
    if (c.dtype != "bool" || c.shape != ins.shape()) {
      fail(ictx, "select cond must be bool with result shape");
    }
    if (t.dtype != ins.dtype() || t.shape != ins.shape() ||
        f.dtype != ins.dtype() || f.shape != ins.shape()) {
      fail(ictx, "select branches must match result dtype/shape");
    }
  } else if (op == "fadd" || op == "fmul") {
    expectOperands(2);
    if (ins.id() < 0) {
      fail(ictx, "negative SSA id");
    }
    if (ins.dtype() != "f32") {
      fail(ictx, "'" + op + "' must be f32");
    }
    for (int o : ins.operands()) {
      const DefInfo& d = lookupDef(defined, o, ictx);
      if (d.dtype != "f32" || d.shape != ins.shape()) {
        fail(ictx, "'" + op + "' operand dtype/shape mismatch");
      }
    }
  } else if (op == "relu") {
    expectOperands(1);
    if (ins.id() < 0) {
      fail(ictx, "negative SSA id");
    }
    if (ins.dtype() != "f32") {
      fail(ictx, "relu must be f32");
    }
    const DefInfo& a = lookupDef(defined, ins.operands()[0], ictx);
    if (a.dtype != "f32" || a.shape != ins.shape()) {
      fail(ictx, "relu operand dtype/shape mismatch");
    }
  } else if (op == "load") {
    expectOperands(2);
    if (ins.id() < 0) {
      fail(ictx, "negative SSA id");
    }
    if (ins.dtype() != "f32" || !isVec) {
      fail(ictx, "load must define an f32 vector");
    }
    if (ins.buffer().empty() || params.find(ins.buffer()) == params.end()) {
      fail(ictx, "load of unknown parameter '" + ins.buffer() + "'");
    }
    const DefInfo& off = lookupDef(defined, ins.operands()[0], ictx);
    const DefInfo& msk = lookupDef(defined, ins.operands()[1], ictx);
    if (off.dtype != "i64" || off.shape != ins.shape()) {
      fail(ictx, "load offset must be i64 with result shape");
    }
    if (msk.dtype != "bool" || msk.shape != ins.shape()) {
      fail(ictx, "load mask must be bool with result shape");
    }
  } else if (op == "store") {
    expectOperands(3);
    if (ins.id() != -1) {
      fail(ictx, "store must not define a value (id must be -1)");
    }
    if (ins.buffer().empty() || params.find(ins.buffer()) == params.end()) {
      fail(ictx, "store to unknown parameter '" + ins.buffer() + "'");
    }
    const DefInfo& v = lookupDef(defined, ins.operands()[0], ictx);
    const DefInfo& off = lookupDef(defined, ins.operands()[1], ictx);
    const DefInfo& msk = lookupDef(defined, ins.operands()[2], ictx);
    if (v.dtype != "f32" || !isVecShape(v.shape, block)) {
      fail(ictx, "store value must be an f32 vector");
    }
    if (off.dtype != "i64" || off.shape != v.shape) {
      fail(ictx, "store offset dtype/shape mismatch with value");
    }
    if (msk.dtype != "bool" || msk.shape != v.shape) {
      fail(ictx, "store mask dtype/shape mismatch with value");
    }
  } else if (op == "cast") {
    expectOperands(1);
    if (ins.id() < 0) {
      fail(ictx, "negative SSA id");
    }
    const DefInfo& a = lookupDef(defined, ins.operands()[0], ictx);
    // Precision-preserving no-op only: dtype AND shape must be identical.
    if (a.dtype != ins.dtype() || a.shape != ins.shape()) {
      fail(
          ictx,
          "cast must preserve dtype/shape exactly (opaque conversions "
          "happen outside generated code)");
    }
  }

  if (op != "store") {
    if (defined.find(ins.id()) != defined.end()) {
      fail(ictx, "duplicate SSA v" + std::to_string(ins.id()));
    }
    defined[ins.id()] = DefInfo{ins.dtype(), ins.shape()};
  }
}

void verifyTiledFunction(const TiledFunction& f) {
  const std::string ctx = "tiled_function '" + f.name() + "'";
  if (f.name().empty()) {
    fail(ctx, "empty function name");
  }
  if (f.grid_size() < 0) {
    fail(ctx, "negative grid_size");
  }
  if (f.block_size() <= 0) {
    fail(ctx, "non-positive block_size");
  }
  if (f.num_warps() <= 0) {
    fail(ctx, "non-positive num_warps");
  }
  if (!isValidNumerics(f.numerics())) {
    fail(
        ctx,
        "unsupported numerics '" + f.numerics() +
            "' (allowed: 'strict_eager', 'inductor')");
  }
  if (f.numerics() == strictNumericsName()) {
    if (f.enable_fp_fusion()) {
      fail(ctx, "enable_fp_fusion must be False (strict eager)");
    }
    if (!f.disable_ftz()) {
      fail(ctx, "disable_ftz must be True (strict eager)");
    }
  }
  std::map<std::string, Buffer> params;
  for (const Buffer& p : f.parameters()) {
    std::string pctx = ctx + " parameter '" + p.name + "'";
    if (p.name.empty()) {
      fail(ctx, "parameter with empty name");
    }
    if (!params.emplace(p.name, p).second) {
      fail(pctx, "duplicate parameter");
    }
    if (p.elements < 0) {
      fail(pctx, "negative elements");
    }
  }
  if (f.blocks().empty()) {
    fail(ctx, "no blocks (expected entry + return)");
  }
  {
    std::set<std::string> bnames;
    for (const TiledBlock& b : f.blocks()) {
      if (b.name().empty()) {
        fail(ctx, "block with empty name");
      }
      if (!bnames.insert(b.name()).second) {
        fail(ctx + " block '" + b.name() + "'", "duplicate block name");
      }
      if (b.terminator() != "return") {
        fail(
            ctx + " block '" + b.name() + "'",
            "terminator must be 'return' (got '" + b.terminator() + "')");
      }
    }
    if (f.blocks()[0].name() != "entry") {
      fail(ctx, "first block must be 'entry'");
    }
  }
  std::map<int, DefInfo> defined;
  for (const TiledBlock& b : f.blocks()) {
    std::string bctx = ctx + " block '" + b.name() + "'";
    for (const TiledInstruction& ins : b.instructions()) {
      verifyTiledInstruction(ins, defined, params, f.block_size(), bctx);
    }
  }
}

} // namespace

void TiledProgram::verify() const {
  if (!isValidNumerics(numerics_)) {
    throw std::invalid_argument(
        "tiled_program: unsupported numerics '" + numerics_ +
        "' (allowed: 'strict_eager', 'inductor')");
  }
  std::set<std::string> fnames;
  for (const TiledFunction& f : functions_) {
    if (f.numerics() != numerics_) {
      throw std::invalid_argument(
          "tiled_program: function '" + f.name() + "' numerics '" +
          f.numerics() + "' does not match program numerics '" + numerics_ +
          "'");
    }
    if (f.name().empty()) {
      throw std::invalid_argument("tiled_program: function with empty name");
    }
    if (!fnames.insert(f.name()).second) {
      throw std::invalid_argument(
          "tiled_program: duplicate function '" + f.name() + "'");
    }
    verifyTiledFunction(f);
  }
}

std::string TiledProgram::dump() const {
  verify();
  std::ostringstream os;
  os << "tiled_program functions=" << functions_.size()
     << " numerics=" << numerics_ << "\n";
  for (const TiledFunction& f : functions_) {
    os << "tiled_function " << f.name() << " grid=" << f.grid_size()
       << " block=" << f.block_size() << " warps=" << f.num_warps()
       << " fp_fusion=" << (f.enable_fp_fusion() ? "true" : "false")
       << " ftz_disabled=" << (f.disable_ftz() ? "true" : "false")
       << " numerics=" << f.numerics() << "\n";
    os << "  parameters [";
    for (size_t i = 0; i < f.parameters().size(); ++i) {
      if (i) {
        os << ", ";
      }
      const Buffer& p = f.parameters()[i];
      os << "(" << p.name << " " << p.dtype << " " << p.elements << ")";
    }
    os << "]\n";
    for (const TiledBlock& b : f.blocks()) {
      os << "  block " << b.name() << " " << b.terminator() << "\n";
      for (const TiledInstruction& ins : b.instructions()) {
        os << "    ";
        if (ins.op() == "store") {
          os << "store " << ins.buffer() << "[v" << ins.operands()[1] << "]"
             << " v" << ins.operands()[0] << " mask=v" << ins.operands()[2];
        } else {
          os << "v" << ins.id() << " = " << ins.op();
          if (ins.op() == "const_int") {
            os << " " << ins.int_value();
          } else if (ins.op() == "const_float") {
            os << " " << formatFloat(ins.float_value());
          }
          if (!ins.buffer().empty()) {
            os << " " << ins.buffer();
          }
          for (int o : ins.operands()) {
            os << " v" << o;
          }
          os << " " << ins.dtype() << " " << shapeStr(ins.shape());
        }
        os << "\n";
      }
    }
  }
  return os.str();
}

// ---------------------------------------------------------------------------
// lower_to_tiled
// ---------------------------------------------------------------------------

namespace {

struct TileBuilder {
  const std::string ctx;
  int64_t block;
  TiledBlock& blk;
  int next = 0;

  explicit TileBuilder(const std::string& c, int64_t b, TiledBlock& blkRef)
      : ctx(c), block(b), blk(blkRef) {}

  int emit(const TiledInstruction& ins) {
    blk.addInstruction(ins);
    return ins.id();
  }
  int emitIntConst(int64_t v) {
    int id = next++;
    emit(TiledInstruction::ConstInt(id, v));
    return id;
  }
  int emitBroadcast(int scalar, const std::string& dtype) {
    int id = next++;
    emit(TiledInstruction::Broadcast(id, scalar, dtype, block));
    return id;
  }
  int emitIntVec(const std::string& op, int lhs, int rhs) {
    int id = next++;
    emit(TiledInstruction::IntBinary(op, id, lhs, rhs, block));
    return id;
  }
  // Lower one LoopIR Index tree to an i64 {block} SSA value. Fresh SSA per
  // Index node (no folding); variables resolve to decoded coordinate vectors.
  int lowerIndexTree(
      const Index& idx,
      const std::map<std::string, int>& coords) {
    const IndexExpr& e = idx.expr();
    switch (e.kind) {
      case IndexExpr::Kind::Constant: {
        int s = emitIntConst(e.value);
        return emitBroadcast(s, "i64");
      }
      case IndexExpr::Kind::Variable: {
        auto it = coords.find(e.name);
        if (it == coords.end()) {
          fail(
              ctx,
              "index variable '" + e.name +
                  "' has no decoded coordinate (unnormalized or reduction "
                  "variable?)");
        }
        return it->second;
      }
      case IndexExpr::Kind::Binary: {
        if (!e.lhs || !e.rhs) {
          fail(ctx, "binary index with empty operand");
        }
        int l = lowerIndexTree(Index(e.lhs), coords);
        int r = lowerIndexTree(Index(e.rhs), coords);
        return emitIntVec(e.op, l, r);
      }
    }
    fail(ctx, "bad index expression");
  }
};

// ---------------------------------------------------------------------------
// Flat-reuse analysis: checked-affine Index coefficients over exact int64
// semantics (no wrap/reassociation license -- LoopIR index math is exact;
// program.verify() runs before this pass and the core verifier proves
// in-bounds offsets on the executed domain). Any non-affine node
// (floordiv/mod), foreign variable, or overflow yields NO MATCH (false),
// never a new failure; the caller keeps coordinate decode.
// ---------------------------------------------------------------------------

struct AffineIndex {
  std::map<std::string, int64_t> coeff; // range variable -> coefficient
  int64_t constant = 0;
};

bool fitCheckedInt64(__int128 v, int64_t& out) {
  if (v > std::numeric_limits<int64_t>::max() ||
      v < std::numeric_limits<int64_t>::min()) {
    return false;
  }
  out = static_cast<int64_t>(v);
  return true;
}

bool analyzeAffineIndex(const Index& idx, AffineIndex& out) {
  const IndexExpr& e = idx.expr();
  switch (e.kind) {
    case IndexExpr::Kind::Constant:
      out.coeff.clear();
      out.constant = e.value;
      return true;
    case IndexExpr::Kind::Variable:
      out.coeff.clear();
      out.coeff[e.name] = 1;
      out.constant = 0;
      return true;
    case IndexExpr::Kind::Binary: {
      if (!e.lhs || !e.rhs) {
        return false;
      }
      AffineIndex l, r;
      if (!analyzeAffineIndex(Index(e.lhs), l)) {
        return false;
      }
      if (!analyzeAffineIndex(Index(e.rhs), r)) {
        return false;
      }
      if (e.op == "add") {
        out = l;
        for (const auto& kv : r.coeff) {
          __int128 s = (__int128)out.coeff[kv.first] + kv.second;
          if (!fitCheckedInt64(s, out.coeff[kv.first])) {
            return false;
          }
        }
        __int128 s = (__int128)out.constant + r.constant;
        return fitCheckedInt64(s, out.constant);
      }
      if (e.op == "mul") {
        // Exactly one side must be constant; var*var is non-affine.
        const AffineIndex* v = nullptr;
        int64_t k = 0;
        if (l.coeff.empty()) {
          v = &r;
          k = l.constant;
        } else if (r.coeff.empty()) {
          v = &l;
          k = r.constant;
        } else {
          return false;
        }
        out.coeff.clear();
        for (const auto& kv : v->coeff) {
          int64_t f = 0;
          if (!fitCheckedInt64((__int128)kv.second * k, f)) {
            return false;
          }
          if (f != 0) {
            out.coeff[kv.first] = f;
          }
        }
        return fitCheckedInt64((__int128)v->constant * k, out.constant);
      }
      return false; // floordiv/mod are non-affine: no match
    }
  }
  return false;
}

// True when the offset tree provably equals flat + base on the masked point
// domain: affine, every pointwise range var present with exactly its
// canonical row-major stride coefficient, no other vars, and
// base = treeConst + sum(start_i * stride_i) fitting int64 (checked).
// Identity proof: on valid lanes flat decodes (row-major division algorithm)
// to coords with flat == sum(raw_i * stride_i); substituting
// coord_i = raw_i + start_i gives tree == flat + base as integers, exact
// because verified in-bounds non-negative lane arithmetic cannot overflow,
// and masked tail lanes never escape through loads/stores.
bool matchCanonicalOffset(
    const Index& idx,
    const std::vector<Range>& ranges,
    const std::vector<int64_t>& strides,
    int64_t& base) {
  if (ranges.empty() || ranges.size() != strides.size()) {
    return false;
  }
  AffineIndex a;
  if (!analyzeAffineIndex(idx, a)) {
    return false;
  }
  if (a.coeff.size() != ranges.size()) {
    return false;
  }
  __int128 b = a.constant;
  for (size_t i = 0; i < ranges.size(); ++i) {
    auto it = a.coeff.find(ranges[i].name);
    if (it == a.coeff.end() || it->second != strides[i]) {
      return false;
    }
    b += (__int128)ranges[i].start * strides[i];
  }
  return fitCheckedInt64(b, base);
}

} // namespace

TiledProgram lower_to_tiled(
    const KernelProgram& program,
    const TileOptions& options) {
  const std::string pctx = "lower_to_tiled program '" + program.name() + "'";
  if (options.block_size <= 0 || options.num_warps <= 0) {
    throw std::invalid_argument(pctx + ": TileOptions must be positive");
  }
  program.verify();
  if (program.numerics() == strictNumericsName()) {
    // Strict keeps explicit rounded arithmetic: no contraction, no FTZ.
    if (options.enable_fp_fusion) {
      throw std::invalid_argument(
          pctx + ": strict_eager rejects enable_fp_fusion=True");
    }
    if (!options.disable_ftz) {
      throw std::invalid_argument(
          pctx + ": strict_eager rejects disable_ftz=False");
    }
  }
  // Inductor policy permits the effective flags carried by TileOptions.
  const int64_t block = options.block_size;

  // Buffer metadata: program inputs + every node result.
  std::map<std::string, Buffer> buffers;
  for (const Buffer& b : program.inputs()) {
    buffers[b.name] = b;
  }
  for (const auto& node : program.nodes()) {
    if (const auto* kf = std::get_if<KernelFunction>(&node)) {
      for (const Compute& r : kf->regions()) {
        for (const Buffer& b : r.results()) {
          buffers[b.name] = b;
        }
      }
    } else if (const auto* ec = std::get_if<ExternCall>(&node)) {
      for (const Buffer& b : ec->results()) {
        buffers[b.name] = b;
      }
    }
  }

  TiledProgram out(program.numerics());
  for (const auto& node : program.nodes()) {
    if (std::get_if<ExternCall>(&node) != nullptr) {
      // Reductions and opaque calls stay eager outside; no tiled function.
      continue;
    }
    const auto* kf = std::get_if<KernelFunction>(&node);
    if (kf == nullptr) {
      fail(pctx, "unknown kernel program node");
    }
    const std::string ctx = "kernel '" + kf->name() + "'";
    if (kf->regions().size() != 1) {
      fail(
          ctx,
          "expected exactly 1 region (got " +
              std::to_string(kf->regions().size()) + ")");
    }
    const Compute& c = kf->regions()[0];
    if (!c.reductionRanges().empty() || !c.accumulators().empty() ||
        !c.reductionBody().empty()) {
      fail(
          ctx,
          "reduction region cannot be tiled (reductions stay eager "
          "extern calls)");
    }

    // Total logical extent with checked math; any non-positive extent means
    // the whole domain is empty (LoopIR stop <= start per range).
    bool empty = false;
    std::vector<int64_t> extents;
    for (const Range& r : c.pointwiseRanges()) {
      int64_t ext = checkedAddSub(ctx, r.stop, r.start, false);
      extents.push_back(ext);
      if (ext <= 0) {
        empty = true;
      }
    }
    int64_t total = 0;
    if (c.pointwiseRanges().empty()) {
      total = 1;
    } else if (empty) {
      total = 0;
    } else {
      __int128 acc = 1;
      for (int64_t e : extents) {
        acc *= e;
        if (acc > std::numeric_limits<int64_t>::max()) {
          fail(ctx, "int64 overflow in total extent");
        }
      }
      total = static_cast<int64_t>(acc);
    }
    int64_t grid = 0;
    if (total > 0) {
      grid = checkedAddSub(ctx, total, block - 1, true) / block;
    }

    const std::vector<Range>& ranges = c.pointwiseRanges();
    const size_t nr = ranges.size();
    // Canonical row-major point-space strides: strides[i] = prod_{j>i}
    // extents[j], checked (identical failure behavior to the historical
    // inline computation). Shared by coordinate decode and the flat-reuse
    // proof below. Only meaningful for nonempty domains with >= 1 range;
    // empty and rank-0 domains keep their existing paths.
    std::vector<int64_t> strides;
    if (!empty && nr >= 1) {
      strides.assign(nr, 1);
      __int128 acc = 1;
      for (size_t i = nr; i-- > 0;) {
        if (acc > std::numeric_limits<int64_t>::max()) {
          fail(ctx, "int64 overflow in tile strides");
        }
        strides[i] = static_cast<int64_t>(acc);
        acc *= extents[i];
      }
    }

    // Launch-order parameters: distinct loaded inputs in first-use order,
    // then stored outputs in store order (outputs included as pointers).
    // Loads/stores of unknown buffers and non-f32 compute data are rejected:
    // opaque conversions happen outside generated code.
    std::vector<std::string> loadOrder, storeOrder;
    for (const Instruction& ins : c.postBody()) {
      switch (ins.kind()) {
        case Instruction::Kind::Load:
          if (std::find(loadOrder.begin(), loadOrder.end(), ins.buffer()) ==
              loadOrder.end()) {
            loadOrder.push_back(ins.buffer());
          }
          break;
        case Instruction::Kind::Store:
          if (std::find(storeOrder.begin(), storeOrder.end(), ins.buffer()) ==
              storeOrder.end()) {
            storeOrder.push_back(ins.buffer());
          }
          break;
        default:
          break;
      }
    }
    std::set<std::string> resultNames;
    for (const Buffer& b : c.results()) {
      resultNames.insert(b.name);
    }
    for (const std::string& s : storeOrder) {
      if (resultNames.find(s) == resultNames.end()) {
        fail(ctx, "store to '" + s + "' which is not an own result");
      }
    }
    std::vector<Buffer> params;
    std::set<std::string> seen;
    for (const std::string& n : loadOrder) {
      auto it = buffers.find(n);
      if (it == buffers.end()) {
        fail(ctx, "load of unknown buffer '" + n + "'");
      }
      if (it->second.dtype != "f32") {
        fail(
            ctx,
            "load of non-f32 buffer '" + n +
                "' (opaque conversions stay "
                "outside generated code)");
      }
      if (seen.insert(n).second) {
        params.push_back(it->second);
      }
    }
    for (const std::string& n : storeOrder) {
      auto it = buffers.find(n);
      if (it == buffers.end()) {
        // Result buffers are declared on the region; fall back to them.
        bool found = false;
        for (const Buffer& b : c.results()) {
          if (b.name == n) {
            if (b.dtype != "f32") {
              fail(ctx, "store to non-f32 result '" + n + "'");
            }
            if (seen.insert(n).second) {
              params.push_back(b);
            }
            found = true;
            break;
          }
        }
        if (!found) {
          fail(ctx, "store to unknown buffer '" + n + "'");
        }
      } else {
        if (it->second.dtype != "f32") {
          fail(ctx, "store to non-f32 buffer '" + n + "'");
        }
        if (seen.insert(n).second) {
          params.push_back(it->second);
        }
      }
    }

    TiledFunction f(
        kf->name(),
        params,
        grid,
        block,
        options.num_warps,
        options.enable_fp_fusion,
        options.disable_ftz,
        program.numerics());
    TiledBlock blk("entry");
    TileBuilder tb(ctx, block, blk);

    // Lane grid: flat = broadcast(pid * BLOCK) + arange; mask = flat < total.
    int pid = tb.next++;
    tb.emit(TiledInstruction::ProgramId(pid));
    int lanes = tb.next++;
    tb.emit(TiledInstruction::Arange(lanes, block));
    int blockScalar = tb.emitIntConst(block);
    int pidBlock = tb.next++;
    tb.emit(TiledInstruction::IntBinary("mul", pidBlock, pid, blockScalar, 0));
    int pidBlockVec = tb.emitBroadcast(pidBlock, "i64");
    int flat = tb.emitIntVec("add", pidBlockVec, lanes);
    int totalScalar = tb.emitIntConst(total);
    int totalVec = tb.emitBroadcast(totalScalar, "i64");
    int mask = tb.next++;
    tb.emit(TiledInstruction::CmpLt(mask, flat, totalVec, block));

    // Decode flat into original range coordinates (row-major). Empty domains
    // use const-0 coordinates with the all-false mask above, so no
    // floordiv/mod by zero is ever emitted. Rank-0 (scalar) domains have no
    // coordinates at all: offsets are const trees.
    //
    // Flat-reuse fast path: when every load/store offset tree provably equals
    // flat + base (checked-affine match against the canonical strides above),
    // memory ops reuse the flat lane SSA directly and decode is skipped. The
    // proof is per-memory-op and integer-only; masked lanes, FP SSA order,
    // and store materialization are untouched. Rank-0 and empty domains are
    // excluded (flat is a lane id / mask is all-false there, not a point).
    bool reuseFlat = false;
    bool haveBase = false;
    int64_t flatBase = 0;
    if (!empty && nr >= 1) {
      reuseFlat = true;
      for (const Instruction& ins : c.postBody()) {
        if (ins.kind() != Instruction::Kind::Load &&
            ins.kind() != Instruction::Kind::Store) {
          continue;
        }
        int64_t b = 0;
        if (!matchCanonicalOffset(ins.offset(), ranges, strides, b)) {
          reuseFlat = false;
          break;
        }
        if (!haveBase) {
          flatBase = b;
          haveBase = true;
        } else if (b != flatBase) {
          // Distinct per-op bases: fall back to decode. (Per-op flat+base
          // emission would also be sound, but one shared skip keeps the
          // transform minimal; mixed layouts are the rare case.)
          reuseFlat = false;
          break;
        }
      }
    }
    int flatBaseVec = -1;
    std::map<std::string, int> coordVec;
    if (reuseFlat) {
      if (flatBase != 0) {
        int s = tb.emitIntConst(flatBase);
        flatBaseVec = tb.emitBroadcast(s, "i64");
      }
    } else if (empty) {
      int zero = tb.emitIntConst(0);
      int zeroVec = tb.emitBroadcast(zero, "i64");
      for (const Range& r : ranges) {
        coordVec[r.name] = zeroVec;
      }
    } else if (nr == 1) {
      int coord = flat;
      if (ranges[0].start != 0) {
        int s = tb.emitIntConst(ranges[0].start);
        int sVec = tb.emitBroadcast(s, "i64");
        coord = tb.emitIntVec("add", flat, sVec);
      }
      coordVec[ranges[0].name] = coord;
    } else if (nr > 1) {
      for (size_t i = 0; i < nr; ++i) {
        int raw = -1;
        if (i == 0) {
          int st = tb.emitIntConst(strides[0]);
          int stVec = tb.emitBroadcast(st, "i64");
          raw = tb.emitIntVec("floordiv", flat, stVec);
        } else if (i == nr - 1) {
          int ex = tb.emitIntConst(extents[i]);
          int exVec = tb.emitBroadcast(ex, "i64");
          raw = tb.emitIntVec("mod", flat, exVec);
        } else {
          int st = tb.emitIntConst(strides[i]);
          int stVec = tb.emitBroadcast(st, "i64");
          int q = tb.emitIntVec("floordiv", flat, stVec);
          int ex = tb.emitIntConst(extents[i]);
          int exVec = tb.emitBroadcast(ex, "i64");
          raw = tb.emitIntVec("mod", q, exVec);
        }
        int coord = raw;
        if (ranges[i].start != 0) {
          int s = tb.emitIntConst(ranges[i].start);
          int sVec = tb.emitBroadcast(s, "i64");
          coord = tb.emitIntVec("add", raw, sVec);
        }
        coordVec[ranges[i].name] = coord;
      }
    }

    // Replay the LoopIR post body verbatim (strict order, no folding).
    std::map<int, int> valMap; // LoopIR SSA -> tiled f32 vector SSA.
    for (const Instruction& ins : c.postBody()) {
      switch (ins.kind()) {
        case Instruction::Kind::Constant: {
          int s = tb.next++;
          tb.emit(TiledInstruction::ConstFloat(s, ins.value()));
          int b = tb.next++;
          tb.emit(TiledInstruction::Broadcast(b, s, "f32", block));
          valMap[ins.id()] = b;
          break;
        }
        case Instruction::Kind::Load: {
          int off = -1;
          if (reuseFlat) {
            // Proven: tree == flat + flatBase on valid lanes. Integer
            // addresses only; the f32 value path below is untouched.
            off = (flatBase == 0) ? flat
                                  : tb.emitIntVec("add", flat, flatBaseVec);
          } else {
            off = tb.lowerIndexTree(ins.offset(), coordVec);
          }
          int id = tb.next++;
          tb.emit(TiledInstruction::Load(id, ins.buffer(), off, mask, block));
          valMap[ins.id()] = id;
          break;
        }
        case Instruction::Kind::Binary: {
          auto itL = valMap.find(ins.lhs());
          auto itR = valMap.find(ins.rhs());
          if (itL == valMap.end() || itR == valMap.end()) {
            fail(ctx, "binary uses undefined SSA value");
          }
          int id = tb.next++;
          tb.emit(
              TiledInstruction::FloatBinary(
                  ins.op() == "add" ? "fadd" : "fmul",
                  id,
                  itL->second,
                  itR->second,
                  block));
          valMap[ins.id()] = id;
          break;
        }
        case Instruction::Kind::Relu: {
          auto it = valMap.find(ins.arg());
          if (it == valMap.end()) {
            fail(ctx, "relu uses undefined SSA value");
          }
          int id = tb.next++;
          tb.emit(TiledInstruction::Relu(id, it->second, block));
          valMap[ins.id()] = id;
          break;
        }
        case Instruction::Kind::Store: {
          auto it = valMap.find(ins.arg());
          if (it == valMap.end()) {
            fail(ctx, "store uses undefined SSA value");
          }
          int off = -1;
          if (reuseFlat) {
            off = (flatBase == 0) ? flat
                                  : tb.emitIntVec("add", flat, flatBaseVec);
          } else {
            off = tb.lowerIndexTree(ins.offset(), coordVec);
          }
          tb.blk.addInstruction(
              TiledInstruction::Store(ins.buffer(), it->second, off, mask));
          break;
        }
        case Instruction::Kind::ViewRead:
          fail(
              ctx, "view_read must be inlined before tiling (run view_inline)");
        case Instruction::Kind::Accumulate:
        case Instruction::Kind::ReadAccumulator:
          fail(
              ctx,
              "reduction instruction cannot be tiled (reductions stay eager "
              "extern calls)");
      }
    }

    f.addBlock(blk);
    out.addFunction(f);
  }
  out.verify();
  return out;
}

} // namespace torch::inductor::loops_ir
