// Verifier + index interval analysis. Throws std::invalid_argument with node
// context on any error. No torch/c10 deps.
// Strict numerics: fixed 'strict_eager'. Compute data f32-only; program
// inputs and extern boundaries allow common opaque dtypes.
//
// Notes on scope (not full proofs):
// - Store coverage: nonempty domain requires >=1 store per result; narrow
//   dense/injective proofs live in passes.cpp for fusion only.
// - Extern calls are opaque; generic output acquisition contract only.

#include "ir.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

namespace torch::inductor::loops_ir {
namespace {

[[noreturn]] void fail(const std::string& ctx, const std::string& msg) {
  throw std::invalid_argument(ctx + ": " + msg);
}

// Checked arithmetic: throws on int64 overflow (never saturates).
__int128 to128(int64_t v) {
  return static_cast<__int128>(v);
}
int64_t checked128(const std::string& ctx, __int128 v) {
  if (v > std::numeric_limits<int64_t>::max() ||
      v < std::numeric_limits<int64_t>::min()) {
    fail(ctx, "int64 overflow in index arithmetic");
  }
  return static_cast<int64_t>(v);
}
int64_t checkedAdd(const std::string& ctx, int64_t a, int64_t b) {
  return checked128(ctx, to128(a) + to128(b));
}
int64_t checkedMul(const std::string& ctx, int64_t a, int64_t b) {
  return checked128(ctx, to128(a) * to128(b));
}

// Floor division (b > 0 required by callers; avoids INT64_MIN/-1 UB since
// divisor is provably positive).
int64_t floorDivPos(int64_t a, int64_t b) {
  int64_t q = a / b;
  int64_t r = a % b;
  if (r != 0 && b > 0 && r < 0) {
    --q;
  }
  return q;
}

void requireFiniteF32(
    const std::string& ctx,
    double v,
    const std::string& what) {
  if (!std::isfinite(v)) {
    fail(ctx, what + " is non-finite double");
  }
  float f = static_cast<float>(v);
  if (!std::isfinite(f)) {
    fail(ctx, what + " overflows f32 (e.g. 1e100 rejected)");
  }
}

void requireF32(
    const std::string& ctx,
    const std::string& dtype,
    const std::string& what) {
  if (dtype == "index") {
    fail(
        ctx,
        what +
            " uses reserved dtype 'index' (integer Index nodes are separate)");
  }
  if (dtype != "f32") {
    fail(ctx, what + " has unsupported dtype '" + dtype + "' (only 'f32')");
  }
}

// Opaque boundary check: program inputs and extern inputs/results may use
// common boundary dtypes. Compute data stays f32-only (requireF32 above).
void requireBoundaryDtype(
    const std::string& ctx,
    const std::string& dtype,
    const std::string& what) {
  if (!isBoundaryDtype(dtype)) {
    if (dtype == "index") {
      fail(
          ctx,
          what +
              " uses reserved dtype 'index' (integer Index nodes are separate)");
    }
    fail(ctx, what + " has unsupported dtype '" + dtype + "'");
  }
}

// Resolve an alias chain (view -> ... -> storage). Returns {base, composed}.
// Throws on unknown names, cycles, or negative composed offsets.
std::pair<std::string, int64_t> resolveAlias(
    const std::string& source,
    int64_t relOffset,
    const std::map<std::string, int64_t>& storage,
    const std::map<std::string, View>& views,
    const std::string& ctx) {
  std::string cur = source;
  __int128 off = relOffset;
  std::set<std::string> seen;
  while (true) {
    auto vit = views.find(cur);
    if (vit == views.end()) {
      if (storage.find(cur) == storage.end()) {
        fail(ctx, "unknown storage or view '" + cur + "'");
      }
      if (off < 0) {
        fail(ctx, "negative composed offset");
      }
      if (off > std::numeric_limits<int64_t>::max()) {
        fail(ctx, "offset overflow");
      }
      return {cur, static_cast<int64_t>(off)};
    }
    if (!seen.insert(cur).second) {
      fail(ctx, "view alias cycle at '" + cur + "'");
    }
    off += vit->second.offset;
    if (off < 0) {
      fail(ctx, "negative composed offset via view '" + cur + "'");
    }
    if (off > std::numeric_limits<int64_t>::max()) {
      fail(ctx, "offset overflow via view '" + cur + "'");
    }
    cur = vit->second.source;
  }
}

// Validate sizes/strides/offset ranks, signs, and physical capacity against
// base storage. Zero sizes mean empty: no max-element access is checked.
// Each shape-span increment is checked against capacity immediately so a
// multi-dim span cannot overflow __int128 reasoning or hide OOB.
void checkShapeBounds(
    const std::string& ctx,
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& strides,
    int64_t offset,
    const std::string& base,
    int64_t composed,
    int64_t baseElements) {
  if (sizes.size() != strides.size()) {
    fail(ctx, "sizes/strides rank mismatch");
  }
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (sizes[i] < 0) {
      fail(ctx, "negative size");
    }
    if (strides[i] < 0) {
      fail(ctx, "negative stride unsupported");
    }
  }
  if (offset < 0) {
    fail(ctx, "negative offset");
  }
  if (composed < 0) {
    fail(ctx, "negative composed offset");
  }
  bool empty = false;
  for (int64_t s : sizes) {
    if (s == 0) {
      empty = true;
    }
  }
  if (!empty) {
    // Rank-0 (no sizes): single element at the composed offset.
    __int128 mx = composed;
    for (size_t i = 0; i < sizes.size(); ++i) {
      __int128 span = (__int128)(sizes[i] - 1) * strides[i];
      __int128 next = mx + span;
      // Incremental capacity check per dim.
      if (next >= (__int128)baseElements) {
        fail(ctx, "shape exceeds storage '" + base + "' capacity");
      }
      if (next > (__int128)std::numeric_limits<int64_t>::max() ||
          next < (__int128)std::numeric_limits<int64_t>::min()) {
        fail(ctx, "shape span int64 overflow");
      }
      mx = next;
    }
    if (mx >= baseElements) {
      fail(ctx, "shape exceeds storage '" + base + "' capacity");
    }
  }
}

void checkIndexVars(
    const Index& idx,
    const std::map<std::string, Interval>& scope,
    const std::string& ictx) {
  idx.walk([&](const IndexExpr& e) {
    if (e.kind == IndexExpr::Kind::Variable &&
        scope.find(e.name) == scope.end()) {
      fail(ictx, "index variable '" + e.name + "' out of scope");
    }
  });
}

Interval evalWithCtx(
    const Index& idx,
    const std::map<std::string, Interval>& scope,
    const std::string& ictx) {
  try {
    return evalIndexInterval(idx, scope);
  } catch (const std::invalid_argument& ex) {
    fail(ictx, std::string("bad index: ") + ex.what());
  }
}

void checkFlatBounds(
    const Index& idx,
    const std::map<std::string, Interval>& scope,
    int64_t elements,
    const std::string& what,
    const std::string& ictx) {
  checkIndexVars(idx, scope, ictx);
  Interval it = evalWithCtx(idx, scope, ictx);
  if (it.lo < 0 || it.hi >= elements) {
    fail(
        ictx,
        what + " index " + idx.toString() + " may be out of bounds [0, " +
            std::to_string(elements) + ")");
  }
}

} // namespace

Interval evalIndexInterval(
    const Index& idx,
    const std::map<std::string, Interval>& varBounds) {
  const std::string ctx = "index '" + idx.toString() + "'";
  const IndexExpr& e = idx.expr();
  switch (e.kind) {
    case IndexExpr::Kind::Constant:
      return Interval{e.value, e.value};
    case IndexExpr::Kind::Variable: {
      auto it = varBounds.find(e.name);
      if (it == varBounds.end()) {
        throw std::invalid_argument(
            "index uses undeclared variable '" + e.name + "'");
      }
      return it->second;
    }
    case IndexExpr::Kind::Binary: {
      if (!e.lhs || !e.rhs) {
        throw std::invalid_argument("binary with empty operand");
      }
      Interval l = evalIndexInterval(Index(e.lhs), varBounds);
      Interval r = evalIndexInterval(Index(e.rhs), varBounds);
      if (l.lo > l.hi || r.lo > r.hi) {
        return Interval{0, -1}; // empty propagates
      }
      if (e.op == "add") {
        // Check all four corner combos for overflow; extremes suffice.
        int64_t mn = checkedAdd(ctx, l.lo, r.lo);
        int64_t mx = checkedAdd(ctx, l.hi, r.hi);
        // For mixed signs the true min/max are still at corners; verify
        // all corners to catch overflow anywhere.
        int64_t corners[4];
        corners[0] = mn;
        corners[1] = checkedAdd(ctx, l.lo, r.hi);
        corners[2] = checkedAdd(ctx, l.hi, r.lo);
        corners[3] = mx;
        int64_t lo = corners[0], hi = corners[0];
        for (int i = 1; i < 4; ++i) {
          lo = std::min(lo, corners[i]);
          hi = std::max(hi, corners[i]);
        }
        return Interval{lo, hi};
      }
      if (e.op == "mul") {
        int64_t c[4] = {
            checkedMul(ctx, l.lo, r.lo),
            checkedMul(ctx, l.lo, r.hi),
            checkedMul(ctx, l.hi, r.lo),
            checkedMul(ctx, l.hi, r.hi)};
        int64_t mn = c[0], mx = c[0];
        for (int i = 1; i < 4; ++i) {
          mn = std::min(mn, c[i]);
          mx = std::max(mx, c[i]);
        }
        return Interval{mn, mx};
      }
      if (e.op == "floordiv") {
        // Divisor must be provably positive (avoids 0 and INT64_MIN/-1 UB).
        if (r.lo <= 0) {
          throw std::invalid_argument(
              "floordiv divisor must be provably positive");
        }
        int64_t c[4] = {
            floorDivPos(l.lo, r.lo),
            floorDivPos(l.lo, r.hi),
            floorDivPos(l.hi, r.lo),
            floorDivPos(l.hi, r.hi)};
        int64_t mn = c[0], mx = c[0];
        for (int i = 1; i < 4; ++i) {
          mn = std::min(mn, c[i]);
          mx = std::max(mx, c[i]);
        }
        return Interval{mn, mx};
      }
      if (e.op == "mod") {
        // Positive constant divisor only.
        if (r.lo == r.hi && r.lo > 0) {
          return Interval{0, r.lo - 1};
        }
        throw std::invalid_argument(
            "mod requires positive constant divisor in bounds check");
      }
      throw std::invalid_argument("unsupported index op '" + e.op + "'");
    }
  }
  throw std::invalid_argument("bad index expression");
}

// --- Instruction factories with typed checks ---

// Structural tree equality. Never raw pointer equality on shared nodes.
namespace {
bool indexExprEquals(
    const std::shared_ptr<const IndexExpr>& a,
    const std::shared_ptr<const IndexExpr>& b) {
  if (!a && !b) {
    return true;
  }
  if (!a || !b) {
    return false;
  }
  if (a->kind != b->kind) {
    return false;
  }
  switch (a->kind) {
    case IndexExpr::Kind::Constant:
      return a->value == b->value;
    case IndexExpr::Kind::Variable:
      return a->name == b->name;
    case IndexExpr::Kind::Binary:
      return a->op == b->op && indexExprEquals(a->lhs, b->lhs) &&
          indexExprEquals(a->rhs, b->rhs);
  }
  return false;
}
} // namespace

bool indexEquals(const Index& a, const Index& b) {
  if (a.empty() && b.empty()) {
    return true;
  }
  if (a.empty() || b.empty()) {
    return false;
  }
  return indexExprEquals(a.ptr(), b.ptr());
}

bool isBoundaryDtype(const std::string& dtype) {
  return dtype == "f16" || dtype == "bf16" || dtype == "f32" ||
      dtype == "f64" || dtype == "i8" || dtype == "u8" || dtype == "i16" ||
      dtype == "i32" || dtype == "i64" || dtype == "bool" || dtype == "c64" ||
      dtype == "c128";
}

bool isComputeDataDtype(const std::string& dtype) {
  return dtype == "f32";
}

Instruction Instruction::Constant(int id, double value) {
  requireFiniteF32(
      "Instruction.constant v" + std::to_string(id), value, "constant");
  Instruction i;
  i.kind_ = Kind::Constant;
  i.id_ = id;
  i.value_ = value;
  return i;
}

Instruction Instruction::Load(int id, std::string buffer, Index offset) {
  if (offset.empty()) {
    throw std::invalid_argument("Instruction.load: empty offset");
  }
  Instruction i;
  i.kind_ = Kind::Load;
  i.id_ = id;
  i.buffer_ = std::move(buffer);
  i.offset_ = std::move(offset);
  return i;
}

Instruction Instruction::ViewRead(
    int id,
    std::string view,
    std::vector<Index> coordinates) {
  for (size_t k = 0; k < coordinates.size(); ++k) {
    if (coordinates[k].empty()) {
      throw std::invalid_argument("Instruction.view_read: empty coordinate");
    }
  }
  Instruction i;
  i.kind_ = Kind::ViewRead;
  i.id_ = id;
  i.view_ = std::move(view);
  i.coords_ = std::move(coordinates);
  return i;
}

Instruction Instruction::Store(std::string buffer, Index offset, int value) {
  if (offset.empty()) {
    throw std::invalid_argument("Instruction.store: empty offset");
  }
  Instruction i;
  i.kind_ = Kind::Store;
  i.id_ = -1;
  i.buffer_ = std::move(buffer);
  i.offset_ = std::move(offset);
  i.arg_ = value;
  return i;
}

std::string Instruction::kindName() const {
  switch (kind_) {
    case Kind::Constant:
      return "constant";
    case Kind::Load:
      return "load";
    case Kind::ViewRead:
      return "view_read";
    case Kind::Binary:
      return "binary";
    case Kind::Relu:
      return "relu";
    case Kind::Accumulate:
      return "accumulate";
    case Kind::ReadAccumulator:
      return "read_accumulator";
    case Kind::Store:
      return "store";
  }
  return "unknown";
}

std::string Index::toString() const {
  const IndexExpr& e = expr();
  switch (e.kind) {
    case IndexExpr::Kind::Constant:
      return std::to_string(e.value);
    case IndexExpr::Kind::Variable:
      return e.name;
    case IndexExpr::Kind::Binary: {
      std::string sym = e.op;
      if (e.op == "add") {
        sym = "+";
      } else if (e.op == "mul") {
        sym = "*";
      }
      return "(" + Index(e.lhs).toString() + " " + sym + " " +
          Index(e.rhs).toString() + ")";
    }
  }
  throw std::invalid_argument("Index: bad expression");
}

// --- ExternCall ---

CallArg CallArg::MakeNone() {
  return CallArg();
}

CallArg CallArg::MakeTensor(std::string name) {
  if (name.empty()) {
    throw std::invalid_argument("CallArg.tensor: empty name");
  }
  CallArg a;
  a.kind_ = Kind::Tensor;
  a.tensor_name_ = std::move(name);
  return a;
}

CallArg CallArg::MakeInteger(int64_t v) {
  CallArg a;
  a.kind_ = Kind::Int;
  a.int_value_ = v;
  return a;
}

CallArg CallArg::MakeFloating(double v) {
  if (std::isnan(v)) {
    throw std::invalid_argument("CallArg.floating: NaN value rejected");
  }
  // Infinities are allowed as static mask constants (e.g. attention -inf);
  // finite values and infinities round-trip exactly via formatDoubleExact.
  // NaN payloads cannot round-trip via %.17g and stay rejected.
  CallArg a;
  a.kind_ = Kind::Float;
  a.float_value_ = v;
  return a;
}

CallArg CallArg::MakeBoolean(bool v) {
  CallArg a;
  a.kind_ = Kind::Bool;
  a.bool_value_ = v;
  return a;
}

CallArg CallArg::MakeString(std::string s) {
  CallArg a;
  a.kind_ = Kind::String;
  a.text_ = std::move(s);
  return a;
}

CallArg CallArg::MakeDtype(std::string d) {
  if (d.empty()) {
    throw std::invalid_argument("CallArg.dtype: empty name");
  }
  CallArg a;
  a.kind_ = Kind::Dtype;
  a.text_ = std::move(d);
  return a;
}

// Validated layout names. Only 'strided' carries full metadata today; other
// layout metadata stays behind the whitelist so the emitter never renders an
// arbitrary layout expression. Declared inline in ir.h.
CallArg CallArg::MakeLayout(std::string name) {
  if (!isValidLayoutName(name)) {
    throw std::invalid_argument(
        "CallArg.layout: unsupported layout '" + name + "'");
  }
  CallArg a;
  a.kind_ = Kind::Layout;
  a.text_ = std::move(name);
  return a;
}

// Validated memory-format names (common contiguous/preserve/channels_last).
// Declared inline in ir.h.
CallArg CallArg::MakeMemoryFormat(std::string name) {
  if (!isValidMemoryFormatName(name)) {
    throw std::invalid_argument(
        "CallArg.memory_format: unsupported memory format '" + name + "'");
  }
  CallArg a;
  a.kind_ = Kind::MemoryFormat;
  a.text_ = std::move(name);
  return a;
}

CallArg CallArg::MakeList(std::vector<CallArg> items) {
  CallArg a;
  a.kind_ = Kind::List;
  a.items_ = std::move(items);
  return a;
}

CallArg CallArg::MakeTuple(std::vector<CallArg> items) {
  CallArg a;
  a.kind_ = Kind::Tuple;
  a.items_ = std::move(items);
  return a;
}

std::string CallArg::kindName() const {
  switch (kind_) {
    case Kind::None:
      return "none";
    case Kind::Tensor:
      return "tensor";
    case Kind::Int:
      return "int";
    case Kind::Float:
      return "float";
    case Kind::Bool:
      return "bool";
    case Kind::String:
      return "string";
    case Kind::Dtype:
      return "dtype";
    case Kind::Layout:
      return "layout";
    case Kind::MemoryFormat:
      return "memory_format";
    case Kind::List:
      return "list";
    case Kind::Tuple:
      return "tuple";
  }
  return "unknown";
}

namespace {
// Collect ordered tensor refs recursively (repeats allowed).
void collectArgTensors(const CallArg& a, std::vector<std::string>& out) {
  if (a.kind() == CallArg::Kind::Tensor) {
    out.push_back(a.tensorName());
  } else if (
      a.kind() == CallArg::Kind::List || a.kind() == CallArg::Kind::Tuple) {
    for (const auto& it : a.items()) {
      collectArgTensors(it, out);
    }
  }
}

// Validate non-tensor payloads exactly (finite floats, nonempty tensor/dtype).
void checkCallArg(const CallArg& a, const std::string& ctx) {
  switch (a.kind()) {
    case CallArg::Kind::None:
    case CallArg::Kind::Int:
    case CallArg::Kind::Bool:
    case CallArg::Kind::String:
      return;
    case CallArg::Kind::Tensor:
      if (a.tensorName().empty()) {
        fail(ctx, "tensor arg with empty name");
      }
      return;
    case CallArg::Kind::Float:
      if (std::isnan(a.floatValue())) {
        fail(ctx, "NaN float arg rejected (payload cannot round-trip)");
      }
      // Infinities allowed as static mask constants; exact bits preserved via
      // formatDoubleExact ("inf"/"-inf" round-trip exactly).
      return;
    case CallArg::Kind::Dtype:
      if (a.text().empty()) {
        fail(ctx, "dtype arg with empty name");
      }
      return;
    case CallArg::Kind::Layout:
      if (!isValidLayoutName(a.text())) {
        fail(ctx, "unsupported layout '" + a.text() + "'");
      }
      return;
    case CallArg::Kind::MemoryFormat:
      if (!isValidMemoryFormatName(a.text())) {
        fail(ctx, "unsupported memory format '" + a.text() + "'");
      }
      return;
    case CallArg::Kind::List:
    case CallArg::Kind::Tuple:
      for (const auto& it : a.items()) {
        checkCallArg(it, ctx);
      }
      return;
  }
  fail(ctx, "unknown call arg kind");
}
} // namespace

ExternCall::ExternCall(
    std::string n,
    std::string tgt,
    std::vector<std::string> in,
    std::vector<Buffer> res,
    std::vector<double> sc,
    std::string eff,
    bool mayAlias,
    std::vector<Descriptor> desc)
    : name_(std::move(n)),
      target_(std::move(tgt)),
      inputs_(std::move(in)),
      results_(std::move(res)),
      scalar_args_(std::move(sc)),
      effects_(std::move(eff)),
      may_alias_inputs_(mayAlias),
      descriptors_(std::move(desc)) {
  // Original path populates typed arguments: Tensor inputs then Float
  // scalar args (preserves construction contract for the Python lowerer).
  for (const auto& t : inputs_) {
    arguments_.push_back(CallArg::MakeTensor(t));
  }
  for (double s : scalar_args_) {
    arguments_.push_back(CallArg::MakeFloating(s));
  }
}

ExternCall ExternCall::Structured(
    std::string n,
    std::string tgt,
    std::vector<CallArg> args,
    std::vector<CallKeyword> kwargs,
    std::vector<Buffer> res,
    std::vector<Descriptor> desc,
    std::string eff,
    bool mayAlias) {
  if (n.empty()) {
    throw std::invalid_argument("ExternCall.structured: empty name");
  }
  if (tgt.empty()) {
    throw std::invalid_argument("ExternCall.structured: empty target");
  }
  if (eff.empty()) {
    throw std::invalid_argument("ExternCall.structured: empty effects");
  }
  std::vector<std::string> in;
  for (const auto& a : args) {
    checkCallArg(a, "extern '" + n + "'");
    collectArgTensors(a, in);
  }
  for (const auto& k : kwargs) {
    if (k.name().empty()) {
      throw std::invalid_argument("ExternCall.structured: empty keyword name");
    }
    checkCallArg(k.value(), "extern '" + n + "'");
    collectArgTensors(k.value(), in);
  }
  ExternCall e;
  e.name_ = std::move(n);
  e.target_ = std::move(tgt);
  e.inputs_ = std::move(in);
  e.results_ = std::move(res);
  e.effects_ = std::move(eff);
  e.may_alias_inputs_ = mayAlias;
  e.descriptors_ = std::move(desc);
  e.arguments_ = std::move(args);
  e.keyword_args_ = std::move(kwargs);
  return e;
}

namespace {

// Resolve the storage dtype behind a buffer-or-view name.
std::string storageDtype(
    const std::string& name,
    const std::map<std::string, std::string>& dtypes,
    const std::map<std::string, View>& views) {
  std::string cur = name;
  std::set<std::string> seen;
  while (true) {
    auto vit = views.find(cur);
    if (vit == views.end()) {
      auto it = dtypes.find(cur);
      return it == dtypes.end() ? std::string() : it->second;
    }
    if (!seen.insert(cur).second) {
      return std::string();
    }
    cur = vit->second.source;
  }
}

void verifyComputeNode(
    const Compute& c,
    std::map<std::string, int64_t>& storage,
    std::map<std::string, std::string>& dtypes,
    const std::map<std::string, View>& views,
    std::set<std::string>& nodeNames,
    const std::string& pctx) {
  const std::string ctx = "compute '" + c.name() + "'";
  if (c.name().empty()) {
    fail(pctx, "compute with empty name");
  }
  if (nodeNames.count(c.name()) || storage.count(c.name()) ||
      views.count(c.name())) {
    fail(ctx, "duplicate node/buffer/view name");
  }

  // Ranges: unique names, lexical scope maps.
  std::set<std::string> rangeNames;
  std::map<std::string, Interval> pwBounds, redScope;
  for (const Range& r : c.pointwiseRanges()) {
    if (r.name.empty()) {
      fail(ctx, "range with empty name");
    }
    if (!rangeNames.insert(r.name).second) {
      fail(ctx, "duplicate range '" + r.name + "'");
    }
    pwBounds[r.name] =
        (r.stop > r.start) ? Interval{r.start, r.stop - 1} : Interval{0, -1};
  }
  redScope = pwBounds;
  for (const Range& r : c.reductionRanges()) {
    if (r.name.empty()) {
      fail(ctx, "range with empty name");
    }
    if (!rangeNames.insert(r.name).second) {
      fail(ctx, "duplicate range '" + r.name + "'");
    }
    redScope[r.name] =
        (r.stop > r.start) ? Interval{r.start, r.stop - 1} : Interval{0, -1};
  }
  bool pwExec = true;
  for (const auto& kv : pwBounds) {
    if (kv.second.lo > kv.second.hi) {
      pwExec = false;
    }
  }
  bool redExec = pwExec;
  for (const Range& r : c.reductionRanges()) {
    auto it = redScope.find(r.name);
    if (it != redScope.end() && it->second.lo > it->second.hi) {
      redExec = false;
    }
  }

  // Accumulators: independent 'add' only; general coupled reducers not
  // implemented (not claimed).
  size_t nacc = c.accumulators().size();
  {
    std::set<std::string> an;
    for (const Accumulator& a : c.accumulators()) {
      std::string actx = ctx + " accumulator '" + a.name + "'";
      if (a.name.empty()) {
        fail(ctx, "accumulator with empty name");
      }
      if (!an.insert(a.name).second) {
        fail(actx, "duplicate accumulator");
      }
      requireF32(actx, a.dtype, "accumulator");
      requireFiniteF32(actx, a.identity, "accumulator identity");
      if (a.combine != "add") {
        fail(
            actx,
            "unsupported combine '" + a.combine +
                "' (only 'add'; coupled reducers not implemented)");
      }
    }
  }
  if (!c.reductionRanges().empty() && nacc == 0) {
    fail(ctx, "reduction ranges without accumulators");
  }
  if (c.reductionRanges().empty() && nacc > 0) {
    fail(ctx, "accumulators with no reduction ranges");
  }
  if (c.reductionRanges().empty() && !c.reductionBody().empty()) {
    fail(ctx, "reduction body must be empty without reduction ranges");
  }

  // Inputs: unique; every load/view_read target must be declared here.
  // Unused declared inputs are OK.
  std::set<std::string> inputSet;
  for (const std::string& in : c.inputs()) {
    std::string ictx = ctx + " input '" + in + "'";
    if (in.empty()) {
      fail(ctx, "empty input name");
    }
    if (!inputSet.insert(in).second) {
      fail(ictx, "duplicate input");
    }
    if (storage.find(in) == storage.end() && views.find(in) == views.end()) {
      fail(ictx, "unknown buffer or view (not topologically available)");
    }
    // Compute loaded/input data must be f32; opaque boundary dtypes (e.g.
    // i64 indices consumed via externs) are rejected here, not silently
    // promoted.
    std::string dt = storageDtype(in, dtypes, views);
    if (dt.empty()) {
      fail(ictx, "unknown dtype for input '" + in + "'");
    }
    if (!isComputeDataDtype(dt)) {
      fail(
          ictx,
          "compute input '" + in + "' has dtype '" + dt +
              "' (compute data must be f32)");
    }
  }

  // Results: new f32 buffers, disjoint from inputs.
  std::set<std::string> resultSet;
  std::map<std::string, int64_t> resultElems;
  for (const Buffer& b : c.results()) {
    std::string rctx = ctx + " result '" + b.name + "'";
    if (b.name.empty()) {
      fail(ctx, "result with empty name");
    }
    if (inputSet.count(b.name)) {
      fail(rctx, "same symbol as input (input/result alias disallowed)");
    }
    if (storage.count(b.name) || views.count(b.name)) {
      fail(rctx, "duplicate buffer name");
    }
    if (!resultSet.insert(b.name).second) {
      fail(rctx, "duplicate result");
    }
    requireF32(rctx, b.dtype, "buffer");
    if (b.elements < 0) {
      fail(rctx, "negative elements");
    }
    resultElems[b.name] = b.elements;
  }

  auto checkBody = [&](const std::vector<Instruction>& body, bool isReduction) {
    const std::string bctx =
        ctx + (isReduction ? " reduction_body" : " post_body");
    // Reduction range vars are in scope only in the reduction body; the
    // post body sees pointwise vars only.
    const std::map<std::string, Interval>& scope =
        isReduction ? redScope : pwBounds;
    const bool execNow = isReduction ? redExec : pwExec;
    std::set<int> defined;
    auto useRef = [&](int id, const std::string& ictx) {
      if (defined.find(id) == defined.end()) {
        fail(
            ictx,
            "use of undefined SSA v" + std::to_string(id) +
                " (must dominate use in this body;"
                " reduction SSA not visible in post body)");
      }
    };
    auto defId = [&](int id, const std::string& ictx) {
      if (id < 0) {
        fail(ictx, "negative SSA id");
      }
      if (!defined.insert(id).second) {
        fail(ictx, "duplicate SSA v" + std::to_string(id));
      }
    };
    for (size_t k = 0; k < body.size(); ++k) {
      const Instruction& ins = body[k];
      const std::string ictx =
          bctx + " instruction[" + std::to_string(k) + "] " + ins.kindName();
      switch (ins.kind()) {
        case Instruction::Kind::Constant:
          defId(ins.id(), ictx);
          requireFiniteF32(ictx, ins.value(), "constant");
          break;
        case Instruction::Kind::Load: {
          defId(ins.id(), ictx);
          if (!inputSet.count(ins.buffer())) {
            fail(
                ictx,
                "load of '" + ins.buffer() +
                    "' not declared in compute inputs");
          }
          if (views.count(ins.buffer())) {
            fail(ictx, "load of view '" + ins.buffer() + "' (use view_read)");
          }
          if (resultSet.count(ins.buffer())) {
            fail(
                ictx,
                "load of own result '" + ins.buffer() + "' before defined");
          }
          auto sit = storage.find(ins.buffer());
          if (sit == storage.end()) {
            fail(ictx, "load of unknown buffer '" + ins.buffer() + "'");
          }
          auto dit = dtypes.find(ins.buffer());
          if (dit == dtypes.end() || !isComputeDataDtype(dit->second)) {
            fail(
                ictx,
                "load of non-f32 buffer '" + ins.buffer() +
                    "' (compute data must be f32)");
          }
          if (execNow) {
            checkFlatBounds(
                ins.offset(),
                scope,
                sit->second,
                "load from '" + ins.buffer() + "'",
                ictx);
          } else {
            checkIndexVars(ins.offset(), scope, ictx);
          }
          break;
        }
        case Instruction::Kind::ViewRead: {
          defId(ins.id(), ictx);
          auto vit = views.find(ins.view());
          if (vit == views.end()) {
            if (storage.count(ins.view())) {
              fail(ictx, "'" + ins.view() + "' is a buffer (use load)");
            }
            fail(ictx, "view_read of unknown view '" + ins.view() + "'");
          }
          if (!inputSet.count(ins.view())) {
            fail(
                ictx,
                "view_read of '" + ins.view() +
                    "' not declared in compute inputs");
          }
          const View& vw = vit->second;
          if (ins.coordinates().size() != vw.sizes.size()) {
            fail(ictx, "coordinate rank mismatch for view '" + vw.name + "'");
          }
          for (size_t d = 0; d < ins.coordinates().size(); ++d) {
            std::string dctx = ictx + " coord c" + std::to_string(d) +
                " of view '" + vw.name + "'";
            checkIndexVars(ins.coordinates()[d], scope, dctx);
            if (execNow) {
              if (vw.sizes[d] == 0) {
                fail(dctx, "coordinate access into zero-size dimension");
              }
              Interval it = evalWithCtx(ins.coordinates()[d], scope, dctx);
              if (it.lo < 0 || it.hi > vw.sizes[d] - 1) {
                fail(dctx, "coordinate out of bounds");
              }
            }
          }
          break;
        }
        case Instruction::Kind::Binary:
          if (ins.op() != "add" && ins.op() != "mul") {
            fail(ictx, "unsupported op '" + ins.op() + "'");
          }
          useRef(ins.lhs(), ictx);
          useRef(ins.rhs(), ictx);
          defId(ins.id(), ictx);
          break;
        case Instruction::Kind::Relu:
          useRef(ins.arg(), ictx);
          defId(ins.id(), ictx);
          break;
        case Instruction::Kind::Accumulate: {
          if (!isReduction) {
            fail(ictx, "accumulate only allowed in reduction_body");
          }
          if (nacc == 0) {
            fail(ictx, "accumulate with no accumulators");
          }
          if (ins.args().size() != nacc) {
            fail(
                ictx,
                "accumulate args count " + std::to_string(ins.args().size()) +
                    " != accumulator count " + std::to_string(nacc));
          }
          for (int a : ins.args()) {
            useRef(a, ictx);
          }
          break;
        }
        case Instruction::Kind::ReadAccumulator: {
          if (isReduction) {
            fail(
                ictx,
                "read_accumulator only allowed in post_body"
                " (accumulators complete after reduction)");
          }
          if (ins.accumulator() < 0 ||
              static_cast<size_t>(ins.accumulator()) >= nacc) {
            fail(ictx, "accumulator index out of range");
          }
          defId(ins.id(), ictx);
          break;
        }
        case Instruction::Kind::Store: {
          if (isReduction) {
            fail(ictx, "store disallowed in reduction body");
          }
          if (!resultSet.count(ins.buffer())) {
            fail(
                ictx,
                "store to '" + ins.buffer() +
                    "' not an own result"
                    " (compute body can only store own outputs)");
          }
          useRef(ins.arg(), ictx);
          if (pwExec) {
            checkFlatBounds(
                ins.offset(),
                scope,
                resultElems[ins.buffer()],
                "store to '" + ins.buffer() + "'",
                ictx);
          } else {
            checkIndexVars(ins.offset(), scope, ictx);
          }
          break;
        }
      }
    }
  };

  checkBody(c.reductionBody(), true);
  checkBody(c.postBody(), false);

  // Nonempty domain requires at least one store per result. This is NOT a
  // full coverage/injectivity proof (documented limitation).
  if (pwExec) {
    std::set<std::string> stored;
    for (const Instruction& ins : c.postBody()) {
      if (ins.kind() == Instruction::Kind::Store) {
        stored.insert(ins.buffer());
      }
    }
    for (const Buffer& b : c.results()) {
      if (!stored.count(b.name)) {
        fail(
            ctx,
            "result '" + b.name +
                "' has no store"
                " (nonempty domain requires at least one store per result)");
      }
    }
  }

  // Results become visible only after this node.
  for (const Buffer& b : c.results()) {
    storage[b.name] = b.elements;
    dtypes[b.name] = b.dtype;
  }
  nodeNames.insert(c.name());
}

void verifyExternNode(
    const ExternCall& e,
    std::map<std::string, int64_t>& storage,
    std::map<std::string, std::string>& dtypes,
    const std::map<std::string, View>& views,
    std::set<std::string>& nodeNames,
    const std::string& pctx) {
  const std::string ctx = "extern '" + e.name() + "'";
  if (e.name().empty()) {
    fail(pctx, "extern with empty name");
  }
  if (e.target().empty()) {
    fail(ctx, "empty target");
  }
  if (nodeNames.count(e.name()) || storage.count(e.name()) ||
      views.count(e.name())) {
    fail(ctx, "duplicate node/buffer/view name");
  }
  if (e.effects().empty()) {
    fail(ctx, "empty effects (must preserve conservative effects string)");
  }
  for (double s : e.scalarArgs()) {
    if (std::isnan(s)) {
      fail(ctx, "NaN scalar arg rejected (payload cannot round-trip)");
    }
    // Infinities allowed as static mask constants; exact bits preserved.
  }
  // Typed structured args must agree with ordered inputs: collect tensor
  // refs recursively (ordered, repeats allowed) and compare exactly.
  for (const auto& a : e.arguments()) {
    checkCallArg(a, ctx);
  }
  for (const auto& k : e.keywordArgs()) {
    if (k.name().empty()) {
      fail(ctx, "keyword with empty name");
    }
    checkCallArg(k.value(), ctx);
  }
  {
    std::vector<std::string> flat;
    for (const auto& a : e.arguments()) {
      collectArgTensors(a, flat);
    }
    for (const auto& k : e.keywordArgs()) {
      collectArgTensors(k.value(), flat);
    }
    if (flat != e.inputs()) {
      fail(ctx, "typed tensor args do not match ordered inputs");
    }
    for (const auto& t : flat) {
      if (storage.find(t) == storage.end() && views.find(t) == views.end()) {
        fail(ctx, "typed tensor arg '" + t + "' not topologically available");
      }
    }
  }
  // Ordered input operands may REPEAT (e.g. x@x): check each position is a
  // known dependency, but only the distinct set matters for descriptors.
  std::set<std::string> distinctInputs;
  for (size_t k = 0; k < e.inputs().size(); ++k) {
    const std::string& in = e.inputs()[k];
    std::string ictx = ctx + " input[" + std::to_string(k) + "] '" + in + "'";
    if (in.empty()) {
      fail(ctx, "empty input name");
    }
    if (storage.find(in) == storage.end() && views.find(in) == views.end()) {
      fail(ictx, "unknown buffer or view (not topologically available)");
    }
    distinctInputs.insert(in);
  }
  std::set<std::string> resultSet;
  for (const Buffer& b : e.results()) {
    std::string rctx = ctx + " result '" + b.name + "'";
    if (b.name.empty()) {
      fail(ctx, "result with empty name");
    }
    if (distinctInputs.count(b.name)) {
      fail(rctx, "same symbol as input (input/result alias disallowed)");
    }
    if (storage.count(b.name) || views.count(b.name)) {
      fail(rctx, "duplicate buffer name");
    }
    if (!resultSet.insert(b.name).second) {
      fail(rctx, "duplicate result");
    }
    // Extern boundaries allow common opaque dtypes for eager execution;
    // Compute data stays f32-only.
    requireBoundaryDtype(rctx, b.dtype, "buffer");
    if (b.elements < 0) {
      fail(rctx, "negative elements");
    }
  }
  // Descriptors must cover every distinct input and every result.
  {
    std::set<std::string> dnames;
    std::set<std::string> covered;
    for (const Descriptor& d : e.descriptors()) {
      std::string dctx = ctx + " descriptor '" + d.name + "'";
      if (d.name.empty()) {
        fail(ctx, "descriptor with empty name");
      }
      if (!dnames.insert(d.name).second) {
        fail(dctx, "duplicate descriptor name");
      }
      // Target must be a known input or own result.
      bool isInput = distinctInputs.count(d.buffer) != 0;
      bool isResult = resultSet.count(d.buffer) != 0;
      if (!isInput && !isResult) {
        fail(dctx, "unknown target '" + d.buffer + "' (not input or result)");
      }
      // Shape bounds: inputs against known storage/views, results against
      // their declared element count.
      if (isInput) {
        if (views.count(d.buffer)) {
          auto base = resolveAlias(d.buffer, d.offset, storage, views, dctx);
          checkShapeBounds(
              dctx,
              d.sizes,
              d.strides,
              d.offset,
              base.first,
              base.second,
              storage[base.first]);
        } else {
          auto sit = storage.find(d.buffer);
          if (sit == storage.end()) {
            fail(dctx, "unknown input buffer '" + d.buffer + "'");
          }
          checkShapeBounds(
              dctx,
              d.sizes,
              d.strides,
              d.offset,
              d.buffer,
              d.offset,
              sit->second);
        }
      } else {
        int64_t elems = -1;
        for (const Buffer& b : e.results()) {
          if (b.name == d.buffer) {
            elems = b.elements;
            break;
          }
        }
        if (elems < 0) {
          fail(dctx, "unknown result '" + d.buffer + "'");
        }
        checkShapeBounds(
            dctx, d.sizes, d.strides, d.offset, d.buffer, d.offset, elems);
      }
      covered.insert(d.buffer);
    }
    for (const std::string& in : distinctInputs) {
      if (!covered.count(in)) {
        fail(ctx, "descriptor missing for input '" + in + "'");
      }
    }
    for (const Buffer& b : e.results()) {
      if (!covered.count(b.name)) {
        fail(ctx, "descriptor missing for result '" + b.name + "'");
      }
    }
  }
  // Effects string preserved as-is; no passes optimize over it.
  for (const Buffer& b : e.results()) {
    storage[b.name] = b.elements;
    dtypes[b.name] = b.dtype;
  }
  nodeNames.insert(e.name());
}

struct NodeVisitor {
  std::map<std::string, int64_t>& storage;
  std::map<std::string, std::string>& dtypes;
  const std::map<std::string, View>& views;
  std::map<std::string, View>& mutViews;
  std::set<std::string>& nodeNames;
  const std::string& pctx;

  void operator()(const View& v) const {
    std::string ctx = "view '" + v.name + "'";
    if (v.name.empty()) {
      fail(pctx, "view with empty name");
    }
    if (storage.count(v.name) || views.count(v.name) ||
        nodeNames.count(v.name)) {
      fail(ctx, "duplicate view/buffer/node name");
    }
    auto base = resolveAlias(v.source, v.offset, storage, mutViews, ctx);
    checkShapeBounds(
        ctx,
        v.sizes,
        v.strides,
        v.offset,
        base.first,
        base.second,
        storage[base.first]);
    mutViews[v.name] = v;
    nodeNames.insert(v.name);
  }
  void operator()(const Compute& c) const {
    verifyComputeNode(c, storage, dtypes, mutViews, nodeNames, pctx);
  }
  void operator()(const ExternCall& e) const {
    verifyExternNode(e, storage, dtypes, mutViews, nodeNames, pctx);
  }
};

} // namespace

Program::Program(std::string n, std::string numerics)
    : name_(std::move(n)), numerics_(std::move(numerics)) {
  if (!isValidNumerics(numerics_)) {
    throw std::invalid_argument(
        "Program: unsupported numerics '" + numerics_ +
        "' (allowed: 'strict_eager', 'inductor')");
  }
}

void Program::verify() const {
  const std::string pctx = "program '" + name_ + "'";
  if (name_.empty()) {
    fail(pctx, "empty program name");
  }
  if (!isValidNumerics(numerics_)) {
    fail(
        pctx,
        "unsupported numerics '" + numerics_ +
            "' (allowed: 'strict_eager', 'inductor')");
  }
  std::map<std::string, int64_t> storage;
  std::map<std::string, std::string> dtypes;
  std::map<std::string, View> views;
  std::set<std::string> nodeNames;

  for (const Buffer& b : inputs_) {
    std::string ctx = pctx + " input '" + b.name + "'";
    if (b.name.empty()) {
      fail(pctx, "input with empty name");
    }
    if (storage.count(b.name)) {
      fail(ctx, "duplicate buffer name");
    }
    // Program inputs are opaque boundaries: allow common boundary dtypes.
    requireBoundaryDtype(ctx, b.dtype, "buffer");
    if (b.elements < 0) {
      fail(ctx, "negative elements");
    }
    storage[b.name] = b.elements;
    dtypes[b.name] = b.dtype;
  }

  {
    std::set<std::string> dnames;
    for (const Descriptor& d : input_descriptors_) {
      std::string ctx = pctx + " input_descriptor '" + d.name + "'";
      if (d.name.empty()) {
        fail(pctx, "input descriptor with empty name");
      }
      if (!dnames.insert(d.name).second) {
        fail(ctx, "duplicate descriptor name");
      }
      if (storage.find(d.buffer) == storage.end()) {
        fail(
            ctx,
            "unknown input buffer '" + d.buffer +
                "' (input descriptors describe program inputs)");
      }
      checkShapeBounds(
          ctx,
          d.sizes,
          d.strides,
          d.offset,
          d.buffer,
          d.offset,
          storage[d.buffer]);
    }
  }

  for (const Node& n : nodes_) {
    std::visit(NodeVisitor{storage, dtypes, views, views, nodeNames, pctx}, n);
  }

  {
    std::set<std::string> onames;
    for (const Descriptor& d : outputs_) {
      std::string ctx = pctx + " output '" + d.name + "'";
      if (d.name.empty()) {
        fail(pctx, "output with empty name");
      }
      if (!onames.insert(d.name).second) {
        fail(ctx, "duplicate output name");
      }
      const bool isView = views.count(d.buffer) != 0;
      const bool isStor = storage.count(d.buffer) != 0;
      if (!isView && !isStor) {
        fail(ctx, "unknown storage or view '" + d.buffer + "'");
      }
      if (isView) {
        // Output wraps the current view pointer: compose the view's own
        // offset with this relative offset for bounds.
        auto base = resolveAlias(d.buffer, d.offset, storage, views, ctx);
        checkShapeBounds(
            ctx,
            d.sizes,
            d.strides,
            d.offset,
            base.first,
            base.second,
            storage[base.first]);
      } else {
        checkShapeBounds(
            ctx,
            d.sizes,
            d.strides,
            d.offset,
            d.buffer,
            d.offset,
            storage[d.buffer]);
      }
    }
  }
}

} // namespace torch::inductor::loops_ir
