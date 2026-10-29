// Logical micro-passes. Strict numerics: preserve all FP op order/rounding.
// No FP constant folding, reassociation, neutral identities, FMA/FTZ.
// Integer-only rewrites use checked arithmetic; failures are loud, never
// silent wrong transforms. Every pass verifies before and after.

#include "passes.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

namespace torch::inductor::loops_ir {
namespace {

[[noreturn]] void passFail(const std::string& pass, const std::string& msg) {
  throw std::invalid_argument(pass + ": " + msg);
}

int64_t checkedAddP(const std::string& pass, int64_t a, int64_t b) {
  __int128 v = (__int128)a + b;
  if (v > std::numeric_limits<int64_t>::max() ||
      v < std::numeric_limits<int64_t>::min()) {
    passFail(pass, "int64 overflow in index rewrite");
  }
  return (int64_t)v;
}

int64_t checkedMulP(const std::string& pass, int64_t a, int64_t b) {
  __int128 v = (__int128)a * b;
  if (v > std::numeric_limits<int64_t>::max() ||
      v < std::numeric_limits<int64_t>::min()) {
    passFail(pass, "int64 overflow in index rewrite");
  }
  return (int64_t)v;
}

int64_t floorDivP(int64_t a, int64_t b) {
  int64_t q = a / b;
  int64_t r = a % b;
  if (r != 0 && b > 0 && r < 0) {
    --q;
  }
  return q;
}

// --- Program/Compute rebuild helpers (Program nodes are append-only) ---

Program rebuildProgram(const Program& src, std::vector<Program::Node> nodes) {
  Program out(src.name(), src.numerics());
  for (const auto& b : src.inputs()) {
    out.addInput(b);
  }
  for (const auto& d : src.inputDescriptors()) {
    out.addInputDescriptor(d);
  }
  for (const auto& n : nodes) {
    if (const auto* v = std::get_if<View>(&n)) {
      out.addView(*v);
    } else if (const auto* c = std::get_if<Compute>(&n)) {
      out.addCompute(*c);
    } else {
      out.addExternCall(*std::get_if<ExternCall>(&n));
    }
  }
  for (const auto& d : src.outputs()) {
    out.addOutput(d);
  }
  return out;
}

Compute rebuildCompute(
    const Compute& c,
    std::vector<std::string> inputs,
    std::vector<Range> pw,
    std::vector<Instruction> red,
    std::vector<Instruction> post) {
  Compute n(
      c.name(),
      std::move(inputs),
      c.results(),
      std::move(pw),
      c.reductionRanges(),
      c.accumulators());
  for (const auto& i : red) {
    n.addReductionInstruction(i);
  }
  for (const auto& i : post) {
    n.addPostInstruction(i);
  }
  return n;
}

// --- Index helpers ---

bool isConstVal(const Index& idx, int64_t& out) {
  if (!idx.empty() && idx.expr().kind == IndexExpr::Kind::Constant) {
    out = idx.expr().value;
    return true;
  }
  return false;
}

// Integer-only simplification: constant folding (checked) + valid integer
// identities (add 0, mul 0/1, div 1, mod 1). No FP involved.
Index simplifyIndexRec(const Index& idx) {
  if (idx.empty()) {
    return idx;
  }
  const IndexExpr& e = idx.expr();
  if (e.kind != IndexExpr::Kind::Binary) {
    return idx;
  }
  Index l = simplifyIndexRec(Index(e.lhs));
  Index r = simplifyIndexRec(Index(e.rhs));
  int64_t lv = 0, rv = 0;
  bool lc = isConstVal(l, lv), rc = isConstVal(r, rv);
  const std::string& op = e.op;
  try {
    if (op == "add") {
      if (rc && rv == 0) {
        return l;
      }
      if (lc && lv == 0) {
        return r;
      }
      if (lc && rc) {
        return Index::Constant(checkedAddP("index_simplify", lv, rv));
      }
    } else if (op == "mul") {
      if ((lc && lv == 0) || (rc && rv == 0)) {
        return Index::Constant(0);
      }
      if (rc && rv == 1) {
        return l;
      }
      if (lc && lv == 1) {
        return r;
      }
      if (lc && rc) {
        return Index::Constant(checkedMulP("index_simplify", lv, rv));
      }
    } else if (op == "floordiv") {
      if (rc && rv == 1) {
        return l;
      }
      if (lc && rc && rv > 0) {
        return Index::Constant(floorDivP(lv, rv));
      }
    } else if (op == "mod") {
      if (rc && rv == 1) {
        return Index::Constant(0);
      }
      // Only fold non-negative dividends (truncation == floor there).
      if (lc && rc && rc && rv > 0 && lv >= 0) {
        return Index::Constant(lv % rv);
      }
    }
    return Index::Binary(op, l, r);
  } catch (const std::invalid_argument&) {
    // Rebuild/simplification invalid here (e.g. zero divisor): keep the
    // original subtree; the verifier judges legality.
    return idx;
  }
}

// Substitute Variable(var) -> repl structurally.
Index substIndexRec(
    const Index& idx,
    const std::string& var,
    const Index& repl) {
  const IndexExpr& e = idx.expr();
  if (e.kind == IndexExpr::Kind::Variable && e.name == var) {
    return repl;
  }
  if (e.kind != IndexExpr::Kind::Binary) {
    return idx;
  }
  Index l = substIndexRec(Index(e.lhs), var, repl);
  Index r = substIndexRec(Index(e.rhs), var, repl);
  return Index::Binary(e.op, l, r);
}

Instruction mapInstructionIndexes(
    const Instruction& ins,
    const std::function<Index(const Index&)>& fn) {
  switch (ins.kind()) {
    case Instruction::Kind::Load:
      return Instruction::Load(ins.id(), ins.buffer(), fn(ins.offset()));
    case Instruction::Kind::ViewRead: {
      std::vector<Index> cs;
      for (const auto& c : ins.coordinates()) {
        cs.push_back(fn(c));
      }
      return Instruction::ViewRead(ins.id(), ins.view(), std::move(cs));
    }
    case Instruction::Kind::Store:
      return Instruction::Store(ins.buffer(), fn(ins.offset()), ins.arg());
    default:
      return ins;
  }
}

} // namespace

// --- view_inline ---

Program view_inline(const Program& program) {
  program.verify();
  std::map<std::string, View> views;
  for (const auto& n : program.nodes()) {
    if (const auto* v = std::get_if<View>(&n)) {
      views[v->name] = *v;
    }
  }
  auto resolveBase = [&](const std::string& name) {
    std::string cur = name;
    __int128 off = 0;
    std::set<std::string> seen;
    while (true) {
      auto it = views.find(cur);
      if (it == views.end()) {
        if (off > std::numeric_limits<int64_t>::max() || off < 0) {
          passFail("view_inline", "offset overflow resolving '" + name + "'");
        }
        return std::make_pair(cur, (int64_t)off);
      }
      if (!seen.insert(cur).second) {
        passFail("view_inline", "alias cycle at '" + cur + "'");
      }
      off += it->second.offset;
      cur = it->second.source;
    }
  };
  std::vector<Program::Node> out;
  for (const auto& n : program.nodes()) {
    if (const auto* c = std::get_if<Compute>(&n)) {
      bool touchesView = false;
      for (const auto* body : {&c->reductionBody(), &c->postBody()}) {
        for (const auto& ins : *body) {
          if (ins.kind() == Instruction::Kind::ViewRead) {
            touchesView = true;
          }
        }
      }
      if (!touchesView) {
        out.emplace_back(*c);
        continue;
      }
      // Retarget view inputs to base storage (order-preserving, deduped).
      std::vector<std::string> inputs;
      for (const auto& in : c->inputs()) {
        std::string base = in;
        if (views.count(in)) {
          base = resolveBase(in).first;
        }
        if (std::find(inputs.begin(), inputs.end(), base) == inputs.end()) {
          inputs.push_back(base);
        }
      }
      auto rewriteBody = [&](const std::vector<Instruction>& body) {
        std::vector<Instruction> nb;
        for (const auto& ins : body) {
          if (ins.kind() != Instruction::Kind::ViewRead) {
            nb.push_back(ins);
            continue;
          }
          auto vit = views.find(ins.view());
          if (vit == views.end()) {
            passFail("view_inline", "unknown view '" + ins.view() + "'");
          }
          const View& vw = vit->second;
          if (ins.coordinates().size() != vw.sizes.size()) {
            passFail(
                "view_inline",
                "coordinate rank mismatch for '" + vw.name + "'");
          }
          auto base = resolveBase(ins.view());
          __int128 flat128 = base.second;
          Index flat = Index::Constant(base.second);
          for (size_t d = 0; d < vw.sizes.size(); ++d) {
            Index term = Index::Binary(
                "mul", ins.coordinates()[d], Index::Constant(vw.strides[d]));
            flat = Index::Binary("add", flat, term);
            flat128 += (__int128)0; // shape only; bounds rechecked by verifier
          }
          (void)flat128;
          nb.push_back(Instruction::Load(ins.id(), base.first, flat));
        }
        return nb;
      };
      out.emplace_back(rebuildCompute(
          *c,
          inputs,
          c->pointwiseRanges(),
          rewriteBody(c->reductionBody()),
          rewriteBody(c->postBody())));
    } else {
      if (const auto* v = std::get_if<View>(&n)) {
        out.emplace_back(*v);
      } else {
        out.emplace_back(*std::get_if<ExternCall>(&n));
      }
    }
  }
  Program result = rebuildProgram(program, std::move(out));
  result.verify();
  return result;
}

// --- index_simplify ---

Program index_simplify(const Program& program) {
  program.verify();
  std::vector<Program::Node> out;
  for (const auto& n : program.nodes()) {
    if (const auto* c = std::get_if<Compute>(&n)) {
      auto simpBody = [&](const std::vector<Instruction>& body) {
        std::vector<Instruction> nb;
        for (const auto& ins : body) {
          nb.push_back(mapInstructionIndexes(ins, simplifyIndexRec));
        }
        return nb;
      };
      out.emplace_back(rebuildCompute(
          *c,
          c->inputs(),
          c->pointwiseRanges(),
          simpBody(c->reductionBody()),
          simpBody(c->postBody())));
    } else if (const auto* v = std::get_if<View>(&n)) {
      out.emplace_back(*v);
    } else {
      out.emplace_back(*std::get_if<ExternCall>(&n));
    }
  }
  Program result = rebuildProgram(program, std::move(out));
  result.verify();
  return result;
}

// --- unit_range_removal ---

Program unit_range_removal(const Program& program) {
  program.verify();
  std::vector<Program::Node> out;
  for (const auto& n : program.nodes()) {
    if (const auto* c = std::get_if<Compute>(&n)) {
      std::set<std::string> unitVars;
      std::vector<Range> pw;
      for (const auto& r : c->pointwiseRanges()) {
        // Signed subtraction can overflow (ranges may span INT64_MIN..MAX);
        // compare via __int128 so extent classification never invokes UB.
        __int128 extent = (__int128)r.stop - (__int128)r.start;
        if (extent == 1) {
          unitVars.insert(r.name);
        } else {
          pw.push_back(r);
        }
      }
      if (unitVars.empty()) {
        out.emplace_back(*c);
        continue;
      }
      // Explicit substitution var -> start in both bodies (reduction body
      // also sees pointwise vars; reduction ranges themselves unchanged).
      std::map<std::string, Index> repl;
      for (const auto& r : c->pointwiseRanges()) {
        if (unitVars.count(r.name)) {
          repl.emplace(r.name, Index::Constant(r.start));
        }
      }
      auto substBody = [&](const std::vector<Instruction>& body) {
        std::vector<Instruction> nb;
        for (const auto& ins : body) {
          Instruction cur = ins;
          for (const auto& kv : repl) {
            try {
              // Substitute var -> start, then normalize the rewritten index
              // (substitution can expose new constant arithmetic, e.g.
              // (p0*3+r0)[p0->0] = 0*3+r0 = r0). Integer-only, checked.
              cur = mapInstructionIndexes(cur, [&](const Index& ix) {
                return simplifyIndexRec(substIndexRec(ix, kv.first, kv.second));
              });
            } catch (const std::invalid_argument& ex) {
              passFail("unit_range_removal", ex.what());
            }
          }
          nb.push_back(cur);
        }
        return nb;
      };
      out.emplace_back(rebuildCompute(
          *c,
          c->inputs(),
          pw,
          substBody(c->reductionBody()),
          substBody(c->postBody())));
    } else if (const auto* v = std::get_if<View>(&n)) {
      out.emplace_back(*v);
    } else {
      out.emplace_back(*std::get_if<ExternCall>(&n));
    }
  }
  Program result = rebuildProgram(program, std::move(out));
  result.verify();
  return result;
}

// --- local_cse_dce ---

namespace {

// Structural index encoding: unambiguous tree encoding. Never toString
// pretty-printing: it collides on legal programs because Range names are
// unrestricted strings, so Variable("10") and Constant(10) both print "10"
// while addressing different elements. Names are length-prefixed so nested
// structure cannot collide either. FP constants still match bit-exactly.
void encodeIndex(const Index& idx, std::string& out) {
  const IndexExpr& e = idx.expr();
  switch (e.kind) {
    case IndexExpr::Kind::Constant:
      out += "C(" + std::to_string(e.value) + ")";
      return;
    case IndexExpr::Kind::Variable:
      out += "V(" + std::to_string(e.name.size()) + ":";
      out += e.name;
      out += ")";
      return;
    case IndexExpr::Kind::Binary:
      out += "B(" + e.op + ",";
      encodeIndex(Index(e.lhs), out);
      out += ",";
      encodeIndex(Index(e.rhs), out);
      out += ")";
      return;
  }
}

std::string cseKey(const Instruction& ins) {
  switch (ins.kind()) {
    case Instruction::Kind::Constant: {
      uint64_t bits = 0;
      double v = ins.value();
      static_assert(sizeof(bits) == sizeof(v), "double size");
      std::memcpy(&bits, &v, sizeof(bits));
      std::ostringstream os;
      os << "const:" << std::hex << bits;
      return os.str();
    }
    case Instruction::Kind::Load: {
      std::string s = "load:";
      s += ins.buffer();
      s += ":";
      encodeIndex(ins.offset(), s);
      return s;
    }
    case Instruction::Kind::ViewRead: {
      std::string s = "view:";
      s += ins.view();
      s += ":";
      for (const auto& c : ins.coordinates()) {
        encodeIndex(c, s);
        s += ";";
      }
      return s;
    }
    case Instruction::Kind::Binary:
      return "bin:" + ins.op() + ":" + std::to_string(ins.lhs()) + ":" +
          std::to_string(ins.rhs());
    case Instruction::Kind::Relu:
      return "relu:" + std::to_string(ins.arg());
    case Instruction::Kind::ReadAccumulator:
      return "racc:" + std::to_string(ins.accumulator());
    default:
      return "";
  }
}

int remapId(const std::map<int, int>& m, int id) {
  auto it = m.find(id);
  return it == m.end() ? id : it->second;
}

Instruction remapOperands(const Instruction& ins, const std::map<int, int>& m) {
  switch (ins.kind()) {
    case Instruction::Kind::Binary:
      return Instruction::Binary(
          ins.id(), ins.op(), remapId(m, ins.lhs()), remapId(m, ins.rhs()));
    case Instruction::Kind::Relu:
      return Instruction::Relu(ins.id(), remapId(m, ins.arg()));
    case Instruction::Kind::Accumulate: {
      std::vector<int> a;
      for (int v : ins.args()) {
        a.push_back(remapId(m, v));
      }
      return Instruction::Accumulate(std::move(a));
    }
    case Instruction::Kind::Store:
      return Instruction::Store(
          ins.buffer(), ins.offset(), remapId(m, ins.arg()));
    default:
      return ins;
  }
}

void markUse(int id, std::set<int>& live) {
  live.insert(id);
}

// One body: CSE identical pure defs (ordered operands, structural index
// equality via tree encoding), then DCE defs not reaching stores/accumulates.
std::vector<Instruction> cseDceBody(const std::vector<Instruction>& body) {
  std::map<std::string, int> canon;
  std::map<int, int> alias; // dropped id -> canonical id
  std::vector<Instruction> kept;
  for (const auto& ins : body) {
    if (!ins.definesValue()) {
      kept.push_back(remapOperands(ins, alias));
      continue;
    }
    Instruction rw = ins;
    // Rewrite operands through aliases first so keys are canonical.
    rw = remapOperands(rw, alias);
    std::string key = cseKey(rw);
    auto it = canon.find(key);
    if (it != canon.end()) {
      alias[ins.id()] = it->second;
      continue;
    }
    canon[key] = rw.id();
    kept.push_back(rw);
  }
  // DCE: seed from stores/accumulates, walk backwards.
  std::set<int> live;
  for (const auto& ins : kept) {
    if (ins.kind() == Instruction::Kind::Store) {
      markUse(remapId(alias, ins.arg()), live);
    } else if (ins.kind() == Instruction::Kind::Accumulate) {
      for (int v : ins.args()) {
        markUse(remapId(alias, v), live);
      }
    }
  }
  std::vector<Instruction> result;
  for (auto it = kept.rbegin(); it != kept.rend(); ++it) {
    const Instruction& ins = *it;
    if (!ins.definesValue()) {
      result.push_back(ins);
      continue;
    }
    if (live.count(ins.id())) {
      result.push_back(ins);
      switch (ins.kind()) {
        case Instruction::Kind::Binary:
          markUse(ins.lhs(), live);
          markUse(ins.rhs(), live);
          break;
        case Instruction::Kind::Relu:
          markUse(ins.arg(), live);
          break;
        default:
          break;
      }
    }
  }
  std::reverse(result.begin(), result.end());
  return result;
}

} // namespace

Program local_cse_dce(const Program& program) {
  program.verify();
  std::vector<Program::Node> out;
  for (const auto& n : program.nodes()) {
    if (const auto* c = std::get_if<Compute>(&n)) {
      out.emplace_back(rebuildCompute(
          *c,
          c->inputs(),
          c->pointwiseRanges(),
          cseDceBody(c->reductionBody()),
          cseDceBody(c->postBody())));
    } else if (const auto* v = std::get_if<View>(&n)) {
      out.emplace_back(*v);
    } else {
      out.emplace_back(*std::get_if<ExternCall>(&n));
    }
  }
  Program result = rebuildProgram(program, std::move(out));
  result.verify();
  return result;
}

// --- dead_prune ---

Program dead_prune(const Program& program) {
  program.verify();
  std::map<std::string, View> views;
  for (const auto& n : program.nodes()) {
    if (const auto* v = std::get_if<View>(&n)) {
      views[v->name] = *v;
    }
  }
  std::set<std::string> needed;
  std::function<void(const std::string&)> needChain =
      [&](const std::string& name) {
        if (!needed.insert(name).second) {
          return;
        }
        auto it = views.find(name);
        if (it != views.end()) {
          needChain(it->second.source);
        }
      };
  for (const auto& o : program.outputs()) {
    needChain(o.buffer);
  }
  // Reverse walk. Externs always kept (conservative effects); their inputs
  // are live. Computes/views kept only when a result is needed.
  std::vector<bool> keep(program.nodes().size(), false);
  for (size_t i = program.nodes().size(); i-- > 0;) {
    const auto& n = program.nodes()[i];
    if (const auto* e = std::get_if<ExternCall>(&n)) {
      keep[i] = true;
      for (const auto& in : e->inputs()) {
        needChain(in);
      }
      for (const auto& b : e->results()) {
        needed.erase(b.name);
      }
    } else if (const auto* c = std::get_if<Compute>(&n)) {
      bool live = false;
      for (const auto& b : c->results()) {
        if (needed.count(b.name)) {
          live = true;
        }
      }
      keep[i] = live;
      if (live) {
        for (const auto& in : c->inputs()) {
          needChain(in);
        }
        for (const auto& b : c->results()) {
          needed.erase(b.name);
        }
      }
    } else {
      const auto* v = std::get_if<View>(&n);
      if (needed.count(v->name)) {
        keep[i] = true;
        needChain(v->source);
      }
    }
  }
  std::vector<Program::Node> out;
  for (size_t i = 0; i < program.nodes().size(); ++i) {
    if (keep[i]) {
      const auto& n = program.nodes()[i];
      if (const auto* v = std::get_if<View>(&n)) {
        out.emplace_back(*v);
      } else if (const auto* c = std::get_if<Compute>(&n)) {
        out.emplace_back(*c);
      } else {
        out.emplace_back(*std::get_if<ExternCall>(&n));
      }
    }
  }
  Program result = rebuildProgram(program, std::move(out));
  result.verify();
  return result;
}

// --- pipeline ---

std::vector<std::string> pass_names() {
  // Order matters for canonical stability: unit_range_removal runs BEFORE
  // index_simplify so substitution-exposed constant arithmetic (e.g. a
  // unit dim replaced by 0 inside (p*stride+r)) is simplified in the same
  // pipeline run; unit_range_removal also normalizes its own rewrites.
  // view_inline first (kernels reject view_read), CSE/DCE after rewrites,
  // dead_prune last on final liveness.
  return {
      "view_inline",
      "unit_range_removal",
      "index_simplify",
      "local_cse_dce",
      "dead_prune"};
}

Program run_pass(const std::string& name, const Program& program) {
  if (name == "view_inline") {
    return view_inline(program);
  }
  if (name == "index_simplify") {
    return index_simplify(program);
  }
  if (name == "unit_range_removal") {
    return unit_range_removal(program);
  }
  if (name == "local_cse_dce") {
    return local_cse_dce(program);
  }
  if (name == "dead_prune") {
    return dead_prune(program);
  }
  throw std::invalid_argument("unknown pass '" + name + "'");
}

Program optimize_program(const Program& program) {
  program.verify();
  Program cur = program;
  for (const auto& name : pass_names()) {
    cur = run_pass(name, cur);
    cur.verify();
  }
  return cur;
}

std::map<std::string, std::string> optimize_program_dumps(
    const Program& program) {
  program.verify();
  std::map<std::string, std::string> dumps;
  Program cur = program;
  for (const auto& name : pass_names()) {
    cur = run_pass(name, cur);
    cur.verify();
    dumps[name] = cur.dump();
  }
  return dumps;
}

// --- dense/injective store proof (fusion legality only) ---

namespace {

// Linear decomposition: offset == base + sum coeff[var]*var. False on any
// nonlinearity (nested mul, floordiv, mod), duplicate vars, or overflow.
bool linearDecomp(
    const Index& idx,
    std::map<std::string, int64_t>& coeffs,
    int64_t& base) {
  if (idx.empty()) {
    return false;
  }
  const IndexExpr& e = idx.expr();
  switch (e.kind) {
    case IndexExpr::Kind::Constant: {
      __int128 v = (__int128)base + e.value;
      if (v > std::numeric_limits<int64_t>::max() ||
          v < std::numeric_limits<int64_t>::min()) {
        return false;
      }
      base = (int64_t)v;
      return true;
    }
    case IndexExpr::Kind::Variable: {
      if (coeffs.count(e.name)) {
        return false;
      }
      coeffs[e.name] = 1;
      return true;
    }
    case IndexExpr::Kind::Binary: {
      if (e.op == "add") {
        return linearDecomp(Index(e.lhs), coeffs, base) &&
            linearDecomp(Index(e.rhs), coeffs, base);
      }
      if (e.op == "mul") {
        int64_t c = 0;
        std::shared_ptr<const IndexExpr> other;
        if (isConstVal(Index(e.lhs), c)) {
          other = e.rhs;
        } else if (isConstVal(Index(e.rhs), c)) {
          other = e.lhs;
        } else {
          return false;
        }
        std::map<std::string, int64_t> sub;
        int64_t subBase = 0;
        if (!linearDecomp(Index(other), sub, subBase)) {
          return false;
        }
        for (const auto& kv : sub) {
          if (coeffs.count(kv.first)) {
            return false;
          }
          __int128 v = (__int128)kv.second * c;
          if (v > std::numeric_limits<int64_t>::max() ||
              v < std::numeric_limits<int64_t>::min()) {
            return false;
          }
          coeffs[kv.first] = (int64_t)v;
        }
        __int128 b = (__int128)base + (__int128)subBase * c;
        if (b > std::numeric_limits<int64_t>::max() ||
            b < std::numeric_limits<int64_t>::min()) {
          return false;
        }
        base = (int64_t)b;
        return true;
      }
      return false;
    }
  }
  return false;
}

} // namespace

bool prove_dense_injective_store(
    const Compute& compute,
    const std::string& result) {
  // Pointwise-only single-store requirement.
  if (!compute.reductionRanges().empty() || !compute.accumulators().empty() ||
      !compute.reductionBody().empty()) {
    return false;
  }
  const Instruction* store = nullptr;
  for (const auto& ins : compute.postBody()) {
    if (ins.kind() == Instruction::Kind::Store && ins.buffer() == result) {
      if (store != nullptr) {
        return false; // multiple stores: no narrow proof
      }
      store = &ins;
    }
  }
  if (store == nullptr) {
    return false;
  }
  int64_t elements = -1;
  for (const auto& b : compute.results()) {
    if (b.name == result) {
      elements = b.elements;
      break;
    }
  }
  if (elements < 0) {
    return false;
  }
  std::map<std::string, int64_t> coeffs;
  int64_t base = 0;
  if (!linearDecomp(store->offset(), coeffs, base) || base != 0) {
    return false;
  }
  // Every pointwise var exactly once; ranges start at 0 (coverage [0, N)).
  std::map<std::string, int64_t> extents;
  for (const auto& r : compute.pointwiseRanges()) {
    if (r.start != 0 || r.stop <= r.start) {
      return false;
    }
    // stop > start >= 0 here (r.start == 0), so the difference fits.
    extents[r.name] = r.stop - r.start;
  }
  if (coeffs.size() != extents.size()) {
    return false;
  }
  for (const auto& kv : extents) {
    auto it = coeffs.find(kv.first);
    if (it == coeffs.end() || it->second <= 0) {
      return false;
    }
  }
  // Exact coverage: product == elements, dense strides ascending.
  __int128 product = 1;
  for (const auto& kv : extents) {
    product *= kv.second;
  }
  if (product != (__int128)elements) {
    return false;
  }
  std::vector<std::pair<int64_t, int64_t>> strideExtent; // (stride, extent)
  for (const auto& kv : extents) {
    strideExtent.emplace_back(coeffs[kv.first], kv.second);
  }
  std::sort(strideExtent.begin(), strideExtent.end());
  __int128 running = 1;
  for (const auto& se : strideExtent) {
    if ((__int128)se.first != running) {
      return false;
    }
    running *= se.second;
  }
  return true;
}

} // namespace torch::inductor::loops_ir
