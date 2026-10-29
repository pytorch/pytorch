// Logical micro-passes. Strict numerics: preserve all FP op order/rounding.
// No FP constant folding, reassociation, neutral identities, FMA/FTZ.
// Integer-only rewrites use checked arithmetic; failures are loud, never
// silent wrong transforms. Every pass verifies before and after.
//
// Real transforms live in detail:: (pass_details.h) and consume the typed
// AnalysisContext. The public functions below are thin shims dispatching
// through pipeline:: (pass_manager.h).

#include "passes.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

#include "analyses.h"
#include "pass_details.h"
#include "pass_manager.h"

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

Program detail::viewInlineImpl(
    const Program& program,
    analyses::AnalysisContext& ctx) {
  program.verify();
  // Consume the shared alias/storage-root facts to DRIVE the rewrite (no
  // local duplicate view map or alias walker): the facts' view metadata
  // supplies per-view sizes/strides, rootOf/composedOffset supply the
  // resolved base + base offset. Every view_read target and compute input
  // must be present in the facts.
  const analyses::AliasRootFacts& alias = ctx.aliasRoots(program);
  const std::map<std::string, View>& views = alias.views;
  for (const auto& n : program.nodes()) {
    if (const auto* c = std::get_if<Compute>(&n)) {
      for (const auto& in : c->inputs()) {
        if (alias.rootOf.find(in) == alias.rootOf.end()) {
          passFail("view_inline", "unresolved alias input '" + in + "'");
        }
      }
      for (const auto* body : {&c->reductionBody(), &c->postBody()}) {
        for (const auto& ins : *body) {
          if (ins.kind() == Instruction::Kind::ViewRead) {
            if (alias.rootOf.find(ins.view()) == alias.rootOf.end()) {
              passFail(
                  "view_inline", "unresolved view_read '" + ins.view() + "'");
            }
            if (views.find(ins.view()) == views.end()) {
              passFail(
                  "view_inline",
                  "view_read '" + ins.view() +
                      "' has no view metadata in alias facts");
            }
          }
        }
      }
    }
  }
  auto resolveBase = [&](const std::string& name) {
    auto rit = alias.rootOf.find(name);
    auto oit = alias.composedOffset.find(name);
    if (rit == alias.rootOf.end() || oit == alias.composedOffset.end()) {
      passFail("view_inline", "unresolved alias '" + name + "'");
    }
    return std::make_pair(rit->second, oit->second);
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

Program detail::indexSimplifyImpl(
    const Program& program,
    analyses::AnalysisContext&) {
  program.verify();
  // Pure local rewrite over index trees; reads no cached analysis
  // (descriptor `required` is empty). Integer-only, checked, verbatim.
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

Program detail::unitRangeRemovalImpl(
    const Program& program,
    analyses::AnalysisContext&) {
  program.verify();
  // Pure local transform: unit-extent classification reads the compute's
  // own pointwise ranges directly (no cached analysis needed; descriptor
  // `required` is empty). Substitution uses integer-only checked rewrites.
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
      s += std::to_string(ins.buffer().size()) + ":";
      s += ins.buffer();
      s += ":";
      encodeIndex(ins.offset(), s);
      return s;
    }
    case Instruction::Kind::ViewRead: {
      std::string s = "view:";
      s += std::to_string(ins.view().size()) + ":";
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

Program detail::localCseDceImpl(
    const Program& program,
    analyses::AnalysisContext&) {
  program.verify();
  // Per-body typed SSA CSE + DCE (structural index equality, bit-exact
  // constants); reads no cached analysis (descriptor `required` is empty).
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

Program detail::deadPruneImpl(
    const Program& program,
    analyses::AnalysisContext& ctx) {
  program.verify();
  // Consume the shared liveness + effects facts to DRIVE the rewrite (no
  // parallel local walk): live.keep[i] decides node i, liveViews/
  // liveComputes name the kept nodes, and every effects barrier
  // (unknown-effects extern, incl. opaque RNG/global side effects) must
  // be kept. Cross-checked per node below; any disagreement fails loudly.
  const analyses::LivenessFacts& live = ctx.liveness(program);
  const analyses::EffectFacts& eff = ctx.effects(program);
  if (live.keep.size() != program.nodes().size()) {
    passFail("dead_prune", "liveness facts disagree with program nodes");
  }
  std::set<std::string> barriers(
      eff.barrierNodes.begin(), eff.barrierNodes.end());
  for (size_t i = 0; i < program.nodes().size(); ++i) {
    const auto& n = program.nodes()[i];
    if (const auto* e = std::get_if<ExternCall>(&n)) {
      if (!live.keep[i]) {
        passFail(
            "dead_prune",
            "extern barrier '" + e->name() + "' not kept by liveness");
      }
      if (!barriers.count(e->name())) {
        passFail(
            "dead_prune",
            "extern '" + e->name() + "' missing from effects facts");
      }
    } else if (const auto* c = std::get_if<Compute>(&n)) {
      bool namedLive = live.liveComputes.count(c->name()) != 0;
      if (live.keep[i] != namedLive) {
        passFail(
            "dead_prune",
            "compute '" + c->name() + "' keep flag disagrees with facts");
      }
    } else {
      const auto* v = std::get_if<View>(&n);
      bool namedLive = live.liveViews.count(v->name) != 0;
      if (live.keep[i] != namedLive) {
        passFail(
            "dead_prune",
            "view '" + v->name + "' keep flag disagrees with facts");
      }
    }
  }
  std::vector<Program::Node> out;
  for (size_t i = 0; i < program.nodes().size(); ++i) {
    if (live.keep[i]) {
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

// --- public compatibility shims (through the formal registry) ---
//
// Each shim keeps its exact legacy signature and dispatches through
// pipeline:: with a fresh AnalysisContext. Detail bodies (detail:: above)
// never call these shims, so there is no recursive public-shim dispatch.

Program view_inline(const Program& program) {
  analyses::AnalysisContext ctx;
  return pipeline::runTypedLoopsPass("view_inline", program, ctx);
}

Program index_simplify(const Program& program) {
  analyses::AnalysisContext ctx;
  return pipeline::runTypedLoopsPass("index_simplify", program, ctx);
}

Program unit_range_removal(const Program& program) {
  analyses::AnalysisContext ctx;
  return pipeline::runTypedLoopsPass("unit_range_removal", program, ctx);
}

Program local_cse_dce(const Program& program) {
  analyses::AnalysisContext ctx;
  return pipeline::runTypedLoopsPass("local_cse_dce", program, ctx);
}

Program dead_prune(const Program& program) {
  analyses::AnalysisContext ctx;
  return pipeline::runTypedLoopsPass("dead_prune", program, ctx);
}

std::vector<std::string> pass_names() {
  // Legacy 5 micro-pass names, order and content identical to the
  // pre-formal pipeline (canonical stability note lives in pass_manager.h).
  return pipeline::legacyPassNamesList();
}

Program run_pass(const std::string& name, const Program& program) {
  // Unknown names and typed stage mismatches throw from the registry.
  analyses::AnalysisContext ctx;
  return pipeline::runTypedLoopsPass(name, program, ctx);
}

Program optimize_program(const Program& program) {
  // Formal 5-stage micro-pipeline with per-pass verification, context,
  // and invalidations. Dumps are not captured here (the registry records
  // empty dumps); use runPipeline for the traced pipeline.
  analyses::AnalysisContext ctx;
  Program cur = program;
  for (const auto& name : pipeline::legacyPassNamesList()) {
    cur = pipeline::runTypedLoopsPass(name, cur, ctx);
    cur.verify();
  }
  return cur;
}

std::map<std::string, std::string> optimize_program_dumps(
    const Program& program) {
  program.verify();
  std::map<std::string, std::string> dumps;
  Program cur = program;
  for (const auto& name : pipeline::legacyPassNamesList()) {
    cur = run_pass(name, cur);
    cur.verify();
    dumps[name] = cur.dump();
  }
  return dumps;
}

// --- dense/injective store proof shim (single home: analyses.h) ---

bool prove_dense_injective_store(
    const Compute& compute,
    const std::string& result) {
  return analyses::denseInjectiveProofOf(compute, result);
}

} // namespace torch::inductor::loops_ir
