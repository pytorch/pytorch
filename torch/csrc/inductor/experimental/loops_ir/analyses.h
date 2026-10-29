#pragma once
// Typed analysis facts + AnalysisContext for the formal LoopsIR pass API.
//
// Owner: native pass-manager worker (formalPassAPI phase). This header is
// self-contained on ir.h + kernel_ir.h only (no pass_manager.h dependency,
// no header cycle). All functions are inline so the current 7-source loader
// keeps linking until the two new translation units (analyses.cpp,
// pass_manager.cpp, owned by this worker) are added to the loader list by
// worker C; those TUs carry build anchors, not duplicate definitions.
//
// Design rules:
// - Every analysis kind is a distinct typed struct (AnalysisKind key). There
//   is no dynamically typed "optional bag" / property map: AnalysisContext
//   exposes one typed getter per kind plus a constrained template fetch().
// - Caches are keyed by content fingerprint AND per-kind epoch. A pass that
//   lists a kind in its descriptor `invalidated` set drops those entries and
//   bumps the kind epoch, so stale facts can never be silently reused even
//   if two different IRs ever hashed alike.
// - Shared algorithms are the verbatim-moved bodies from passes.cpp /
//   kernel_ir.cpp (alias resolution, interval evaluation reuse,
//   liveness walk, dependency construction, effect scan, dense/injective
//   proof). No competing duplicate proofs are introduced: detail pass bodies
//   and the public prove_dense_injective_store shim all call into these
//   helpers.
// - Unknown extern effects stay conservative barriers (effects facts only
//   ever record "unknown"; no pass optimizes over an extern).

#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ir.h"
#include "kernel_ir.h"

namespace torch::inductor::loops_ir {

// Forward declaration; full definition lives in tiled_ir.h. This header
// never includes tiled_ir.h (which forward-declares Kernel types from
// kernel_ir.h) so there is no header cycle. Only analyses.cpp includes
// tiled_ir.h for the TiledProgram key.
class TiledProgram;

namespace analyses {

// ---------------------------------------------------------------------------
// Kinds
// ---------------------------------------------------------------------------

// Reusable analysis cache kinds. Only kinds a pass genuinely reads may
// appear in its descriptor `required` set; only kinds it (re)computes may
// appear in `computed`. See pass_manager.h.
enum class AnalysisKind {
  AliasStorageRoots,
  IndexIntervals,
  Liveness,
  Dependencies,
  Effects,
  DenseInjective,
};

inline std::string toString(AnalysisKind kind) {
  switch (kind) {
    case AnalysisKind::AliasStorageRoots:
      return "alias_storage_roots";
    case AnalysisKind::IndexIntervals:
      return "index_intervals";
    case AnalysisKind::Liveness:
      return "liveness";
    case AnalysisKind::Dependencies:
      return "dependencies";
    case AnalysisKind::Effects:
      return "effects";
    case AnalysisKind::DenseInjective:
      return "dense_injective";
  }
  return "unknown";
}

inline AnalysisKind analysisKindFromString(const std::string& name) {
  if (name == "alias_storage_roots") {
    return AnalysisKind::AliasStorageRoots;
  }
  if (name == "index_intervals") {
    return AnalysisKind::IndexIntervals;
  }
  if (name == "liveness") {
    return AnalysisKind::Liveness;
  }
  if (name == "dependencies") {
    return AnalysisKind::Dependencies;
  }
  if (name == "effects") {
    return AnalysisKind::Effects;
  }
  if (name == "dense_injective") {
    return AnalysisKind::DenseInjective;
  }
  throw std::invalid_argument("unknown analysis kind '" + name + "'");
}

// Typed IR stage kind. No catch-all IR state: every pass descriptor names
// exactly one input and one output kind.
enum class IRKind {
  Loops,
  Kernel,
  Tiled,
};

inline std::string toString(IRKind kind) {
  switch (kind) {
    case IRKind::Loops:
      return "loops_ir";
    case IRKind::Kernel:
      return "kernel_ir";
    case IRKind::Tiled:
      return "tiled_ir";
  }
  return "unknown";
}

inline IRKind irKindFromString(const std::string& name) {
  if (name == "loops_ir") {
    return IRKind::Loops;
  }
  if (name == "kernel_ir") {
    return IRKind::Kernel;
  }
  if (name == "tiled_ir") {
    return IRKind::Tiled;
  }
  throw std::invalid_argument("unknown IR kind '" + name + "'");
}

// ---------------------------------------------------------------------------
// Canonical semantic keys + trace fingerprints.
//
// Cache soundness (review F1): caches are keyed by FULL canonical byte
// strings, compared with complete string equality — never by a 64-bit
// hash. Two semantically different programs can never share a cache entry
// (hash collisions cannot reuse wrong facts; epochs additionally guard
// invalidation). The canonical serializer below is a structural visitor
// over every typed IR field (NOT the pretty-printer dump, which omits
// kernel views/extern details and truncates doubles):
//   - names, numerics, dtypes, element counts, sizes/strides/offsets,
//     range bounds, accumulator combine + bit-exact identity,
//   - every instruction field: SSA ids, op names, all operands, bit-exact
//     double bits of constants/scalar args, structural index trees
//     (never pretty-printed strings, which collide),
//   - extern targets, ordered inputs/results, full descriptors, recursive
//     structured args/kwargs (bit-exact floats), effects strings,
//     may_alias flags,
//   - kernel views, kernel functions + regions, dependency edges,
//   - tiled functions/blocks/instructions incl. grid/flags/numerics.
// Strings are length-prefixed so no payload can collide across fields.
// Trace fingerprints are short FNV-1a labels over the canonical key
// (human-scale trace only; never cache keys).
// ---------------------------------------------------------------------------

inline uint64_t fnv1a64(const std::string& text) {
  uint64_t hash = 14695981039346656037ULL;
  for (unsigned char c : text) {
    hash ^= c;
    hash *= 1099511628211ULL;
  }
  return hash;
}

inline std::string toHex16(uint64_t v) {
  static const char* kDigits = "0123456789abcdef";
  std::string out(16, '0');
  for (int i = 15; i >= 0; --i) {
    out[(size_t)i] = kDigits[v & 0xF];
    v >>= 4;
  }
  return out;
}

// Full canonical semantic keys (defined in analyses.cpp). Valid-IR only.
std::string canonicalKeyOf(const Program& program);
std::string canonicalKeyOf(const KernelProgram& program);
std::string canonicalKeyOf(const TiledProgram& program);
std::string canonicalKeyOf(const Compute& compute);

// Short trace labels (never cache keys).
inline std::string fingerprintOf(const Program& program) {
  return "loops:" + program.numerics() + ":" + toHex16(fnv1a64(canonicalKeyOf(program)));
}

inline std::string fingerprintOf(const KernelProgram& program) {
  return "kernel:" + program.numerics() + ":" + toHex16(fnv1a64(canonicalKeyOf(program)));
}

// Out of line (defined in canonical_keys.cpp, which includes tiled_ir.h);
// analyses.h only forward-declares TiledProgram to avoid a header cycle.
std::string fingerprintOf(const TiledProgram& program);

// ---------------------------------------------------------------------------
// Typed fact structs (one per AnalysisKind).
// ---------------------------------------------------------------------------

// Alias/storage-root facts: verbatim-moved resolution from passes.cpp
// (resolveBase) and kernel_ir.cpp (viewRoot). rootOf maps every known
// view/buffer to its storage root; composedOffset carries the summed
// relative offset from that root. views carries the full view metadata
// (source/sizes/strides/offset) so transforms consume these facts to
// drive rewrites instead of rebuilding local view maps (review F5).
struct AliasRootFacts {
  std::map<std::string, std::string> rootOf;
  std::map<std::string, int64_t> composedOffset;
  std::map<std::string, View> views;
};

// Index-interval facts: per-compute closed bounds for every range variable,
// mirroring the verifier's scope construction in ir.cpp (nonempty range
// [start, stop-1], empty range lo > hi).
struct IndexIntervalFacts {
  // compute name -> (variable -> closed interval).
  std::map<std::string, std::map<std::string, Interval>> bounds;
  // compute name -> (variable -> (start, stop)) for extent classification.
  std::map<std::string, std::map<std::string, std::pair<int64_t, int64_t>>> ranges;
};

// Liveness facts: verbatim-moved reverse walk from dead_prune
// (passes.cpp). keep[i] decides Program node i; liveComputes holds the
// kept compute names, liveViews the kept view names, liveBuffers the
// surviving buffer roots (outputs + extern-observed inputs + chains).
// Transforms consume these facts (no parallel local walkers; review F5).
struct LivenessFacts {
  std::vector<bool> keep;
  std::set<std::string> liveComputes;
  std::set<std::string> liveBuffers;
  std::set<std::string> liveViews;
};

// Dependency facts: program-level DAG edges. Constructed by running the
// existing KernelProgram::buildDependencies algorithm on a copy (no
// competing duplicate construction).
struct DependencyFacts {
  std::vector<std::vector<size_t>> edges;
};

// Effect facts: conservative extern barriers. Every ExternCall is a
// barrier (effects strings are opaque/unknown); barrierNodes holds the
// node names of extern calls in order. No pass may optimize over these.
struct EffectFacts {
  bool hasUnknownEffectExtern = false;
  std::vector<std::string> barrierNodes;
};

// ---------------------------------------------------------------------------
// Shared algorithms (verbatim logic, single home).
// ---------------------------------------------------------------------------

namespace detail {

// Resolve one name through a view map to (storage root, composed offset).
// Verbatim from passes.cpp resolveBase / kernel_ir.cpp viewRoot, unified:
// views that resolve to non-storage names terminate at that name (passes.cpp
// behavior for compute inputs); unknown roots throw loudly.
inline std::pair<std::string, int64_t> resolveRoot(const std::string& name, const std::map<std::string, View>& views) {
  std::string cur = name;
  __int128 off = 0;
  std::set<std::string> seen;
  while (true) {
    auto it = views.find(cur);
    if (it == views.end()) {
      if (off > std::numeric_limits<int64_t>::max() || off < 0) {
        throw std::invalid_argument("alias analysis: offset overflow resolving '" + name + "'");
      }
      return {cur, (int64_t)off};
    }
    if (!seen.insert(cur).second) {
      throw std::invalid_argument("alias analysis: alias cycle at '" + cur + "'");
    }
    off += it->second.offset;
    cur = it->second.source;
  }
}

// Alias closure of a buffer: itself plus every view whose transitive root
// is that buffer. Verbatim from kernel_ir.cpp aliasClosure.
inline std::set<std::string> aliasClosureSet(const std::string& buffer, const std::map<std::string, View>& viewMap) {
  std::set<std::string> out{buffer};
  for (const auto& kv : viewMap) {
    if (resolveRoot(kv.first, viewMap).first == buffer) {
      out.insert(kv.first);
    }
  }
  return out;
}

// Linear decomposition: offset == base + sum coeff[var]*var. False on any
// nonlinearity (nested mul, floordiv, mod), duplicate vars, or overflow.
// Verbatim from passes.cpp linearDecomp (fusion-legality helper).
inline bool linearDecompImpl(const Index& idx, std::map<std::string, int64_t>& coeffs, int64_t& base) {
  if (idx.empty()) {
    return false;
  }
  const IndexExpr& e = idx.expr();
  switch (e.kind) {
    case IndexExpr::Kind::Constant: {
      __int128 v = (__int128)base + e.value;
      if (v > std::numeric_limits<int64_t>::max() || v < std::numeric_limits<int64_t>::min()) {
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
        return linearDecompImpl(Index(e.lhs), coeffs, base) && linearDecompImpl(Index(e.rhs), coeffs, base);
      }
      if (e.op == "mul") {
        int64_t c = 0;
        std::shared_ptr<const IndexExpr> other;
        bool lhsConst = !Index(e.lhs).empty() && Index(e.lhs).expr().kind == IndexExpr::Kind::Constant;
        bool rhsConst = !Index(e.rhs).empty() && Index(e.rhs).expr().kind == IndexExpr::Kind::Constant;
        if (lhsConst) {
          c = Index(e.lhs).expr().value;
          other = e.rhs;
        } else if (rhsConst) {
          c = Index(e.rhs).expr().value;
          other = e.lhs;
        } else {
          return false;
        }
        std::map<std::string, int64_t> sub;
        int64_t subBase = 0;
        if (!linearDecompImpl(Index(other), sub, subBase)) {
          return false;
        }
        for (const auto& kv : sub) {
          if (coeffs.count(kv.first)) {
            return false;
          }
          __int128 v = (__int128)kv.second * c;
          if (v > std::numeric_limits<int64_t>::max() || v < std::numeric_limits<int64_t>::min()) {
            return false;
          }
          coeffs[kv.first] = (int64_t)v;
        }
        __int128 b = (__int128)base + (__int128)subBase * c;
        if (b > std::numeric_limits<int64_t>::max() || b < std::numeric_limits<int64_t>::min()) {
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

} // namespace detail

// Narrow dense/injective store proof for fusion. Returns true only when the
// result store is proven to cover exactly [0, elements) injectively.
// Conservative: false means "no proof", never "proven false". Verbatim from
// passes.cpp prove_dense_injective_store; this is the single home of the
// proof (the public shim and fusion both call here).
inline bool denseInjectiveProofOf(const Compute& compute, const std::string& result) {
  // Pointwise-only single-store requirement.
  if (!compute.reductionRanges().empty() || !compute.accumulators().empty() || !compute.reductionBody().empty()) {
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
  if (!detail::linearDecompImpl(store->offset(), coeffs, base) || base != 0) {
    return false;
  }
  // Every pointwise var exactly once; ranges start at 0 (coverage [0, N)).
  std::map<std::string, int64_t> extents;
  for (const auto& r : compute.pointwiseRanges()) {
    if (r.start != 0 || r.stop <= r.start) {
      return false;
    }
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

// Compute alias/storage-root facts for a Loops program (verbatim
// resolution; throws loudly on cycles/overflow, never silently wrong).
// Also carries the full view metadata so transforms consume these facts
// to drive rewrites (no parallel local view maps).
inline AliasRootFacts computeAliasRootFacts(const Program& program) {
  std::map<std::string, View> views;
  for (const auto& n : program.nodes()) {
    if (const auto* v = std::get_if<View>(&n)) {
      views[v->name] = *v;
    }
  }
  AliasRootFacts facts;
  facts.views = views;
  for (const auto& b : program.inputs()) {
    facts.rootOf[b.name] = b.name;
    facts.composedOffset[b.name] = 0;
  }
  for (const auto& kv : views) {
    auto resolved = detail::resolveRoot(kv.first, views);
    facts.rootOf[kv.first] = resolved.first;
    facts.composedOffset[kv.first] = resolved.second;
  }
  for (const auto& n : program.nodes()) {
    if (const auto* c = std::get_if<Compute>(&n)) {
      for (const auto& b : c->results()) {
        facts.rootOf[b.name] = b.name;
        facts.composedOffset[b.name] = 0;
      }
    } else if (const auto* e = std::get_if<ExternCall>(&n)) {
      for (const auto& b : e->results()) {
        facts.rootOf[b.name] = b.name;
        facts.composedOffset[b.name] = 0;
      }
    }
  }
  return facts;
}

// Compute per-compute range facts (mirrors the verifier scope in ir.cpp).
inline IndexIntervalFacts computeIndexIntervalFacts(const Program& program) {
  IndexIntervalFacts facts;
  for (const auto& n : program.nodes()) {
    if (const auto* c = std::get_if<Compute>(&n)) {
      std::map<std::string, Interval> bounds;
      std::map<std::string, std::pair<int64_t, int64_t>> ranges;
      for (const auto& r : c->pointwiseRanges()) {
        bounds[r.name] = (r.stop > r.start) ? Interval{r.start, r.stop - 1} : Interval{0, -1};
        ranges[r.name] = {r.start, r.stop};
      }
      for (const auto& r : c->reductionRanges()) {
        bounds[r.name] = (r.stop > r.start) ? Interval{r.start, r.stop - 1} : Interval{0, -1};
        ranges[r.name] = {r.start, r.stop};
      }
      facts.bounds[c->name()] = std::move(bounds);
      facts.ranges[c->name()] = std::move(ranges);
    }
  }
  return facts;
}

// Compute program liveness (verbatim dead_prune reverse walk). keep[i]
// decides Program node i; extern calls are always kept (conservative
// effects) and their inputs are live.
inline LivenessFacts computeLivenessFacts(const Program& program) {
  std::map<std::string, View> views;
  for (const auto& n : program.nodes()) {
    if (const auto* v = std::get_if<View>(&n)) {
      views[v->name] = *v;
    }
  }
  std::set<std::string> needed;
  std::function<void(const std::string&)> needChain = [&](const std::string& name) {
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
  LivenessFacts facts;
  facts.keep.assign(program.nodes().size(), false);
  for (size_t i = program.nodes().size(); i-- > 0;) {
    const auto& n = program.nodes()[i];
    if (const auto* e = std::get_if<ExternCall>(&n)) {
      facts.keep[i] = true;
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
      facts.keep[i] = live;
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
        facts.keep[i] = true;
        needChain(v->source);
      }
    }
  }
  facts.liveBuffers = needed;
  for (size_t i = 0; i < program.nodes().size(); ++i) {
    if (!facts.keep[i]) {
      continue;
    }
    const auto& n = program.nodes()[i];
    if (const auto* c = std::get_if<Compute>(&n)) {
      facts.liveComputes.insert(c->name());
    } else if (const auto* v = std::get_if<View>(&n)) {
      facts.liveViews.insert(v->name);
    }
  }
  return facts;
}

// Compute dependency edges by running the existing
// KernelProgram::buildDependencies algorithm on a copy (single home for
// dependency construction; no duplicate proof).
inline DependencyFacts computeDependencyFacts(const KernelProgram& program) {
  KernelProgram copy = program;
  copy.buildDependencies();
  DependencyFacts facts;
  facts.edges = copy.dependencies();
  return facts;
}

// Scan extern barriers of a Loops program. Every ExternCall is a barrier
// (effects are opaque/unknown); may_alias_inputs defaults true.
inline EffectFacts computeEffectFacts(const Program& program) {
  EffectFacts facts;
  for (const auto& n : program.nodes()) {
    if (const auto* e = std::get_if<ExternCall>(&n)) {
      facts.barrierNodes.push_back(e->name());
      if (e->effects() == "unknown") {
        facts.hasUnknownEffectExtern = true;
      }
    }
  }
  return facts;
}

// Scan extern barriers of a Kernel program (same conservative contract).
inline EffectFacts computeKernelEffectFacts(const KernelProgram& program) {
  EffectFacts facts;
  for (const auto& n : program.nodes()) {
    if (const auto* e = std::get_if<ExternCall>(&n)) {
      facts.barrierNodes.push_back(e->name());
      if (e->effects() == "unknown") {
        facts.hasUnknownEffectExtern = true;
      }
    }
  }
  return facts;
}

// Compute alias/storage-root facts for a Kernel program. Same verbatim
// resolution over KernelProgram.views; every input and every node result
// (kernel regions + externs) is its own storage root. Carries the full
// view metadata so fusion consumes these facts (no local view maps).
inline AliasRootFacts computeKernelAliasRootFacts(const KernelProgram& program) {
  std::map<std::string, View> views;
  for (const auto& v : program.views()) {
    views[v.name] = v;
  }
  AliasRootFacts facts;
  facts.views = views;
  for (const auto& b : program.inputs()) {
    facts.rootOf[b.name] = b.name;
    facts.composedOffset[b.name] = 0;
  }
  for (const auto& kv : views) {
    auto resolved = detail::resolveRoot(kv.first, views);
    facts.rootOf[kv.first] = resolved.first;
    facts.composedOffset[kv.first] = resolved.second;
  }
  for (const auto& n : program.nodes()) {
    if (const auto* f = std::get_if<KernelFunction>(&n)) {
      for (const auto& reg : f->regions()) {
        for (const auto& b : reg.results()) {
          facts.rootOf[b.name] = b.name;
          facts.composedOffset[b.name] = 0;
        }
      }
    } else {
      for (const auto& b : std::get_if<ExternCall>(&n)->results()) {
        facts.rootOf[b.name] = b.name;
        facts.composedOffset[b.name] = 0;
      }
    }
  }
  return facts;
}

// ---------------------------------------------------------------------------
// AnalysisContext: typed, epoch-guarded caches.
// ---------------------------------------------------------------------------

template <typename Facts>
struct CacheEntry {
  Facts facts;
  uint64_t kindEpoch = 0;
};

class AnalysisContext {
 public:
  AnalysisContext() = default;

  uint64_t epoch() const {
    return epoch_;
  }

  uint64_t kindEpoch(AnalysisKind kind) const {
    auto it = kindEpochs_.find(kind);
    return it == kindEpochs_.end() ? 0 : it->second;
  }

  // Number of times `kind` was genuinely (re)computed through this
  // context (cache misses). The pipeline executor snapshots these counts
  // around each pass so trace `analyses_computed` lists exactly the kinds
  // computed during that pass — never placeholders.
  uint64_t computationCount(AnalysisKind kind) const {
    auto it = computeCounts_.find(kind);
    return it == computeCounts_.end() ? 0 : it->second;
  }

  std::map<AnalysisKind, uint64_t> snapshotComputationCounts() const {
    return computeCounts_;
  }

  // Drop cached entries of the listed kinds and bump their kind epochs
  // (plus the global epoch). Any later fetch recomputes from the current
  // IR; stale facts can never be silently reused.
  void invalidate(const std::vector<AnalysisKind>& kinds) {
    for (AnalysisKind kind : kinds) {
      kindEpochs_[kind] = kindEpoch(kind) + 1;
      switch (kind) {
        case AnalysisKind::AliasStorageRoots:
          aliasCache_.clear();
          break;
        case AnalysisKind::IndexIntervals:
          intervalCache_.clear();
          break;
        case AnalysisKind::Liveness:
          livenessCache_.clear();
          break;
        case AnalysisKind::Dependencies:
          dependencyCache_.clear();
          break;
        case AnalysisKind::Effects:
          effectCache_.clear();
          kernelEffectCache_.clear();
          break;
        case AnalysisKind::DenseInjective:
          denseCache_.clear();
          break;
      }
    }
    ++epoch_;
  }

  void invalidateAll() {
    invalidate({
        AnalysisKind::AliasStorageRoots,
        AnalysisKind::IndexIntervals,
        AnalysisKind::Liveness,
        AnalysisKind::Dependencies,
        AnalysisKind::Effects,
        AnalysisKind::DenseInjective,
    });
  }

  // Constrained typed template fetch: one instantiation per Facts type,
  // each backed by its own cache. No untyped bag: a caller asking for an
  // unsupported Facts type fails to compile (static_assert below).
  template <typename Facts>
  const Facts& fetch(const std::string& fingerprint, std::function<Facts()> compute) {
    if constexpr (std::is_same<Facts, AliasRootFacts>::value) {
      return fetchInto(aliasCache_, AnalysisKind::AliasStorageRoots, fingerprint, std::move(compute));
    } else if constexpr (std::is_same<Facts, IndexIntervalFacts>::value) {
      return fetchInto(intervalCache_, AnalysisKind::IndexIntervals, fingerprint, std::move(compute));
    } else if constexpr (std::is_same<Facts, LivenessFacts>::value) {
      return fetchInto(livenessCache_, AnalysisKind::Liveness, fingerprint, std::move(compute));
    } else if constexpr (std::is_same<Facts, DependencyFacts>::value) {
      return fetchInto(dependencyCache_, AnalysisKind::Dependencies, fingerprint, std::move(compute));
    } else if constexpr (std::is_same<Facts, EffectFacts>::value) {
      // Effects has two IR-typed caches (Loops vs Kernel keys are disjoint
      // by their "program"/"kernel_program" tags); callers must use
      // effects()/kernelEffects().
      static_assert(!std::is_same<Facts, Facts>::value,
                    "AnalysisContext: use effects() or kernelEffects(), not "
                    "fetch<EffectFacts>()");
      throw std::logic_error("unreachable");
    } else if constexpr (std::is_same<Facts, bool>::value) {
      return fetchInto(denseCache_, AnalysisKind::DenseInjective, fingerprint, std::move(compute));
    } else {
      static_assert(!std::is_same<Facts, Facts>::value, "AnalysisContext::fetch: unsupported Facts type");
      throw std::logic_error("unreachable");
    }
  }

  // Typed getters (each validates the kind epoch on hit). Cache keys
  // are FULL canonical semantic strings (complete equality); trace
  // fingerprints stay short labels.
  const AliasRootFacts& aliasRoots(const Program& program) {
    std::string key = canonicalKeyOf(program);
    return fetch<AliasRootFacts>(key, [&]() { return computeAliasRootFacts(program); });
  }

  const AliasRootFacts& kernelAliasRoots(const KernelProgram& program) {
    std::string key = canonicalKeyOf(program);
    return fetch<AliasRootFacts>(key, [&]() { return computeKernelAliasRootFacts(program); });
  }

  const IndexIntervalFacts& indexIntervals(const Program& program) {
    std::string key = canonicalKeyOf(program);
    return fetch<IndexIntervalFacts>(key, [&]() { return computeIndexIntervalFacts(program); });
  }

  const LivenessFacts& liveness(const Program& program) {
    std::string key = canonicalKeyOf(program);
    return fetch<LivenessFacts>(key, [&]() { return computeLivenessFacts(program); });
  }

  const DependencyFacts& dependencies(const KernelProgram& program) {
    std::string key = canonicalKeyOf(program);
    return fetch<DependencyFacts>(key, [&]() { return computeDependencyFacts(program); });
  }

  const EffectFacts& effects(const Program& program) {
    std::string key = canonicalKeyOf(program);
    auto it = effectCache_.find(key);
    if (it != effectCache_.end() && it->second.kindEpoch == kindEpoch(AnalysisKind::Effects)) {
      return it->second.facts;
    }
    ++computeCounts_[AnalysisKind::Effects];
    CacheEntry<EffectFacts> entry;
    entry.facts = computeEffectFacts(program);
    entry.kindEpoch = kindEpoch(AnalysisKind::Effects);
    effectCache_[key] = std::move(entry);
    return effectCache_[key].facts;
  }

  const EffectFacts& kernelEffects(const KernelProgram& program) {
    std::string key = canonicalKeyOf(program);
    auto it = kernelEffectCache_.find(key);
    if (it != kernelEffectCache_.end() && it->second.kindEpoch == kindEpoch(AnalysisKind::Effects)) {
      return it->second.facts;
    }
    ++computeCounts_[AnalysisKind::Effects];
    CacheEntry<EffectFacts> entry;
    entry.facts = computeKernelEffectFacts(program);
    entry.kindEpoch = kindEpoch(AnalysisKind::Effects);
    kernelEffectCache_[key] = std::move(entry);
    return kernelEffectCache_[key].facts;
  }

  // Cached dense/injective proof for one (scope, compute, result) key.
  // Same verdict as the stateless proof; the cache only memoizes. The
  // scope key is the FULL canonical compute string (not a hash), so
  // post-rewrite SSA changes can never hit a stale entry.
  bool denseInjective(const Compute& compute, const std::string& result, const std::string& scopeKey) {
    std::string key = scopeKey + '\x00' + compute.name() + '\x00' + result;
    return fetch<bool>(key, [&]() { return denseInjectiveProofOf(compute, result); });
  }

 private:
  template <typename Facts>
  const Facts& fetchInto(std::map<std::string, CacheEntry<Facts>>& cache,
                         AnalysisKind kind,
                         const std::string& fingerprint,
                         std::function<Facts()> compute) {
    auto it = cache.find(fingerprint);
    if (it != cache.end() && it->second.kindEpoch == kindEpoch(kind)) {
      return it->second.facts;
    }
    ++computeCounts_[kind];
    CacheEntry<Facts> entry;
    entry.facts = compute();
    entry.kindEpoch = kindEpoch(kind);
    cache[fingerprint] = std::move(entry);
    return cache[fingerprint].facts;
  }

  uint64_t epoch_ = 0;
  std::map<AnalysisKind, uint64_t> kindEpochs_;
  std::map<AnalysisKind, uint64_t> computeCounts_;
  std::map<std::string, CacheEntry<AliasRootFacts>> aliasCache_;
  std::map<std::string, CacheEntry<IndexIntervalFacts>> intervalCache_;
  std::map<std::string, CacheEntry<LivenessFacts>> livenessCache_;
  std::map<std::string, CacheEntry<DependencyFacts>> dependencyCache_;
  std::map<std::string, CacheEntry<EffectFacts>> effectCache_;
  std::map<std::string, CacheEntry<EffectFacts>> kernelEffectCache_;
  std::map<std::string, CacheEntry<bool>> denseCache_;
};

} // namespace analyses
} // namespace torch::inductor::loops_ir
