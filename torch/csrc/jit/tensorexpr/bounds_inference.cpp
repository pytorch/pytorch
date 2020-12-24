#include <torch/csrc/jit/tensorexpr/bounds_inference.h>
#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>
#include <torch/csrc/jit/tensorexpr/stmt.h>

namespace torch {
namespace jit {
namespace tensorexpr {

using namespace analysis;

template <typename Container>
BoundsInfo mergeTensorAccesses(
    const Container& accesses,
    const std::unordered_map<const Var*, const Buf*>& varToBuf,
    bool distinctAccessKinds) {
  BoundsInfo ret;
  for (auto& access : accesses) {
    if (access->type() == AccessType::Input ||
        access->type() == AccessType::Output) {
      continue;
    }

    auto vtbIt = varToBuf.find(access->var());
    TORCH_INTERNAL_ASSERT(vtbIt != varToBuf.end());
    const Buf* buf = vtbIt->second;
    std::vector<TensorAccessBoundsInfo>& infos = ret[buf];

    bool added = false;
    // This loop should be small, max of 2 (kLoad, kStore).
    for (auto& TABI : infos) {
      TensorAccessKind kind = access->isWrite() ? kStore : kLoad;
      if (!distinctAccessKinds || kind == TABI.kind) {
        TORCH_INTERNAL_ASSERT(TABI.start.size() == access->bounds().size());
        TORCH_INTERNAL_ASSERT(TABI.stop.size() == access->bounds().size());
        for (size_t i = 0; i < TABI.start.size(); ++i) {
          TABI.start[i] = IRSimplifier::simplify(
              new Min(TABI.start[i], access->bounds()[i].start, true));
          TABI.stop[i] = IRSimplifier::simplify(
              new Max(TABI.stop[i], access->bounds()[i].end, true));
          added = true;

          if (kind != TABI.kind) {
            TABI.kind = kMutate;
          }
        }
      }
    }

    if (!added) {
      TensorAccessBoundsInfo info;
      info.kind = access->isWrite() ? kStore : kLoad;

      for (auto& b : access->bounds()) {
        info.start.push_back(b.start);
        info.stop.push_back(b.end);
      }

      infos.push_back(info);
    }
  }

  return ret;
}

std::unordered_map<const Var*, const Buf*> getAllBufs(Stmt* s) {
  std::unordered_map<const Var*, const Buf*> varToBuf;

  auto bufs = NodeFinder<const Buf>::find(s);
  auto calls = NodeFinder<FunctionCall>::find(s);
  for (auto* c : calls) {
    bufs.push_back(c->tensor()->buf());
  }

  for (auto* b : bufs) {
    varToBuf[b->base_handle()] = b;
  }
  return varToBuf;
}

BoundsInfo inferBounds(Stmt* s, bool distinctAccessKinds) {
  auto varToBuf = getAllBufs(s);

  MemDependencyChecker checker;
  s->accept(&checker);

  return mergeTensorAccesses(
      checker.getHistory(), varToBuf, distinctAccessKinds);
}

BoundsInfo getInferredBounds(
    MemDependencyChecker& analyzer,
    Stmt* s,
    bool distinctAccessKinds) {
  return mergeTensorAccesses(
      analyzer.accessesWithin(s), getAllBufs(s), distinctAccessKinds);
}

void printBoundsInfo(const BoundsInfo& v) {
  std::cerr << "Access vector {\n";
  for (auto& pair : v) {
    std::cerr << *pair.first << " in [";
    bool first = true;
    for (const auto& b : pair.second) {
      if (!first) {
        std::cerr << ", ";
      }
      std::cerr << ((b.kind == kLoad) ? "LOAD" : "STORE") << "(";
      int i = 0;
      if (b.start.empty()) {
        std::cerr << "0";
      }
      for (const auto& s : b.start) {
        if (i != 0) {
          std::cerr << ", ";
        }
        std::cerr << *s;
        i++;
      }
      std::cerr << "; ";
      i = 0;
      if (b.stop.empty()) {
        std::cerr << "0";
      }
      for (const auto& s : b.stop) {
        if (i != 0) {
          std::cerr << ", ";
        }
        std::cerr << *s;
        i++;
      }
      std::cerr << ")";
      first = false;
    }
    std::cerr << "]\n";
  }
  std::cerr << "}\n";
}

std::vector<const Expr*> getBoundExtents(
    const std::vector<TensorAccessBoundsInfo>& infos) {
  std::vector<const Expr*> starts;
  std::vector<const Expr*> stops;

  // Find the safe size of the temprorary buffer by determining the outer
  // extents of a union of all bounds.
  for (const TensorAccessBoundsInfo& p : infos) {
    for (size_t i = 0; i < p.start.size(); i++) {
      if (starts.size() <= i) {
        starts.push_back(p.start[i]);
      } else {
        starts[i] =
            IRSimplifier::simplify(new Min(starts[i], p.start[i], true));
      }

      if (stops.size() <= i) {
        stops.push_back(p.stop[i]);
      } else {
        stops[i] = IRSimplifier::simplify(new Max(stops[i], p.stop[i], true));
      }
    }
  }

  std::vector<const Expr*> extents;
  for (size_t i = 0; i < starts.size(); ++i) {
    const Expr* dim = IRSimplifier::simplify(
        new Add(new Sub(stops[i], starts[i]), new IntImm(1)));

    extents.push_back(dim);
  }

  return extents;
}

using BoundSet = std::unordered_set<Bound, BoundHash>;

BoundSet convertBounds(
    const std::vector<TensorAccessBoundsInfo>& bounds,
    TensorAccessKind filter = kMutate) {
  BoundSet ret;
  for (auto& TABI : bounds) {
    if (filter == kMutate || TABI.kind == filter) {
      for (size_t i = 0; i < TABI.start.size(); ++i) {
        ret.insert(Bound(TABI.start[i], TABI.stop[i]));
      }
    }
  }
  return ret;
}

BoundSet convertBounds(
    BoundsInfo& bounds,
    const Buf* buf,
    TensorAccessKind filter = kMutate) {
  auto it = bounds.find(buf);
  if (it == bounds.end()) {
    return BoundSet();
  }

  return convertBounds(it->second, filter);
}

HazardKind getPotentialHazards(
    MemDependencyChecker& analyzer,
    Stmt* A,
    Stmt* B) {
  BoundsInfo aBounds = getInferredBounds(analyzer, A, true);
  BoundsInfo bBounds = getInferredBounds(analyzer, B, true);

  BoundSet aWrites;
  BoundSet aReads;

  for (auto& pair : bBounds) {
    const Buf* buf = pair.first;
    if (aBounds.find(buf) == aBounds.end()) {
      continue;
    }

    auto aWrites = convertBounds(aBounds, buf, kStore);
    auto aReads = convertBounds(aBounds, buf, kLoad);

    auto bWrites = convertBounds(pair.second, kStore);
    auto bReads = convertBounds(pair.second, kLoad);

    // First, RAW.
    for (auto& bR : bReads) {
      for (auto& aW : aWrites) {
        if (boundOverlap(bR, aW) != NoOverlap) {
          return HazardKind::ReadAfterWrite;
        }
      }
    }

    // Then WAR.
    for (auto& bW : bWrites) {
      for (auto& aR : aReads) {
        if (boundOverlap(bW, aR) != NoOverlap) {
          return HazardKind::WriteAfterRead;
        }
      }
    }

    // Then WAW.
    for (auto& bW : bWrites) {
      for (auto& aW : aWrites) {
        if (boundOverlap(bW, aW) != NoOverlap) {
          return HazardKind::WriteAfterWrite;
        }
      }
    }
  }

  return HazardKind::NoDependency;
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
