#include <torch/csrc/jit/tensorexpr/bounds_inference.h>

#include <torch/csrc/jit/tensorexpr/bounds_overlap.h>
#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>
#include <torch/csrc/jit/tensorexpr/stmt.h>

#include <c10/util/irange.h>

namespace torch::jit::tensorexpr {

using namespace analysis;

template <typename Container>
BoundsInfo mergeTensorAccesses(
    const Container& accesses,
    const std::unordered_map<VarPtr, BufPtr>& varToBuf,
    bool distinctAccessKinds) {
  BoundsInfo ret;
  for (auto& access : accesses) {
    if (access->type() == AccessType::Input ||
        access->type() == AccessType::Output) {
      continue;
    }

    auto vtbIt = varToBuf.find(access->var());
    TORCH_INTERNAL_ASSERT(vtbIt != varToBuf.end(), buildErrorMessage());
    BufPtr buf = vtbIt->second;
    std::vector<TensorAccessBoundsInfo>& infos = ret[buf];

    bool added = false;
    // This loop should be small, max of 2 (kLoad, kStore).
    for (auto& TABI : infos) {
      TensorAccessKind kind = access->isWrite() ? kStore : kLoad;
      if (!distinctAccessKinds || kind == TABI.kind) {
        TORCH_INTERNAL_ASSERT(
            TABI.start.size() == access->bounds().size(), buildErrorMessage());
        TORCH_INTERNAL_ASSERT(
            TABI.stop.size() == access->bounds().size(), buildErrorMessage());
        for (size_t i = 0; i < TABI.start.size(); ++i) {
          TABI.start[i] = IRSimplifier::simplify(
              alloc<Min>(TABI.start[i], access->bounds()[i].start, true));
          TABI.stop[i] = IRSimplifier::simplify(
              alloc<Max>(TABI.stop[i], access->bounds()[i].end, true));
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

std::unordered_map<VarPtr, BufPtr> getAllBufs(StmtPtr s) {
  std::unordered_map<VarPtr, BufPtr> varToBuf;

  auto bufs = NodeFinder<Buf>::find(s);
  for (const auto& b : bufs) {
    varToBuf[b->base_handle()] = b;
  }
  return varToBuf;
}

std::unordered_map<VarPtr, BufPtr> getAllBufs(ExprPtr e) {
  std::unordered_map<VarPtr, BufPtr> varToBuf;

  auto bufs = NodeFinder<Buf>::find(e);
  for (const auto& b : bufs) {
    varToBuf[b->base_handle()] = b;
  }
  return varToBuf;
}

BoundsInfo inferBounds(StmtPtr s, bool distinctAccessKinds) {
  auto varToBuf = getAllBufs(s);

  MemDependencyChecker checker;
  s->accept(&checker);

  return mergeTensorAccesses(
      checker.getHistory(), varToBuf, distinctAccessKinds);
}

BoundsInfo getInferredBounds(
    MemDependencyChecker& analyzer,
    StmtPtr s,
    bool distinctAccessKinds) {
  return mergeTensorAccesses(
      analyzer.accessesWithin(s), getAllBufs(s), distinctAccessKinds);
}

BoundsInfo getInferredBounds(
    MemDependencyChecker& analyzer,
    ExprPtr e,
    bool distinctAccessKinds) {
  return mergeTensorAccesses(
      analyzer.accessesWithin(e), getAllBufs(e), distinctAccessKinds);
}

void printBoundsInfo(const BoundsInfo& v) {
  std::cerr << "Access vector {\n";
  for (auto& pair : v) {
    std::cerr << *pair.first << " in [";
    bool first = true;
    for (auto& b : pair.second) {
      if (!first) {
        std::cerr << ", ";
      }
      std::cerr << ((b.kind == kLoad) ? "LOAD" : "STORE") << "(";
      int i = 0;
      if (b.start.empty()) {
        std::cerr << "0";
      }
      for (auto& s : b.start) {
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
      for (auto& s : b.stop) {
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

std::vector<ExprPtr> getBoundExtents(
    const std::vector<TensorAccessBoundsInfo>& infos) {
  std::vector<ExprPtr> starts;
  std::vector<ExprPtr> stops;

  // Find the safe size of the temporary buffer by determining the outer
  // extents of a union of all bounds.
  for (const TensorAccessBoundsInfo& p : infos) {
    for (const auto i : c10::irange(p.start.size())) {
      if (starts.size() <= i) {
        starts.push_back(p.start[i]);
      } else {
        starts[i] =
            IRSimplifier::simplify(alloc<Min>(starts[i], p.start[i], true));
      }

      if (stops.size() <= i) {
        stops.push_back(p.stop[i]);
      } else {
        stops[i] =
            IRSimplifier::simplify(alloc<Max>(stops[i], p.stop[i], true));
      }
    }
  }

  std::vector<ExprPtr> extents;
  for (size_t i = 0; i < starts.size(); ++i) {
    ExprPtr dim = IRSimplifier::simplify(
        alloc<Add>(alloc<Sub>(stops[i], starts[i]), immLike(stops[i], 1)));

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
    BufPtr buf,
    TensorAccessKind filter = kMutate) {
  auto it = bounds.find(buf);
  if (it == bounds.end()) {
    return BoundSet();
  }

  return convertBounds(it->second, filter);
}

HazardKind getPotentialHazards(
    MemDependencyChecker& analyzer,
    StmtPtr A,
    StmtPtr B) {
  BoundsInfo aBounds = getInferredBounds(analyzer, A, true);
  BoundsInfo bBounds = getInferredBounds(analyzer, B, true);

  for (auto& pair : bBounds) {
    BufPtr buf = pair.first;
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
        if (boundOverlap(bR, aW) != OverlapKind::NoOverlap) {
          return HazardKind::ReadAfterWrite;
        }
      }
    }

    // Then WAR.
    for (auto& bW : bWrites) {
      for (auto& aR : aReads) {
        if (boundOverlap(bW, aR) != OverlapKind::NoOverlap) {
          return HazardKind::WriteAfterRead;
        }
      }
    }

    // Then WAW.
    for (auto& bW : bWrites) {
      for (auto& aW : aWrites) {
        if (boundOverlap(bW, aW) != OverlapKind::NoOverlap) {
          return HazardKind::WriteAfterWrite;
        }
      }
    }
  }

  return HazardKind::NoDependency;
}

IndexBounds getIndexBounds(const TensorAccessBoundsInfo& tabi) {
  TORCH_INTERNAL_ASSERT(
      tabi.start.size() == tabi.stop.size(), buildErrorMessage());
  IndexBounds ret(tabi.start.size());
  if (tabi.start.empty()) {
    return ret;
  }
  for (size_t i = 0; i < tabi.start.size(); ++i) {
    ret[i] = Bound(tabi.start[i], tabi.stop[i]);
  }
  return ret;
}

std::vector<IndexBounds> getIndexBounds(
    const std::vector<TensorAccessBoundsInfo>& vTABI,
    TensorAccessKind filter = kMutate) {
  std::vector<IndexBounds> bounds;
  for (auto& TABI : vTABI) {
    if (filter == kMutate || TABI.kind == filter) {
      bounds.push_back(getIndexBounds(TABI));
    }
  }
  return bounds;
}

bool hasConflictingOverlap(
    const BoundsInfo& aBounds,
    const BoundsInfo& bBounds,
    TensorAccessKind aFilter = kMutate,
    TensorAccessKind bFilter = kMutate) {
  using IndexBoundsInfo = std::unordered_map<BufPtr, std::vector<IndexBounds>>;
  IndexBoundsInfo aIndexBoundsInfo;
  for (auto& aBound : aBounds) {
    aIndexBoundsInfo[aBound.first] = getIndexBounds(aBound.second, aFilter);
  }
  IndexBoundsInfo bIndexBoundsInfo;
  for (auto& bBound : bBounds) {
    bIndexBoundsInfo[bBound.first] = getIndexBounds(bBound.second, bFilter);
  }

  for (auto& aBound : aBounds) {
    auto bIt = bBounds.find(aBound.first);
    if (bIt == bBounds.end()) {
      continue;
    }
    auto aIndexBounds = aIndexBoundsInfo[aBound.first];
    auto bIndexBounds = bIndexBoundsInfo[bIt->first];
    auto aTABIs = aBound.second;
    auto bTABIs = bIt->second;
    for (size_t i = 0; i < aTABIs.size(); ++i) {
      for (size_t j = 0; j < bTABIs.size(); ++j) {
        auto aTABI = aTABIs[i];
        auto bTABI = bTABIs[j];
        if (aTABI.kind == kLoad && bTABI.kind == kLoad) {
          continue;
        }
        auto overlap = overlaps(aIndexBounds[i], bIndexBounds[j]);
        if (overlap != OverlapKind::NoOverlap) {
          return true;
        }
      }
    }
  }
  return false;
}

bool hasConflictingOverlap(
    analysis::MemDependencyChecker& analyzer,
    StmtPtr A,
    StmtPtr B) {
  BoundsInfo aBounds = getInferredBounds(analyzer, A, true);
  BoundsInfo bBounds = getInferredBounds(analyzer, B, true);
  return hasConflictingOverlap(aBounds, bBounds);
}

bool isOverlapping(
    analysis::MemDependencyChecker& analyzer,
    StorePtr S1,
    StorePtr S2) {
  BoundsInfo s1Bounds = getInferredBounds(analyzer, S1, true);
  BoundsInfo s2Bounds = getInferredBounds(analyzer, S2, true);
  return hasConflictingOverlap(s1Bounds, s2Bounds, kStore, kStore);
}

bool isOverlapping(
    analysis::MemDependencyChecker& analyzer,
    StorePtr S,
    LoadPtr L) {
  BoundsInfo sBounds = getInferredBounds(analyzer, S, true);
  BoundsInfo lBounds = getInferredBounds(analyzer, L, true);
  return hasConflictingOverlap(sBounds, lBounds, kStore, kLoad);
}

} // namespace torch::jit::tensorexpr
