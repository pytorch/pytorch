#include <torch/csrc/jit/codegen/cuda/transform_iter.h>
#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>

namespace torch {
namespace jit {
namespace fuser {

TensorDomain* TransformIter::replayBackward(Split* split, TensorDomain* td) {
  return split->in();
}

TensorDomain* TransformIter::replayBackward(Merge* merge, TensorDomain* td) {
  return merge->in();
}

TensorDomain* TransformIter::replayBackward(
    Reorder* reorder,
    TensorDomain* td) {
  return reorder->in();
}

TensorDomain* TransformIter::replayBackward(Expr* expr, TensorDomain* td) {
  TORCH_INTERNAL_ASSERT(
      expr->isExpr(),
      "Dispatch in transform iteration is expecting Exprs only.");
  switch (*(expr->getExprType())) {
    case (ExprType::Split):
      return replayBackward(static_cast<Split*>(expr), td);
    case (ExprType::Merge):
      return replayBackward(static_cast<Merge*>(expr), td);
    case (ExprType::Reorder):
      return replayBackward(static_cast<Reorder*>(expr), td);
    default:
      TORCH_INTERNAL_ASSERT(
          false, "Could not detect expr type in replayBackward.");
  }
}

std::vector<Expr*> TransformIter::getHistory(TensorDomain* td) {
  std::vector<Expr*> ops;
  TensorDomain* root = td; // backward running td
  Fusion* fusion = FusionGuard::getCurFusion();

  // Get my origin
  Expr* orig = fusion->origin(root);
  std::set<Expr*> visited_exprs;

  // If I'm not back to the original td
  while (orig != nullptr) {
    if (visited_exprs.find(orig) != visited_exprs.end())
      TORCH_INTERNAL_ASSERT(
          false,
          "TransformReplay::runBackward is not traversing a correct history.");
    ops.push_back(orig);
    visited_exprs.emplace(orig);
    TensorDomain* previous_td = nullptr;
    // Check inputs of this operation, make sure there isn't more than one TD
    // I can only record operations that only take this TD as an input.
    for (Val* inp : orig->inputs())
      if (inp->getValType() == ValType::TensorDomain) {
        if (previous_td != nullptr)
          TORCH_INTERNAL_ASSERT(
              false,
              "TransformReplay::runBackward could not decifer transform history of a TensorDomain.");

        // Traverse back
        root = static_cast<TensorDomain*>(inp);
        orig = fusion->origin(root);
      }
  }
  return std::vector<Expr*>(ops.rbegin(), ops.rend());
}

TensorDomain* TransformIter::runBackward(TensorDomain* td) {
  std::vector<Expr*> ops = getHistory(td);

  // We want to iterate backwards, reverse history.
  ops = std::vector<Expr*>(ops.rbegin(), ops.rend());

  TensorDomain* running_td = td;
  for (Expr* op : ops)
    running_td = replayBackward(op, running_td);

  return running_td;
}

TensorDomain* TransformIter::replay(Split* expr, TensorDomain* td) {
  return td->split(
      expr->axis(), static_cast<Int*>(expr->factor())->value().value());
}

TensorDomain* TransformIter::replay(Merge* expr, TensorDomain* td) {
  return td->merge(expr->axis());
}

TensorDomain* TransformIter::replay(Reorder* expr, TensorDomain* td) {
  std::unordered_map<int, int> old2new;
  for (decltype(expr->new2old().size()) i{0}; i < expr->new2old().size(); i++)
    old2new[expr->new2old()[i]] = i;
  return td->reorder(old2new);
}

TensorDomain* TransformIter::replay(Expr* expr, TensorDomain* td) {
  TORCH_INTERNAL_ASSERT(expr->isExpr());
  switch (*(expr->getExprType())) {
    case (ExprType::Split):
      return replay(static_cast<Split*>(expr), td);
    case (ExprType::Merge):
      return replay(static_cast<Merge*>(expr), td);
    case (ExprType::Reorder):
      return replay(static_cast<Reorder*>(expr), td);
    default:
      TORCH_INTERNAL_ASSERT(false, "Could not detect expr type in replay.");
  }
}

TensorDomain* TransformIter::runReplay(
    TensorDomain* td,
    const std::vector<Expr*>& history) {
  for (Expr* op : history)
    td = TransformIter::replay(op, td);
  return td;
}

namespace {

void validate_axis_map(int nDims, const std::vector<int>& axis_map) {
  TORCH_INTERNAL_ASSERT(
      axis_map.size() == (unsigned int)nDims,
      "Invalid axis map in replay transform. NDims doesn't match.");

  TORCH_INTERNAL_ASSERT(
      !std::any_of(
          axis_map.begin(),
          axis_map.end(),
          [nDims](int i) { return i < -1 || i >= nDims; }),
      "Invalid axis map in replay transform, map goes outside domains of provided TensorDomain.");
}

void validate_history_entry(Expr* expr, int nDims) {
  TORCH_INTERNAL_ASSERT(
      expr->input(0)->getValType().value() == ValType::TensorDomain &&
          static_cast<TensorDomain*>(expr->input(0))->nDims() ==
              (unsigned int)nDims,
      "Invalid history, or invalid axis_map in TransformIter.");
}

struct Influence : public TransformIter {
 private:
  // BACKWARD INFLUENCE

  TensorDomain* replayBackward(Split* split, TensorDomain* td) override {
    int axis = split->axis();

    TORCH_INTERNAL_ASSERT(
        (unsigned int)(axis + 1) < influence.size(),
        "Error during replay backwards, td/influence size mismatch.");
    influence[axis] = influence[axis] | influence[axis + 1];
    influence.erase(influence.begin() + axis + 1);

    return split->in();
  }

  TensorDomain* replayBackward(Merge* merge, TensorDomain* td) override {
    int axis = merge->axis();
    TORCH_INTERNAL_ASSERT(
        (unsigned int)axis < influence.size(),
        "Error during replay backwards, td/influence size mismatch.");
    influence.insert(influence.begin() + axis + 1, influence[axis]);

    return merge->in();
  }

  TensorDomain* replayBackward(Reorder* reorder, TensorDomain* td) override {
    // new2old[new_pos] = old_pos Generate new old2new map
    const std::vector<int>& new2old = reorder->new2old();

    std::vector<bool> reorder_influence(influence.size(), false);
    for (decltype(new2old.size()) i = 0; i < new2old.size(); i++) {
      int new_pos = i;
      int old_pos = new2old[i];
      TORCH_INTERNAL_ASSERT(
          new_pos < (int)influence.size() &&
              old_pos < (int)reorder_influence.size(),
          "Error during replay backwards, td/influence size mismatch.");
      reorder_influence[old_pos] = influence[new_pos];
    }

    influence = reorder_influence;
    return reorder->in();
  }

  // FORWARD INFLUENCE

  TensorDomain* replay(Split* split, TensorDomain* td) override {
    int axis = split->axis();
    TORCH_INTERNAL_ASSERT(
        (unsigned int)axis < influence.size(),
        "Error during replay, td/influence size mismatch.");
    influence.insert(influence.begin() + axis + 1, influence[axis]);
    return nullptr;
  }

  TensorDomain* replay(Merge* merge, TensorDomain* td) override {
    int axis = merge->axis();
    TORCH_INTERNAL_ASSERT(
        axis >= 0 && (unsigned int)(axis + 1) < influence.size(),
        "Error during replay, td/influence size mismatch.");
    influence[axis] = influence[axis] | influence[axis + 1];
    influence.erase(influence.begin() + axis + 1);
    return nullptr;
  }

  TensorDomain* replay(Reorder* reorder, TensorDomain* td) override {
    // new2old[new_pos] = old_pos Generate new old2new map
    const std::vector<int>& new2old = reorder->new2old();

    std::vector<bool> reorder_influence(influence.size(), false);
    for (decltype(new2old.size()) i = 0; i < new2old.size(); i++) {
      int new_pos = i;
      int old_pos = new2old[i];
      TORCH_INTERNAL_ASSERT(
          new_pos < (int)influence.size() &&
              old_pos < (int)reorder_influence.size(),
          "Error during replay, td/influence size mismatch.");
      reorder_influence[new_pos] = influence[old_pos];
    }

    influence = reorder_influence;
    return nullptr;
  }

  // INTERFACE

  std::vector<bool> influence;

  Influence(std::vector<bool> td_influence)
      : influence(std::move(td_influence)) {}

  using TransformIter::replayBackward;
  using TransformIter::runReplay;

 public:
  static std::vector<bool> computeBackward(
      const std::vector<Expr*>& history,
      const std::vector<bool>& td_influence) {
    if (history.empty())
      return td_influence;

    Val* last_val = history[history.size() - 1]->output(0);
    TORCH_INTERNAL_ASSERT(
        last_val->getValType().value() == ValType::TensorDomain &&
            static_cast<TensorDomain*>(last_val)->nDims() ==
                td_influence.size(),
        "Tried to compute influence, but recieved an influence vector that does not match the expected size.");

    Influence inf(td_influence);
    std::vector<Expr*> ops(history.rbegin(), history.rend());
    for (Expr* op : ops)
      inf.replayBackward(op, nullptr);
    return inf.influence;
  }

  static std::vector<bool> computeForward(
      const std::vector<Expr*>& history,
      const std::vector<bool>& td_influence) {
    if (history.empty())
      return td_influence;

    TORCH_INTERNAL_ASSERT(
        history[0]->input(0)->getValType().value() == ValType::TensorDomain &&
            static_cast<TensorDomain*>(history[0]->input(0))->nDims() ==
                td_influence.size(),
        "Tried to compute influence, but recieved an influence vector that does not match the expected size.");
    Influence inf(td_influence);
    inf.runReplay(nullptr, history);
    return inf.influence;
  }

}; // struct Influence

struct Replay : public TransformIter {
  /*
   * Replay functions, takes a TensorDomain and steps through the operations in
   * "record" based on influence axes. Will also update influence and propagate
   * it forward.
   */
  TensorDomain* replay(Split* split, TensorDomain* td) override {
    int saxis = split->axis();

    TORCH_INTERNAL_ASSERT(
        saxis >= 0 && (unsigned int)saxis < axis_map.size(),
        "TransformReplay tried to modify an axis out of range, recieved ",
        saxis,
        " but this value should be >=0 and <",
        axis_map.size());

    // Axis relative to td
    int axis = axis_map[saxis];

    if (axis == -1) {
      // don't modify path, we need an extra axis as there would have been one
      // there, but we shouldn't modify it.
      axis_map.insert(axis_map.begin() + saxis + 1, -1);
      return td;
    }

    // Move indices up as we now have an extra axis
    std::transform(
        axis_map.begin(), axis_map.end(), axis_map.begin(), [axis](int i) {
          return i > axis ? i + 1 : i;
        });

    // Insert new axis in map
    axis_map.insert(axis_map.begin() + saxis + 1, axis + 1);

    TORCH_INTERNAL_ASSERT(
        split->factor()->isConst(),
        "Cannot replay split as it's not based on a const value.");
    td = td->split(axis, split->factor()->value().value());

    return td;
  }

  TensorDomain* replay(Merge* merge, TensorDomain* td) override {
    int maxis = merge->axis();

    TORCH_INTERNAL_ASSERT(
        maxis >= 0 && (unsigned int)(maxis + 1) < axis_map.size(),
        "TransformReplay tried to modify an axis out of range, recieved ",
        maxis,
        " but this value should be >= 0 and < axis_map.size()");

    // Get axis relative to what we actually have in td.
    int axis = axis_map[maxis];
    int axis_p_1 = axis_map[maxis + 1];
    // If either dim is not to be touch, set both not to be touched
    axis = axis_p_1 == -1 ? -1 : axis;
    axis_map[maxis] = axis;

    // Remove axis from axis_map as in original transformations it didn't exist
    axis_map.erase(axis_map.begin() + maxis + 1);

    // Don't modify:
    if (axis == -1)
      return td;

    // Move indices down as we're removing an axis
    std::transform(
        axis_map.begin(), axis_map.end(), axis_map.begin(), [axis](int i) {
          return i > axis ? i - 1 : i;
        });

    return td->merge(axis);
  }

  // This transform requires reordering axes in td, then updating the axis_map
  // We want to replay axes in td, not marked with -1, to match that in the
  // provided reorder. This must be done because there may be a reorder that's
  // required for a merge, as merge is specified by the first axes and merges
  // the next consecutive axis.
  //
  // Once we transform td, we need to update axis_map or the mapping to provide:
  // reorder->in()->axis(i) == reorder->axis(axis_map[i])
  //
  // Axes not marked with -1 should be placed in the outer most dimensions in
  // the relative order specified by reorder. Remaining axes should be placed in
  // the inner most dimensions maintaining their original relative positioning.
  TensorDomain* replay(Reorder* reorder, TensorDomain* td) override {
    // convert to old2new as it makes this easier to do, and we need that map
    // anyways in the end to replay reorder
    const std::vector<int>& new2old_orig = reorder->new2old();
    std::vector<int> old2new_orig(new2old_orig.size());
    for (decltype(new2old_orig.size()) i{0}; i < new2old_orig.size(); i++)
      old2new_orig[new2old_orig[i]] = i;

    // old2new_orig: reorder->in()->axis(i) ==
    // reorder->out()->axis(old2new_orig[i])

    // We would like old2new: td->axis(i) == td_out->axis(old2new[i])
    // if td->axis(i) will participate in the reorder defined by "reorder"
    auto extent = reorder->in()->nDims() > td->nDims() ? reorder->in()->nDims()
                                                       : td->nDims();
    std::vector<int> old2new(extent, -1);
    for (decltype(old2new_orig.size()) i{0}; i < old2new_orig.size(); i++) {
      int old_pos = axis_map[i];
      int new_pos = old2new_orig[i];
      if (old_pos != -1)
        old2new[old_pos] = new_pos;
    }

    // We want to push to the left the new positions in td_out therefore if our
    // map looks like:
    //
    // Going to move all new_pos to the left so there's no gaps, for example if
    // we have: old2new = 2 -1 4 -1 0 (0 -> 2, 2 -> 4, 4->0) we will the new
    // positions down to: old2new = 1 -1 2 -1 0 (0 -> 1, 2 -> 2, 4->0)
    //  0 -1 -1  3 -1 -1  6
    //  0 -1 -1  0 -1 -1  0
    //  0 -1 -2 -2 -3 -4 -4
    // offset[0]  =  0 0->0
    // offset[3]  = -2 3->1
    // offset[6]  = -4 6->2
    // --------------------
    // -1 -1 -1  3 -1 -1  6
    // -1 -1 -1  0 -1 -1  0
    // -1 -2 -3 -3 -4 -5 -5
    // offset[3]  = -3 3->0
    // offset[6]  = -5 6->1

    std::vector<int> offset(old2new.size(), -1);
    for (decltype(offset.size()) i{0}; i < offset.size(); i++) {
      // If we process this axis
      if (old2new[i] != -1)
        // we wouldn't offset based on this value
        offset[old2new[i]] = 0;
    }

    // Prefix sum offset
    for (decltype(offset.size()) i{1}; i < offset.size(); i++) {
      offset[i] += offset[i - 1];
    }
    // Offset is now computed

    // Apply offset
    for (decltype(old2new.size()) i{0}; i < old2new.size(); i++) {
      if (old2new[i] == -1)
        continue;
      old2new[i] += offset[old2new[i]];
    }

    /*
     * old2new should now be the output of what we mention ^^ for offset, i.e.
     * old2new = 2 -1 4 -1 0 (0 -> 2, 2 -> 4, 4->0)
     * should now be:
     * old2new = 1 -1 2 -1 0 (0 -> 1, 2->2, 4->0)
     * OR:
     * old2new = 1 -1 4 -1 -1 (0->1, 2->4)
     * should now be:
     * old2new = 0 -1 1 -1 -1 (0->0, 2->1)
     * Now we want to fill in -1 positions in relative order, i.e.
     * old2new = 1 -1 2 -1 0 (0 -> 1, 2->2, 4->0)
     * we want as:
     * old2new = 1 3 2 4 0 (0 -> 1, 1->3, 2->2, 3->4, 4->0)
     * OR:
     * old2new = 0 -1 1 -1 -1 (0->0, 2->1)
     * we want as:
     * old2new = 0 2 1 3 4 (0->0, 1->2, 2->1, 3->3, 4->4)
     */
    // grab the highest index in new_pos
    int max_new_pos = *std::max_element(old2new.begin(), old2new.end());
    // Fill in the -1 values in order
    for (decltype(old2new.size()) i{0}; i < old2new.size(); i++)
      if (old2new[i] == -1)
        old2new[i] = ++max_new_pos;
    old2new.erase(old2new.begin() + td->nDims(), old2new.end());

    std::set<int> missed_pos;
    for (decltype(old2new.size()) i{0}; i < old2new.size(); i++)
      missed_pos.emplace(i);

    for (decltype(old2new.size()) i{0}; i < old2new.size(); i++) {
      TORCH_INTERNAL_ASSERT(
          missed_pos.find(i) != missed_pos.end(),
          "Duplicate entries in replayed reorder map.");
      missed_pos.erase(i);
    }

    TORCH_INTERNAL_ASSERT(
        missed_pos.empty(),
        "It's a real mystery how we ended up here. Congrats.");

    // Check if this is a null opt i.e. no actual reordering needs to be done
    bool nullopt = true;
    std::unordered_map<int, int> old2new_map;
    for (decltype(td->nDims()) i{0}; i < td->nDims(); i++) {
      if (old2new[i] != (int)i) {
        nullopt = false;
      }
      old2new_map[i] = old2new[i];
    }

    // Even if null opt, I'm worried we could have a desynced axis_map as some
    // how reorder wasn't a null opt, but after axis_map it was. I'm uncertain
    // if this can happen but we can reorder axis_map anyways.

    // HAVE:
    // td->axis(old2new[i]) == td_out->axis(i)
    // reorder->in()->axis(old2new_orig[i]) = reorder->out()->axis(i)
    // reorder->in()->axis(i) ~= td->axis(axis_map[i])
    // NEED:
    // td_out->axis(reorder_axis_map[i]) ~= reorder->out()->axis(i)
    decltype(axis_map) reordered_axis_map(axis_map.size(), -1);
    for (decltype(axis_map.size()) i{0}; i < axis_map.size(); i++) {
      int reorder_in_axis = i;
      int td_axis = axis_map[i];
      if (td_axis == -1)
        continue;

      int reorder_out_axis = old2new_orig[reorder_in_axis];
      int td_axis_out = old2new[td_axis];
      reordered_axis_map[reorder_out_axis] = td_axis_out;
    }

    axis_map = reordered_axis_map;

    // If null opt do nothing, return td
    if (nullopt)
      return td;

    // Rerun reorder
    return td->reorder(old2new_map);
  }

  std::vector<int> axis_map;
  Replay(std::vector<int> _axis_map) : axis_map(std::move(_axis_map)) {}

 public:
  // Replays history provided on td, axis_map is the mapping from td axes to
  // those expected in history, if an axis shouldn't be transformed, it needs to
  // be marked as -1 in the axis_map
  static TensorDomain* replay(
      TensorDomain* td,
      const std::vector<Expr*>& history,
      const std::vector<int>& axis_map) {
    if (history.empty())
      return td;

    Replay r(axis_map);
    return r.runReplay(td, history);
  }

}; // struct Replay

struct ReplaySelf : public TransformIter {
  /*
   * Replay functions, takes a TensorDomain and steps through its own history
   * and reapplies it based on influence axes. Will replay rfactor axes
   * correctly as well.
   */
  TensorDomain* replay(Split* split, TensorDomain* td) override {
    int saxis = split->axis();

    TORCH_INTERNAL_ASSERT(
        saxis >= 0 && (unsigned int)saxis < axis_map.size(),
        "TransformReplay tried to modify an axis out of range, recieved ",
        saxis,
        " but this value should be >=0 and <",
        axis_map.size());

    // Axis relative to td
    int axis = axis_map[saxis];

    if (axis == -1) {
      // don't modify path, we need an extra axis as there would have been one
      // there, but we shouldn't modify it.
      axis_map.insert(axis_map.begin() + saxis + 1, -1);
      return td;
    }

    // Move indices up as we now have an extra axis
    std::transform(
        axis_map.begin(), axis_map.end(), axis_map.begin(), [axis](int i) {
          return i > axis ? i + 1 : i;
        });

    // Insert new axis in map
    axis_map.insert(axis_map.begin() + saxis + 1, axis + 1);

    TORCH_INTERNAL_ASSERT(
        split->factor()->isConst(),
        "Cannot replay split as it's not based on a const value.");

    // Create new domain reflecting split
    std::vector<IterDomain*> new_domain;
    for (decltype(td->nDims()) i{0}; i < td->nDims(); i++) {
      if ((int)i == axis) {
        // We want to support cases where our root domain has changed sizes
        // this happens in lowering when we replace sizes with runtime look ups
        IterDomain* td_axis = td->axis(axis);
        IterDomain* saxis_1 = split->out()->axis(saxis);
        IterDomain* saxis_2 = split->out()->axis(saxis + 1);
        // manually replay split domains using td extent, otherwise matching
        // split axes params.
        TORCH_CHECK(
            td_axis->start()->isZeroInt(),
            "Splitting IterDomains with starting values that aren't 0, is not supported at this time.");

        IterDomain* ido = new IterDomain(
            new Int(0),
            ceilDiv(td_axis->extent(), split->factor()),
            saxis_1->parallel_method(),
            saxis_1->isReduction(),
            saxis_1->isRFactorProduct());
        new_domain.push_back(ido);

        // inner loop IterDomain
        IterDomain* idi = new IterDomain(
            new Int(0),
            split->factor(),
            saxis_2->parallel_method(),
            saxis_2->isReduction(),
            saxis_2->isRFactorProduct());
        new_domain.push_back(idi);
      } else {
        // Add in all other axes, these may not match the input td to the split.
        new_domain.push_back(td->axis(i));
      }
    }

    TensorDomain* replayed = new TensorDomain(new_domain);
    new Split(replayed, td, axis, split->factor());
    return replayed;
  }

  TensorDomain* replay(Merge* merge, TensorDomain* td) override {
    int maxis = merge->axis();

    TORCH_INTERNAL_ASSERT(
        maxis >= 0 && (unsigned int)(maxis + 1) < axis_map.size(),
        "TransformReplay tried to modify an axis out of range, recieved ",
        maxis,
        " but this value should be >= 0 and < axis_map.size()");

    // Get axis relative to what we actually have in td.
    int axis = axis_map[maxis];
    int axis_p_1 = axis_map[maxis + 1];
    // If either dim is not to be touch, set both not to be touched
    axis = axis_p_1 == -1 ? -1 : axis;
    axis_map[maxis] = axis;

    // Remove axis from axis_map as in original transformations it didn't exist
    axis_map.erase(axis_map.begin() + maxis + 1);

    // Don't modify:
    if (axis == -1)
      return td;

    // Move indices down as we're removing an axis
    std::transform(
        axis_map.begin(), axis_map.end(), axis_map.begin(), [axis](int i) {
          return i > axis ? i - 1 : i;
        });

    // Create new domain reflecting post-merge
    std::vector<IterDomain*> new_domain;
    for (decltype(td->nDims()) i{0}; i < td->nDims(); i++) {
      if ((int)i == axis) {
        // We want to support cases where our root domain has changed sizes
        // this happens in lowering when we replace sizes with runtime look ups
        IterDomain* td_axis1 = td->axis(axis);
        IterDomain* td_axis2 = td->axis(axis_p_1);
        IterDomain* m_axis = merge->out()->axis(maxis);

        TORCH_INTERNAL_ASSERT(
            td_axis1->start()->isZeroInt() && td_axis2->start()->isZeroInt(),
            "Splitting IterDomains with starting values that aren't 0, is not supported at this time.");

        IterDomain* merged = new IterDomain(
            new Int(0),
            mul(td_axis1->extent(), td_axis2->extent()),
            m_axis->parallel_method(),
            m_axis->isReduction(),
            m_axis->isRFactorProduct());
        new_domain.push_back(merged);

      } else if ((int)i != axis_p_1) {
        // Add in all other axes, these may not match the input td to the split.
        new_domain.push_back(td->axis(i));
      }
    }

    TensorDomain* replayed = new TensorDomain(new_domain);
    new Merge(replayed, td, axis);
    return replayed;
  }

  // TODO: This is the same as Replay::replay, should work towards code reuse.
  TensorDomain* replay(Reorder* reorder, TensorDomain* td) override {
    // convert to old2new as it makes this easier to do, and we need that map
    // anyways in the end to replay reorder
    const std::vector<int>& new2old_orig = reorder->new2old();
    std::vector<int> old2new_orig(new2old_orig.size());
    for (decltype(new2old_orig.size()) i{0}; i < new2old_orig.size(); i++)
      old2new_orig[new2old_orig[i]] = i;

    // old2new_orig: reorder->in()->axis(i) ==
    // reorder->out()->axis(old2new_orig[i])

    // We would like old2new: td->axis(i) == td_out->axis(old2new[i])
    // if td->axis(i) will participate in the reorder defined by "reorder"
    auto extent = reorder->in()->nDims() > td->nDims() ? reorder->in()->nDims()
                                                       : td->nDims();
    std::vector<int> old2new(extent, -1);
    for (decltype(old2new_orig.size()) i{0}; i < old2new_orig.size(); i++) {
      int old_pos = axis_map[i];
      int new_pos = old2new_orig[i];
      if (old_pos != -1)
        old2new[old_pos] = new_pos;
    }

    // We want to push to the left the new positions in td_out therefore if our
    // map looks like:
    //
    // Going to move all new_pos to the left so there's no gaps, for example if
    // we have: old2new = 2 -1 4 -1 0 (0 -> 2, 2 -> 4, 4->0) we will the new
    // positions down to: old2new = 1 -1 2 -1 0 (0 -> 1, 2 -> 2, 4->0)
    //  0 -1 -1  3 -1 -1  6
    //  0 -1 -1  0 -1 -1  0
    //  0 -1 -2 -2 -3 -4 -4
    // offset[0]  =  0 0->0
    // offset[3]  = -2 3->1
    // offset[6]  = -4 6->2
    // --------------------
    // -1 -1 -1  3 -1 -1  6
    // -1 -1 -1  0 -1 -1  0
    // -1 -2 -3 -3 -4 -5 -5
    // offset[3]  = -3 3->0
    // offset[6]  = -5 6->1

    std::vector<int> offset(old2new.size(), -1);
    for (decltype(offset.size()) i{0}; i < offset.size(); i++) {
      // If we process this axis
      if (old2new[i] != -1)
        // we wouldn't offset based on this value
        offset[old2new[i]] = 0;
    }

    // Prefix sum offset
    for (decltype(offset.size()) i{1}; i < offset.size(); i++) {
      offset[i] = offset[i] + offset[i - 1];
    }
    // Offset is now computed

    // Apply offset
    for (decltype(old2new.size()) i{0}; i < old2new.size(); i++) {
      if (old2new[i] == -1)
        continue;
      old2new[i] += offset[old2new[i]];
    }

    /*
     * old2new should now be the output of what we mention ^^ for offset, i.e.
     * old2new = 2 -1 4 -1 0 (0 -> 2, 2 -> 4, 4->0)
     * should now be:
     * old2new = 1 -1 2 -1 0 (0 -> 1, 2->2, 4->0)
     * OR:
     * old2new = 1 -1 4 -1 -1 (0->1, 2->4)
     * should now be:
     * old2new = 0 -1 1 -1 -1 (0->0, 2->1)
     * Now we want to fill in -1 positions in relative order, i.e.
     * old2new = 1 -1 2 -1 0 (0 -> 1, 2->2, 4->0)
     * we want as:
     * old2new = 1 3 2 4 0 (0 -> 1, 1->3, 2->2, 3->4, 4->0)
     * OR:
     * old2new = 0 -1 1 -1 -1 (0->0, 2->1)
     * we want as:
     * old2new = 0 2 1 3 4 (0->0, 1->2, 2->1, 3->3, 4->4)
     */
    // grab the highest index in new_pos
    int max_new_pos = -1;
    for (decltype(old2new.size()) i{0}; i < old2new.size(); i++)
      max_new_pos = max_new_pos > old2new[i] ? max_new_pos : old2new[i];
    // Fill in the -1 values in order
    for (decltype(old2new.size()) i{0}; i < old2new.size(); i++)
      if (old2new[i] == -1)
        old2new[i] = ++max_new_pos;
    old2new.erase(old2new.begin() + td->nDims(), old2new.end());

    std::set<int> missed_pos;
    for (decltype(old2new.size()) i{0}; i < old2new.size(); i++)
      missed_pos.emplace(i);

    for (decltype(old2new.size()) i{0}; i < old2new.size(); i++) {
      TORCH_INTERNAL_ASSERT(
          missed_pos.find(i) != missed_pos.end(),
          "Duplicate entries in replayed reorder map.");
      missed_pos.erase(i);
    }

    TORCH_INTERNAL_ASSERT(
        missed_pos.empty(),
        "It's a real mystery how we ended up here. Congrats.");

    // Check if this is a null opt i.e. no actual reordering needs to be done
    bool nullopt = true;
    std::unordered_map<int, int> old2new_map;
    for (decltype(td->nDims()) i{0}; i < td->nDims(); i++) {
      if (old2new[i] != (int)i) {
        nullopt = false;
      }
      old2new_map[i] = old2new[i];
    }

    // Even if null opt, I'm worried we could have a desynced axis_map as some
    // how reorder wasn't a null opt, but after axis_map it was. I'm uncertain
    // if this can happen but we can reorder axis_map anyways.

    // HAVE:
    // td->axis(old2new[i]) == td_out->axis(i)
    // reorder->in()->axis(old2new_orig[i]) = reorder->out()->axis(i)
    // reorder->in()->axis(i) ~= td->axis(axis_map[i])
    // NEED:
    // td_out->axis(reorder_axis_map[i]) ~= reorder->out()->axis(i)
    decltype(axis_map) reordered_axis_map(axis_map.size(), -1);
    for (decltype(axis_map.size()) i{0}; i < axis_map.size(); i++) {
      int reorder_in_axis = i;
      int td_axis = axis_map[i];
      if (td_axis == -1)
        continue;

      int reorder_out_axis = old2new_orig[reorder_in_axis];
      int td_axis_out = old2new[td_axis];
      reordered_axis_map[reorder_out_axis] = td_axis_out;
    }

    axis_map = reordered_axis_map;

    // If null opt do nothing, return td
    if (nullopt)
      return td;

    // Rerun reorder
    return td->reorder(old2new_map);
  }

  std::vector<int> axis_map;
  ReplaySelf(std::vector<int> _axis_map) : axis_map(std::move(_axis_map)) {}

 public:
  // Replays history provided on td, axis_map is the mapping from td axes to
  // those expected in history, if an axis shouldn't be transformed, it needs to
  // be marked as -1 in the axis_map
  static TensorDomain* replay(
      TensorDomain* td,
      const std::vector<Expr*>& history,
      const std::vector<int>& axis_map) {
    ReplaySelf r(axis_map);
    return r.runReplay(TransformIter::getRoot(td), history);
  }

}; // struct ReplaySelf

struct TransformBackward : public TransformIter {
 private:
  // axis_map goes from the transform position to the position in our modified
  // td.
  TensorDomain* replayBackward(Split* split, TensorDomain* td) override {
    int saxis = split->axis();

    TORCH_INTERNAL_ASSERT(
        saxis >= 0 && (unsigned int)saxis < axis_map.size(),
        "TransformBackward tried to modify an axis out of range, recieved ",
        saxis,
        " but this value should be >= 0 and < axis_map.size()");

    // Get axis relative to what we actually have in td.
    int axis = axis_map[saxis];
    int axis_p_1 = axis_map[saxis + 1];
    // If either dim is not to be touch, set both not to be touched
    axis = axis_p_1 == -1 ? -1 : axis;
    axis_map[saxis] = axis;

    // Remove axis from axis_map as in original transformations it didn't exist
    axis_map.erase(axis_map.begin() + saxis + 1);

    // Don't modify:
    if (axis == -1)
      return td;

    // Move indices down as previously we didn't have the split axis
    std::transform(
        axis_map.begin(), axis_map.end(), axis_map.begin(), [axis](int i) {
          return i > axis ? i - 1 : i;
        });

    // Create new domain reflecting pre-split
    std::vector<IterDomain*> new_domain;
    for (decltype(td->nDims()) i{0}; i < td->nDims(); i++) {
      if ((int)i == axis) {
        IterDomain* orig_axis = split->in()->axis(saxis);
        // Insert pre-split axis, make sure isReduction matches what is expected
        new_domain.push_back(new IterDomain(
            orig_axis->start(),
            orig_axis->extent(),
            orig_axis->parallel_method(),
            td->axis(axis)->isReduction(),
            td->axis(axis)->isRFactorProduct()));
      } else if ((int)i != axis_p_1) {
        // Add in all other axes, these may not match the input td to the split.
        new_domain.push_back(td->axis(i));
      }
    }

    TensorDomain* replayed_inp = new TensorDomain(new_domain);
    new Split(td, replayed_inp, axis, split->factor());
    return replayed_inp;
  }

  TensorDomain* replayBackward(Merge* merge, TensorDomain* td) override {
    /*
     * Remember axis_map goes from merge information -> how it's stored in td
     * When we're done we want axis_map to match the returned td before or not
     * before the merge depending on should_modify.
     */

    int maxis = merge->axis();

    TORCH_INTERNAL_ASSERT(
        maxis >= 0 && (unsigned int)maxis < axis_map.size(),
        "TransformBackward tried to modify an axis out of range, recieved ",
        maxis,
        " but this value should be >=0 and <",
        axis_map.size());

    if (axis_map[maxis] == -1) {
      // don't modify path, we need an extra axis as there was previously one
      // there, but we shouldn't modify it.
      axis_map.insert(axis_map.begin() + maxis + 1, -1);
      return td;
    }

    // Recreate the merge, axis is relative to the td
    int axis = axis_map[maxis];
    // Move indices up as previously we had an extra axis
    std::transform(
        axis_map.begin(), axis_map.end(), axis_map.begin(), [axis](int i) {
          return i > axis ? i + 1 : i;
        });

    // Insert pre-merged axis back into map
    axis_map.insert(axis_map.begin() + maxis + 1, axis_map[maxis] + 1);

    // Create new domain reflecting pre-merge
    std::vector<IterDomain*> new_domain;
    for (decltype(td->nDims()) i{0}; i < td->nDims(); i++) {
      if ((int)i == axis) {
        IterDomain* td_axis = td->axis(axis);
        IterDomain* maxis_1 = merge->in()->axis(maxis);
        IterDomain* maxis_2 = merge->in()->axis(maxis + 1);
        new_domain.push_back(new IterDomain(
            maxis_1->start(),
            maxis_1->extent(),
            ParallelType::Serial,
            td_axis->isReduction(),
            td_axis->isRFactorProduct()));
        new_domain.push_back(new IterDomain(
            maxis_2->start(),
            maxis_2->extent(),
            ParallelType::Serial,
            td_axis->isReduction(),
            td_axis->isRFactorProduct()));
      } else {
        // Add in all other axes, these may not match the input td to the split.
        new_domain.push_back(td->axis(i));
      }
    }

    TensorDomain* replayed_inp = new TensorDomain(new_domain);
    new Merge(td, replayed_inp, axis);
    return replayed_inp;
  }

  TensorDomain* replayBackward(Reorder* reorder, TensorDomain* td) override {
    const std::vector<int>& new2old_orig = reorder->new2old();

    // We want to convert new2old to something with td->nDims which it isn't
    // guarenteed to be
    std::vector<int> new2old(td->nDims(), -1);

    for (decltype(new2old_orig.size()) i{0}; i < new2old_orig.size(); i++) {
      int new_pos = axis_map[i]; // position in td
      int old_pos = new2old_orig[i]; // position it should be at before td

      if (new_pos != -1)
        new2old[new_pos] = old_pos;
    }

    // We want to push to the RIGHT the modified positions in td_in. This is
    // in comparison with forward replay which moves modified positions to the
    // left.

    // Easiest to start by moving to left like forward replay

    std::vector<int> new2old_offset(new2old_orig.size(), -1);
    // Create offset map
    for (decltype(new2old.size()) i{0}; i < new2old.size(); i++)
      if (new2old[i] != -1)
        new2old_offset[new2old[i]] = 0;

    // Prefix sum new2old_offset
    for (decltype(new2old_offset.size()) i{1}; i < new2old_offset.size(); i++)
      new2old_offset[i] += new2old_offset[i - 1];
    // Apply offset
    for (decltype(new2old.size()) i{0}; i < new2old.size(); i++) {
      if (new2old[i] == -1)
        continue;
      new2old[i] += new2old_offset[new2old[i]];
    }

    int max_elem = *std::max_element(new2old.begin(), new2old.end());
    // Now lets push all elements to the right
    int right_offset = ((int)td->nDims()) - max_elem - 1;
    TORCH_INTERNAL_ASSERT(
        right_offset >= 0,
        "Error during backward replay, couldn't move modified axes to the right in reorder.");

    // Move to the right
    for (decltype(new2old.size()) i{0}; i < new2old.size(); i++) {
      if (new2old[i] == -1)
        continue;
      new2old[i] += right_offset;
    }

    // Fill in unmodified positions in order to the left
    int it = 0;
    for (decltype(td->nDims()) i{0}; i < td->nDims(); i++)
      if (new2old[i] == -1)
        new2old[i] = it++;

    // Trim new2old to match td
    new2old.erase(new2old.begin() + td->nDims(), new2old.end());

    // new2old_orig[reorder->out()->pos] = reorder->in()->pos
    // axis_map[reorder->out()->pos] = td->pos
    // new2old[td->pos] = old_td->pos
    // NEED: new_axis_map[reorder->in()->pos] = old_td->pos

    std::vector<int> new_axis_map(axis_map.size(), -1);
    for (decltype(new_axis_map.size()) i{0}; i < new_axis_map.size(); i++) {
      int reorder_out_pos = i;
      int reorder_in_pos = new2old_orig[reorder_out_pos];
      int td_pos = axis_map[reorder_out_pos];
      int old_td_pos = td_pos == -1 ? -1 : new2old[td_pos];

      new_axis_map[reorder_in_pos] = old_td_pos;
    }

    axis_map = new_axis_map;

    std::vector<IterDomain*> old_td(td->nDims(), nullptr);
    for (decltype(new2old.size()) i{0}; i < new2old.size(); i++) {
      // new2old[new] = old relative to td
      int new_pos = i; // position in td
      int old_pos = new2old[i]; // position it should be at before td
      old_td[old_pos] = td->axis(new_pos);
    }

    TensorDomain* replayed_inp = new TensorDomain(old_td);
    new Reorder(td, replayed_inp, new2old);
    return replayed_inp;
  }

  // Entry for backward influence propagation on td following record, history
  // should be present -> past as you go through the vector
  TensorDomain* replayBackward(
      TensorDomain* td,
      const std::vector<Expr*>& history) {
    TensorDomain* running_td = td;

    std::vector<Expr*> rev_history(history.rbegin(), history.rend());
    for (Expr* op : rev_history)
      running_td = TransformIter::replayBackward(op, running_td);
    return running_td;
  }

  std::vector<int> axis_map;

  TransformBackward(std::vector<int> _axis_map)
      : axis_map(std::move(_axis_map)){};

 public:
  static TensorDomain* replay(
      TensorDomain* td,
      const std::vector<Expr*>& history,
      const std::vector<int>& axis_map) {
    TransformBackward tb(axis_map);
    return tb.replayBackward(td, history);
  }
};

struct RFactorRoot : public TransformIter {
  bool found_non_rfactor_op = false;

  TensorDomain* replay(Split* split, TensorDomain*) final {
    if (!split->in()->axis(split->axis())->isRFactorProduct())
      found_non_rfactor_op = true;
    return split->out();
  }

  TensorDomain* replay(Merge* merge, TensorDomain*) final {
    if (!merge->in()->axis(merge->axis())->isRFactorProduct())
      found_non_rfactor_op = true;
    return merge->out();
  }

  TensorDomain* replay(Reorder* reorder, TensorDomain*) final {
    return reorder->out();
  }

  // Replay forward until we hit an operation that doesn't involve an rfactor
  // axis
  TensorDomain* runReplay(TensorDomain*, const std::vector<Expr*>& history)
      final {
    TORCH_INTERNAL_ASSERT(
        !history.empty(), "No history provided to find rfactor root domain.");

    auto last_rfactor_op = history.begin();
    auto running_op = history.begin();

    for (auto it = history.begin(); it != history.end(); it++) {
      TransformIter::replay(*it, nullptr);
      if (found_non_rfactor_op)
        break;
      running_op = it;
      if ((*it)->getExprType() != ExprType::Reorder) {
        last_rfactor_op = it;
      }
    }

    // We need to make sure the rfactor root is ordered correctly.
    bool found_valid_rfactor_root = false;

    Val* val;

    while (!found_valid_rfactor_root && last_rfactor_op != history.end()) {
      // Try next val
      val = (*last_rfactor_op++)->output(0);
      TORCH_INTERNAL_ASSERT(
          val->getValType().value() == ValType::TensorDomain,
          "Invalid history to find rfactor root.");

      TensorDomain* td = static_cast<TensorDomain*>(val);
      bool found_rfactor_dim = false;
      for (decltype(td->nDims()) i{0}; i < td->nDims(); i++) {
        if (found_rfactor_dim) {
          if (!td->axis(i)->isRFactorProduct())
            break;
        } else {
          if (td->axis(i)->isRFactorProduct())
            found_rfactor_dim = true;
        }
        if (i == td->nDims() - 1)
          found_valid_rfactor_root = true;
      }
    }
    TORCH_INTERNAL_ASSERT(
        found_valid_rfactor_root, "Could not find a valid rfactor root.");
    return static_cast<TensorDomain*>(val);
  }

 public:
  static TensorDomain* get(TensorDomain* td) {
    auto history = TransformIter::getHistory(td);
    if (history.empty())
      return td;
    RFactorRoot rfr;
    return rfr.runReplay(nullptr, history);
  }
};

} // namespace

// API INTO TRANSFORM ITER

std::vector<bool> TransformIter::getRootInfluence(
    TensorDomain* td,
    const std::vector<bool>& td_influence) {
  return Influence::computeBackward(
      TransformIter::getHistory(td), td_influence);
}

std::vector<bool> TransformIter::replayBackwardInfluence(
    const std::vector<Expr*>& history,
    const std::vector<bool>& td_influence) {
  return Influence::computeBackward(history, td_influence);
}

std::vector<bool> TransformIter::replayInfluence(
    const std::vector<Expr*>& history,
    const std::vector<bool>& td_influence) {
  if (history.empty())
    return td_influence;

  return Influence::computeForward(history, td_influence);
}

TensorDomain* TransformIter::replay(
    TensorDomain* td,
    const std::vector<Expr*>& history,
    const std::vector<int>& axis_map) {
  if (history.empty())
    return td;
  if (std::none_of(
          axis_map.begin(), axis_map.end(), [](int i) { return i > -1; }))
    return td;

  validate_history_entry(history[0], axis_map.size());
  return Replay::replay(td, history, axis_map);
}

TensorDomain* TransformIter::replaySelf(
    TensorDomain* td,
    const std::vector<Expr*>& history,
    const std::vector<int>& axis_map) {
  if (std::none_of(
          axis_map.begin(), axis_map.end(), [](int i) { return i > -1; }))
    return TransformIter::getRoot(td);

  validate_axis_map(TransformIter::getRoot(td)->nDims(), axis_map);
  return ReplaySelf::replay(td, history, axis_map);
}

TensorDomain* TransformIter::replayBackward(
    TensorDomain* td,
    const std::vector<Expr*>& history,
    const std::vector<int>& axis_map) {
  if (history.empty())
    return td;
  if (std::none_of(
          axis_map.begin(), axis_map.end(), [](int i) { return i > -1; }))
    return td;

  TORCH_INTERNAL_ASSERT(
      history[history.size() - 1]->output(0)->getValType().value() ==
              ValType::TensorDomain &&
          static_cast<TensorDomain*>(history[history.size() - 1]->output(0))
                  ->nDims() == axis_map.size(),
      "Invalid history, or invalid axis_map in TransformIter.");

  return TransformBackward::replay(td, history, axis_map);
}

TensorDomain* TransformIter::getRFactorRoot(TensorDomain* td) {
  auto td_root = TransformIter::getRoot(td);
  if (std::none_of(
          td_root->domain().begin(),
          td_root->domain().end(),
          [](IterDomain* id) { return id->isRFactorProduct(); }))
    return td_root;

  auto ret = RFactorRoot::get(td);
  return ret;
}

} // namespace fuser
} // namespace jit
} // namespace torch
