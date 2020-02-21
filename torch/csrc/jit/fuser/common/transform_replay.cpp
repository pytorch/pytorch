#include <torch/csrc/jit/fuser/common/ir.h>
#include <torch/csrc/jit/fuser/common/iter_visitor.h>
#include <torch/csrc/jit/fuser/common/tensor.h>
#include <torch/csrc/jit/fuser/common/transform_replay.h>

#include <algorithm>
#include <vector>

// Could be in a .cpp file:
#include <torch/csrc/jit/fuser/common/fusion.h>

// For debug:
/**/
#include <torch/csrc/jit/fuser/common/iriostream.h>
/**/
namespace torch {
namespace jit {
namespace fuser {

// For debug:
/**/
std::ostream& operator<<(std::ostream& os, std::vector<bool> vec) {
  os << "<";
  for (int i = 0; i < vec.size(); i++) {
    if (vec[i])
      os << " t";
    else
      os << " f";
    if (i == vec.size() - 1)
      os << ">";
    else
      os << ",";
  }
  return os;
}
/**/

/*
 * Functions to backward propagate influence from split/merge/reorder
 */
void TransformReplay::compute_influence(const Split* expr) {
  int axis = expr->axis();
  influence[axis] = influence[axis] | influence[axis + 1];
  influence.erase(influence.begin() + axis + 1);
}

void TransformReplay::compute_influence(const Merge* expr) {
  int axis = expr->axis();
  influence.insert(influence.begin() + axis + 1, influence[axis]);
}

void TransformReplay::compute_influence(const Reorder* expr) {
  const std::vector<int>& pos2axis = expr->pos2axis();

  std::vector<bool> reorder_influence(influence.size(), false);
  for (decltype(pos2axis.size()) i = 0; i < pos2axis.size(); i++) {
    int old_pos = i;
    int new_pos = pos2axis[i];
    reorder_influence[old_pos] = influence[new_pos];
  }

  influence = reorder_influence;
}

// Backward influence propagate dispatch
void TransformReplay::compute_influence(const Expr* expr) {
  TORCH_CHECK(expr->isExpr());
  switch (*(expr->getExprType())) {
    case (ExprType::Split):
      compute_influence(static_cast<const Split*>(expr));
      break;
    case (ExprType::Merge):
      compute_influence(static_cast<const Merge*>(expr));
      break;
    case (ExprType::Reorder):
      compute_influence(static_cast<const Reorder*>(expr));
      break;
    default:
      throw std::runtime_error(
          "Could not detect expr type in compute_influence.");
  }
}

// Entry for backward influence propagation on td following record
void TransformReplay::compute_influence(const TensorDomain* td) {
  influence = std::vector<bool>(td->size(), false);
  for (int i = 0; i < compute_at_axis; i++)
    influence[i] = true;

  for (auto it = record.rbegin(); it < record.rend(); ++it) {
    compute_influence(*it);
  }
}

// Trace back the history of td, record the Expr's that made this td (split,
// merge, reorder)
const TensorDomain* TransformReplay::get_root(
    const TensorDomain* td,
    bool create_record) {
  if (create_record)
    record = std::vector<const Expr*>();

  const TensorDomain* root = td; // backward running td
  Fusion* fusion = FusionGuard::getCurFusion();

  // Get my origin
  const Expr* orig = fusion->origin(root);
  std::set<const Expr*> visited_exprs;

  // If I'm not back to the original td
  while (orig != nullptr) {
    if (visited_exprs.find(orig) != visited_exprs.end())
      throw std::runtime_error(
          "TransformReplay::get_root is not traversing a correct history.");

    visited_exprs.emplace(orig);
    const TensorDomain* previous_td = nullptr;
    // Check inputs of this operation, make sure there isn't more than one TD
    // I can only record operations that only take this TD as an input.
    for (const Val* inp : orig->inputs())
      if (inp->getValType() == ValType::TensorDomain) {
        if (previous_td != nullptr)
          throw std::runtime_error(
              "TransformReplay::get_root could not decifer transform history of a TensorDomain.");

        // Place transform op on top of stack.
        if (create_record)
          record.push_back(orig);

        // Traverse back
        root = static_cast<const TensorDomain*>(inp);
        orig = fusion->origin(root);
      }
  }
  if (create_record)
    std::reverse(record.begin(), record.end());

  return root;
}

/*
 * Replay functions, takes a TensorView and steps through the operations in
 * "record" based on influence axes. Will also update influence and propagate
 * it forward.
 */
const TensorView* TransformReplay::replay(
    const Split* expr,
    const TensorView* tv) {
  int axis = expr->axis();
  // Forward prop influence
  influence.insert(influence.begin() + axis + 1, influence[axis]);
  // Replay split
  TORCH_CHECK(expr->factor()->isConst());
  return split(tv, expr->axis(), *(expr->factor()->value()));

  return tv;
}

const TensorView* TransformReplay::replay(
    const Merge* expr,
    const TensorView* tv) {
  int axis = expr->axis();
  // Forward prop influence
  influence[axis] = influence[axis] || influence[axis + 1];
  influence.erase(influence.begin() + axis + 1);
  // Replay merge
  return merge(tv, axis);
  return tv;
}

const TensorView* TransformReplay::replay(
    const Reorder* expr,
    const TensorView* tv) {
  // axis2pos[old_pos] = new_pos is sent to reorder, Reorder holds
  // pos2axis[new_pos] = old_pos Generate new axis2pos map
  std::unordered_map<int, int> axis2pos;
  const std::vector<int>& pos2axis = expr->pos2axis();

  std::vector<bool> reordered_influence(pos2axis.size(), false);

  for (decltype(pos2axis.size()) i = 0; i < pos2axis.size(); i++) {
    int new_pos = i;
    int old_pos = pos2axis[i];
    // Map to replay reorder
    axis2pos.emplace(std::pair<int, int>{old_pos, new_pos});
    // Forward prop influence
    reordered_influence[new_pos] = influence[old_pos];
  }
  // Replay reorder
  const TensorView* reordered_view = reorder(tv, axis2pos);
  influence = reordered_influence;
  return reordered_view;
}

// Dispatch for replay functions
const TensorView* TransformReplay::replay(
    const Expr* expr,
    const TensorView* tv) {
  TORCH_CHECK(expr->isExpr());
  switch (*(expr->getExprType())) {
    case (ExprType::Split):
      return replay(static_cast<const Split*>(expr), tv);
    case (ExprType::Merge):
      return replay(static_cast<const Merge*>(expr), tv);
    case (ExprType::Reorder):
      return replay(static_cast<const Reorder*>(expr), tv);
    default:
      throw std::runtime_error("Could not detect expr type in replay.");
  }
}

// Entry point for replay on a TensorView, will relpay all ops from "replay"
const TensorView* TransformReplay::replay(const TensorView* target) {
  const TensorView* tv = target;
  for (auto it = record.begin(); it < record.end(); ++it) {
    tv = replay(*it, tv);
  }
  return tv;
}

/*
 * 1) Take the reference, trace back its domain history to get all the
 * split/merge/reorder calls, as well as its original domain. Get the
 * original domain of the target as well.
 *
 * 2) We only need compute_at_axis and earlier dimensions to match for
 * compute_at. Therefore, we want to find all original axes that must have
 * been modified in order to produce the axes below compute_at_axis. We take a
 * bool vector called influence, and mark axes below compute_at_axis as true,
 * and all others as false. This vector is propagated up through
 * split/merge/reorder if split/merge/reorder output a marked axis, their
 * input will be marked as well. This marks all original axes required to be
 * modified to produce the axes below compute_at_axis.
 *
 * 3) We take the ordered list of split/merge/reorder and the influence vector
 * on the inputs and we apply all split/merge/reorder operations on the
 * replay_target. We also forward propagate the influence vector again, as
 * this time it could be different than originally marked.
 *
 * 4) We take all axes produced by the forward replay that are marked as
 * influenced and push them into a new domain. We then take all original axes
 * that are not marked by the second propagate of the influence vector and
 * push those as well.
 */
const TensorView* TransformReplay::runReplay(
    const TensorView* replay_ref,
    const TensorView* replay_target,
    int compute_at_axis) {
  /* STEP 1 */
  // Trace back to the root TensorDomain's of ref and target
  const TensorDomain* target_root = get_root(replay_target->domain());
  // As we trace the ref, record the operations to go from replay_ref ->
  // ref_root, save in "record"
  const TensorDomain* ref_root = get_root(replay_ref->domain(), true);
  // Domain sizes must match at root for replay!
  TORCH_CHECK(target_root->size() == ref_root->size());
  this->compute_at_axis = compute_at_axis;

  /* STEP 2 */
  // Mark compute_at_axis and below as "influenced", trace back through
  // operations, and map these axes to the ref root axis that were modified to
  // produce these axis
  compute_influence(replay_ref->domain());
  // We're going to save a copy of this vector, class member influnce will be
  // used during replay to forward propagate influence.
  std::vector<bool> root_influence_vector = influence;

  /* STEP 3 */
  // Replay operations while forward propagating influence. The resulting
  // influence can be different, than just the compute_at axis depending on
  // the combination of merge/split/reorder nodes
  const TensorView* full_replay =
      replay(new TensorView(replay_target->tensor(), replay_target->domain()));

  /* STEP 4 */
  // We've replayed all operations from ref on target. However, based on
  // influence, some of these are unlikely needed. Create a new TensorDomain,
  // push all axes from replay that were influenced, then push all root axes
  // that were not influenced. This is the correct resulting tensor.

  std::vector<const IterDomain*> compute_at_domain;
  auto dom_size = full_replay->domain()->size();
  for (decltype(dom_size) i = 0; i < dom_size; i++) {
    if (influence[i])
      compute_at_domain.push_back(full_replay->domain()->axis(i));
  }

  // Something could have gone wrong in backward/forward influence
  // propagation.
  TORCH_CHECK(target_root->size() == root_influence_vector.size());

  for (decltype(target_root->size()) i = 0; i < target_root->size(); i++)
    if (!root_influence_vector[i])
      compute_at_domain.push_back(target_root->axis(i));
  // Return the newly produced view!
  return new TensorView(
      replay_target->tensor(), new TensorDomain(compute_at_domain));
}

} // namespace fuser
} // namespace jit
} // namespace torch