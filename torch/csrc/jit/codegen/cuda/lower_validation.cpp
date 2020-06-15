#include <torch/csrc/jit/codegen/cuda/lower_validation.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>
#include <torch/csrc/jit/codegen/cuda/type.h>

namespace torch {
namespace jit {
namespace fuser {

namespace {

// Some pre-compilation checks
void IRValidate(Fusion* fusion) {
  FusionGuard fg(fusion);
  fusion->validateInputs();
  for (Val* val : fusion->vals()) {
    if (ir_utils::isTV(val)) {
      TensorView* tv = ir_utils::asTV(val);
      for (decltype(tv->nDims()) i{0}; i < tv->nDims(); i++) {
        IterDomain* id = tv->getComputeAtAxis(i).first;

        if (id->isBlockDim()) {
          TORCH_CHECK(
              !id->isBroadcast(),
              "Parallelization across blocks on broadcast axes is not supported, but found on, ",
              tv,
              ".");
        }
      }
    } // if ir_utils::isTV
  } // for(Val* val : fusion->vals())
} // validate

} // namespace

// Remove circular computeAt references
void IRFixComputeAt(Fusion* fusion) {
  FusionGuard fg(fusion);

  std::vector<Expr*> exprs = fusion->exprs(true);
  std::set<TensorView*> visited;
  for (auto it = exprs.rbegin(); it != exprs.rend(); it++) {
    Expr* expr = *it;
    if (!ir_utils::isTVOp(expr))
      continue;

    TensorView* tv = ir_utils::asTV(expr->output(0));
    TensorView* ctv = tv->getComputeAtView();

    if (ctv != nullptr && visited.find(ctv) == visited.end()) {
      ctv->setComputeAt(tv, (int)tv->getThisComputeAtAxis());
      tv->clearComputeAt();
    }
    visited.emplace(tv);
  }
}

// TensorViews are all based on symbolic sizes. When we first initialize them we
// don't know if they're inputs or outputs which would mean that they have
// runtime shapes. Intermediate tensors (those not going to global memory) do
// not have this information. Since we need to have the correct information in
// the kernel being fetched for shapes, we want to replace input and output
// tensors to reference the runtime structure containing sizes.
void IRReplaceSizes() {
  Fusion* fusion = FusionGuard::getCurFusion();
  // Sizes of inputs/outputs -> T.size[...]
  std::unordered_map<Val*, Val*> size_map;

  // Grab inputs and outputs
  std::vector<TensorView*> orig_inp_out;
  std::vector<TensorView*> all_tvs;

  for (auto* val : fusion->inputs())
    if (ir_utils::isTV(val))
      orig_inp_out.push_back(ir_utils::asTV(val));

  for (auto* val : fusion->outputs())
    if (ir_utils::isTV(val))
      orig_inp_out.push_back(ir_utils::asTV(val));

  for (auto* val : fusion->deterministic_vals()) {
    if (ir_utils::isTV(val)) {
      all_tvs.push_back(ir_utils::asTV(val));
    }
  }

  // Run through inputs and outputs first. Since we're replacing full
  // tensorviews their names are going to change. We need  the new referenc
  // name for the inputs/outputs. This way we won't reference the wrong tensor
  // view. For example T0 may be translated to T9. We don't want our new
  // variable to be T0->size[...] we need it to be T9->size[...]
  //
  // This could be done in a better way but changing split/merge to be a
  // TensorDomain focused operation, then we could simply do this process on
  // domains, instead of tensorviews. This would have the benefit that the
  // TensorView wouldn't change, so users pointers will remain valid. The other
  // option which seems less elegant but would also work is build up the domain
  // on the new tensor, and then simply replace it into the original one.
  for (TensorView* tv : orig_inp_out) {
    // Replace the domain with one based on Ti.size[j]
    std::vector<IterDomain*> new_domain_iters;
    const std::vector<IterDomain*>& root_td = tv->getRootDomain();

    for (decltype(root_td.size()) i{0}; i < root_td.size(); i++) {
      // Output sizes could have reduction axes, which isn't what gets output.
      if (root_td[i]->isReduction())
        continue;

      Val* orig_size = root_td[i]->extent();

      std::stringstream ss;
      ss << "T" << tv->name() << ".size[" << i << "]";
      Val* new_size =
          new NamedScalar(ss.str(), orig_size->getDataType().value());
      if (!orig_size->sameAs(new_size) ||
          size_map.find(orig_size) == size_map.end())
        size_map[orig_size] = new_size;
    }
  }

  // If we already lowered all inputs/outputs we can just return.
  if (size_map.size() == 0)
    return;

  // Set domains to be based on symbolic sizes (i.e. Ti.size[...])
  for (TensorView* tv : all_tvs) {
    std::vector<IterDomain*> new_domain_iters;
    const std::vector<IterDomain*>& root_td = tv->getRootDomain();

    for (decltype(root_td.size()) i{0}; i < root_td.size(); i++) {
      Val* new_size = root_td[i]->extent();
      if (size_map.find(new_size) != size_map.end())
        new_size = size_map[new_size];

      new_domain_iters.push_back(new IterDomain(
          root_td[i]->start(),
          new_size,
          root_td[i]->parallel_method(),
          root_td[i]->isReduction(),
          root_td[i]->isRFactorProduct(),
          root_td[i]->isBroadcast()));
    }

    TensorDomain* old_domain = tv->domain();
    TensorDomain* new_domain = new TensorDomain(new_domain_iters);

    // We should just be able to replace sizes in place, but mutator is setup to
    // do that as it set up to replace vals in Exprs, but
    // IterDomain/TensorDomain are vals.

    new_domain = TransformReplay::fullSelfReplay(new_domain, old_domain);

    TORCH_INTERNAL_ASSERT(
        old_domain->nDims() == new_domain->nDims(),
        "Tried to set symbolic sizes through the kernel, but hit a snag, Replayed domain should be the same size as the target domain, but got ",
        new_domain->nDims(),
        " and ",
        old_domain->nDims());
    // Parallelize all iter domains
    for (decltype(new_domain->nDims()) i{0}; i < new_domain->nDims(); i++)
      new_domain->axis(i)->parallelize(old_domain->axis(i)->parallel_method());

    tv->setDomain(new_domain);
  }

  // Adjust memory types to make sure they are valid
  for (TensorView* tv : all_tvs) {
    if (fusion->hasInput(tv) || fusion->hasOutput(tv)) {
      tv->setMemoryType(MemoryType::Global);
    } else {
      if (tv->getMemoryType() == MemoryType::Global)
        tv->setMemoryType(MemoryType::Local);
    }
  }
}

void PrepareForLowering(Fusion* fusion) {
  FusionGuard fg(fusion);

  IRFixComputeAt(fusion);
  IRValidate(fusion);
  IRReplaceSizes();
}

} // namespace fuser
} // namespace jit
} // namespace torch
