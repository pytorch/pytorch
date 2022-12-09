#include <torch/csrc/jit/codegen/cuda/ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>

#include <torch/csrc/jit/codegen/cuda/grouped_reduction.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

// Return if ref and other are transformed in the same way.
bool hasMatchingTransformations(TensorView* ref, TensorView* other) {
  std::unordered_map<IterDomain*, IterDomain*> ref_2_other;
  for (const auto i : c10::irange(ref->getRootDomain().size())) {
    ref_2_other.emplace(
        ref->getRootDomain().at(i), other->getRootDomain().at(i));
  }

  auto replay =
      BestEffortReplay(
          other->domain()->domain(), ref->domain()->domain(), ref_2_other)
          .getReplay();

  for (const auto i : c10::irange(ref->nDims())) {
    auto ref_id = ref->axis(i);
    auto other_id = other->axis(i);
    auto it = replay.find(ref_id);
    if (it == replay.end() || it->second != other_id) {
      return false;
    }
  }

  return true;
}

// Validate grouping of reductions and return a new max producer position
void validateReductionGrouping(
    const std::vector<Val*>& inputs,
    const std::vector<Val*>& outputs) {
  TORCH_INTERNAL_ASSERT(inputs.size() == outputs.size());
  TORCH_INTERNAL_ASSERT(!inputs.empty());

  auto fusion = dynamic_cast<Fusion*>(outputs[0]->container());
  TORCH_INTERNAL_ASSERT(
      fusion != nullptr, "Grouping of reductions must be done within a Fusion");

  ExactRootDomainMap exact_map(fusion);

  // Pick the first output TV as a reference and compare it with the
  // rest. Do not allow grouping if any mismatch is detected.
  auto ref_tv = outputs[0]->as<TensorView>();
  const auto ref_domain = ref_tv->getRootDomain();
  const auto num_root_dims = ref_domain.size();
  const auto num_dims = ref_tv->nDims();
  const auto ref_ca_pos = ref_tv->getComputeAtPosition();
  for (const auto i : c10::irange(inputs.size())) {
    auto output_tv = outputs.at(i)->as<TensorView>();
    const auto& output_domain = output_tv->getRootDomain();
    if (ref_tv == output_tv) {
      continue;
    }
    TORCH_INTERNAL_ASSERT(
        output_domain.size() == num_root_dims,
        "Invalid grouped reduction due to mismatched number of root dimensions. "
        "Expected: ",
        num_root_dims,
        ". Detected: ",
        output_domain.size(),
        ". Invalid output tensor: ",
        output_tv->toString());
    TORCH_INTERNAL_ASSERT(
        output_tv->nDims() == num_dims,
        "Invalid grouped reduction due to mismatched number of dimensions. "
        "Expected: ",
        num_dims,
        ". Detected: ",
        output_tv->nDims(),
        ". Invalid output tensor: ",
        output_tv->toString());
    for (const auto i : c10::irange(num_root_dims)) {
      auto ref_id = ref_domain.at(i);
      auto output_id = output_domain.at(i);
      // If an IterDomain is broadcast, require the other
      // corresponding IterDomains are also broadcast. This may not be
      // necessary but not completely certain.
      TORCH_INTERNAL_ASSERT(
          ref_id->isBroadcast() == output_id->isBroadcast(),
          "Invalid grouped reduction due to mismatched broadcast root domains. ",
          "Reference domain: ",
          ref_id->toString(),
          ". Mismatched domain: ",
          output_id->toString(),
          ". Invalid tensor: ",
          output_tv->toString());
      if (ref_id->isBroadcast()) {
        continue;
      }
      TORCH_INTERNAL_ASSERT(
          ref_id->isReduction() == output_id->isReduction(),
          "Invalid grouped reduction due to mismatched reduction root domains. ",
          "Reference domain: ",
          ref_id->toString(),
          ". Mismatched domain: ",
          output_id->toString(),
          ". Invalid tensor: ",
          output_tv->toString());
      TORCH_INTERNAL_ASSERT(
          exact_map.areMapped(ref_id, output_id) || ref_id->sameAs(output_id),
          "Invalid grouped reduction due to mismatched root domains. ",
          "Reference domain: ",
          ref_id->toString(),
          ". Mismatched domain: ",
          output_id->toString(),
          ". Invalid tensor: ",
          output_tv->toString());
    }

    TORCH_INTERNAL_ASSERT(
        hasMatchingTransformations(ref_tv, output_tv),
        "Invalid grouped reduction due to mismatched transformations. ",
        "Reference tensor: ",
        ref_tv->toString(),
        ". Mismatched tensor: ",
        output_tv->toString());

    // Must have the same computeAt position
    TORCH_INTERNAL_ASSERT(
        output_tv->getComputeAtPosition() == ref_ca_pos,
        "Invalid grouped reduction due to mismatched computeAt position. ",
        "Reference tensor: ",
        ref_tv->toString(),
        ". Mismatched tensor: ",
        output_tv->toString());
  }

  // Must not have any data dependency from outputs to inputs
  const auto all_dep_vals = DependencyCheck::getAllValsBetween(
      {outputs.begin(), outputs.end()}, inputs);
  if (!all_dep_vals.empty()) {
    std::stringstream ss;
    ss << "Invalid dependency:";
    for (auto val : all_dep_vals) {
      ss << " " << val->toString();
    }
    TORCH_INTERNAL_ASSERT(all_dep_vals.empty(), ss.str());
  }
}

} // namespace

void groupReductions(const std::vector<TensorView*>& reduction_outputs) {
  TORCH_CHECK(!reduction_outputs.empty(), "No tensor is given");

  auto container = reduction_outputs[0]->container();

  const auto num_reductions = reduction_outputs.size();

  std::vector<BinaryOpType> op_types(num_reductions);
  std::vector<Val*> init_vals(num_reductions);
  std::vector<Val*> outputs(num_reductions);
  std::vector<Val*> inputs(num_reductions);

  for (const auto i : c10::irange(num_reductions)) {
    auto reduction_out = reduction_outputs.at(i);
    TORCH_CHECK(
        reduction_out->definition() != nullptr,
        "Invalid tensor to group: ",
        reduction_out->toString(),
        ". Definition not found");
    auto rop = dynamic_cast<ReductionOp*>(reduction_out->definition());
    TORCH_CHECK(
        rop != nullptr,
        "Invalid tensor to group: ",
        reduction_out->toString(),
        ". Not an output of a ReductionOp: ",
        reduction_out->definition()->toString());
    // Fused reduction is only enabled during the lowering, so at this
    // point it should be false.
    TORCH_INTERNAL_ASSERT(
        !rop->isAllreduce(), "Invalid ReductionOp: ", rop->toString());
    op_types.at(i) = rop->getReductionOpType();
    init_vals.at(i) = rop->init();
    outputs.at(i) = rop->out();
    inputs.at(i) = rop->in();
  }

  validateReductionGrouping(inputs, outputs);

  IrBuilder::create<GroupedReductionOp>(
      container, op_types, init_vals, outputs, inputs);

  for (auto output : ir_utils::filterByType<TensorView>(outputs)) {
    output->updateMaxProducerPosition();
  }
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
