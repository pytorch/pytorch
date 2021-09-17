#include <torch/csrc/jit/codegen/cuda/evaluator_common.h>
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/kernel_expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

template <typename VALTYPE>
std::vector<VALTYPE*> getImmediateProducers(VALTYPE* val) {
  if (val->definition()) {
    auto expr = val->definition();
    return expr->inputs();
  } else {
    return {};
  }
}

//! IR-Generic utility, collects all the producers required for the
//!  given list of IR values and returns them along with the original
//!  list in topological order.
template <typename VALTYPE>
std::vector<VALTYPE*> makeSortedEvaluationList(std::vector<VALTYPE*> input) {
  // Deduplicate
  std::vector<VALTYPE*> to_sort;
  std::unordered_set<VALTYPE*> visited;
  for (auto val : input) {
    if (!visited.count(val)) {
      to_sort.push_back(val);
      visited.insert(val);
    }
  }

  std::vector<VALTYPE*> sorted;
  visited.clear();

  // Topological Sort
  //  Note: didn't explicitly exclude producers that are not in the original
  //   list. This should be acceptable for the intended use.
  while (!to_sort.empty()) {
    auto top_val = to_sort.back();
    if (visited.count(top_val)) {
      to_sort.pop_back();
    } else {
      bool ready_to_pop = true;
      for (auto producer : getImmediateProducers(top_val)) {
        if (!visited.count(producer)) {
          ready_to_pop = false;
          to_sort.push_back(producer);
        }
      }
      if (ready_to_pop) {
        visited.insert(top_val);
        sorted.push_back(top_val);
        to_sort.pop_back();
      }
    }
  }

  return sorted;
}

//! Kernel IR utility, collects all the symbolic integers
//!  used in allocation nodes.
void collectBufferSizes(
    std::vector<kir::Val*>& into,
    const std::vector<kir::Expr*>& exprs) {
  for (auto expr : exprs) {
    if (auto allocate = dynamic_cast<kir::Allocate*>(expr)) {
      into.push_back(allocate->size());
    } else if (auto for_loop = dynamic_cast<kir::ForLoop*>(expr)) {
      collectBufferSizes(into, for_loop->body().exprs());
    } else if (auto ite = dynamic_cast<kir::IfThenElse*>(expr)) {
      collectBufferSizes(into, ite->thenBody().exprs());
      collectBufferSizes(into, ite->elseBody().exprs());
    }
  }
}

//! Kernel IR utility, collects all the kir symbolic
//!  integers we will need at runtime, i.e. after the
//!  generated cuda kernel has already been compiled.
//!  The values are to be used for runtime logic, like
//!  `computeLaunchparams`.
std::vector<kir::Val*> collectRuntimeUsedIntegers(
    Fusion* fusion,
    GpuLower* lower) {
  std::vector<kir::Val*> ret;

  // Collect extent and integer inputs
  for (auto val : fusion->usedMathVals()) {
    auto kir_val = lower->lowerValue(val);
    if (auto kir_tv = dynamic_cast<kir::TensorView*>(kir_val)) {
      for (auto id : kir_tv->domain()->domain()) {
        ret.push_back(id->extent());
      }
    } else if (val->isFusionInput()) {
      if (kir_val->isA<kir::Int>()) {
        ret.push_back(kir_val);
      }
    }
  }

  // Collect allocation sizes:
  collectBufferSizes(ret, lower->kernel()->topLevelExprs());

  return makeSortedEvaluationList(ret);
}
//! Fusion IR utility, collects all the fusionIR symbolic
//!  integers we will need at runtime, i.e. after the
//!  generated cuda kernel has already been compiled.
//!  The values are to be used for runtime logic, like
//!  `canSchedule` in heuristic look up.
std::vector<Val*> collectRuntimeUsedIntegers(Fusion* fusion) {
  std::vector<Val*> ret;

  // Collect extent and integer inputs
  for (auto val : fusion->usedMathVals()) {
    if (auto tv = dynamic_cast<TensorView*>(val)) {
      for (auto id : tv->domain()->domain()) {
        ret.push_back(id->extent());
      }
    } else if (val->isFusionInput()) {
      if (val->isA<Int>()) {
        ret.push_back(val);
      }
    }
  }

  return makeSortedEvaluationList(ret);
}

} // namespace

template <typename IRContext>
void PrecomputedIntegersBase<IRContext>::initializeValueList(
    typename IRContext::EVALUATOR_TYPE& const_evaluator,
    const std::vector<IR_VAL*>& sorted_value_list) {
  // Initialize workspace
  num_of_values_ = sorted_value_list.size();
  defined_ = std::vector<bool>(num_of_values_, false);
  is_constant_ = std::vector<bool>(num_of_values_, false);
  values_ = std::vector<int64_t>(num_of_values_, -1);

  // Fill in constants and assign evaluator indices
  for (int i = 0; i < num_of_values_; i++) {
    // Use an expression evaluator to test if value is const
    auto const_val = const_evaluator.evaluate(sorted_value_list[i]);
    if (const_val.has_value()) {
      is_constant_[i] = true;
      values_[i] = const_val.value();
    }
    sorted_value_list[i]->setEvaluatorIndex(i);
  }
}

template <typename IRContext>
c10::optional<int64_t> PrecomputedIntegersBase<IRContext>::getMaybeValueFor(
    const IR_VAL* val) {
  auto index = val->evaluatorIndex();
  if (index < 0) {
    return c10::nullopt;
  }
  if (!defined_[index] && !is_constant_[index]) {
    return c10::nullopt;
  }
  return values_[index];
}

template <typename IRContext>
void PrecomputedIntegersBase<IRContext>::evaluate() {
  FUSER_PERF_SCOPE("PrecomputedIntegers::Evaluate");
  integer_machine_->run();
  validate();
}

template <typename IRContext>
void PrecomputedIntegersBase<IRContext>::invalidate() {
  // clear binding values
  binding_log_.clear();

  // invalidate value entries
  std::fill(defined_.begin(), defined_.end(), false);

  // invalidate flag
  has_valid_values_ = false;
}

template <typename IRContext>
void PrecomputedIntegersBase<IRContext>::validate() {
  FUSER_PERF_SCOPE("PrecomputedIntegers::Validate");
  for (auto it : binding_log_) {
    TORCH_INTERNAL_ASSERT(values_[it.first] == it.second);
  }
  has_valid_values_ = true;
}

template <typename IRContext>
NaiveIntegerMachine<IRContext>::NaiveIntegerMachine(
    PrecomputedIntegersBase<IRContext>& precomputed_integers)
    : precomputed_integers_(precomputed_integers) {
  num_of_instructions_ = 0;
  for (auto val : precomputed_integers_.symbols_) {
    auto def = val->definition();
    if (def) {
      if (auto uop = dynamic_cast<typename IRContext::UNARY_OP_TYPE*>(def)) {
        makeUnaryOp(uop);
      } else if (
          auto bop = dynamic_cast<typename IRContext::BINARY_OP_TYPE*>(def)) {
        makeBinaryOp(bop);
      } else {
        TORCH_INTERNAL_ASSERT(false, "Unsupported expr");
      }
    }
  }
}

template <typename IRContext>
void NaiveIntegerMachine<IRContext>::run() {
  for (int i = 0; i < num_of_instructions_; i++) {
    // Skip this instruction if the dest location
    //  has already been computed or is constant.
    if (precomputed_integers_.defined_[dest_[i]] ||
        precomputed_integers_.is_constant_[dest_[i]]) {
      continue;
    }
    runInstruction(i);
  }
}

template <typename IRContext>
void NaiveIntegerMachine<IRContext>::makeUnaryOp(
    typename IRContext::UNARY_OP_TYPE* uop) {
  int in = uop->inputs()[0]->evaluatorIndex();
  int out = uop->outputs()[0]->evaluatorIndex();
  TORCH_INTERNAL_ASSERT(in >= 0, "Integer Machine: unknown input: ", uop);
  TORCH_INTERNAL_ASSERT(out >= 0, "Integer Machine: unknown out: ", uop);

  int index = makeInstructionEntry();
  inst_type_[index] = InstructionType::UNARY_OP;
  uop_type_[index] = IRContext::getOpType(uop);
  src0_[index] = in;
  dest_[index] = out;
}

template <typename IRContext>
void NaiveIntegerMachine<IRContext>::makeBinaryOp(
    typename IRContext::BINARY_OP_TYPE* bop) {
  int in0 = bop->inputs()[0]->evaluatorIndex();
  int in1 = bop->inputs()[1]->evaluatorIndex();
  int out = bop->outputs()[0]->evaluatorIndex();

  TORCH_INTERNAL_ASSERT(in0 >= 0, "Integer Machine: unknown lhs: ", bop);
  TORCH_INTERNAL_ASSERT(in1 >= 0, "Integer Machine: unknown rhs: ", bop);
  TORCH_INTERNAL_ASSERT(out >= 0, "Integer Machine: unknown out: ", bop);

  int index = makeInstructionEntry();
  inst_type_[index] = InstructionType::BINARY_OP;
  bop_type_[index] = IRContext::getOpType(bop);
  src0_[index] = in0;
  src1_[index] = in1;
  dest_[index] = out;
}

template <typename IRContext>
int NaiveIntegerMachine<IRContext>::makeInstructionEntry() {
  int index = num_of_instructions_++;
  inst_type_.push_back(InstructionType::UNARY_OP);
  uop_type_.push_back(UnaryOpType::Abs);
  bop_type_.push_back(BinaryOpType::Add);
  src0_.push_back(-1);
  src1_.push_back(-1);
  dest_.push_back(-1);
  return index;
}

template <typename IRContext>
void NaiveIntegerMachine<IRContext>::runInstruction(int index) {
  switch (inst_type_[index]) {
    case InstructionType::UNARY_OP:
      runUnaryOp(index);
      break;
    case InstructionType::BINARY_OP:
      runBinaryOp(index);
      break;
  }
}

template <typename IRContext>
void NaiveIntegerMachine<IRContext>::runUnaryOp(int index) {
  int src_index = src0_[index];
  bool src_defined = precomputed_integers_.defined_[src_index];
  bool src_is_const = precomputed_integers_.is_constant_[src_index];
  if (!src_defined && !src_is_const) {
    return;
  }

  int dest_index = dest_[index];

  auto& src = precomputed_integers_.values_[src_index];
  auto& dest = precomputed_integers_.values_[dest_index];

  switch (uop_type_[index]) {
    case UnaryOpType::Neg:
      dest = -src;
      break;
    case UnaryOpType::Cast:
      dest = src;
      break;
    default:
      TORCH_CHECK(!"Unexpected operator type");
  }

  precomputed_integers_.defined_[dest_index] = true;
}

template <typename IRContext>
void NaiveIntegerMachine<IRContext>::runBinaryOp(int index) {
  int src0_index = src0_[index];
  int src1_index = src1_[index];
  bool src0_is_const = precomputed_integers_.is_constant_[src0_index];
  bool src1_is_const = precomputed_integers_.is_constant_[src1_index];

  bool src_defined =
      (precomputed_integers_.defined_[src0_index] || src0_is_const) &&
      (precomputed_integers_.defined_[src1_index] || src1_is_const);

  if (!src_defined) {
    return;
  }
  int dest_index = dest_[index];

  auto& lhs = precomputed_integers_.values_[src0_index];
  auto& rhs = precomputed_integers_.values_[src1_index];
  auto& dest = precomputed_integers_.values_[dest_index];

  switch (bop_type_[index]) {
    case BinaryOpType::Add:
      dest = lhs + rhs;
      break;
    case BinaryOpType::Sub:
      dest = lhs - rhs;
      break;
    case BinaryOpType::Mul:
      dest = lhs * rhs;
      break;
    case BinaryOpType::Div:
      TORCH_CHECK(rhs != 0);
      dest = lhs / rhs;
      break;
    case BinaryOpType::Mod:
      TORCH_CHECK(rhs != 0);
      dest = lhs % rhs;
      break;
    case BinaryOpType::CeilDiv:
      TORCH_CHECK(rhs != 0);
      dest = (lhs + rhs - 1) / rhs;
      break;
    case BinaryOpType::And:
      dest = Int::ScalarType(lhs && rhs);
      break;
    case BinaryOpType::Max:
      dest = lhs > rhs ? lhs : rhs;
      break;
    case BinaryOpType::Min:
      dest = lhs < rhs ? lhs : rhs;
      break;
    default:
      TORCH_CHECK(!"Unexpected operator type");
  }

  precomputed_integers_.defined_[dest_index] = true;
}

KernelPrecomputedIntegers::KernelPrecomputedIntegers(
    Fusion* fusion,
    GpuLower& lower)
    : lower_(&lower) {
  loadSymbols(collectRuntimeUsedIntegers(fusion, lower_));
  kir::ExpressionEvaluator evaluator;
  initializeValueList(evaluator, symbols());
  initializeNamedScalars();
  initializeIntegerMachine();
}

void KernelPrecomputedIntegers::bindTensorMetaData(
    kir::TensorView* tv,
    const at::Tensor& at_tensor) {
  std::vector<std::pair<kir::Val*, int64_t>> ret;
  const auto root_domain =
      kir::TensorDomain::noReductions(tv->domain()->rootDomain());
  TORCH_INTERNAL_ASSERT(
      at_tensor.ndimension() == static_cast<int>(root_domain.size()),
      "Something went wrong configuring launch. Inputs do not match.");

  for (size_t dim = 0; dim < root_domain.size(); dim++) {
    auto extent = root_domain[dim]->extent();
    auto value = at_tensor.sizes()[dim];
    bindValue(extent->evaluatorIndex(), value);
  }
}

namespace {

//! Compares the name of given scalar with thread size strings
//!  and returns the corresponding parallel type if a match
//!  is found.
c10::optional<ParallelType> getMaybeThreadSizeParallelType(
    kir::NamedScalar* named_scalar) {
  auto& var_name = named_scalar->name();
  for (auto ptype : kParallelTypeThreads) {
    if (var_name == stringifyThreadSize(ptype)) {
      return ptype;
    }
  }
  return c10::nullopt;
}

} // namespace

void KernelPrecomputedIntegers::initializeNamedScalars() {
  for (auto val : symbols()) {
    if (auto named_scalar = dynamic_cast<kir::NamedScalar*>(val)) {
      auto maybe_parallel_type = getMaybeThreadSizeParallelType(named_scalar);
      if (maybe_parallel_type.has_value()) {
        auto& index_list =
            thread_dim_value_indices_[maybe_parallel_type.value()];
        if (!index_list) {
          index_list = std::make_unique<std::vector<int>>();
        }
        index_list->push_back(val->evaluatorIndex());
      }
    }
  }
}

void KernelPrecomputedIntegers::bindKernelInputs(
    const at::ArrayRef<IValue>& aten_inputs) {
  if (hasValidValues()) {
    invalidate();
  }

  auto kernel = lower_->kernel();
  const auto& inputs = kernel->inputs();

  for (size_t i = 0; i < inputs.size(); i++) {
    const auto input = inputs[i];
    if (auto tensor_input = dynamic_cast<kir::TensorView*>(input)) {
      const auto aten_tensor = aten_inputs[i].toTensor();
      bindTensorMetaData(tensor_input, aten_tensor);
    } else if (input->isScalar() && input->dtype() == DataType::Int) {
      bindValue(input->evaluatorIndex(), aten_inputs[i].toInt());
    }
  }
}

void KernelPrecomputedIntegers::bindParallelExtents(
    const ParallelExtentMap& parallel_extents,
    const LaunchParams& launch_constraint) {
  // Bind integer values of extents of parallelized
  //  iterdomains from launch_constraint when applicable.
  // Consistency will be checked at validate().
  for (const auto& it : parallel_extents) {
    auto raw_val = launch_constraint.getRawVal(it.first);
    if (raw_val > 0) {
      for (auto extent : it.second) {
        bindValue(extent->evaluatorIndex(), raw_val);
      }
    }
  }
}

void KernelPrecomputedIntegers::bindConcreteParallelTypeValue(
    ParallelType pt,
    int64_t value) {
  auto index_list_it = thread_dim_value_indices_.find(pt);
  if (index_list_it != thread_dim_value_indices_.end()) {
    for (auto index : *(index_list_it->second)) {
      bindValue(index, value);
    }
  }
}

FusionPrecomputedIntegers::FusionPrecomputedIntegers(Fusion* fusion)
    : fusion_(fusion) {
  loadSymbols(collectRuntimeUsedIntegers(fusion));
  ExpressionEvaluator evaluator(fusion);
  initializeValueList(evaluator, symbols());
  initializeIntegerMachine();
}

void FusionPrecomputedIntegers::bindTensorMetaData(
    TensorView* tv,
    const at::Tensor& at_tensor) {
  const auto root_domain =
      TensorDomain::noReductions(tv->getMaybeRFactorDomain());
  TORCH_INTERNAL_ASSERT(
      at_tensor.ndimension() == static_cast<int>(root_domain.size()),
      "Something went wrong configuring launch. Inputs do not match.");

  for (size_t dim = 0; dim < root_domain.size(); dim++) {
    auto extent = root_domain[dim]->extent();
    auto value = at_tensor.sizes()[dim];
    precomputedIntegersBaseType::bindValue(extent->evaluatorIndex(), value);
  }
}

void FusionPrecomputedIntegers::bindFusionInputs(
    const at::ArrayRef<IValue>& aten_inputs) {
  if (hasValidValues()) {
    precomputedIntegersBaseType::invalidate();
  }

  const auto& inputs = fusion_->inputs();

  for (size_t i = 0; i < inputs.size(); i++) {
    const auto input = inputs[i];
    if (auto tensor_input = dynamic_cast<TensorView*>(input)) {
      const auto aten_tensor = aten_inputs[i].toTensor();
      bindTensorMetaData(tensor_input, aten_tensor);
    } else if (input->isScalar() && input->getDataType() == DataType::Int) {
      precomputedIntegersBaseType::bindValue(
          input->evaluatorIndex(), aten_inputs[i].toInt());
    }
  }
}

template class PrecomputedIntegersBase<FusionIRContext>;
template class PrecomputedIntegersBase<KernelIRContext>;

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
