#include <expr_evaluator.h>
#include <instrumentation.h>
#include <ir_utils.h>
#include <kernel_expr_evaluator.h>
#include <lower2device.h>

#include <evaluator_common.h>

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

//! Kernel IR utility, collects all the symbolic values
//!  used in allocation nodes.
void collectBufferSizes(
    std::vector<Val*>& into,
    const std::vector<Expr*>& exprs) {
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

//! Kernel IR utility, collects all the kernel symbolic
//!  values we will need at runtime, i.e. after the
//!  generated cuda kernel has already been compiled.
//!  The values are to be used for runtime logic, like
//!  `computeLaunchparams`.
std::vector<Val*> collectRuntimeUsedValues(kir::Kernel* kernel) {
  std::vector<Val*> ret;
  auto all_tvs = ir_utils::allTvs(kernel);
  // Collect extent and inputs
  for (auto tv : all_tvs) {
    for (auto id : tv->domain()->domain()) {
      ret.push_back(id->extent());
    }
  }
  for (auto inp : kernel->inputs()) {
    if (inp->isA<Int>() || inp->isA<Double>()) {
      ret.push_back(inp);
    }
  }
  // Collect allocation sizes:
  collectBufferSizes(ret, kernel->topLevelExprs());
  return makeSortedEvaluationList(ret);
}

std::vector<Val*> collectRuntimeUsedValues(Fusion* fusion) {
  std::vector<Val*> ret;
  auto all_tvs = ir_utils::allTvs(fusion);
  // Collect extent and inputs
  for (auto tv : all_tvs) {
    for (auto id : tv->domain()->domain()) {
      ret.push_back(id->extent());
    }
  }
  for (auto inp : fusion->inputs()) {
    if (inp->isA<Int>() || inp->isA<Double>()) {
      ret.push_back(inp);
    }
  }
  return makeSortedEvaluationList(ret);
}

} // namespace

template <typename IRContext>
void PrecomputedValuesBase<IRContext>::initializeValueList(
    typename IRContext::EVALUATOR_TYPE& const_evaluator,
    const std::vector<Val*>& sorted_value_list) {
  // Initialize workspace
  num_of_values_ = sorted_value_list.size();
  defined_ = std::vector<bool>(num_of_values_, false);
  is_constant_ = std::vector<bool>(num_of_values_, false);
  values_ = std::vector<IntOrDouble>(num_of_values_, -1);

  // Fill in constants and assign evaluator indices
  for (const auto i : c10::irange(num_of_values_)) {
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
c10::optional<IntOrDouble> PrecomputedValuesBase<IRContext>::getMaybeValueFor(
    const Val* val) {
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
void PrecomputedValuesBase<IRContext>::print() const {
  std::cout << "Precomputed Values:\n";
  for (auto i : c10::irange(symbols_.size())) {
    if (defined_[i]) {
      std::cout << symbols_[i]->toInlineString() << " = " << values_[i]
                << std::endl;
    }
  }
}

template <typename IRContext>
void PrecomputedValuesBase<IRContext>::evaluate() {
  FUSER_PERF_SCOPE("PrecomputedValues::Evaluate");
  value_machine_->run();
  validate();
}

template <typename IRContext>
void PrecomputedValuesBase<IRContext>::invalidate() {
  // clear binding values
  binding_log_.clear();

  // invalidate value entries
  std::fill(defined_.begin(), defined_.end(), false);

  // invalidate flag
  has_valid_values_ = false;
}

template <typename IRContext>
void PrecomputedValuesBase<IRContext>::validate() {
  FUSER_PERF_SCOPE("PrecomputedValuess::Validate");
  for (auto it : binding_log_) {
    TORCH_INTERNAL_ASSERT(
        values_[it.first] == it.second,
        "Precomputed values failed to validate.",
        "\nSomething unexpected changed between the compilation and execution.\n",
        values_[it.first],
        " != ",
        it.second);
  }
  has_valid_values_ = true;
}

template <typename IRContext>
NaiveValueMachine<IRContext>::NaiveValueMachine(
    PrecomputedValuesBase<IRContext>& precomputed_values)
    : precomputed_values_(precomputed_values) {
  num_of_instructions_ = 0;
  for (auto val : precomputed_values_.symbols_) {
    auto def = val->definition();
    if (def) {
      if (auto uop = dynamic_cast<UnaryOp*>(def)) {
        makeUnaryOp(uop);
      } else if (auto bop = dynamic_cast<BinaryOp*>(def)) {
        makeBinaryOp(bop);
      } else {
        TORCH_INTERNAL_ASSERT(false, "Unsupported expr");
      }
    }
  }
}

template <typename IRContext>
void NaiveValueMachine<IRContext>::run() {
  for (const auto i : c10::irange(num_of_instructions_)) {
    // Skip this instruction if the dest location
    //  has already been computed or is constant.
    if (precomputed_values_.defined_[dest_[i]] ||
        precomputed_values_.is_constant_[dest_[i]]) {
      continue;
    }
    runInstruction(i);
  }
}

template <typename IRContext>
void NaiveValueMachine<IRContext>::makeUnaryOp(UnaryOp* uop) {
  int in = uop->inputs()[0]->evaluatorIndex();
  int out = uop->outputs()[0]->evaluatorIndex();
  TORCH_INTERNAL_ASSERT(in >= 0, "Integer Machine: unknown input: ", uop);
  TORCH_INTERNAL_ASSERT(out >= 0, "Integer Machine: unknown out: ", uop);

  int index = makeInstructionEntry();
  inst_type_[index] = InstructionType::UNARY_OP;
  uop_type_[index] = IRContext::getOpType(uop);
  if (uop_type_[index] == UnaryOpType::Cast) {
    data_type_[index] = uop->out()->getDataType().value();
  }
  src0_[index] = in;
  dest_[index] = out;
}

template <typename IRContext>
void NaiveValueMachine<IRContext>::makeBinaryOp(BinaryOp* bop) {
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
int NaiveValueMachine<IRContext>::makeInstructionEntry() {
  int index = num_of_instructions_++;
  inst_type_.push_back(InstructionType::UNARY_OP);
  uop_type_.push_back(UnaryOpType::Abs);
  bop_type_.push_back(BinaryOpType::Add);
  data_type_.push_back(DataType::Null);
  src0_.push_back(-1);
  src1_.push_back(-1);
  dest_.push_back(-1);
  return index;
}

template <typename IRContext>
void NaiveValueMachine<IRContext>::runInstruction(int index) {
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
void NaiveValueMachine<IRContext>::runUnaryOp(int index) {
  using namespace IntOrDouble_functions;
  int src_index = src0_[index];
  bool src_defined = precomputed_values_.defined_[src_index];
  bool src_is_const = precomputed_values_.is_constant_[src_index];
  if (!src_defined && !src_is_const) {
    return;
  }

  int dest_index = dest_[index];

  auto& src = precomputed_values_.values_[src_index];
  auto& dest = precomputed_values_.values_[dest_index];

  switch (uop_type_[index]) {
    case UnaryOpType::Neg:
      dest = -src;
      break;
    case UnaryOpType::Set:
      dest = src;
      break;
    case UnaryOpType::Cast:
      if (data_type_[index] == DataType::Double) {
        dest = src.template cast<double>();
      } else if (data_type_[index] == DataType::Int) {
        dest = src.template cast<int64_t>();
      } else {
        TORCH_INTERNAL_ASSERT(false, "dtype not supported in evaluator");
      }
      break;
    case UnaryOpType::Abs:
      dest = abs(src);
      break;
    default:
      TORCH_CHECK(!"Unexpected operator type ", uop_type_[index]);
  }

  precomputed_values_.defined_[dest_index] = true;
}

template <typename IRContext>
void NaiveValueMachine<IRContext>::runBinaryOp(int index) {
  using namespace IntOrDouble_functions;
  int src0_index = src0_[index];
  int src1_index = src1_[index];
  bool src0_is_const = precomputed_values_.is_constant_[src0_index];
  bool src1_is_const = precomputed_values_.is_constant_[src1_index];

  bool src_defined =
      (precomputed_values_.defined_[src0_index] || src0_is_const) &&
      (precomputed_values_.defined_[src1_index] || src1_is_const);

  if (!src_defined) {
    return;
  }
  int dest_index = dest_[index];

  auto& lhs = precomputed_values_.values_[src0_index];
  auto& rhs = precomputed_values_.values_[src1_index];
  auto& dest = precomputed_values_.values_[dest_index];

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
      dest = ceildiv(lhs, rhs);
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

  precomputed_values_.defined_[dest_index] = true;
}

KernelPrecomputedValues::KernelPrecomputedValues(kir::Kernel* kernel) {
  loadSymbols(collectRuntimeUsedValues(kernel));
  kir::ExpressionEvaluator evaluator;
  initializeValueList(evaluator, symbols());
  initializeNamedScalars();
  initializeIntegerMachine();
}

// TODO: put this to base class
void KernelPrecomputedValues::bindTensorMetaData(
    TensorView* tv,
    const TensorArgAbstract* tensor_arg_abstract) {
  const auto root_domain =
      TensorDomain::noReductions(tv->domain()->getMaybeRFactorDomain());
  TORCH_INTERNAL_ASSERT(
      tensor_arg_abstract->getRank() == static_cast<int>(root_domain.size()),
      "Something went wrong configuring launch. Inputs do not match.");

  for (const auto dim : c10::irange(root_domain.size())) {
    auto extent = root_domain[dim]->extent();
    auto value = tensor_arg_abstract->getSize(dim);
    bindValue(extent->evaluatorIndex(), value);
  }
}

namespace {

//! Compares the name of given scalar with thread size strings
//!  and returns the corresponding parallel type if a match
//!  is found.
c10::optional<ParallelType> getMaybeThreadSizeParallelType(
    NamedScalar* named_scalar) {
  auto& var_name = named_scalar->name();
  for (auto ptype : kParallelTypeThreads) {
    if (var_name == stringifyThreadSize(ptype)) {
      return ptype;
    }
  }
  return c10::nullopt;
}

} // namespace

void KernelPrecomputedValues::initializeNamedScalars() {
  for (auto val : symbols()) {
    if (auto named_scalar = dynamic_cast<NamedScalar*>(val)) {
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

// TODO: merge this one with above.
void KernelPrecomputedValues::bindKernelInputs(
    kir::Kernel* kernel,
    const KernelArgumentHolder& args) {
  if (hasValidValues()) {
    invalidate();
  }

  const auto& inputs = kernel->inputs();
  TORCH_INTERNAL_ASSERT(
      args.size() == inputs.size(), "kernel inputs size does not match args");

  for (const auto i : c10::irange(inputs.size())) {
    auto arg = args[i];
    const auto input = inputs[i];
    if (auto tensor_input = dynamic_cast<TensorView*>(input)) {
      if (const auto& tensor_arg_abstract =
              dynamic_cast<const TensorArgAbstract*>(arg)) {
        bindTensorMetaData(tensor_input, tensor_arg_abstract);
      } else {
        // TODO: cpu scalar of int type should be bound as scalar int as well
        TORCH_CHECK(
            arg->isType(ArgType::CpuScalarTensor),
            "binding input to TensorView expects input arg to be of tensor type");
      }
    } else if (input->isScalar()) {
      if (input->dtype() == DataType::Int) {
        TORCH_CHECK(
            arg->isType(ArgType::Long),
            "binding input to integer type expects input arg to be a scalar of Long type");
        precomputedValuesBaseType::bindValue(
            input->evaluatorIndex(), *static_cast<const int64_t*>(arg->arg()));
      } else if (input->dtype() == DataType::Double) {
        TORCH_CHECK(
            arg->isType(ArgType::Double),
            "binding input to double type expects input arg to be a scalar of Double type");
        precomputedValuesBaseType::bindValue(
            input->evaluatorIndex(), *static_cast<const double*>(arg->arg()));
      }
    }
  }
}

void KernelPrecomputedValues::bindParallelExtents(
    const ParallelExtentMap& parallel_extents,
    const LaunchParams& launch_constraint) {
  // Bind values of extents of parallelized
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

void KernelPrecomputedValues::bindConcreteParallelTypeValue(
    ParallelType pt,
    int64_t value) {
  auto index_list_it = thread_dim_value_indices_.find(pt);
  if (index_list_it != thread_dim_value_indices_.end()) {
    for (auto index : *(index_list_it->second)) {
      bindValue(index, value);
    }
  }
}

FusionPrecomputedValues::FusionPrecomputedValues(Fusion* fusion)
    : fusion_(fusion) {
  loadSymbols(collectRuntimeUsedValues(fusion));
  ExpressionEvaluator evaluator(fusion);
  initializeValueList(evaluator, symbols());
  initializeIntegerMachine();
}

// TODO: put this to base class
void FusionPrecomputedValues::bindTensorMetaData(
    TensorView* tv,
    const TensorArgAbstract* tensor_arg_abstract) {
  const auto root_domain =
      TensorDomain::noReductions(tv->getMaybeRFactorDomain());
  TORCH_INTERNAL_ASSERT(
      tensor_arg_abstract->getRank() == static_cast<int>(root_domain.size()),
      "Something went wrong configuring launch. Inputs do not match.");

  for (const auto dim : c10::irange(root_domain.size())) {
    auto extent = root_domain[dim]->extent();
    auto value = tensor_arg_abstract->getSize(dim);
    precomputedValuesBaseType::bindValue(extent->evaluatorIndex(), value);
  }
}

void FusionPrecomputedValues::bindFusionInputs(
    const KernelArgumentHolder& args) {
  if (hasValidValues()) {
    precomputedValuesBaseType::invalidate();
  }

  const auto& inputs = fusion_->inputs();
  TORCH_INTERNAL_ASSERT(
      args.size() == inputs.size(), "kernel inputs size does not match args");

  for (const auto i : c10::irange(inputs.size())) {
    const auto input = inputs[i];
    const ArgAbstract* arg = args[i];
    if (auto tensor_input = dynamic_cast<TensorView*>(input)) {
      if (const auto& tensor_arg_abstract =
              dynamic_cast<const TensorArgAbstract*>(arg)) {
        bindTensorMetaData(tensor_input, tensor_arg_abstract);
      } else {
        TORCH_CHECK(
            arg->isType(ArgType::CpuScalarTensor),
            "binding input to TensorView expects input arg to be of tensor type");
      }
    } else if (input->isScalar()) {
      if (input->getDataType() == DataType::Int) {
        TORCH_CHECK(
            arg->isType(ArgType::Long),
            "binding input to integer type expects input arg to be a scalar of Long type");
        precomputedValuesBaseType::bindValue(
            input->evaluatorIndex(), *static_cast<const int64_t*>(arg->arg()));
      } else if (input->getDataType() == DataType::Double) {
        TORCH_CHECK(
            arg->isType(ArgType::Double),
            "binding input to double type expects input arg to be a scalar of Double type");
        precomputedValuesBaseType::bindValue(
            input->evaluatorIndex(), *static_cast<const double*>(arg->arg()));
      }
    }
  }
}

template class PrecomputedValuesBase<FusionIRContext>;
template class PrecomputedValuesBase<KernelIRContext>;

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
