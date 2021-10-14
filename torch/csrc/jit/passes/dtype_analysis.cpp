#include <ATen/core/function_schema.h>
#include <ATen/core/interned_strings.h>
#include <ATen/core/jit_type.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/dtype_analysis.h>
#include <torch/library.h>
#include <torch/types.h>
#include <memory>
#include <stdexcept>
#include "c10/core/Scalar.h"
#include <c10/util/ArrayRef.h>
#include <jit/runtime/operator.h>

namespace torch {
namespace jit {

namespace {

using ArgumentCreator = std::function<c10::optional<Stack>(Node*)>;

std::unique_ptr<Stack> MTensorArgumentCreator(Node* n)  {
  auto stack = std::make_unique<std::vector<IValue>>();
  for (Value* inp : n->inputs()) {
    if (auto tp = inp->type()->cast<TensorType>()) {
      // TODO: Need to implement Rank based Tensor argument creation
      at::IntArrayRef tensor_size = {1, 1};
      stack->emplace_back(at::empty(
          tensor_size, at::TensorOptions(at::kMeta).dtype(*tp->scalarType())));
      continue;
    }
    // Someday Todo: Fill in concrete values that we know.
    if (inp->type() == FloatType::get()) {
      stack->emplace_back(0.);
    } else if (inp->type() == IntType::get()) {
      stack->emplace_back(0);
    } else if (inp->type() == BoolType::get()) {
      // For now just not support bool until we know it is safe.
      throw std::runtime_error("Unsupported input type for Tensor argument");
      stack->emplace_back(false);
    }else {
      // maybe do: Actually return nullptr and escape out

      // Arrays of values are specifically not handled due
      // to the fact that naive default vaules would likely be
      // incorrect anyways.
      throw std::runtime_error("Unsupported input type for Tensor argument");
    }
  }
  return stack;
};

/*
bool TryRegisterArgumentCreatorFor(
    const char* schema,
    ArgumentCreator creator) {
  static std::mutex mut;
  std::lock_guard<std::mutex> lock(mut);
  auto op = getOperatorForLiteral(schema);
  if (!op) {
    return false;
  }

  auto& map = getArgumentCreatorMap();
  if (map.contains(*op)) {
    return false;
  }
  getArgumentCreatorMap().insert(op, creator);
  return true;
}

void RegisterArgumentCreatorFor(const char* schema, ArgumentCreator creator) {
  if (!TryRegisterArgumentCreatorFor(schema, creator)) {
    throw std::runtime_error(
        std::string("Couldn't register ArgumentCreator for ") + schema);
  }
}
*/

bool MTensorValidNodeArg(Value* value){
  if (!value->type()->cast<TensorType>()){
    return true;
  }
  if (!value->type()->expect<TensorType>()->scalarType().has_value()){
    GRAPH_DEBUG("Argument missing Dtype");
    return false;
  }
  // TODO: Check that Tensor args have dimmensions specified.
  return true;
}

static bool canBeInferredWithMetaTensor(Node* n) {
  // Not a guarantee that the metatensor will not error out
  // Do not have a allowlist for now and let things error out in execution.
  // Has Tensor output is checked in another place
  bool args_have_dtypes =
      std::all_of(n->inputs().begin(), n->inputs().end(), MTensorValidNodeArg);

  if (!args_have_dtypes) {
    return false;
  }
  if (n->outputs().size() != 1) {
    // Currently not supporting multiple outputs
    return false;
  }
  auto opt_op = n->maybeOperator();
  if (!opt_op) {
    GRAPH_DEBUG("not registered with Meta");
    return false;
  }
  return true;
}

c10::optional<Tensor> inferWithMetaTensor(Node* n) {
  GRAPH_DEBUG("inferWithMetaTensor", getHeader(n));
  if (!canBeInferredWithMetaTensor(n)) {
    return c10::nullopt;
  }
  auto stack = MTensorArgumentCreator(n);
  Operation op = n->getOperation();
  try {
    GRAPH_DEBUG("Running op for ", getHeader(n));
    op(*stack);
    GRAPH_DEBUG("op run successfully", getHeader(n));
    GRAPH_DEBUG("After receive!");
    return stack->back().toTensor();

  } catch (...) {
    GRAPH_DEBUG("caught exception with Metatensor run!");
  };
  return c10::nullopt;
}

/*
  void setDtype(Value* value, const IValue& ival) override {
    auto scalarType = ival.toTensor().scalar_type();
    _setDtype(value, scalarType);
  }

  void setDtype(Value* dst, const Value* src) override {
    auto scalarType = *src->type()->expect<TensorType>()->scalarType();
    _setDtype(dst, scalarType);
  }
  */


bool setDtype(Value* value, ScalarType scalarType) {
  // returns if anything was changed
  auto tensor_type = value->type()->cast<TensorType>();
  TORCH_INTERNAL_ASSERT(tensor_type, "Expecting a tensor type");
  if (!tensor_type->scalarType().has_value()) {
    value->setType(tensor_type->withScalarType(scalarType));
    return true;
  }
  if (tensor_type->scalarType().value() != scalarType) {
    value->setType(tensor_type->withScalarType(scalarType));
    return true;
  }
  return false;
}

bool tryApplyDtypeMetaTensor(Node* n)
{
  // returns if anything was changed
  auto return_tensor = inferWithMetaTensor(n);
  if (!return_tensor) {
    return false;
  }
  GRAPH_DEBUG("Received ", toString(return_tensor->scalar_type()));
  return setDtype(n->output(), return_tensor->scalar_type());
}

 class TensorPropertyInferrer {
 public:
  TensorPropertyInferrer();
  bool virtual hasProperty(Value* v) = 0;
  void virtual setDtype(Value* v, const IValue& ival) = 0;
  void virtual setDtype(Value* dst, const Value* src) = 0;

/*
  bool supportsNodeWithMeta(Node* n) {
    const Operator* opt_op = n->maybeOperator();
    GRAPH_DEBUG("Checking ", getHeader(n));
    if (!opt_op) {
      GRAPH_DEBUG("Not a valid operator");
      return false;
    }
    return true;
  }
  */
/*
  bool virtual processValueWithCopy(Value* src, Value* dst) {
    if (!hasProperty(src)) {
      return false;
    }

    setDtype(dst, src);
    return true;
  }
*/
/*
  bool virtual processNodeWithMeta(Node* n) {
    bool args_have_dtypes =
        std::all_of(n->inputs().begin(), n->inputs().end(), [this](Value* v) {
          return this->hasProperty(v);
        });
    if (!args_have_dtypes) {
      return false;
    }

    try {
      Stack stack = mTensorArgumentCreator(n);
      auto op = n->getOperation();
      op(&stack);
      const auto num_inputs = n->outputs().size();
      for (auto i : c10::irange(num_inputs)) {
        setDtype(n->output(i), peek(stack, i, num_inputs));
      }
      return true;
    } catch (...) {
      GRAPH_DEBUG("caught exception!");
      return false;
    };
  }
  */
};


/*
using dtype_func_t = std::function<c10::optional<ScalarType>(Node*)>;
static std::vector<std::pair<OperatorSet, dtype_func_t>>
    dtype_transfer_functions;
    */

// TensorPropertyPropagationPass is an analysis pass that walks through a graph
// in topological order and forward propagate TensorProperties from graph inputs
// (expressed in input_descriptors) to all output tensor nodes in the graph.
// The inferred TensorProperties of an output tensor will be checked against
// the original TensorProperties of the tensor node:
//  - if inferred property is incongruent with the original property, an error
//  is issued
//  - otherwise if the inferred property is more precise, original property of
//  the output tensor will be updated
struct DTypePropagationPass {
  DTypePropagationPass(
      std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)) {}

  // returns true if at least one node has its scalar type set on a tensor node
  bool run() {
    return processBlocks(graph_->block());
  }

 private:
 /*
  static c10::optional<at::ScalarType> getScalarType(const Value* value) {
    c10::optional<at::ScalarType> stype;
    const auto& type = value->type();
    if (type->kind() == TypeKind::TensorType) {
      stype = type->castRaw<TensorType>()->scalarType();
    } else {
      stype = tryScalarTypeFromJitType(type);
    }
    if (stype.has_value()) {
      GRAPH_DEBUG("getScalarType = ", stype.value());
    }
    return stype;
  }
  */

  bool processBlocks(at::ArrayRef<Block*> blocks) {
    bool changed = false;
    for (auto block : blocks) {
      changed |= processBlock(block);
    }
    return changed;
  }

  bool processBlock(Block* block) {
    GRAPH_DEBUG("processBlock");
    bool changed = false;
    for (auto it = block->nodes().begin(); it != block->nodes().end(); it++) {
      changed |= processNode(*it);
    }
    return changed;
  }

  bool processNode(Node* n) {
    GRAPH_DEBUG("processNode");
    switch (n->kind()) {
      case prim::If:
        return processIf(n);
      case prim::Loop:
      case prim::CallMethod:
      case prim::CallFunction:
        TORCH_INTERNAL_ASSERT(false, "Loop/Call not handled now");
      default:
        break;
    }

    TORCH_INTERNAL_ASSERT(
        n->blocks().empty(), "Do not handle this block structure");

    // process non-block nodes
    bool has_tensor_output = false;
    for (size_t i = 0; i < n->outputs().size(); i++) {
      auto type = n->output(i)->type();
      if (auto tt = type->castRaw<TensorType>()) {
        // TODO: Check if the castRaw is actually what we want
        // EG Bool might be castable but is not needed to be propagated.
        has_tensor_output = true;
        break;
      }
    }
    // if output contains no tensor, nothing to propagate
    if (!has_tensor_output) {
      return false;
    }

    bool changed = false;
    // Main dispatch loop for the abstract interpreter
    switch (n->kind()) {
      case prim::Constant:
        // This seems to already been propagated by something else in freezing
        return false;
      case prim::ListConstruct:
      case prim::ListUnpack:
        TORCH_INTERNAL_ASSERT(false, "not supported IR");
        break;
      default:
        if (n->kind().is_aten()) {
          changed = processAtenOps(n);
        } else {
          TORCH_INTERNAL_ASSERT(false, "not supported IR");
        }
    }
    return changed;
  }

  bool mergeTensorProperties(
      const at::ArrayRef<Value*>& oneList,
      const at::ArrayRef<Value*>& anotherList) {
    // This is currently a placeholder for MobileNet
    // After Month1: implement the merge function
    TORCH_INTERNAL_ASSERT(oneList.size() == 0, "Not implemented yet");
    return false;
    /*
    TORCH_INTERNAL_ASSERT(oneList.size() == anotherList.size());

    for (auto i : c10::irange(oneList.size())) {
    }
    return false;
    */
  }

  bool processIf(Node* node) {
    GRAPH_DEBUG("processIf");
    bool changed = false;
    auto blocks = node->blocks();
    auto true_block = blocks.at(0);
    auto false_block = blocks.at(1);

    changed |= processBlock(true_block);
    changed |= processBlock(false_block);

    changed |=
        mergeTensorProperties(true_block->outputs(), false_block->outputs());

    return changed;
  }

  bool checkSchemaReturnsTensors(const c10::FunctionSchema* schema) {
    const std::vector<Argument>& return_args = schema->returns();
    bool has_tensor_output = false;
    for (const auto& arg : return_args) {
       if (arg.type()->castRaw<TensorType>()) {
        has_tensor_output = true;
        break;
      }
    }
    return has_tensor_output;
  }

  // for efficiency
  bool processAtenOps(Node* n) {
    GRAPH_DEBUG("processAtenOps");

    auto schema_opt = n->maybeSchema();
    if (!schema_opt || !checkSchemaReturnsTensors(schema_opt)) {
      GRAPH_DEBUG("schema not found or op does not return tensors");
      return false;
    }
    // TODO: Add custom rule support here

    GRAPH_DEBUG("case = ", n->kind(), " ", *n);
    bool changed = tryApplyDtypeMetaTensor(n);
    /*
    switch (n->kind()) {
      case aten::append:
        // auto elementType =
        // n->input()->type()->expect<ListType>()->containedTypes()[0]; if
        // (auto itp  = elementType->cast<TensorType>()) {
        //   //itp->
        //   auto otp = elementType->cast<TensorType>()->containedTypes()[0];
        //   ListType::create(otp->withScalarType(i))
        // }
        break;
    }
    */

    return changed;
  }
  /*
  AliasDb* getOrCreateAliasDb() {
    if (!aliasDb_) {
      aliasDb_ = std::make_unique<AliasDb>(graph_);
    }
    return aliasDb_.get();
  }
  */

  /*
  // This one is a special rule -- mean take the ScalarType if specified,
  otherwise Tensor type
  // This is an example where the dtype rule has to be extracted from the
  implementation
  "aten::mean(Tensor self, *, ScalarType? dtype) -> Tensor",
  */

  std::shared_ptr<Graph> graph_;
  // lazily initialized if using aliasing_types, otherwise not initialized
  //std::unique_ptr<AliasDb> aliasDb_ = nullptr;
  // std::shared_ptr<TensorPropertyInferrer> node_type_inferrer_ = nullptr;
};

} // anonymous namespace

// This analysis propagates input dtypes (if any) throughout the
// graph.
bool DtypePropagation(std::shared_ptr<Graph>& graph) {
  DTypePropagationPass tp = DTypePropagationPass(graph);
  bool changed = tp.run();
  if (changed) {
    GRAPH_DUMP("After TensorPropertyPropagation pass:", graph);
  }
  return changed;
}

} // namespace jit
} // namespace torch
