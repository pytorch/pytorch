#include <torch/csrc/jit/passes/utils/check_alias_annotation.h>

#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/normalize_ops.h>
#include <torch/csrc/jit/runtime/operator.h>

#include <c10/util/irange.h>

namespace torch {
namespace jit {
namespace {

IValue deepCopy(const IValue& self) {
  // primitive types can be copied directly
  if (!self.isPtrType()) {
    return self;
  }

  // Tensors need special handling, since copy assignment creates an alias
  if (self.isTensor()) {
    return IValue(self.toTensor().clone(at::MemoryFormat::Preserve));
  }

  // Lists of ivalues should recursively deep copy their contents
  if (self.isList()) {
    // NOLINTNEXTLINE(performance-move-const-arg)
    auto source = std::move(self).toList();
    auto newList = c10::impl::GenericList(source.elementType());
    newList.reserve(source.size());
    // NOLINTNEXTLINE(performance-implicit-conversion-in-loop)
    for (const IValue& value : source) {
      newList.push_back(deepCopy(value));
    }
    return newList;
  }

  // Regular lists can copy assign
  if (self.isIntList()) {
    return IValue(self.toIntList().copy());
  } else if (self.isDoubleList()) {
    return IValue(self.toDoubleList().copy());
  } else if (self.isComplexDoubleList()) {
    return IValue(self.toComplexDoubleList().copy());
  } else if (self.isBoolList()) {
    return IValue(self.toBoolList().copy());
  } else if (self.isString()) {
    return IValue(self.toStringRef());
  }

  // If in the future we add more reference types that are used in aten ops,
  // we'll have to add them as cases here.
  AT_ASSERT(false);
}

Stack deepCopy(const Stack& stack) {
  Stack ret;
  ret.reserve(stack.size());
  for (const auto& v : stack) {
    ret.push_back(deepCopy(v));
  }
  return ret;
}

bool deepEquals(const IValue& lhs, const IValue& rhs) {
  if (lhs.isTensor() && rhs.isTensor()) {
    return lhs.toTensor().equal(rhs.toTensor());
  }

  if (lhs.isTensorList() && rhs.isTensorList()) {
    const auto a = lhs.toTensorList();
    const auto b = rhs.toTensorList();
    if (a.size() != b.size()) {
      return false;
    }
    for (auto i = decltype(a.size()){0}; i < a.size(); ++i) {
      if (!a[i].equal(b[i])) {
        return false;
      }
    }
    return true;
  }

  return lhs == rhs;
}

struct AliasAndIValue {
  AliasAndIValue(const at::AliasInfo* aliasInfo, IValue iValue)
      : aliasInfo(aliasInfo), iValue(std::move(iValue)) {}

  const at::AliasInfo* aliasInfo;
  const IValue iValue;
};

// No inputs should alias each other
void checkInputPreconditions(const Stack& inputs) {
  for (const auto i : c10::irange(inputs.size())) {
    for (const auto j : c10::irange(inputs.size())) {
      if (i == j) {
        continue;
      }
      const auto& lhs = inputs.at(i);
      const auto& rhs = inputs.at(j);
      AT_ASSERT(!lhs.isAliasOf(rhs));
    }
  }
}

// If two ivalues alias, they must share an alias set
void checkAliases(
    const std::vector<AliasAndIValue>& inputs,
    const std::vector<AliasAndIValue>& outputs) {
  for (const auto& output : outputs) {
    // if this output aliases any input, make sure that they share an alias set
    for (const auto& input : inputs) {
      if (output.iValue.isAliasOf(input.iValue)) {
        const auto* inputSet = input.aliasInfo;
        const auto* outputSet = output.aliasInfo;
        AT_ASSERT(inputSet && outputSet);
        bool found = false;
        for (const auto& set : inputSet->beforeSets()) {
          if (outputSet->beforeSets().count(set)) {
            found = true;
            break;
          }
        }
        AT_ASSERT(found);
      }
    }
  }
}

// If we didn't specify that we write to an input value, it must have not
// changed
void checkWrites(
    const std::vector<AliasAndIValue>& inputs,
    const std::vector<IValue>& deepCopiedInputs) {
  AT_ASSERT(inputs.size() == deepCopiedInputs.size());
  for (const auto i : c10::irange(inputs.size())) {
    const auto& input = inputs[i];
    const auto& deepCopiedInput = deepCopiedInputs[i];
    if (!input.aliasInfo || !input.aliasInfo->isWrite()) {
      AT_ASSERT(deepEquals(input.iValue, deepCopiedInput));
    }
  }
}

const Node* findNodeForOp(
    const Graph& g,
    const std::string& unqualifiedOpName) {
  const auto opName = Symbol::fromQualString("aten::" + unqualifiedOpName);
  for (const auto* node : g.nodes()) {
    if (node->kind() == opName) {
      return node;
    }
  }

  // Check for alias-ed operator names
  const auto aliasOp = torch::jit::getOperatorAliasMap().find(opName);
  if (aliasOp != torch::jit::getOperatorAliasMap().end()) {
    for (const auto* node : g.nodes()) {
      if (node->kind() == aliasOp->second) {
        return node;
      }
    }
  }

  // Ideally, there will be only one ATen operator that has tensor outputs in
  // the graph. Let's use that as the last resolve to make checkAliasAnnotation
  // more robust.
  for (const auto* node : g.nodes()) {
    if (!node->maybeOperator()) {
      continue;
    }
    if (!node->getOperator().isC10Op()) {
      continue;
    }

    for (const auto* output : node->outputs()) {
      if (output->type()->kind() == TypeKind::TensorType) {
        return node;
      }
    }
  }

  AT_ASSERT(false);
}

// Handle a few special cases where we need to propagate constants
// manually
// TODO(suo): we should be able to move this stuff to constant prop
std::optional<IValue> toIValueProp(const Value* v) {
  if (v->node()->kind() == prim::ListConstruct) {
    std::vector<IValue> genericList;
    for (auto input : v->node()->inputs()) {
      if (auto elem = toIValue(input)) {
        genericList.push_back(*elem);
      } else {
        // One of the list elements isn't constant.
        return c10::nullopt;
      }
    }

    // Specialize the list based on ListConstruct's return type
    auto listType = v->node()->output()->type();
    auto containedType = listType->containedTypes().at(0);
    if (containedType == IntType::get()) {
      return IValue(
          fmap(genericList, [](const IValue& v) { return v.toInt(); }));
    } else if (containedType == FloatType::get()) {
      return IValue(
          fmap(genericList, [](const IValue& v) { return v.toDouble(); }));
    } else if (containedType->isSubtypeOf(*TensorType::get())) {
      return IValue(
          fmap(genericList, [](const IValue& v) { return v.toTensor(); }));
    } else {
      return c10::nullopt;
    }
  }

  if (v->node()->kind() == aten::Float) {
    if (auto maybe_stack = runNodeIfInputsAreConstant(v->node())) {
      return maybe_stack->at(0);
    }
  }
  return c10::nullopt;
}

// batch_norm and instance_norm have incorrect annotations, because
// (a!)? annotations aren't supported, so these checks would fail.
// Their behavior also varies depending on the `training` and
// `use_input_stats` arguments.
// There are custom implementations in alias_analysis.cpp for these ops.
bool shouldIgnoreNode(const Node* n) {
  switch (n->kind()) {
    case aten::batch_norm:
    case aten::instance_norm:
      return true;
    default:
      return false;
  }
}
} // namespace

void checkAliasAnnotation(
    const std::shared_ptr<Graph>& graph,
    std::vector<IValue> pythonInputs,
    const std::string& unqualifiedOpName) {
  // Find the node that corresponds to our op name
  const auto node = findNodeForOp(*graph, unqualifiedOpName);
  if (shouldIgnoreNode(node)) {
    return;
  }

  // Build the stack to use as input to the op
  Stack stack;
  for (const auto input : node->inputs()) {
    if (input->node() == graph->param_node()) {
      // This value was passed as an input in python
      push(stack, pythonInputs.at(input->offset()));
    } else {
      // This a generated constant, which we need to evaluate
      auto inputValue = toIValue(input);
      if (!inputValue) {
        inputValue = toIValueProp(input);
      }

      if (inputValue) {
        push(stack, *inputValue);
      } else {
        AT_ASSERT(input->type()->kind() == TypeKind::OptionalType);
        push(stack, IValue());
      }
    }
  }

  // Precondition: no inputs should alias each other. So if we find an alias,
  // it was created by the op.
  checkInputPreconditions(stack);

  const auto& schema = node->schema();

  std::vector<AliasAndIValue> inputsToCheck;
  for (const auto i : c10::irange(schema.arguments().size())) {
    inputsToCheck.emplace_back(
        schema.arguments().at(i).alias_info(), stack.at(i));
  }

  // Save a copy of the inputs so we can check whether the original inputs were
  // written to.
  const auto inputsDeepCopy = deepCopy(stack);

  // Run the op
  node->getOperation()(stack);

  const auto outputs = std::move(stack);

  std::vector<AliasAndIValue> outputsToCheck;
  for (const auto i : c10::irange(schema.returns().size())) {
    outputsToCheck.emplace_back(
        schema.returns().at(i).alias_info(), outputs.at(i));
  }

  // Check that if any alias was created, we annotated it properly.
  checkAliases(inputsToCheck, outputsToCheck);

  // Check that if nothing was accidentally written to.
  checkWrites(inputsToCheck, inputsDeepCopy);
}

} // namespace jit
} // namespace torch
