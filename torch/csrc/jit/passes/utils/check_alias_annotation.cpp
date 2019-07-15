#include <torch/csrc/jit/passes/utils/check_alias_annotation.h>
#include <torch/csrc/jit/operator.h>

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
    return IValue(self.toTensor().clone());
  }
  if (self.isTensorList()) {
    c10::List<at::Tensor> newList;
    for (const at::Tensor& oldTensor : self.toTensorListRef()) {
      newList.push_back(oldTensor.clone());
    }
    return newList;
  }

  // Lists of ivalues should recursively deep copy their contents
  if (self.isGenericList()) {
    auto newList = c10::impl::GenericList(c10::impl::deprecatedUntypedList());
    newList.reserve(self.toGenericListRef().size());
    for (const IValue& value : self.toGenericListRef()) {
      newList.push_back(deepCopy(value));
    }
    return newList;
  }

  // Regular lists can copy assign
  if (self.isIntList()) {
    return IValue(self.toIntList().copy());
  } else if (self.isDoubleList()) {
    return IValue(self.toDoubleList().copy());
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
  if (lhs.isInt() && rhs.isInt()) {
    return lhs.toInt() == rhs.toInt();
  } else if (lhs.isDouble() && rhs.isDouble()) {
    return lhs.toDouble() == rhs.toDouble();
  } else if (lhs.isNone() && rhs.isNone()) {
    return true;
  } else if (lhs.isIntList() && rhs.isIntList()) {
    return lhs.toIntListRef().equals(rhs.toIntListRef());
  } else if (lhs.isTensor() && rhs.isTensor()) {
    return lhs.toTensor().equal(rhs.toTensor());
  }

  throw std::runtime_error("Deep equals not implemented for type");
}

struct AliasAndIValue {
  AliasAndIValue(c10::optional<at::AliasInfo> aliasInfo, IValue iValue)
      : aliasInfo(std::move(aliasInfo)), iValue(std::move(iValue)) {}

  const c10::optional<at::AliasInfo> aliasInfo;
  const IValue iValue;
};

// No inputs should alias each other
void checkInputPreconditions(const Stack& inputs) {
  for (size_t i = 0; i < inputs.size(); i++) {
    for (size_t j = 0; j < inputs.size(); j++) {
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
        const auto inputSet = input.aliasInfo;
        const auto outputSet = output.aliasInfo;
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
  for (size_t i = 0; i < inputs.size(); i++) {
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
  for (const auto node : g.nodes()) {
    if (node->kind() == opName) {
      return node;
    }
  }
  AT_ASSERT(false);
}

// Handle a few special cases where we need to propagate constants
// manually
// TODO(suo): we should be able to move this stuff to constant prop
c10::optional<IValue> toIValueProp(const Value* v) {
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
      return c10::impl::toList(fmap(genericList, [](const IValue& v) { return v.toInt(); }));
    } else if (containedType == FloatType::get()) {
      return c10::impl::toList(fmap(genericList, [](const IValue& v) { return v.toDouble(); }));
    } else if (containedType->isSubtypeOf(TensorType::get())) {
      return c10::impl::toList(fmap(genericList, [](const IValue& v) { return v.toTensor(); }));
    } else {
      return c10::nullopt;
    }
  }

  if (v->node()->kind() == aten::Float) {
    auto op = getOperation(v->node());
    if (auto input = toIValue(v->node()->input())) {
      auto op = getOperation(v->node());
      Stack stack;
      push(stack, *input);
      op(stack);
      return stack.back();
    } else {
      return c10::nullopt;
    }
  }
  return c10::nullopt;
}
} // namespace

void checkAliasAnnotation(
    const std::shared_ptr<Graph>& graph,
    std::vector<IValue> pythonInputs,
    const std::string& unqualifiedOpName) {
  // Find the node that corresponds to our op name
  const auto node = findNodeForOp(*graph, unqualifiedOpName);

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

  const auto schema = node->schema();

  std::vector<AliasAndIValue> inputsToCheck;
  for (size_t i = 0; i < schema.arguments().size(); i++) {
    inputsToCheck.emplace_back(
        schema.arguments().at(i).alias_info(), stack.at(i));
  }

  // Save a copy of the inputs so we can check whether the original inputs were
  // written to.
  const auto inputsDeepCopy = deepCopy(stack);

  // Run the op
  getOperation(node)(stack);

  const auto outputs = std::move(stack);

  std::vector<AliasAndIValue> outputsToCheck;
  for (size_t i = 0; i < schema.returns().size(); i++) {
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
