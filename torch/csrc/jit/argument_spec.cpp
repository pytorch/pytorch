
#include <torch/csrc/jit/argument_spec.h>

namespace torch {
namespace jit {

void ArgumentSpecCreator::scan(
    const TypePtr& typ,
    size_t depth,
    const WrittenSlots& written_slots) {
  auto finishAggregate = [&](size_t pos) {
    // it is possible after all the work we did to scan this aggregate,
    // we found no tensors or optionals to specialize. In this case, just
    // generate a skip for the whole aggregate.
    bool any_spec = std::any_of(
        instructions_.begin() + pos, instructions_.end(), [](Inst i) {
          return i == SPECIALIZE_TENSOR || i == SPECIALIZE_OPTIONAL ||
              i == SPECIALIZE_OPTIONAL_TENSOR;
        });
    if (!any_spec) {
      instructions_[pos] = SKIP;
      instructions_.resize(pos + 1);
    } else {
      instructions_.emplace_back(LEAVE);
    }
  };
  // the simple vm that scans instructions_ has a limited stack depth,
  // this prevents going deeper than that.
  if (depth >= DEPTH_LIMIT) {
    instructions_.emplace_back(SKIP);
  }
  if (typ->isSubtypeOf(TensorType::get())) {
    num_tensors_++;
    instructions_.emplace_back(SPECIALIZE_TENSOR);
  } else if (typ->isSubtypeOf(OptionalType::ofTensor())) {
    num_tensors_++;
    num_optionals_++;
    instructions_.emplace_back(SPECIALIZE_OPTIONAL_TENSOR);
  } else if (typ->kind() == TypeKind::OptionalType) {
    // note that Optional[Tuple] or Optional[Class] will just register
    // as optional (previously they didn't at all, so it's not a regression).
    num_optionals_++;
    instructions_.emplace_back(SPECIALIZE_OPTIONAL);
  } else if (auto tup = typ->cast<TupleType>()) {
    size_t pos = instructions_.size();
    instructions_.emplace_back(ENTER_TUPLE);
    for (const auto& elem : tup->containedTypes()) {
      scan(elem, depth + 1, written_slots);
    }
    finishAggregate(pos);
  } else if (auto cls = typ->cast<ClassType>()) {
    size_t pos = instructions_.size();
    instructions_.emplace_back(ENTER_OBJECT);
    for (size_t i = 0; i < cls->numAttributes(); ++i) {
      auto key = cls->name()->qualifiedName() + cls->attributeNames().at(i);
      // it is only safe to specialize because someone might have written to it
      if (!written_slots.count(key)) {
        scan(cls->containedTypes().at(i), depth + 1, written_slots);
      } else {
        instructions_.emplace_back(SKIP);
      }
    }
    finishAggregate(pos);
  } else {
    instructions_.emplace_back(SKIP);
  }
};

// this is a coarse-grained guarentee that the slots of a class will not be
// modified by the function. It works fine for things that used be read-only
// modules, but will be overly conservative when some classes are written to.
// Doing alias analysis and looking for writes to the class would be more
// accurate.
static void scanWrittenSlots(
    Block* block,
    ArgumentSpecCreator::WrittenSlots& written_slots) {
  for (Node* n : block->nodes()) {
    if (n->kind() == prim::SetAttr) {
      if (auto cls = n->inputs().at(0)->type()->cast<ClassType>()) {
        written_slots.insert(cls->name()->qualifiedName() + n->s(attr::name));
      }
    }
    for (Block* subblock : n->blocks()) {
      scanWrittenSlots(subblock, written_slots);
    }
    if (n->hasAttribute(attr::Subgraph)) {
      scanWrittenSlots(n->g(attr::Subgraph)->block(), written_slots);
    }
  }
}

ArgumentSpecCreator::ArgumentSpecCreator(Graph& graph)
    : num_inputs_(graph.inputs().size()) {
  WrittenSlots written_slots;
  scanWrittenSlots(graph.block(), written_slots);
  for (Value* input : graph.inputs()) {
    scan(input->type(), 0, written_slots);
  }
}

void ArgumentSpecCreator::dump() const {
  for (Inst inst : instructions_) {
    switch (inst) {
      case LEAVE:
        std::cout << "] ";
        break;
      case ENTER_TUPLE:
        std::cout << "Tuple[";
        break;
      case ENTER_OBJECT:
        std::cout << "Object[";
        break;
      case SKIP:
        std::cout << "Skip ";
        break;
      case SPECIALIZE_TENSOR:
        std::cout << "SpecializeTensor ";
        break;
      case SPECIALIZE_OPTIONAL_TENSOR:
        std::cout << "SpecializeOptionalTensor ";
        break;
      case SPECIALIZE_OPTIONAL:
        std::cout << "SpecializeOptional ";
        break;
    }
  }
  std::cout << "\n";
}

ArgumentSpec ArgumentSpecCreator::create(bool with_grad, const Stack& input)
    const {
  ArgumentSpec spec(num_tensors_, num_optionals_);
  const IValue* stack[DEPTH_LIMIT]; // The stack of IValue lists
  // The stack gets initialized with the input list
  stack[0] = last(input, num_inputs_).begin();
  size_t stack_top = 0; // offset to the top of the stack
  for (Inst inst : instructions_) {
    switch (inst) {
      case SPECIALIZE_OPTIONAL_TENSOR: {
        // consume a tensor optional and add to the argspec
        auto& arg = *stack[stack_top]++;
        spec.addOptional(arg);
        if (!arg.isNone()) {
          spec.addTensor(arg, with_grad);
        }
      } break;
      case SPECIALIZE_TENSOR:
        // consume a tensor and add to the argspec
        spec.addTensor(*stack[stack_top]++, with_grad);
        break;
      case SPECIALIZE_OPTIONAL:
        // consume a non-tensor optional and add to the argspec
        spec.addOptional(*stack[stack_top]++);
        break;
      case ENTER_TUPLE: {
        // consume tuple
        const IValue* iv = stack[stack_top]++;
        AT_ASSERT(iv->isTuple(), "Expected Tuple but got ", iv->tagKind());
        auto p = *reinterpret_cast<const at::ivalue::Tuple* const*>(iv);
        auto tup_ptr = &p->elements()[0];
        // push list of tuple elements to the stack
        stack[++stack_top] = tup_ptr;
      } break;
      case ENTER_OBJECT: {
        // consume object
        const IValue* iv = stack[stack_top]++;
        AT_ASSERT(iv->isObject(), "Expected Object but got ", iv->tagKind());
        auto obj_ptr = &iv->toObjectRef().slots()[0];
        // push list of object elements to the stack
        stack[++stack_top] = obj_ptr;
      } break;
      case SKIP:
        // consume and skip an element
        stack[stack_top]++;
        break;
      case LEAVE:
        --stack_top;
        break;
    }
  }
  return spec;
}

// For every input of a given graph, returns a most detailed type that can be
// inferred for it based on this ArgumentSpec.
void ArgumentSpecCreator::specializeTypes(
    Graph& graph,
    const ArgumentSpec& spec) const {
  auto input_types =
      fmap(graph.inputs(), [](Value* input) { return input->type(); });
  std::vector<std::vector<TypePtr>> result_stack;
  result_stack.emplace_back();
  std::vector<const TypePtr*> input_stack = {input_types.data()};
  std::vector<std::function<TypePtr()>> aggregate_creators;

  size_t tensor_arg_spec_offset =
      0; // number of specialized tensors seen so far
  size_t optional_arg_spec_offset =
      0; // number of specialized optionals seen so far

  auto dim_tensor_type_from_arg = [](const ArgumentInfo& arg) {
    return DimensionedTensorType::create(
        arg.type(),
        ConvertIntToCPUOrCUDA(arg.device()),
        arg.dim(),
        arg.requires_grad());
  };
  for (Inst inst : instructions_) {
    switch (inst) {
      case SPECIALIZE_OPTIONAL_TENSOR: {
        auto& input_type = *input_stack.back()++;
        auto is_present = spec.isPresent(optional_arg_spec_offset++);
        if (!is_present) {
          result_stack.back().emplace_back(input_type);
          break;
        }
        auto& arg = spec.tensorAt(tensor_arg_spec_offset++);
        AT_ASSERT(arg.defined());
        result_stack.back().emplace_back(dim_tensor_type_from_arg(arg));
      } break;
      case SPECIALIZE_TENSOR: {
        input_stack.back()++;
        auto& arg = spec.tensorAt(tensor_arg_spec_offset++);
        if (!arg.defined()) {
          result_stack.back().emplace_back(AutogradZeroTensorType::get());
        } else {
          result_stack.back().emplace_back(dim_tensor_type_from_arg(arg));
        }
      } break;
      case SPECIALIZE_OPTIONAL: {
        auto is_present = spec.isPresent(optional_arg_spec_offset++);
        auto ot = (*input_stack.back()++)->expect<OptionalType>();
        if (!is_present) {
          result_stack.back().emplace_back(ot);
        } else {
          result_stack.back().emplace_back(ot->getElementType());
        }
      } break;
      case ENTER_TUPLE: {
        auto tup = (*input_stack.back()++)->expect<TupleType>();
        input_stack.emplace_back(tup->elements().data());
        result_stack.emplace_back();
        aggregate_creators.emplace_back(
            [&] { return TupleType::create(result_stack.back()); });
      } break;
      case ENTER_OBJECT: {
        auto cls = (*input_stack.back()++)->expect<ClassType>();
        input_stack.emplace_back(cls->containedTypes().data());
        result_stack.emplace_back();
        aggregate_creators.emplace_back(
            [&result_stack, cls] { return cls->refine(result_stack.back()); });
      } break;
      case SKIP:
        result_stack.back().emplace_back(*input_stack.back()++);
        break;
      case LEAVE:
        TypePtr result = aggregate_creators.back()();
        result_stack.pop_back();
        aggregate_creators.pop_back();
        input_stack.pop_back();
        result_stack.back().emplace_back(std::move(result));
        break;
    }
  }
  AT_ASSERT(result_stack.size() == 1);
  // FIXME: by doing this only on the inputs, we only capture graph inputs and
  // not
  //        optionals in tuples or objects. For that to work, we would have
  //        to investigate the uses of the inputs in detail to change the
  //        accesses/ unwrapping
  auto inputs = graph.inputs();
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto t = result_stack.back()[i];
    if (auto ot = t->cast<OptionalType>()) {
      // if an optional input hasn't been specialized above, it is None
      // so we disconnect the input here and replace its uses with
      // a constant
      WithInsertPoint guard(*graph.nodes().begin());
      auto c = graph.insertConstant({}, ot);
      inputs[i]->replaceAllUsesWith(c);
    } else {
      inputs[i]->setType(t);
    }
  }
}

} // namespace jit
} // namespace torch
