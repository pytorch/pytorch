#include <torch/csrc/jit/passes/shape_analysis.h>

#include <c10/util/Exception.h>
#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/exception_message.h>
#include <torch/csrc/jit/runtime/operator.h>

#include <torch/csrc/autograd/variable.h>

#include <ATen/DeviceGuard.h>
#include <ATen/ExpandUtils.h>

#include <exception>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

namespace torch {
namespace jit {

namespace prim {
using namespace ::c10::prim;
}

struct propagation_error : std::exception {};

#define SHAPE_ASSERT(cond) \
  if (!(cond))             \
  throw propagation_error()

namespace {

bool isValidArgumentForRunning(Value* v) {
  // allow constants
  if (toIValue(v))
    return true;
  if (TensorTypePtr tt = v->type()->cast<TensorType>()) {
    if (!tt->scalarType()) {
      return false;
    }
    return !at::isIntegralType(*tt->scalarType(), /*includeBool=*/false);
  }
  return v->type()->isSubtypeOf(FloatType::get());
}

bool isValidReturnForRunning(Value* v) {
  return v->type()->isSubtypeOf(TensorType::get()) ||
      v->type()->isSubtypeOf(NumberType::get());
}

bool containsTensorType(const TypePtr& t) {
  auto n_contained = t->containedTypes().size();
  if (n_contained == 1) {
    return t->containedTypes().at(0)->isSubtypeOf(TensorType::get());
  } else if (n_contained > 1) {
    return std::any_of(
        t->containedTypes().begin(),
        t->containedTypes().end(),
        containsTensorType);
  }
  return false;
}

class ShapePropagator {
 public:
  explicit ShapePropagator(const std::shared_ptr<Graph>& graph)
      : aliasDb_(graph) {
    collectResizeSet(graph->block());
  }

  void PropagateShapeOnBlock(Block* block, bool insert_expands = true) {
    for (Node* node : block->nodes()) {
      try {
        PropagateShapeOnNode(node, insert_expands);
      } catch (propagation_error& e) {
        setUnshapedType(node);
      } catch (std::exception& e) {
        throw ErrorReport(node->sourceRange())
            << ExceptionMessage(e)
            << "\nThe above operation failed shape propagation in this context";
      }
    }
  }

 private:
  ValueSet resized_alias_set;
  const AliasDb aliasDb_;

  bool resizesInput(Node* n) {
    static std::unordered_set<Symbol> resize_ops{
        aten::resize_,
        aten::resize_as_,
        aten::copy_,
        aten::set_,
        aten::unsqueeze_,
        aten::t_,
        aten::transpose_,
    };

    if (resize_ops.count(n->kind()))
      return true;

    if (!n->maybeSchema())
      return false;

    // ops which take the result and write to input "out"
    if (auto out_arg_index = n->schema().argumentIndexWithName("out")) {
      auto arg = n->schema().arguments().at(*out_arg_index);
      return arg.kwarg_only() && arg.type()->isSubtypeOf(TensorType::get());
    }
    return false;
  }

  void collectResizeSet(Block* block) {
    for (Node* n : block->nodes()) {
      for (Block* b : n->blocks()) {
        collectResizeSet(b);
      }
      if (resizesInput(n)) {
        for (const auto input : n->inputs()) {
          if (aliasDb_.writesToAlias(n, {input})) {
            resized_alias_set.insert(input);
          }
        }
      }
    }
  }

  void setUnshapedType(Value* o) {
    o->setType(unshapedType(o->type()));
  }

  void setUnshapedType(Node* node) {
    for (auto o : node->outputs()) {
      setUnshapedType(o);
    }
  }

  int64_t wrapDim(int64_t dim, at::IntArrayRef sizes) {
    if (dim < 0) {
      dim += sizes.size();
    }
    return dim;
  }

  // TODO: Would be better to make JIT not assume that CUDA devices
  // are the only thing that exist.
  static at::Device jitDeviceIndexToDevice(int device) {
    return device == -1 ? at::kCPU : at::Device(at::kCUDA, device);
  }

  IValue representativeValue(Value* v) {
    TypePtr type_ = v->type();
    // if the value is actually constant, just use it!
    if (auto iv = toIValue(v)) {
      return *iv;
    }
    if (TensorTypePtr type = type_->cast<TensorType>()) {
      if (type->isComplete()) {
        auto attype = type->device()->is_cpu() ? at::CPU(*type->scalarType())
                                               : at::CUDA(*type->scalarType());
        at::DeviceGuard device_guard(*type->device());
        return at::empty_strided(
                   *type->sizes().concrete_sizes(),
                   *type->strides().concrete_sizes(),
                   attype.options())
            .zero_();
      }
      // fallthrough
    } else if (type_->isSubtypeOf(FloatType::get())) {
      return 0.f;
    }
    // we should not get here because isValidArgumentForRunning should have
    // prevented it
    std::stringstream ss;
    ss << "unable to create representative value for: " << type_->str()
       << ". File a bug report";
    throw std::runtime_error(ss.str());
  }

  // for each node in the schema with type Tensor, extract the T type
  // returns c10::nullopt if any Tensor in the schema does not have a known
  // shape ignores non-tensor in the list of inputs
  c10::optional<std::vector<TensorTypePtr>> gatherTensorTypes(
      Node* node,
      bool complete = false) {
    std::vector<TensorTypePtr> tensor_types;

    auto schema_opt = node->maybeSchema();
    if (!schema_opt) {
      return c10::nullopt;
    }
    auto& schema = *schema_opt;
    auto& args = schema.arguments();
    // can't handle varargs primitives because we don't know what should be a
    // Tensor
    if (schema.is_vararg()) {
      return c10::nullopt;
    }
    for (size_t i = 0; i < args.size(); ++i) {
      if (args[i].type()->isSubtypeOf(ListType::ofTensors())) {
        return c10::nullopt;
      } else if (args[i].type()->isSubtypeOf(TensorType::get())) {
        if (auto type = node->input(i)->type()->cast<TensorType>()) {
          if (complete && !type->isComplete()) {
            return c10::nullopt;
          }
          tensor_types.push_back(type);
        } else {
          return c10::nullopt;
        }
      } else /* non-tensor type */ {
        continue;
      }
    }
    return tensor_types;
  }

  c10::ScalarType unionScalarTypes(
      c10::ScalarType original,
      c10::ScalarType next) {
    if (original == c10::ScalarType::Undefined) {
      return next;
    } else {
      return c10::promoteTypes(original, next);
    }
  }

  // Promotes result types for arithmetic operations on Tensor operands using
  // new type promotion logic. See tensor_attributes.rst for details.
  // This doesn't handle the case of arithmetic ops with Scalar arguments (when
  // `Tensor.getUnsafeTensorImpl()->is_wrapped_nubmer()` would return true)
  c10::optional<c10::ScalarType> getPromotedTypeForArithmeticOp(Node* node) {
    c10::ScalarType dimmed = c10::ScalarType::Undefined;
    c10::ScalarType zerodim = c10::ScalarType::Undefined;
    // binary arithmetic ops, more than 2 args is alpha.
    for (size_t i = 0; i < 2; i++) {
      auto dtt = node->inputs()[i]->type()->expect<TensorType>();
      auto inputDtype = dtt->scalarType();
      if (!dtt || !inputDtype) {
        return c10::nullopt;
      }
      if (dtt->dim() && *dtt->dim() > 0) {
        dimmed = unionScalarTypes(dimmed, *inputDtype);
      } else if (!isFloatingType(dimmed)) {
        // if no dimensions
        zerodim = unionScalarTypes(zerodim, *inputDtype);
      }
    }
    // if a tensor with dimensions is already of the highest category, don't
    // need to check zero-dim tensors.
    if (isFloatingType(dimmed)) {
      return dimmed;
    }
    // int_tensor * zero_dim_floating -> floating_tensor
    if (isIntegralType(dimmed, false) && isFloatingType(zerodim)) {
      return zerodim;
    }
    // bool_tensor * non_bool_scalar -> non_bool_tensor
    if (c10::ScalarType::Bool == dimmed &&
        c10::ScalarType::Undefined != zerodim) {
      return zerodim;
    }
    // types of dimensioned tensors generally take precedence over zero-dim
    // tensors if not promoting due to category. e.g.:
    // int_tensor * long -> int_tensor
    if (c10::ScalarType::Undefined != dimmed) {
      return dimmed;
    }

    // no dimmed tensors. e.g. zero_dim_tensor + zero_dim_tensor.
    return zerodim;
  }

  bool mergeTypes(
      ArrayRef<Value*> lhs,
      ArrayRef<Value*> rhs,
      ArrayRef<Value*> outputs) {
    AT_ASSERT(lhs.size() == rhs.size() && rhs.size() == outputs.size());
    bool changed = false;
    for (size_t i = 0; i < lhs.size(); ++i) {
      auto old_output_type = outputs[i]->type();
      auto new_type =
          unifyTypes(lhs[i]->type(), rhs[i]->type(), /*default_to_any=*/true);
      AT_ASSERT(new_type);
      outputs[i]->setType(*new_type);
      if (*old_output_type != *outputs[i]->type())
        changed = true;
    }
    return changed;
  }

  void broadcastBinary(
      Node* node,
      std::vector<TensorTypePtr>& types,
      size_t idx1,
      size_t idx2) {
    auto expected_size = at::infer_size(
        *types[idx1]->sizes().concrete_sizes(),
        *types[idx2]->sizes().concrete_sizes());
    auto broadcast = [&](size_t input_idx) {
      TensorTypePtr input_type = types.at(input_idx);
      if (input_type->sizes() == expected_size)
        return;
      auto graph = node->owningGraph();
      WithInsertPoint point_guard{node};
      Node* expand = graph
                         ->create(
                             aten::expand,
                             {node->inputs().at(input_idx),
                              graph->insertConstant(expected_size),
                              graph->insertConstant(false)})
                         ->insertBefore(node);
      PropagateShapeOnNode(expand);
      node->replaceInput(input_idx, expand->output());
    };
    broadcast(idx1);
    broadcast(idx2);
    types[0] = node->inputs().at(idx1)->type()->expect<TensorType>();
    types[1] = node->inputs().at(idx2)->type()->expect<TensorType>();
  }

  OperatorSet cannot_propagate_shape_by_running_it = {
      "aten::solve(Tensor self, Tensor A) -> (Tensor, Tensor)",
      "aten::inverse(Tensor self) -> Tensor",
  };

  // Check if this node depends on a value that has been mutated previously. If
  // it has, then it's not safe to run this node in isolation, since we don't
  // know whether the dependency has been executed.
  std::unordered_map<Node*, bool> dependsOnMutationMemo_;
  bool dependsOnMutation(Node* node) {
    if (dependsOnMutationMemo_.count(node) != 0) {
      return dependsOnMutationMemo_[node];
    }

    if (aliasDb_.hasWriters(node)) {
      // If something could have written to a value used by this node, we can't
      // guarantee the result is the same when running it in isolation.
      dependsOnMutationMemo_[node] = true;
      return true;
    }

    // recursively check the producers of its inputs. We need to do this if the
    // mutable value has been laundered through a pure function:
    //   a += 1
    //   c = a + b
    //   d = c + 1
    // In this case, `d` cares whether `a` has been mutated even though it's not
    // a direct input.
    auto depends = false;
    for (auto input : node->inputs()) {
      depends |= dependsOnMutation(input->node());
    }

    dependsOnMutationMemo_[node] = depends;
    return depends;
  }

  bool canPropagateShapeByRunningIt(Node* node) {
    if (node->isMemberOf(cannot_propagate_shape_by_running_it)) {
      return false;
    }

    if (dependsOnMutation(node)) {
      return false;
    }

    bool valid_args = std::all_of(
        node->inputs().begin(),
        node->inputs().end(),
        isValidArgumentForRunning);
    if (!valid_args)
      return false;

    bool valid_returns = std::all_of(
        node->outputs().begin(),
        node->outputs().end(),
        isValidReturnForRunning);
    if (!valid_returns)
      return false;

    return true;
  }

  // If there's no Tensor in outputs, e.g float / float,
  // we don't need to propagate shape.
  bool DoesntRefineOutputs(Node* node) {
    auto outputs = node->outputs();
    for (auto& out : outputs) {
      if (containsTensorType(out->type())) {
        return false;
      }
    }
    return true;
  }

  bool PropagateShapeOnNodeByRunningIt(Node* node, Operation op = nullptr) {
    if (!canPropagateShapeByRunningIt(node))
      return false;

    if (!op)
      op = node->getOperation();

    Stack stack;

    for (auto input : node->inputs()) {
      stack.push_back(representativeValue(input));
    }

    // XXX: we're not catching any exceptions from the op for now. This
    // is to uncover any mistakes we could make when editing this code,
    // and eventually it shouldn't matter, because this phase should be
    // preceded by schema checking.
    op(&stack);

    AT_ASSERT(stack.size() == node->outputs().size());
    for (size_t i = 0; i < stack.size(); ++i) {
      // some ops may have mixed tensor/primitive outputs
      // for primitives, we don't need to change the type because it is already
      // its most constrained form.
      auto tensor_type = node->outputs()[i]->type()->cast<TensorType>();
      if (stack[i].isTensor() && tensor_type) {
        // gradient information isn't always available or part of represenative
        // inputs, maintain original grad property
        auto tensor_grad = tensor_type->requiresGrad();
        node->outputs()[i]->setType(TensorType::create(stack[i].toTensor())
                                        ->withRequiresGrad(tensor_grad));
      }
    }
    return true;
  }

  void PropagateCatShape(Node* cat_node) {
    static const auto propagate_complete =
        [this](Node* node, at::ArrayRef<Value*> tensors) -> bool {
      auto input_types =
          fmap(tensors, [](Value* v) { return v->type()->cast<TensorType>(); });
      if (!std::all_of(
              input_types.begin(),
              input_types.end(),
              [](const TensorTypePtr& tp) {
                return tp != nullptr && tp->isComplete();
              })) {
        return false;
      }
      if (!node->is_constant(attr::dim))
        return false;
      std::vector<int64_t> sizes = *input_types[0]->sizes().concrete_sizes();
      const int64_t dim = wrapDim(node->get<int64_t>(attr::dim).value(), sizes);
      const int64_t ndim = sizes.size();

      if (dim < 0 || dim >= ndim)
        return false;

      sizes[dim] = 0;
      for (auto& tp : input_types) {
        auto tp_sizes = tp->sizes().concrete_sizes().value();
        if (sizes.size() != tp_sizes.size())
          return false;
        for (int64_t i = 0; i < ndim; ++i) {
          if (sizes[i] != tp_sizes[i] && i != dim) {
            return false;
          }
        }
        sizes[dim] += tp_sizes[dim];
      }
      node->output()->setType(input_types[0]->withSizes(sizes));
      return true;
    };
    static const auto propagate = [](Node* node,
                                     at::ArrayRef<Value*> tensors) -> bool {
      for (Value* v : tensors) {
        if (auto type = v->type()->cast<TensorType>()) {
          node->output()->setType(type->dimensionedOnly());
          return true;
        }
      }
      return false;
    };
    auto list_node =
        ((cat_node->kind() == prim::FusedConcat)
             ? cat_node
             : cat_node->namedInput(attr::tensors)->node());
    if (list_node->kind() == prim::ListConstruct ||
        cat_node->kind() == prim::FusedConcat) {
      auto tensors = list_node->inputs();
      if (!tensors.empty()) {
        // NOLINTNEXTLINE(bugprone-branch-clone)
        if (propagate_complete(cat_node, tensors)) {
          return;
        } else if (propagate(cat_node, tensors)) {
          return;
        }
      }
    }
    setUnshapedType(cat_node);
  }

  void propagateTorchTensorShape(Node* node) {
    auto input_type = node->inputs().at(0)->type();

    size_t dims = 0;
    auto input_base_type = input_type;
    auto list_type = input_type->cast<ListType>();
    while (list_type) {
      dims++;
      input_base_type = list_type->getElementType();
      list_type = input_base_type->cast<ListType>();
    }

    at::optional<at::ScalarType> default_type =
        tryScalarTypeFromJitType(input_base_type);
    if (auto grad_index = node->schema().argumentIndexWithName("dtype")) {
      auto inp = toIValue(node->inputs().at(*grad_index));
      if (inp == c10::nullopt) {
        return;
      } else if (!inp->isNone()) {
        default_type = inp->toScalarType();
      }
    }

    at::Device default_device = at::kCPU;
    if (auto device_index = node->schema().argumentIndexWithName("device")) {
      auto inp = toIValue(node->inputs().at(*device_index));
      if (inp == c10::nullopt) {
        return;
      } else if (!inp->isNone()) {
        default_device = inp->toDevice();
      }
    }
    node->output()->setType(TensorType::create(
        default_type, default_device, dims, /*requires_grad=*/c10::nullopt));
  }

  // returns whether any such values were found
  bool setUnshapedTypeIfAliasResizedSet(at::ArrayRef<Value*> vs) {
    bool in_resize = false;
    for (auto v : vs) {
      if (aliasDb_.mayAlias(ValueSet{v}, resized_alias_set)) {
        setUnshapedType(v);
        in_resize = true;
      }
    }
    return in_resize;
  }

  void PropagateShapeOnNode(Node* node, bool insert_expands = true) {
    // Certain ops like resize_ change the input tensors size. Because our
    // analysis is flow invariant, we set any Tensor that can alias a resized
    // Tensor to the base Tensor Type without size information.
    if (setUnshapedTypeIfAliasResizedSet(node->inputs())) {
      return setUnshapedType(node);
    }

    // These don't require the types, and have complicated schema. Return early
    // after we process them.
    switch (node->kind()) {
      case prim::If: {
        auto then_block = node->blocks().at(0);
        auto else_block = node->blocks().at(1);
        PropagateShapeOnBlock(then_block);
        PropagateShapeOnBlock(else_block);
        mergeTypes(
            then_block->outputs(), else_block->outputs(), node->outputs());
        return;
      }
      case prim::Loop: {
        auto body_block = node->blocks().at(0);
        // propagate counter type
        body_block->inputs().at(0)->setType(node->inputs().at(0)->type());
        // propagate loop-carried input types to block inputs
        auto loop_carried_inputs = node->inputs().slice(2); // skip max, cond
        auto loop_carried_block = body_block->inputs().slice(1); // skip trip
        for (size_t i = 0; i < loop_carried_inputs.size(); ++i) {
          loop_carried_block[i]->setType(loop_carried_inputs[i]->type());
        }
        auto loop_carried_outputs = body_block->outputs().slice(1); // skip cond

        do {
          PropagateShapeOnBlock(body_block, /*insert_expands=*/false);
          // note: inserting expands is unsafe at this point, we don't know
          // if the types are stable yet, so the arguments to expand may change
        } while (mergeTypes(
            loop_carried_block, loop_carried_outputs, loop_carried_block));

        // now that the types are stable, we can insert the expands
        PropagateShapeOnBlock(body_block, /*insert_expands=*/true);

        for (size_t i = 0; i < loop_carried_inputs.size(); ++i) {
          node->outputs()[i]->setType(loop_carried_block[i]->type());
        }
        return;
      }
      case aten::Bool:
      case aten::Int:
      case aten::Float:
      case aten::ScalarImplicit:
      case aten::FloatImplicit:
      case aten::IntImplicit:
        return; // correct num type is already set
      case prim::NumToTensor: {
        TypePtr typ = node->input()->type();
        if (typ->isSubtypeOf(IntType::get()) ||
            typ->isSubtypeOf(BoolType::get())) {
          node->output()->setType(TensorType::create(
              at::kLong, at::kCPU, 0, /*requires_grad=*/c10::nullopt));
        } else if (node->input()->type()->isSubtypeOf(FloatType::get())) {
          node->output()->setType(TensorType::create(
              at::kDouble, at::kCPU, 0, /*requires_grad=*/c10::nullopt));
        }
        return;
      }
      case aten::tensor:
      case aten::as_tensor: {
        // as_tensor has an overloaded schema and can either have a tensor or
        // a list as the first input, if the input is a tensor, we delegate
        // the shape propagation in PropagateTensorShapeOnNode
        if (node->inputs().at(0)->type()->isSubtypeOf(TensorType::get())) {
          break;
        }
        return propagateTorchTensorShape(node);
      }
      case prim::TupleConstruct: {
        // We refresh the tuple type, because the input types could have been
        // refined.
        auto orig_type = node->output()->type()->expect<TupleType>();
        auto new_types =
            fmap(node->inputs(), [](Value* v) { return v->type(); });
        node->output()->setType(
            orig_type->createWithContained(std::move(new_types)));
        return;
      }
      case prim::TupleUnpack: {
        auto tuple_type = node->input()->type()->cast<TupleType>();
        AT_ASSERT(
            tuple_type &&
            tuple_type->elements().size() == node->outputs().size());
        auto elems = tuple_type->elements();
        for (size_t i = 0; i < node->outputs().size(); ++i) {
          node->output(i)->setType(elems[i]);
        }
        return;
      }
      case prim::Constant: {
        if (node->output()->type()->isSubtypeOf(TensorType::get())) {
          node->output()->inferTypeFrom(node->t(attr::value));
        }
        return;
      }
      case prim::unchecked_unwrap_optional: {
        // If we have specialized the optional type to the element type,
        // we want to pass it down. We write this as input.isSubtypeOf(output)
        // to be sure that we don't screw up nested optionals.
        if (node->input()->type()->isSubtypeOf(node->output()->type())) {
          node->output()->setType(node->input()->type());
        }
        return;
      }
      case prim::ConstantChunk: {
        Value* tensor = node->input();
        if (auto type = tensor->type()->cast<TensorType>()) {
          type = type->dimensionedOnly();
          for (Value* output : node->outputs()) {
            output->setType(type);
          }
        } else {
          setUnshapedType(node);
        }
        return;
      }
      case prim::grad: {
        auto tt = node->input()->type()->expect<TensorType>();
        // grad may be undefined
        // requires_grad may be required
        auto grad_type = TensorType::get()->withPossiblyUndefined();
        node->output()->setType(grad_type);
        return;
      }
      case prim::CallFunction:
      case prim::CallMethod:
      case prim::AutogradZero: {
        setUnshapedType(node);
        return;
      }
      case prim::GetAttr: {
        auto cls = node->input()->type()->expect<ClassType>();
        // propagate any type specializations encoded in the type of the class
        node->output()->setType(cls->getAttribute(node->s(attr::name)));
        return;
      }
      case aten::_unwrap_optional: {
        // If we have specialized the optional type to the element type,
        // we want to pass it down. We write this as input.isSubtypeOf(output)
        // to be sure that we don't screw up nested optionals.
        if (node->input()->type()->isSubtypeOf(node->output()->type())) {
          node->output()->setType(node->input()->type());
        }
        return;
      }
      default:
        break; // fall-through
    }

    if (node->hasSideEffects()) {
      return;
    }

    if (node->matches("aten::cat(Tensor[] tensors, int dim) -> Tensor") ||
        node->kind() == prim::FusedConcat) {
      return PropagateCatShape(node);
    }

    if (auto maybe_complete_types =
            gatherTensorTypes(node, /*complete=*/true)) {
      if (PropagateCompleteShapeOnNode(
              node, insert_expands, std::move(*maybe_complete_types))) {
        return;
      }
    }

    if (PropagateTensorShapeOnNode(node, insert_expands)) {
      return;
    }

    if (DoesntRefineOutputs(node)) {
      return;
    }

    if (PropagateShapeOnNodeByRunningIt(node)) {
      return;
    }
    return setUnshapedType(node);
  }

  static c10::optional<size_t> determineListSize(Value* list) {
    AT_ASSERT(list->type()->cast<ListType>());
    if (auto shape = constant_as<c10::List<int64_t>>(list)) {
      return shape->size();
    }
    auto input_node = list->node();
    if (input_node->kind() == prim::ListConstruct) {
      return input_node->inputs().size();
    }
    return c10::nullopt;
  }

  // is it ok to try to run the op
  // If an input is a constant, then we assume that the input is valid
  // and we can try to run it.
  // Otherwise:
  // Integral typed _inputs_ are often an indicator that we're indexing into
  // a tensor, so we should special-case these ops in the shape propagation.
  // Additionally, passing in a zero representative tensor into an integer
  // division op causes divide-by-zero errors
  // _Outputs_ must be tensors or primitives
  // We will call inferTypeFrom on the tensors, and ignore the primitives.
  // However, we allow primitive returns because we want to support mixed
  // primitive/tensor outputs.

  bool PropagateTensorShapeOnNode(Node* node, bool insert_expands) {
    static const auto broadcast =
        [](std::vector<TensorTypePtr>& tensor_types,
           c10::optional<at::ScalarType> t) -> TensorTypePtr {
      if (tensor_types.size() == 1) {
        return tensor_types[0]->dimensionedOnly()->withScalarType(t);
      }
      AT_ASSERT(!tensor_types.empty());
      auto any_type = tensor_types[0];
      auto max_dims = any_type->dim();
      for (auto& type : tensor_types) {
        if (!max_dims || !type->dim()) {
          max_dims = c10::nullopt;
        } else {
          max_dims = std::max(*max_dims, *type->dim());
        }
      }
      return TensorType::create(
          t,
          any_type->device(),
          max_dims,
          /*requires_grad=*/c10::nullopt);
    };

    using type_vec_t = std::vector<TensorTypePtr>;
    // Formula is expected to return a vector of length equal to the number of
    // tensor outputs of the node, or an empty vector which implies that it
    // failed to propagate.
    using formula_t = std::function<type_vec_t(Node*)>;
    static std::mutex shape_formulas_mutex;
    static std::vector<std::pair<OperatorSet, formula_t>> shape_formulas;
    struct register_formula_for {
      register_formula_for(OperatorSet operators, formula_t formula) {
        std::unique_lock<std::mutex> lock{shape_formulas_mutex};
        shape_formulas.emplace_back(std::move(operators), std::move(formula));
      }
    };

    // Requirements:
    //   dims           : preserved
    //   scalar type    : preserved
    //   device         : preserved
    //   tensor inputs  : 1
    //   tensor outputs : 1
    // Additionally:
    //   - First input should be the only tensor input
    static const register_formula_for simple_unary_ops{
        {
            "aten::acos(Tensor self) -> Tensor",
            "aten::neg(Tensor self) -> Tensor",
            "aten::t(Tensor self) -> Tensor",
            "aten::sigmoid(Tensor self) -> Tensor",
            "aten::logit(Tensor self, float? eps=None) -> Tensor",
            "aten::tanh(Tensor self) -> Tensor",
            "aten::relu(Tensor self) -> Tensor",
            "aten::asin(Tensor self) -> Tensor",
            "aten::atan(Tensor self) -> Tensor",
            "aten::ceil(Tensor self) -> Tensor",
            "aten::clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor",
            "aten::contiguous(Tensor(a) self, *, MemoryFormat memory_format=contiguous_format) -> Tensor(a)",
            "aten::bernoulli(Tensor self, *, Generator? generator) -> Tensor",
            "aten::celu(Tensor self, Scalar alpha) -> Tensor",
            "aten::clamp(Tensor self, Scalar? min, Scalar? max) -> Tensor",
            "aten::clamp_max(Tensor self, Scalar max) -> Tensor",
            "aten::clamp_min(Tensor self, Scalar min) -> Tensor",
            "aten::alpha_dropout(Tensor input, float p, bool train) -> Tensor",
            "aten::bernoulli(Tensor self, float p, *, Generator? generator) -> Tensor",
            "aten::cos(Tensor self) -> Tensor",
            "aten::cosh(Tensor self) -> Tensor",
            "aten::digamma(Tensor self) -> Tensor",
            "aten::dropout(Tensor input, float p, bool train) -> Tensor",
            "aten::elu(Tensor self, Scalar alpha, Scalar scale, Scalar input_scale) -> Tensor",
            "aten::erf(Tensor self) -> Tensor",
            "aten::erfc(Tensor self) -> Tensor",
            "aten::erfinv(Tensor self) -> Tensor",
            "aten::exp(Tensor self) -> Tensor",
            "aten::expm1(Tensor self) -> Tensor",
            "aten::log(Tensor self) -> Tensor",
            "aten::log10(Tensor self) -> Tensor",
            "aten::log1p(Tensor self) -> Tensor",
            "aten::log2(Tensor self) -> Tensor",
            "aten::log_sigmoid(Tensor self) -> Tensor",
            "aten::floor(Tensor self) -> Tensor",
            "aten::frac(Tensor self) -> Tensor",
            "aten::flip(Tensor self, int[] dims) -> Tensor",
            "aten::feature_alpha_dropout(Tensor input, float p, bool train) -> Tensor",
            "aten::feature_dropout(Tensor input, float p, bool train) -> Tensor",
            "aten::hardshrink(Tensor self, Scalar lambd) -> Tensor",
            "aten::hardtanh(Tensor self, Scalar min_val, Scalar max_val) -> Tensor",
            "aten::glu(Tensor self, int dim) -> Tensor",
            "aten::inverse(Tensor self) -> Tensor",
            "aten::leaky_relu(Tensor self, Scalar negative_slope) -> Tensor",
            "aten::lgamma(Tensor self) -> Tensor",
            "aten::mvlgamma(Tensor self, int p) -> Tensor",
            "aten::normal(float mean, Tensor std, *, Generator? generator) -> Tensor",
            "aten::normal(Tensor mean, float std, *, Generator? generator) -> Tensor",
            "aten::permute(Tensor self, int[] dims) -> Tensor",
            "aten::pin_memory(Tensor(a) self) -> Tensor(a)",
            "aten::pinverse(Tensor self, float rcond) -> Tensor",
            "aten::reciprocal(Tensor self) -> Tensor",
            "aten::relu(Tensor self) -> Tensor",
            "aten::round(Tensor self) -> Tensor",
            "aten::rrelu(Tensor self, Scalar lower, Scalar upper, bool training, Generator? generator) -> Tensor",
            "aten::rsqrt(Tensor self) -> Tensor",
            "aten::selu(Tensor self) -> Tensor",
            "aten::gelu(Tensor self) -> Tensor",
            "aten::sigmoid(Tensor self) -> Tensor",
            "aten::sign(Tensor self) -> Tensor",
            "aten::sin(Tensor self) -> Tensor",
            "aten::sinh(Tensor self) -> Tensor",
            "aten::softplus(Tensor self, Scalar beta, Scalar threshold) -> Tensor",
            "aten::softshrink(Tensor self, Scalar lambd) -> Tensor",
            "aten::sqrt(Tensor self) -> Tensor",
            "aten::tan(Tensor self) -> Tensor",
            "aten::tanh(Tensor self) -> Tensor",
            "aten::threshold(Tensor self, Scalar threshold, Scalar value) -> Tensor",
            "aten::transpose(Tensor self, int dim0, int dim1) -> Tensor",
            "aten::tril(Tensor self, int diagonal) -> Tensor",
            "aten::triu(Tensor self, int diagonal) -> Tensor",
            "aten::trunc(Tensor self) -> Tensor",
            "aten::rot90(Tensor self, int k, int[] dims) -> Tensor",
            "aten::narrow(Tensor self, int dim, int start, int length) -> Tensor",
            "aten::slice(Tensor self, int dim, int? start=None, int? end=None, int step=1) -> Tensor",
            "aten::alias(Tensor self) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          auto input_type = node->input(0)->type()->cast<TensorType>();
          return input_type ? type_vec_t{input_type->dimensionedOnly()}
                            : type_vec_t{};
        }};

    // Requirements:
    //   dims           : preserved
    //   scalar type    : preserved, except complex maps to float
    //   device         : preserved
    //   tensor inputs  : 1
    //   tensor outputs : 1
    // Additionally:
    //   - First input should be the only tensor input
    static const register_formula_for simple_unary_ops_complex_to_float{
        {
            "aten::abs(Tensor self) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          auto input_type = node->input(0)->type()->cast<TensorType>();

          // Maps complex -> float
          if (input_type->scalarType()) {
            const auto scalar_type = *(input_type->scalarType());
            if (isComplexType(scalar_type)) {
              const auto out_type = c10::toValueType(scalar_type);
              return type_vec_t{
                  input_type->dimensionedOnly()->withScalarType(out_type)};
            }
          }

          return input_type ? type_vec_t{input_type->dimensionedOnly()}
                            : type_vec_t{};
        }};

    // Requirements:
    //   dims           : broadcast all tensor args
    //   scalar type    : promoted from input dtypes
    //   device         : always matching and preserved
    //   tensor inputs  : *
    //   tensor outputs : 1
    static const register_formula_for broadcasting_ops_arithmetic{
        {
            // Tensor-Tensor operators
            "aten::add(Tensor self, Tensor other, *, Scalar alpha) -> Tensor",
            "aten::sub(Tensor self, Tensor other, *, Scalar alpha) -> Tensor",
            "aten::mul(Tensor self, Tensor other) -> Tensor",
            "aten::div(Tensor self, Tensor other) -> Tensor",
        },
        [this](Node* node) -> type_vec_t {
          if (auto maybe_tensor_types = gatherTensorTypes(node)) {
            AT_ASSERT(maybe_tensor_types->size() >= 2);
            auto dtype = getPromotedTypeForArithmeticOp(node);
            return {broadcast(*maybe_tensor_types, dtype)};
          }
          return {};
        }};

    // Requirements:
    //   dims           : broadcast all tensor args
    //   scalar type    : always matching and preserved
    //   device         : always matching and preserved
    //   tensor inputs  : *
    //   tensor outputs : 1
    static const register_formula_for broadcasting_ops{
        {
            "aten::pow(Tensor self, Tensor exponent) -> Tensor",
            "aten::fmod(Tensor self, Tensor other) -> Tensor",
            "aten::remainder(Tensor self, Tensor other) -> Tensor",
            "aten::lerp(Tensor self, Tensor end, Scalar weight) -> Tensor",
            "aten::lerp(Tensor self, Tensor end, Tensor weight) -> Tensor",
            "aten::max(Tensor self, Tensor other) -> Tensor",
            "aten::min(Tensor self, Tensor other) -> Tensor",
            "aten::__and__(Tensor self, Tensor other) -> Tensor",
            "aten::__or__(Tensor self, Tensor other) -> Tensor",
            "aten::__xor__(Tensor self, Tensor other) -> Tensor",
            "aten::__lshift__(Tensor self, Tensor other) -> Tensor",
            "aten::__rshift__(Tensor self, Tensor other) -> Tensor",
            "aten::__iand__(Tensor self, Tensor other) -> Tensor",
            "aten::__ior__(Tensor self, Tensor other) -> Tensor",
            "aten::__ixor__(Tensor self, Tensor other) -> Tensor",
            "aten::__ilshift__(Tensor self, Tensor other) -> Tensor",
            "aten::__irshift__(Tensor self, Tensor other) -> Tensor",

            // Ops with Tensor-Tensor overloads only
            "aten::atan2(Tensor self, Tensor other) -> Tensor",
        },
        [this](Node* node) -> type_vec_t {
          if (auto maybe_tensor_types = gatherTensorTypes(node)) {
            AT_ASSERT(maybe_tensor_types->size() >= 2);
            auto first_scalar_type = (*maybe_tensor_types)[0]->scalarType();
            auto second_scalar_type = (*maybe_tensor_types)[1]->scalarType();
            if (!first_scalar_type || !second_scalar_type) {
              return {};
            }
            size_t arg_for_type = 0;
            if (c10::promoteTypes(*first_scalar_type, *second_scalar_type) !=
                first_scalar_type) {
              arg_for_type = 1;
            }
            auto t = (*maybe_tensor_types)[arg_for_type]->scalarType();
            return {broadcast(*maybe_tensor_types, *t)};
          }
          return {};
        }};

    static const register_formula_for fused_accum_binary_ops{
        {
            // Non-binary ops
            "aten::addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value) -> Tensor",
            "aten::addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value) -> Tensor",
        },
        [this](Node* node) -> type_vec_t {
          if (auto maybe_tensor_types = gatherTensorTypes(node)) {
            auto dtype = (*maybe_tensor_types)[0]->scalarType();
            if (!dtype) {
              return {};
            }
            return {broadcast(*maybe_tensor_types, *dtype)};
          }
          return {};
        }};

    static const register_formula_for broadcasting_tensor_scalar_ops_arithmetic{
        {
            // Tensor-Scalar operators
            "aten::add(Tensor self, Scalar other, Scalar alpha) -> Tensor",
            "aten::sub(Tensor self, Scalar other, Scalar alpha) -> Tensor",
            "aten::mul(Tensor self, Scalar other) -> Tensor",
            "aten::div(Tensor self, Scalar other) -> Tensor",
        },
        [this](Node* node) -> type_vec_t {
          if (auto maybe_tensor_types = gatherTensorTypes(node)) {
            auto first_scalar_type = (*maybe_tensor_types)[0]->scalarType();
            auto second_scalar_type =
                tryScalarTypeFromJitType(node->inputs()[1]->type());
            if (!first_scalar_type || !second_scalar_type) {
              return {};
            }
            if (isIntegralType(*first_scalar_type, false) &&
                isFloatingType(*second_scalar_type)) {
              auto default_dtype =
                  at::typeMetaToScalarType(caffe2::get_default_dtype());
              return {broadcast(*maybe_tensor_types, default_dtype)};
            }
            if (c10::ScalarType::Bool == *first_scalar_type &&
                c10::ScalarType::Bool != *second_scalar_type) {
              auto result_type =
                  c10::promoteTypes(*first_scalar_type, *second_scalar_type);
              return {broadcast(*maybe_tensor_types, result_type)};
            }
            return {broadcast(*maybe_tensor_types, first_scalar_type)};
          }
          return {};
        }};

    // NB: we always take the scalar type of the Tensor
    static const register_formula_for broadcasting_tensor_scalar_ops{
        {

            "aten::pow(Tensor self, Scalar exponent) -> Tensor",
            "aten::fmod(Tensor self, Scalar other) -> Tensor",
            "aten::remainder(Tensor self, Scalar other) -> Tensor",
            "aten::pow(Scalar self, Tensor exponent) -> Tensor",
            "aten::__and__(Tensor self, Scalar other) -> Tensor",
            "aten::__or__(Tensor self, Scalar other) -> Tensor",
            "aten::__xor__(Tensor self, Scalar other) -> Tensor",
            "aten::__lshift__(Tensor self, Scalar other) -> Tensor",
            "aten::__rshift__(Tensor self, Scalar other) -> Tensor",
            "aten::__iand__(Tensor self, Scalar other) -> Tensor",
            "aten::__ior__(Tensor self, Scalar other) -> Tensor",
            "aten::__ixor__(Tensor self, Scalar other) -> Tensor",
            "aten::__ilshift__(Tensor self, Scalar other) -> Tensor",
            "aten::__irshift__(Tensor self, Scalar other) -> Tensor",
        },
        [this](Node* node) -> type_vec_t {
          if (auto maybe_tensor_types = gatherTensorTypes(node)) {
            return {broadcast(
                *maybe_tensor_types, (*maybe_tensor_types)[0]->scalarType())};
          }
          return {};
        }};

    // aten::where is special in that its return type is the second argument's
    // (self) type rather than the that of condition
    static const register_formula_for where_op{
        {
            "aten::where(Tensor condition, Tensor self, Tensor other) -> Tensor",
        },
        [this](Node* node) -> type_vec_t {
          if (auto maybe_tensor_types = gatherTensorTypes(node)) {
            return {broadcast(
                *maybe_tensor_types, (*maybe_tensor_types)[1]->scalarType())};
          }
          return {};
        }};

    static const auto any_tensor_type = [](Node* node) -> TensorTypePtr {
      for (Value* input : node->inputs()) {
        if (auto type = input->type()->cast<TensorType>()) {
          if (type->dim().has_value()) {
            return type;
          }
        }
      }
      return nullptr;
    };

    // Requirements:
    //   dims           : always matching and preserved
    //   scalar type    : always matching and preserved
    //   device         : always matching and preserved
    //   tensor inputs  : 2
    //   tensor outputs : 1
    static const register_formula_for binary_ops_strict_match{
        {
            "aten::normal(Tensor mean, Tensor std, *, Generator? generator) -> Tensor",
            "aten::mm(Tensor self, Tensor mat2) -> Tensor",
            "aten::bmm(Tensor self, Tensor mat2) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          if (auto type = any_tensor_type(node)) {
            return {type};
          }
          return {};
        }};

    // Requirements:
    //   dims           : all tensor args are broadcast
    //   scalar type    : byte/uint8
    //   device         : always matching and preserved
    //   tensor inputs  : *
    //   tensor outputs : 1
    static const register_formula_for comparison_ops{
        {
            "aten::lt(Tensor self, Tensor other) -> Tensor",
            "aten::le(Tensor self, Tensor other) -> Tensor",
            "aten::gt(Tensor self, Tensor other) -> Tensor",
            "aten::ge(Tensor self, Tensor other) -> Tensor",
            "aten::eq(Tensor self, Tensor other) -> Tensor",
            "aten::ne(Tensor self, Tensor other) -> Tensor",
            "aten::lt(Tensor self, Scalar other) -> Tensor",
            "aten::le(Tensor self, Scalar other) -> Tensor",
            "aten::gt(Tensor self, Scalar other) -> Tensor",
            "aten::ge(Tensor self, Scalar other) -> Tensor",
            "aten::eq(Tensor self, Scalar other) -> Tensor",
            "aten::ne(Tensor self, Scalar other) -> Tensor",
        },
        [this](Node* node) -> type_vec_t {
          if (auto maybe_tensor_types = gatherTensorTypes(node)) {
            return {broadcast(*maybe_tensor_types, at::kBool)};
          }
          return {};
        }};

    // Requirements:
    //   dims           : preserved from the first argument
    //   scalar type    : preserved from the first argument (doesn't have to
    //   match other arguments) device         : always matching and preserved
    //   tensor inputs  : *
    //   tensor outputs : 1
    // NB: those ops (with slight adjustments) are good candidates for restarts.
    //     Knowing the type and device of weights or biases is usually enough to
    //     infer the output type.
    static const register_formula_for nn_ops_first_input_preserving{
        {
            "aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor",
            "aten::conv1d(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor",
            "aten::conv2d(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor",
            "aten::conv3d(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor",
            "aten::conv_tbc(Tensor self, Tensor weight, Tensor bias, int pad) -> Tensor",
            "aten::conv_transpose1d(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] output_padding, int groups, int[] dilation) -> Tensor",
            "aten::conv_transpose2d(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] output_padding, int groups, int[] dilation) -> Tensor",
            "aten::conv_transpose3d(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] output_padding, int groups, int[] dilation) -> Tensor",
            "aten::convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups) -> Tensor",
            "aten::_convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled) -> Tensor", // deprecated _convolution
            "aten::_convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32) -> Tensor",
            "aten::adaptive_avg_pool1d(Tensor self, int[] output_size) -> Tensor",
            "aten::adaptive_avg_pool2d(Tensor self, int[] output_size) -> Tensor",
            "aten::adaptive_avg_pool3d(Tensor self, int[] output_size) -> Tensor",
            "aten::avg_pool1d(Tensor self, int[] kernel_size, int[] stride, int[] padding, bool ceil_mode, bool count_include_pad) -> Tensor",
            "aten::avg_pool2d(Tensor self, int[] kernel_size, int[] stride, int[] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor",
            "aten::avg_pool3d(Tensor self, int[] kernel_size, int[] stride, int[] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor",
            "aten::max_pool1d(Tensor self, int[] kernel_size, int[] stride, int[] padding, int[] dilation, bool ceil_mode) -> Tensor",
            "aten::max_pool2d(Tensor self, int[] kernel_size, int[] stride, int[] padding, int[] dilation, bool ceil_mode) -> Tensor",
            "aten::max_pool3d(Tensor self, int[] kernel_size, int[] stride, int[] padding, int[] dilation, bool ceil_mode) -> Tensor",
            "aten::max_unpool2d(Tensor self, Tensor indices, int[] output_size) -> Tensor",
            "aten::max_unpool3d(Tensor self, Tensor indices, int[] output_size, int[] stride, int[] padding) -> Tensor",
            "aten::reflection_pad1d(Tensor self, int[] padding) -> Tensor",
            "aten::reflection_pad2d(Tensor self, int[] padding) -> Tensor",
            "aten::replication_pad1d(Tensor self, int[] padding) -> Tensor",
            "aten::replication_pad2d(Tensor self, int[] padding) -> Tensor",
            "aten::replication_pad3d(Tensor self, int[] padding) -> Tensor",
            "aten::upsample_bilinear2d(Tensor self, int[] output_size, bool align_corners, float? scales_h, float? scales_w) -> Tensor",
            "aten::upsample_linear1d(Tensor self, int[] output_size, bool align_corners, float? scales) -> Tensor",
            "aten::upsample_nearest1d(Tensor self, int[] output_size, float? scales) -> Tensor",
            "aten::upsample_nearest2d(Tensor self, int[] output_size, float? scales_h, float? scales_w) -> Tensor",
            "aten::upsample_nearest3d(Tensor self, int[] output_size, float? scales_d, float? scales_h, float? scales_w) -> Tensor",
            "aten::upsample_trilinear3d(Tensor self, int[] output_size, bool align_corners, float? scales_d, float? scales_h, float? scales_w) -> Tensor",
            "aten::prelu(Tensor self, Tensor weight) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          if (auto type = node->input(0)->type()->cast<TensorType>()) {
            return {type->dimensionedOnly()};
          }
          return {};
        }};

    // Requirements:
    //   dims           : 0
    //   scalar type    : preserved
    //   device         : preserved
    //   tensor inputs  : 1
    //   tensor outputs : 1
    // Additionally:
    //   - First input should be the only tensor input
    static const register_formula_for all_reduce_ops{
        {
            "aten::det(Tensor self) -> Tensor",
            "aten::logdet(Tensor self) -> Tensor",
            "aten::max(Tensor self) -> Tensor",
            "aten::min(Tensor self) -> Tensor",
            "aten::median(Tensor self) -> Tensor",
            "aten::nanmedian(Tensor self) -> Tensor",
            "aten::norm(Tensor self, Scalar p) -> Tensor",
            "aten::std(Tensor self, bool unbiased) -> Tensor",
            "aten::trace(Tensor self) -> Tensor",
            "aten::var(Tensor self, bool unbiased) -> Tensor",
            "aten::all(Tensor self) -> Tensor",
            "aten::any(Tensor self) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          if (auto type = node->input(0)->type()->cast<TensorType>()) {
            return {type->withDim(0)};
          }
          return {};
        }};

    // Requirements:
    //   dims           : 0
    //   scalar type    : dtype if specified, else preserved
    //   device         : preserved
    //   tensor inputs  : 1
    //   tensor outputs : 1
    // Additionally:
    //   - First input should be the only tensor input
    static const register_formula_for reduce_ops_with_opt_dtype{
        {"aten::mean(Tensor self, *, int? dtype) -> Tensor"},
        [](Node* node) -> type_vec_t {
          at::optional<IValue> maybe_dtype_option = node->get(attr::dtype);
          if (auto type = node->input(0)->type()->cast<TensorType>()) {
            auto ret = type->withDim(0);
            if (maybe_dtype_option && !maybe_dtype_option->isNone()) {
              return {ret->withScalarType(maybe_dtype_option->toScalarType())};
            } else {
              return {ret};
            }
          }
          return {};
        }};

    // Requirements:
    //   dims           : 0
    //   scalar type    : dtype if specified, else preserved if floating point,
    //   otherwise long/int64 device         : preserved tensor inputs  : 1
    //   tensor outputs : 1
    // Additionally:
    //   - First input should be the only tensor input
    static const register_formula_for
        all_reduce_ops_with_integer_upcast_and_dtype{
            {
                "aten::sum(Tensor self, *, int? dtype) -> Tensor",
                "aten::prod(Tensor self, *, int? dtype) -> Tensor",
            },
            [](Node* node) -> type_vec_t {
              if (auto type = node->input(0)->type()->cast<TensorType>()) {
                type = type->withDim(0);
                at::optional<IValue> maybe_dtype_option =
                    node->get(attr::dtype);
                if (maybe_dtype_option && !maybe_dtype_option->isNone()) {
                  return {
                      type->withScalarType(maybe_dtype_option->toScalarType())};
                }
                if (type->scalarType()) {
                  return {
                      at::isFloatingType(*type->scalarType())
                          ? type
                          : type->withScalarType(at::kLong)};
                } else {
                  return {type};
                }
              }
              return {};
            }};

    static const auto reduce_op_handler = [](Node* node,
                                             int64_t num_reduced_dim = 0,
                                             bool upcast_integer = false,
                                             c10::optional<IValue> opt_dtype =
                                                 c10::nullopt) -> type_vec_t {
      if (auto type = node->input(0)->type()->cast<TensorType>()) {
        if (!type->scalarType() || !type->dim()) {
          return {};
        }
        if (opt_dtype && !opt_dtype->isNone()) {
          type = type->withScalarType(opt_dtype->toScalarType());
        } else if (upcast_integer && !at::isFloatingType(*type->scalarType())) {
          type = type->withScalarType(at::kLong);
        }
        // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
        if (*type->dim() >= num_reduced_dim && num_reduced_dim > 0) {
          return {type->withDim(*type->dim() - num_reduced_dim)};
        } else {
          return {type};
        }
      }
      return {};
    };

    static const auto multidim_reduce_with_keepdim =
        [](Node* node,
           int64_t num_reduced_dim,
           bool upcast_integer) -> type_vec_t {
      auto maybe_keepdim = node->get<bool>(attr::keepdim);
      if (!maybe_keepdim)
        return {};
      return reduce_op_handler(
          node, *maybe_keepdim ? 0 : num_reduced_dim, upcast_integer);
    };

    // Requirements:
    //   dims           : 0 if dim is None, otherwise preserved if keepdim ==
    //   false or 1 smaller otherwise scalar type    : preserved device :
    //   preserved tensor inputs  : 1 tensor outputs : 1
    // Additionally:
    //   - First input should be the only tensor input
    //   - Has a bool keepdim argument
    static const register_formula_for argminmax{
        {
            "aten::argmax(Tensor self, int? dim, bool keepdim) -> Tensor",
            "aten::argmin(Tensor self, int? dim, bool keepdim) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          if (auto type = node->input(0)->type()->cast<TensorType>()) {
            if (node->input(1)->type()->kind() == c10::TypeKind::NoneType) {
              return {type->withDim(0)};
            } else {
              return multidim_reduce_with_keepdim(
                  node, /*num_reduced_dim=*/1, /*upcast_integer=*/false);
            }
          }
          return {};
        }};

    // Requirements:
    //   dims           : preserved if keepdim == false, 1 smaller otherwise
    //   scalar type    : preserved for first output, byte/uint8 for second
    //   output if exists device         : preserved tensor inputs  : 1 tensor
    //   outputs : 1 or 2
    // Additionally:
    //   - First input should be the only tensor input
    //   - Has a bool keepdim argument
    static const register_formula_for dim_reduce_ops{
        {
            "aten::all(Tensor self, int dim, bool keepdim) -> Tensor",
            "aten::any(Tensor self, int dim, bool keepdim) -> Tensor",

            // Ops returning indices as second output
            "aten::kthvalue(Tensor self, int k, int dim, bool keepdim) -> (Tensor, Tensor)",
            "aten::max(Tensor self, int dim, bool keepdim) -> (Tensor, Tensor)",
            "aten::min(Tensor self, int dim, bool keepdim) -> (Tensor, Tensor)",
            "aten::median(Tensor self, int dim, bool keepdim) -> (Tensor, Tensor)",
            "aten::nanmedian(Tensor self, int dim, bool keepdim) -> (Tensor, Tensor)",
            "aten::mode(Tensor self, int dim, bool keepdim) -> (Tensor, Tensor)",
        },
        [](Node* node) -> type_vec_t {
          // NB: Note that while this function is generally meant to be used
          // with ops that have a single output, we will fix up its return right
          // below.
          auto output_types = multidim_reduce_with_keepdim(
              node, /*num_reduced_dim=*/1, /*upcast_integer=*/false);
          if (!output_types.empty() && node->outputs().size() == 2) {
            output_types.push_back(
                output_types.back()->withScalarType(at::kLong));
          }
          return output_types;
        }};

    // Requirements:
    //   dims           : preserved if keepdim == false, 1 smaller otherwise
    //   scalar type    : dtype if specified. preserved if floating point,
    //   otherwise long/int64 device         : preserved tensor inputs  : 1
    //   tensor outputs : 1
    // Additionally:
    //   - First input should be the only tensor input
    //   - has a bool keepdim argument
    static const register_formula_for dim_reduce_ops_with_integer_upcast{
        {
            "aten::prod(Tensor self, int dim, bool keepdim, *, int? dtype) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          auto maybe_keepdim = node->get<bool>(attr::keepdim);
          at::optional<IValue> opt_dtype = node->get(attr::dtype);
          return reduce_op_handler(
              node,
              /*num_reduce_dim=*/*maybe_keepdim ? 0 : 1,
              /*integer_upcast=*/true,
              opt_dtype);
        }};

    // Requirements:
    //   dims           : preserved
    //   scalar type    : dtype if specified, preserved if floating point,
    //    otherwise long/int64
    //   device         : preserved
    //   tensor inputs  : 1
    //   tensor outputs : 1
    // Additionally:
    //   - First input should be the only tensor input
    static const register_formula_for dim_reduce_ops_dtype{
        {"aten::cumprod(Tensor self, int dim, *, int? dtype) -> Tensor",
         "aten::cumsum(Tensor self, int dim, *, int? dtype) -> Tensor",
         "aten::log_softmax(Tensor self, int dim, int? dtype) -> Tensor"},
        [](Node* node) -> type_vec_t {
          at::optional<IValue> opt_dtype = node->get(attr::dtype);
          return reduce_op_handler(
              node, /*num_reduce_dim=*/0, /*integer_upcast=*/true, opt_dtype);
        }};

    // Requirements:
    //   dims           : preserved
    //   scalar type    : dtype if specified, otherwise preserved
    //   device         : preserved
    //   tensor inputs  : 1
    //   tensor outputs : 1
    // Additionally:
    //   - has bool keepdim and int[] dim arguments
    static const register_formula_for register_softmax{
        {"aten::softmax(Tensor self, int dim, int? dtype) -> Tensor"},
        [](Node* node) -> type_vec_t {
          at::optional<IValue> opt_dtype = node->get(attr::dtype);
          return reduce_op_handler(
              node, /*num_reduced_dim=*/0, /*upcast_integer=*/false, opt_dtype);
        }};

    static const auto factory_with_ndim = [](Node* node,
                                             int dim) -> type_vec_t {
      at::optional<IValue> maybe_layout_option = node->get(attr::layout);
      if (!maybe_layout_option)
        return {};

      at::optional<IValue> maybe_device_option = node->get(attr::device);
      if (!maybe_device_option)
        return {};
      auto device =
          (maybe_device_option->isNone() ? at::kCPU
                                         : maybe_device_option->toDevice());

      at::optional<IValue> maybe_dtype_option = node->get(attr::dtype);
      if (!maybe_dtype_option)
        return {};
      auto dtype =
          (maybe_dtype_option->isNone() ? at::kDouble
                                        : maybe_dtype_option->toScalarType());

      return {TensorType::create(
          dtype, device, dim, /*requires_grad=*/c10::nullopt)};
    };

    static const auto factory_like_with_ndim = [](Node* node,
                                                  int dim) -> type_vec_t {
      auto tt = node->input(0)->type()->expect<TensorType>();
      auto in_type = tt->scalarType();
      auto in_dev = tt->device();

      at::optional<IValue> maybe_layout_option = node->get(attr::layout);
      if (!maybe_layout_option)
        return {};

      at::optional<IValue> maybe_device_option = node->get(attr::device);
      if (!maybe_device_option)
        return {};

      if (!maybe_device_option->isNone()) {
        in_dev = maybe_device_option->toDevice();
      }

      at::optional<IValue> maybe_dtype_option = node->get(attr::dtype);
      if (!maybe_dtype_option)
        return {};

      if (!maybe_dtype_option->isNone()) {
        in_type = maybe_dtype_option->toScalarType();
      }

      return {TensorType::create(
          in_type, in_dev, dim, /*requires_grad=*/c10::nullopt)};
    };

    // Requirements:
    //   dims           : preserved
    //   scalar type    : equal to value of dtype
    //   device         : equal to value of device
    //   tensor inputs  : 1
    //   tensor outputs : 1
    // Additionally:
    //   - has ScalarType dtype, Layeout layout and Device device arguments
    static const register_formula_for like_factories_with_options{
        {
            "aten::empty_like(Tensor self, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
            "aten::full_like(Tensor self, Scalar fill_value, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
            "aten::ones_like(Tensor self, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
            "aten::rand_like(Tensor self, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
            "aten::randint_like(Tensor self, int high, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
            "aten::randint_like(Tensor self, int low, int high, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
            "aten::randn_like(Tensor self, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
            "aten::zeros_like(Tensor self, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          if (auto type =
                  node->namedInput(attr::self)->type()->cast<TensorType>()) {
            if (type->dim()) {
              return factory_like_with_ndim(node, *type->dim());
            }
          }
          return {};
        }};

    // Requirements:
    //   dims           : equal to number of elements in size
    //   scalar type    : equal to value of dtype
    //   device         : equal to value of device
    //   tensor inputs  : 1
    //   tensor outputs : 1
    // Additionally:
    //   - has int[] size, ScalarType dtype, Layeout layout and Device device
    //   arguments
    static const register_formula_for size_factories_with_options{
        {
            "aten::empty(int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory, MemoryFormat? memory_format=contiguous_format) -> Tensor",
            "aten::full(int[] size, Scalar fill_value, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
            "aten::ones(int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
            "aten::rand(int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
            "aten::randn(int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
            "aten::zeros(int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
            "aten::randint(int high, int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
            "aten::randint(int low, int high, int[] size, *, int? dtype, int? layout, Device? device, bool? pin_memory) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          if (auto maybe_size = node->get<c10::List<int64_t>>(attr::size)) {
            return factory_with_ndim(node, maybe_size->size());
          }
          return {};
        }};

    static const auto get_cast_scalar_type = [](Node* node) -> at::ScalarType {
      switch (node->kind()) {
        case aten::_cast_Byte:
          return at::kByte;
        case aten::_cast_Char:
          return at::kChar;
        case aten::_cast_Double:
          return at::kDouble;
        case aten::_cast_Float:
          return at::kFloat;
        case aten::_cast_Half:
          return at::kHalf;
        case aten::_cast_Int:
          return at::kInt;
        case aten::_cast_Long:
          return at::kLong;
        case aten::_cast_Short:
          return at::kShort;
        default:
          AT_ASSERTM(
              false,
              "unknown node kind in get_cast_scalar_type: ",
              node->kind().toQualString());
      }
    };
    static const register_formula_for cast_ops{
        {
            "aten::_cast_Byte(Tensor self, bool non_blocking) -> Tensor",
            "aten::_cast_Char(Tensor self, bool non_blocking) -> Tensor",
            "aten::_cast_Double(Tensor self, bool non_blocking) -> Tensor",
            "aten::_cast_Float(Tensor self, bool non_blocking) -> Tensor",
            "aten::_cast_Half(Tensor self, bool non_blocking) -> Tensor",
            "aten::_cast_Int(Tensor self, bool non_blocking) -> Tensor",
            "aten::_cast_Long(Tensor self, bool non_blocking) -> Tensor",
            "aten::_cast_Short(Tensor self, bool non_blocking) -> Tensor",
        },
        [](Node* node) -> type_vec_t {
          if (auto type =
                  node->namedInput(attr::self)->type()->cast<TensorType>()) {
            return {type->withScalarType(get_cast_scalar_type(node))};
          }
          return {};
        }};

    // First, try to match one of the registered formulas to their operator
    // sets.
    for (auto& entry : shape_formulas) {
      if (node->isMemberOf(entry.first)) {
        auto types = entry.second(node);
        if (types.empty()) {
          return false;
        } else {
          auto outputs = node->outputs();
          AT_ASSERT(types.size() == outputs.size());
          for (size_t i = 0; i < types.size(); ++i) {
            AT_ASSERT(outputs[i]->type()->isSubtypeOf(TensorType::get()));
            outputs[i]->setType(types[i]);
          }
          return true;
        }
      }
    }

    // This section implements shape prop for an assorted set of nodes that only
    // need partial information about their input types.
    const auto input_type = [node](size_t index) {
      auto result = node->input(index)->type()->cast<TensorType>();
      if (result) {
        result = result->dimensionedOnly();
      }
      return result;
    };
    if (node->matches(
            "aten::masked_select(Tensor self, Tensor mask) -> Tensor")) {
      if (auto type = input_type(0)) {
        node->output()->setType(type->withDim(1));
        return true;
      }
    } else if (node->matches("aten::detach(Tensor(a) self) -> Tensor(a)")) {
      if (auto type = input_type(0)) {
        node->output()->setType(type->withRequiresGrad(false));
        return true;
      }
    } else if (
        node->matches(
            "aten::batch_norm_stats(Tensor input, float eps) -> (Tensor, Tensor)")) {
      if (auto type = input_type(0)) {
        if (type->scalarType() == at::kHalf) {
          type = type->withScalarType(at::kFloat);
        }
        type = type->withDim(1);
        node->outputs()[0]->setType(type);
        node->outputs()[1]->setType(type);
        return true;
      }
    } else if (node->matches(
                   "aten::dot(Tensor self, Tensor tensor) -> Tensor")) {
      if (auto type = any_tensor_type(node)) {
        node->output()->setType(type->withDim(0));
        return true;
      }
    } else if (
        node->matches("aten::mv(Tensor self, Tensor vec) -> Tensor") ||
        node->matches(
            "aten::addmv(Tensor self, Tensor mat, Tensor vec, *, Scalar beta, Scalar alpha) -> Tensor")) {
      if (auto type = any_tensor_type(node)) {
        node->output()->setType(type->withDim(1));
        return true;
      }
    } else if (
        node->matches(
            "aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta, Scalar alpha) -> Tensor") ||
        node->matches(
            "aten::addbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta, Scalar alpha) -> Tensor") ||
        node->matches(
            "aten::addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta, Scalar alpha) -> Tensor")) {
      if (auto type = any_tensor_type(node)) {
        node->output()->setType(type->withDim(2));
        return true;
      }
    } else if (
        node->matches(
            "aten::baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta, Scalar alpha) -> Tensor")) {
      if (auto type = any_tensor_type(node)) {
        node->output()->setType(type->withDim(3));
        return true;
      }
    } else if (
        node->matches(
            "aten::index_select(Tensor self, int dim, Tensor index) -> Tensor")) {
      auto type = input_type(0);
      auto index_type = input_type(1);
      // index_select behaves very weirdly when self.dim() == 0. It allows both
      // 0D and 1D indices, and returns a value that has as many dimensions as
      // index.
      if (type && index_type && type->dim()) {
        if (*type->dim() == 0) {
          node->output()->setType(type->withDim(index_type->dim()));
        } else {
          node->output()->setType(type);
        }
        return true;
      }
    } else if (
        node->matches(
            "aten::gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor")) {
      auto type = input_type(0);
      auto index_type = input_type(1);
      // Gather has this annoying edge case where index always needs to match
      // the number of dims of self, **except** when self is 1D and index is 0D
      // in which case we return a 0D output.
      if (type && index_type && index_type->dim()) {
        if (*index_type->dim() == 0) {
          node->output()->setType(type->withDim(0));
        } else {
          node->output()->setType(type);
        }
        return true;
      }
    } else if (
        node->matches(
            "aten::embedding(Tensor weight, Tensor indices, int padding_idx, bool scale_grad_by_freq, bool sparse) -> Tensor")) {
      auto weight_type = input_type(0);
      auto indices_type = input_type(1);
      if (weight_type && indices_type && indices_type->dim()) {
        node->output()->setType(weight_type->withDim(*indices_type->dim() + 1));
        return true;
      }
    } else if (
        node->matches(
            "aten::bilinear(Tensor input1, Tensor input2, Tensor weight, Tensor? bias) -> Tensor")) {
      if (auto type = input_type(0)) {
        node->output()->setType(type);
        return true;
      }
      if (auto type = input_type(1)) {
        node->output()->setType(type);
        return true;
      }
    } else if (
        node->matches(
            "aten::dist(Tensor self, Tensor other, Scalar p) -> Tensor")) {
      if (auto type = any_tensor_type(node)) {
        node->output()->setType(type->withDim(0));
        return true;
      }
    }

    // The code below implements formulas that need type information for all
    // their tensor inputs, and have exactly one output.
    std::vector<TensorTypePtr> tensor_types;
    static const auto reshape_prop =
        [](Node* node,
           Symbol shape_input,
           const std::vector<TensorTypePtr>& tensor_types) -> TensorTypePtr {
      if (auto list_size = determineListSize(node->namedInput(shape_input))) {
        return tensor_types.at(0)->withDim(*list_size);
      }
      return nullptr;
    };
    const auto getSingleOutputType = [&]() -> TypePtr {
      if (node->matches("aten::type_as(Tensor self, Tensor other) -> Tensor")) {
        return tensor_types.at(0)->withScalarType(
            tensor_types.at(1)->scalarType());
      } else if (
          node->matches(
              "aten::view_as(Tensor(a) self, Tensor other) -> Tensor(a)") ||
          node->matches(
              "aten::expand_as(Tensor(a) self, Tensor other) -> Tensor(a)") ||
          node->matches(
              "aten::reshape_as(Tensor(a) self, Tensor other) -> Tensor(a)")) {
        return tensor_types.at(0)->withDim(tensor_types.at(1)->dim());
      } else if (
          node->matches("aten::view(Tensor self, int[] size) -> Tensor") ||
          node->matches(
              "aten::expand(Tensor self, int[] size, *, bool implicit) -> Tensor") ||
          node->matches(
              "aten::as_strided(Tensor self, int[] size, int[] stride, int? storage_offset) -> Tensor")) {
        return reshape_prop(node, attr::size, tensor_types);
      } else if (
          node->matches(
              "aten::as_tensor(Tensor data, *, ScalarType? dtype, Device? device) -> Tensor")) {
        TypePtr input_type = node->inputs().at(0)->type();
        if (auto type = input_type->cast<TensorType>()) {
          if (type->scalarType() && type->device()) {
            at::ScalarType default_type = *type->scalarType();
            c10::Device default_device = *type->device();
            if (auto dtype_index =
                    node->schema().argumentIndexWithName("dtype")) {
              auto inp = toIValue(node->inputs().at(*dtype_index));
              if (inp == c10::nullopt) {
                return nullptr;
              }
              if (!inp->isNone()) {
                default_type = inp->toScalarType();
              }
            }
            if (auto device_index =
                    node->schema().argumentIndexWithName("device")) {
              auto inp = toIValue(node->inputs().at(*device_index));
              if (inp == c10::nullopt) {
                return nullptr;
              }
              if (!inp->isNone()) {
                default_device = inp->toDevice();
              }
            }
            node->output()->setType(TensorType::create(
                default_type,
                default_device,
                type->dim(),
                /*requires_grad=*/c10::nullopt));
          }
        }
        return nullptr;
      } else if (
          node->matches(
              "aten::reshape(Tensor(a) self, int[] shape) -> Tensor(a)")) {
        return reshape_prop(node, attr::shape, tensor_types);
      } else if (node->matches(
                     "aten::repeat(Tensor self, int[] repeats) -> Tensor")) {
        return reshape_prop(node, attr::repeats, tensor_types);
      } else if (node->matches(
                     "aten::unsqueeze(Tensor self, int dim) -> Tensor")) {
        auto& t = tensor_types.at(0);
        if (!t->dim()) {
          return t;
        }
        return t->withDim(*t->dim() + 1);
      } else if (
          node->matches(
              "aten::select(Tensor self, int dim, int index) -> Tensor") ||
          node->matches(
              "aten::diagonal(Tensor self, int offset, int dim1, int dim2) -> Tensor")) {
        auto& t = tensor_types.at(0);
        return t->dim() && *t->dim() > 0 ? t->withDim(*t->dim() - 1) : nullptr;
      } else if (node->matches(
                     "aten::matmul(Tensor self, Tensor other) -> Tensor")) {
        if (!tensor_types.at(0)->dim() || !tensor_types.at(1)->dim()) {
          return nullptr;
        }
        int dim1 = *tensor_types.at(0)->dim();
        int dim2 = *tensor_types.at(1)->dim();
        if (dim1 == 1 && dim2 == 1) {
          // Dot product
          return tensor_types.at(0)->withDim(0);
          // NOLINTNEXTLINE(bugprone-branch-clone)
        } else if (dim1 == 2 && dim2 == 2) {
          // Matrix multiply
          return tensor_types.at(0);
        } else if (dim1 == 1 && dim2 == 2) {
          // Unsqueeze + matrix multiply + squeeze
          return tensor_types.at(0);
        } else if (dim1 == 2 && dim2 == 1) {
          // Matrix vector multiply
          return tensor_types.at(1);
        } else {
          // Batched matrix multiply (possibly with squeeze + unsqueeze if one
          // argument is 1D)
          auto type = broadcast(tensor_types, tensor_types[0]->scalarType());
          if (dim1 == 1 || dim2 == 1) {
            type = type->withDim(type->dim().value() - 1);
          }
          return type;
        }
      } else if (node->matches("aten::nonzero(Tensor self) -> Tensor")) {
        return tensor_types.at(0)->dimensionedOnly()->withScalarType(at::kLong);
      } else if (node->matches(
                     "aten::take(Tensor self, Tensor index) -> Tensor")) {
        return tensor_types.at(1)->dimensionedOnly()->withScalarType(
            tensor_types.at(0)->scalarType());
      } else if (node->matches(
                     "aten::diagflat(Tensor self, int offset) -> Tensor")) {
        return tensor_types.at(0)->withDim(2);
      } else if (node->matches(
                     "aten::diag(Tensor self, int diagonal) -> Tensor")) {
        auto& t = tensor_types.at(0);
        if (t->dim() && *t->dim() == 1) {
          return t->withDim(2);
        } else if (t->dim() && *t->dim() == 2) {
          return t->withDim(1);
        } else {
          return nullptr;
        }
      } else if (
          node->matches(
              "aten::unfold(Tensor self, int dimension, int size, int step) -> Tensor")) {
        auto& t = tensor_types.at(0);
        if (!t->dim()) {
          return nullptr;
        }
        return t->withDim(*t->dim() + 1);
      } else if (node->matches(
                     "aten::polygamma(int n, Tensor self) -> Tensor")) {
        return tensor_types.at(0);
      }
      return nullptr;
    };
    if (auto maybe_tensor_types = gatherTensorTypes(node)) {
      tensor_types = std::move(*maybe_tensor_types);
    } else {
      return false;
    }
    if (node->outputs().size() == 1) {
      if (auto type = getSingleOutputType()) {
        node->output()->setType(type);
        return true;
      }
    }
    return false;
  }

  bool PropagateCompleteShapeOnNode(
      Node* node,
      bool insert_expands,
      std::vector<TensorTypePtr> tensor_types) {
    // For expensive ops we can directly encode their shape propagation
    // here, otherwise we fallback to running a fake version of the op
    // to get a quick and dirty propagation.
    if (node->matches(
            "aten::add(Tensor self, Tensor other, *, Scalar alpha) -> Tensor") ||
        node->matches(
            "aten::sub(Tensor self, Tensor other, *, Scalar alpha) -> Tensor") ||
        node->matches("aten::mul(Tensor self, Tensor other) -> Tensor")) {
      // These nodes handle tensors of different shapes internally, so there's
      // no need to insert explicit expand nodes.
      return PropagateShapeOnNodeByRunningIt(node);
    } else if (node->matches(
                   "aten::div(Tensor self, Tensor other) -> Tensor")) {
      // "div" handle tensors of different shapes internally, so there's no need
      // to insert explicit expand nodes.
      // Note that this function could be merged to the one above , but "div" is
      // not always safe to run by itself due to integer divide-by-zero.
      // We fake the execution by running "mul" operation instead.
      auto op = getOperatorForLiteral(
                    "aten::mul(Tensor self, Tensor other) -> Tensor")
                    ->getOperation();
      return PropagateShapeOnNodeByRunningIt(node, op);
    } else if (node->matches(
                   "aten::pow(Tensor self, Scalar exponent) -> Tensor")) {
      node->output()->setType(tensor_types.at(0));
      return true;
    } else if (
        node->matches(
            "aten::add(Tensor self, Scalar other, Scalar alpha) -> Tensor") ||
        node->matches(
            "aten::sub(Tensor self, Scalar other, Scalar alpha) -> Tensor") ||
        node->matches("aten::div(Tensor self, Scalar other) -> Tensor") ||
        node->matches("aten::mul(Tensor self, Scalar other) -> Tensor")) {
      auto first_scalar_type = (tensor_types)[0]->scalarType();
      auto second_scalar_type =
          tryScalarTypeFromJitType(node->inputs()[1]->type());
      if (!first_scalar_type || !second_scalar_type) {
        return false;
      }
      if (isIntegralType(*first_scalar_type, false) &&
          isFloatingType(*second_scalar_type)) {
        auto default_dtype =
            at::typeMetaToScalarType(caffe2::get_default_dtype());
        auto type = tensor_types[0]->withScalarType(default_dtype);
        node->output()->setType(type);
        return true;
      }
      if (c10::ScalarType::Bool == *first_scalar_type &&
          c10::ScalarType::Bool != *second_scalar_type) {
        auto result_type =
            c10::promoteTypes(*first_scalar_type, *second_scalar_type);
        auto type = tensor_types[0]->withScalarType(result_type);
        node->output()->setType(type);
        return true;
      }
      auto type = tensor_types[0]->withScalarType(first_scalar_type);
      node->output()->setType(type);
      return true;
    } else if (
        insert_expands &&
        (node->matches("aten::pow(Tensor self, Tensor exponent) -> Tensor") ||
         node->matches("aten::min(Tensor self, Tensor other) -> Tensor") ||
         node->matches("aten::max(Tensor self, Tensor other) -> Tensor") ||
         node->matches("aten::lt(Tensor self, Tensor other) -> Tensor") ||
         node->matches("aten::le(Tensor self, Tensor other) -> Tensor") ||
         node->matches("aten::gt(Tensor self, Tensor other) -> Tensor") ||
         node->matches("aten::ge(Tensor self, Tensor other) -> Tensor") ||
         node->matches("aten::eq(Tensor self, Tensor other) -> Tensor") ||
         node->matches("aten::ne(Tensor self, Tensor other) -> Tensor"))) {
      // Binary broadcasting ops
      // NB: we don't handle the nodes in any other way (note the lack of
      // return!), because the type casting logic in scalar cases is
      // non-trivial. It's better to just run them.
      broadcastBinary(node, tensor_types, 0, 1);
      return PropagateShapeOnNodeByRunningIt(node);
    } else if (
        node->matches(
            "aten::logit(Tensor self, float? eps = None) -> Tensor") ||
        node->matches("aten::neg(Tensor self) -> Tensor") ||
        node->matches("aten::sigmoid(Tensor self) -> Tensor") ||
        node->matches("aten::tanh(Tensor self) -> Tensor")) {
      node->output()->setType(tensor_types.at(0)->contiguous());
      return true;
    } else if (node->matches("aten::mm(Tensor self, Tensor mat2) -> Tensor")) {
      auto lhs_type = tensor_types.at(0);
      auto rhs_type = tensor_types.at(1);
      auto lhs_sizes = lhs_type->sizes().concrete_sizes().value();
      auto rhs_sizes = rhs_type->sizes().concrete_sizes().value();
      SHAPE_ASSERT(
          *lhs_type->sizes().size() == 2 && *rhs_type->sizes().size() == 2);
      node->output()->setType(TensorType::createContiguous(
          *lhs_type->scalarType(),
          *lhs_type->device(),
          at::IntArrayRef{lhs_sizes[0], rhs_sizes[1]}));
      return true;
    } else if (node->matches("aten::t(Tensor self) -> Tensor")) {
      auto tp = tensor_types.at(0);
      auto sizes = tp->sizes().concrete_sizes().value();
      auto strides = tp->strides().concrete_sizes().value();
      SHAPE_ASSERT(sizes.size() == 2);
      std::swap(sizes.at(0), sizes.at(1));
      std::swap(strides.at(0), strides.at(1));
      node->output()->setType(tp->withSizesStrides(sizes, strides));
      return true;
    } else if (
        node->matches(
            "aten::narrow(Tensor self, int dim, int start, int length) -> Tensor",
            /*const_inputs=*/{attr::dim, attr::length})) {
      auto tp = tensor_types.at(0);
      auto sizes = tp->sizes().concrete_sizes().value();
      int64_t dim = node->get<int64_t>(attr::dim).value();
      int64_t length = node->get<int64_t>(attr::length).value();
      SHAPE_ASSERT(dim >= 0 && static_cast<size_t>(dim) < sizes.size());
      sizes.at(dim) = length;
      node->output()->setType(
          tp->withSizesStrides(sizes, tp->strides().concrete_sizes().value()));
      return true;
    } else if (node->matches(
                   "aten::sum(Tensor self, *, int? dtype) -> Tensor")) {
      node->output()->setType(tensor_types.at(0)->withSizes({}));
      return true;
    } else if (
        node->matches(
            "aten::sum(Tensor self, int[] dim, bool keepdim, *, int? dtype) -> Tensor",
            /*const_inputs=*/{attr::dim, attr::keepdim})) {
      auto& tp = tensor_types.at(0);
      auto sizes = tp->sizes().concrete_sizes().value();
      auto dims = node->get<c10::List<int64_t>>(attr::dim).value();
      bool keepdim = node->get<bool>(attr::keepdim).value();
      std::reverse(dims.begin(), dims.end());
      for (int64_t dim : dims) {
        SHAPE_ASSERT(dim >= 0 && static_cast<size_t>(dim) < sizes.size());
        if (keepdim) {
          sizes.at(dim) = 1;
        } else {
          sizes.erase(sizes.begin() + dim);
        }
      }
      node->output()->setType(tp->withSizes(sizes));
      return true;
    } else if (node->matches(
                   "aten::squeeze(Tensor self, int dim) -> Tensor",
                   /*const_inputs=*/attr::dim)) {
      auto& tp = tensor_types.at(0);
      auto sizes = tp->sizes().concrete_sizes().value();
      auto strides = tp->strides().concrete_sizes().value();
      int64_t dim = wrapDim(node->get<int64_t>(attr::dim).value(), sizes);
      SHAPE_ASSERT(dim >= 0 && static_cast<size_t>(dim) < sizes.size());
      if (sizes.at(dim) == 1) {
        sizes.erase(sizes.begin() + dim);
        strides.erase(strides.begin() + dim);
      }
      node->output()->setType(tp->withSizesStrides(sizes, strides));
      return true;
    } else if (node->matches(
                   "aten::unsqueeze(Tensor self, int dim) -> Tensor",
                   /*const_inputs=*/attr::dim)) {
      auto& tp = tensor_types.at(0);
      auto sizes = tp->sizes().concrete_sizes().value();
      auto strides = tp->strides().concrete_sizes().value();
      int64_t dim = wrapDim(node->get<int64_t>(attr::dim).value(), sizes);
      SHAPE_ASSERT(dim >= 0 && static_cast<size_t>(dim) <= sizes.size());
      int64_t new_stride = dim >= static_cast<int64_t>(sizes.size())
          ? 1
          : sizes.at(dim) * strides.at(dim);
      sizes.insert(sizes.begin() + dim, 1);
      strides.insert(strides.begin() + dim, new_stride);
      node->output()->setType(tp->withSizesStrides(sizes, strides));
      return true;
    } else if (node->matches(
                   "aten::view(Tensor self, int[] size) -> Tensor",
                   /*const_inputs=*/attr::size)) {
      auto sizes = node->get<c10::List<int64_t>>(attr::size).value();
      bool inferred = false;
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      size_t inferred_idx;
      int64_t size_product = 1;
      for (size_t i = 0; i < sizes.size(); ++i) {
        if (sizes.get(i) == -1) {
          if (inferred)
            throw propagation_error();
          inferred = true;
          inferred_idx = i;
        } else {
          size_product *= sizes.get(i);
        }
      }

      if (inferred) {
        SHAPE_ASSERT(size_product != 0);
        size_t numel = 1;
        auto concrete_sizes =
            tensor_types.at(0)->sizes().concrete_sizes().value();
        for (int64_t s : concrete_sizes)
          numel *= s;
        int64_t inferred_size = numel / size_product;
        sizes[inferred_idx] = inferred_size;
      }
      node->output()->setType(tensor_types.at(0)->withSizes(sizes.vec()));
      return true;
    } else if (node->matches(
                   "aten::type_as(Tensor self, Tensor other) -> Tensor")) {
      if (tensor_types.at(0)->scalarType() ==
          tensor_types.at(1)->scalarType()) {
        node->output()->setType(node->namedInput(attr::self)->type());
      } else {
        // This will be a copy, so the result will be contiguous
        node->output()->setType(tensor_types.at(1)->withSizes(
            tensor_types.at(0)->sizes().concrete_sizes().value()));
      }
      return true;
    } else if (
        node->matches(
            "aten::expand(Tensor self, int[] size, *, bool implicit) -> Tensor",
            /*const_inputs=*/attr::size)) {
      auto tp = tensor_types.at(0);
      auto sizesAndStrides = at::inferExpandGeometry_dimvector(
          tp->sizes().concrete_sizes().value(),
          tp->strides().concrete_sizes().value(),
          node->get<c10::List<int64_t>>(attr::size).value().vec());
      node->output()->setType(
          tp->withSizesStrides(sizesAndStrides.sizes, sizesAndStrides.strides));
      return true;
    } else if (
        node->matches(
            "aten::index_select(Tensor self, int dim, Tensor index) -> Tensor",
            /*const_inputs=*/attr::dim)) {
      auto ten = tensor_types.at(0);
      auto index = tensor_types.at(1);
      int64_t dim = node->get<int64_t>(attr::dim).value();
      SHAPE_ASSERT(*index->sizes().size() == 1);
      SHAPE_ASSERT(dim >= 0 && static_cast<size_t>(dim) < ten->sizes().size());
      std::vector<int64_t> sizes = ten->sizes().concrete_sizes().value();
      sizes[dim] = index->sizes()[0].value();
      node->output()->setType(ten->withSizes(sizes));
      return true;
    } else if (node->matches(
                   "aten::chunk(Tensor self, int chunks, int dim) -> Tensor[]",
                   /*const_inputs=*/{attr::chunks, attr::dim})) {
      auto input_type = tensor_types.at(0);
      auto sizes = input_type->sizes().concrete_sizes().value();
      auto strides = input_type->strides().concrete_sizes().value();
      int64_t dim = node->get<int64_t>(attr::dim).value();
      int64_t chunks = node->get<int64_t>(attr::chunks).value();
      sizes[dim] /= chunks;
      for (Value* output : node->outputs()) {
        output->setType(input_type->withSizesStrides(sizes, strides));
      }
      if (*input_type->sizes()[dim] % chunks != 0) {
        sizes[dim] = *input_type->sizes()[dim] % chunks;
        node->outputs().back()->setType(
            input_type->withSizesStrides(sizes, strides));
      }
      return true;
    } else if (node->kind() == ::c10::onnx::Shape) {
      SHAPE_ASSERT(node->inputs().size() == 1 && node->outputs().size() == 1);
      std::vector<int64_t> dim_vec = {
          (int64_t)*tensor_types.at(0)->sizes().size()};
      at::IntArrayRef dims(dim_vec);
      node->output()->setType(
          TensorType::createContiguous(at::kLong, at::kCPU, dims));
      return true;
    } else if (node->kind() == ::c10::onnx::Reshape) {
      setUnshapedType(node);
      return true;
    }
    setUnshapedType(node);
    return false;
  }
};
} // anonymous namespace

void PropagateInputShapes(const std::shared_ptr<Graph>& graph) {
  ShapePropagator(graph).PropagateShapeOnBlock(graph->block());
}

namespace {

using TypeCache = std::unordered_map<TypePtr, TypePtr>;

TypePtr getOrCreateUnshapedType(TypePtr type, TypeCache& unshaped_type_cache);

TypePtr unshapedTypeImpl(TypePtr type, TypeCache& unshaped_type_cache) {
  if (type->isSubtypeOf(TensorType::get())) {
    return TensorType::get();
  }
  std::vector<TypePtr> unshaped_contained_types;
  for (const auto& contained_type : type->containedTypes()) {
    unshaped_contained_types.push_back(
        getOrCreateUnshapedType(contained_type, unshaped_type_cache));
  }
  return type->withContained(unshaped_contained_types);
}

TypePtr getOrCreateUnshapedType(TypePtr type, TypeCache& unshaped_type_cache) {
  auto maybe_cached_type = unshaped_type_cache.find(type);
  if (maybe_cached_type != unshaped_type_cache.end()) {
    return maybe_cached_type->second;
  }
  auto unshaped_type = unshapedTypeImpl(type, unshaped_type_cache);
  unshaped_type_cache[type] = unshaped_type;
  return unshaped_type;
}

void EraseShapeInformation(
    const std::shared_ptr<Graph>& graph,
    TypeCache& unshaped_type_cache);

void EraseShapeInformation(
    at::ArrayRef<Value*> vals,
    TypeCache& unshaped_type_cache) {
  for (Value* v : vals) {
    v->setType(getOrCreateUnshapedType(v->type(), unshaped_type_cache));
  }
}

void EraseShapeInformation(Block* b, TypeCache& unshaped_type_cache) {
  EraseShapeInformation(b->inputs(), unshaped_type_cache);
  EraseShapeInformation(b->outputs(), unshaped_type_cache);
  for (Node* n : b->nodes()) {
    EraseShapeInformation(n->outputs(), unshaped_type_cache);
    for (Block* sb : n->blocks()) {
      EraseShapeInformation(sb, unshaped_type_cache);
    }
    if (n->hasAttribute(attr::Subgraph)) {
      EraseShapeInformation(n->g(attr::Subgraph), unshaped_type_cache);
    }
  }
}

void EraseShapeInformation(
    const std::shared_ptr<Graph>& graph,
    TypeCache& unshaped_type_cache) {
  EraseShapeInformation(graph->block(), unshaped_type_cache);
}

} // anonymous namespace

void EraseShapeInformation(const std::shared_ptr<Graph>& graph) {
  TypeCache unshaped_type_cache;
  EraseShapeInformation(graph->block(), unshaped_type_cache);
}
} // namespace jit
} // namespace torch
