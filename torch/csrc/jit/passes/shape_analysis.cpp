#include "torch/csrc/jit/passes/shape_analysis.h"

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/constants.h"
#include "torch/csrc/jit/argument_spec.h"
#include "torch/csrc/jit/operator.h"
#include "torch/csrc/jit/assertions.h"

#include "torch/csrc/autograd/variable.h"

#include <ATen/DeviceGuard.h>
#include <ATen/ExpandUtils.h>

#include <exception>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

namespace torch { namespace jit {

struct propagation_error : std::exception {};

#define SHAPE_ASSERT(cond) if (!(cond)) throw propagation_error()

namespace {

void setUnshapedType(Node * node) {
  for(auto o : node->outputs()) {
    o->setType(unshapedType(o->type()));
  }
}

int64_t wrapDim(int64_t dim, at::IntList sizes) {
  if (dim < 0) {
    dim += sizes.size();
  }
  return dim;
}

IValue representativeValue(Value* v) {
  TypePtr type_ = v->type();
  // if the value is actually constant, just use it!
  if(auto iv = toIValue(v)) {
    return *iv;
  }
  if (CompleteTensorTypePtr type = type_->cast<CompleteTensorType>()) {
    auto backend = type->device() == -1 ? at::Backend::CPU : at::Backend::CUDA;
    at::DeviceGuard device_guard(type->device());
    auto& attype = at::getNonVariableType(backend, type->scalarType());
    auto t = attype.tensor(type->sizes(), type->strides()).zero_();
    return autograd::make_variable(t, /*requires_grad=*/false);
  } else if (type_->isSubtypeOf(FloatType::get())) {
    return 0.f;
  }
  // we should not get here because isValidArgumentForRunning should have
  // prevented it
  std::stringstream ss;
  ss << "unable to create representative value for: " << type_->str()
     << ". File a bug report.";
  throw std::runtime_error(ss.str());
}

void PropagateShapeOnBlock(Block * block, bool insert_expands=true);

// for each node in the schema with type Tensor, extract the T type
// returns at::nullopt if any Tensor in the schema does not have a known shape
// ignores non-tensor in the list of inputs
template<typename T>
at::optional<std::vector<std::shared_ptr<T>>> gatherTensorTypes(Node *node) {
  std::vector<std::shared_ptr<T>> tensor_types;

  auto & schema = node->schema();
  auto & args = schema.arguments;
  // can't handle varargs primitives because we don't know what should be a Tensor
  if (schema.is_vararg) {
    return at::nullopt;
  }
  for (size_t i = 0; i < args.size(); ++i) {
    if (args[i].type->isSubtypeOf(ListType::ofTensors())) {
      return at::nullopt;
    } else if (args[i].type->isSubtypeOf(DynamicType::get())) {
      if (auto type = node->input(i)->type()->cast<T>()) {
        tensor_types.push_back(type);
      } else {
        return at::nullopt;
      }
    } else /* non-tensor type */ {
      continue;
    }
  }

  return tensor_types;
}

bool mergeTypes(ArrayRef<Value*> lhs, ArrayRef<Value*> rhs, ArrayRef<Value*> outputs) {
  JIT_ASSERT(lhs.size() == rhs.size() && rhs.size() == outputs.size());
  bool changed = false;
  for(size_t i = 0; i < lhs.size(); ++i) {
    auto old_output_type = outputs[i]->type();
    auto new_type = unifyTypes(lhs[i]->type(), rhs[i]->type());
    JIT_ASSERT(new_type);
    outputs[i]->setType(*new_type);
    if(*old_output_type != *outputs[i]->type())
      changed = true;
  }
  return changed;
}

void PropagateShapeOnNode(Node * node, bool insert_expands=true);

void broadcastBinary(Node *node, std::vector<CompleteTensorTypePtr>& types, size_t idx1, size_t idx2) {
  auto expected_size = at::infer_size(types[idx1]->sizes(), types[idx2]->sizes());
  auto broadcast = [&](size_t input_idx) {
    CompleteTensorTypePtr input_type = types.at(input_idx);
    if (input_type->sizes() == expected_size)
      return;
    auto graph = node->owningGraph();
    WithInsertPoint point_guard { node };
    Node *expand = graph->create(aten::expand,
                                 {node->inputs().at(input_idx),
                                  graph->insertConstant(expected_size),
                                  graph->insertConstant(false)})
                        ->insertBefore(node);
    PropagateShapeOnNode(expand);
    node->replaceInput(input_idx, expand->output());
  };
  broadcast(idx1);
  broadcast(idx2);
  types[0] = node->inputs().at(idx1)->type()->expect<CompleteTensorType>();
  types[1] = node->inputs().at(idx2)->type()->expect<CompleteTensorType>();
}

bool isValidArgumentForRunning(Value* v) {
  // allow constants
  if(toIValue(v))
    return true;
  if(CompleteTensorTypePtr tt = v->type()->cast<CompleteTensorType>()) {
    return !at::isIntegralType(tt->scalarType());
  }
  return v->type()->isSubtypeOf(FloatType::get());
}
bool isValidReturnForRunning(Value* v) {
  return v->type()->isSubtypeOf(DynamicType::get()) ||
      v->type()->isSubtypeOf(NumberType::get());
}

OperatorSet cannot_propagate_shape_by_running_it = {
  "aten::gesv(Tensor self, Tensor A) -> (Tensor, Tensor)",
  "aten::inverse(Tensor self) -> Tensor",
};

bool canPropagateShapeByRunningIt(Node* node) {
  if(cannot_propagate_shape_by_running_it.find(node)) {
    return false;
  }

  bool valid_args = std::all_of(
      node->inputs().begin(), node->inputs().end(), isValidArgumentForRunning);
  if (!valid_args)
    return false;

  bool valid_returns = std::all_of(
      node->outputs().begin(), node->outputs().end(), isValidReturnForRunning);
  if (!valid_returns)
    return false;

  return true;
}

bool PropagateShapeOnNodeByRunningIt(Node* node) {
  if (!canPropagateShapeByRunningIt(node))
    return false;
  auto op = getOperation(node);
  Stack stack;

  for (auto input : node->inputs()) {
    stack.push_back(representativeValue(input));
  }

  // XXX: we're not catching any exceptions from the op for now. This
  // is to uncover any mistakes we could make when editing this code,
  // and eventually it shouldn't matter, because this phase should be
  // preceded by schema checking.
  op(stack);

  JIT_ASSERT(stack.size() == node->outputs().size());
  for (size_t i = 0; i < stack.size(); ++i) {
    // some ops may have mixed tensor/primitive outputs
    // for primitives, we don't need to change the type because it is already
    // its most constrained form.
    if(stack[i].isTensor())
      node->outputs()[i]->inferTypeFrom(stack[i].toTensor());
  }
  return true;
}

// is it ok to try to run the op
// If an input is a constant, then we assume that the input is valid
// and we can try to run it.
// Otherwise:
// Integral typed _inputs_ are often an indicator that we're indexing into
// a tensor, so we should special-case these ops in the shape propagation.
// Additionally, passing in a zero representative tensor into an integer
// division op causes divide-by-zero errors
// _Outputs_ must be tensors or primtives
// We will call inferTypeFrom on the tensors, and ignore the primitives.
// However, we allow primitive returns because we want to support mixed
// primitive/tensor outputs.

bool PropagateTensorShapeOnNode(
    Node * node, bool insert_expands, std::vector<TensorTypePtr> types);
bool PropagateCompleteShapeOnNode(
    Node * node, bool insert_expands, std::vector<CompleteTensorTypePtr> types);

void PropagateCatShape(Node * cat_node) {
  static const auto propagate_complete = [](Node * node, at::ArrayRef<Value*> tensors) -> bool {
    auto input_types = fmap(tensors, [](Value *v) { return v->type()->cast<CompleteTensorType>(); });
    if (!std::all_of(input_types.begin(), input_types.end(),
        [](const CompleteTensorTypePtr& tp) { return tp != nullptr; })) {
      return false;
    }
    if (!node->is_constant(attr::dim)) return false;
    std::vector<int64_t> sizes = input_types[0]->sizes();
    const int64_t dim = wrapDim(node->get<int64_t>(attr::dim).value(), sizes);
    const int64_t ndim = sizes.size();

    if (dim < 0 || dim >= ndim)
      return false;

    sizes[dim] = 0;
    for (auto & tp : input_types) {
      auto & tp_sizes = tp->sizes();
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
  static const auto propagate = [](Node * node, at::ArrayRef<Value*> tensors) -> bool {
    for (Value * v : tensors) {
      if (auto type = v->type()->cast<TensorType>()) {
        node->output()->setType(type);
        return true;
      }
    }
    return false;
  };
  auto list_node = cat_node->namedInput(attr::tensors)->node();
  if (list_node->kind() == prim::ListConstruct) {
    auto tensors = list_node->inputs();
    if (!tensors.empty()) {
      if (propagate_complete(cat_node, tensors)) {
        return;
      } else if (propagate(cat_node, tensors)) {
        return;
      }
    }
  }
  setUnshapedType(cat_node);
}

void PropagateShapeOnNode(Node * node, bool insert_expands) {
  // These don't require the types, and have complicated schema. Return early after we process them.
  switch(node->kind()) {
    case prim::If: {
      auto then_block = node->blocks().at(0);
      auto else_block = node->blocks().at(1);
      PropagateShapeOnBlock(then_block);
      PropagateShapeOnBlock(else_block);
      mergeTypes(then_block->outputs(), else_block->outputs(), node->outputs());
      return;
    }
    case prim::Loop: {
      auto body_block = node->blocks().at(0);
      // propagate counter type
      body_block->inputs().at(0)->setType(node->inputs().at(0)->type());
      // propagate loop-carried input types to block inputs
      auto loop_carried_inputs = node->inputs().slice(2); // skip max, cond
      auto loop_carried_block = body_block->inputs().slice(1); // skip trip
      for(size_t i = 0; i < loop_carried_inputs.size(); ++i) {
        loop_carried_block[i]->setType(loop_carried_inputs[i]->type());
      }
      auto loop_carried_outputs = body_block->outputs().slice(1); // skip cond

      do {
        PropagateShapeOnBlock(body_block, /*insert_expands=*/false);
        // note: inserting expands is unsafe at this point, we don't know
        // if the types are stable yet, so the arguments to expand may change
      } while(mergeTypes(loop_carried_block, loop_carried_outputs, loop_carried_block));

      // now that the types are stable, we can insert the expands
      PropagateShapeOnBlock(body_block, /*insert_expands=*/true);


      for(size_t i = 0; i < loop_carried_inputs.size(); ++i) {
        node->outputs()[i]->setType(loop_carried_block[i]->type());
      }
      return;
    }
    case prim::ImplicitTensorToNum:
    case prim::TensorToNum:
    case prim::NumToTensor:
      return; // correct num type is already set
    case prim::Constant: {
      if(node->output()->type()->isSubtypeOf(DynamicType::get())) {
        node->output()->inferTypeFrom(node->t(attr::value));
      }
      return;
    }
    case prim::ConstantChunk: {
      Value *tensor = node->input();
      if (auto type = tensor->type()->cast<TensorType>()) {
        for (Value * output : node->outputs()) {
          output->setType(type);
        }
      } else {
        setUnshapedType(node);
      }
      return;
    }
    case prim::PythonOp:
    case prim::Print:
    case prim::Undefined: {
      setUnshapedType(node);
      return;
    }
    default:
      break; // fall-through
  }
  if (node->matches("aten::cat(Tensor[] tensors, int dim) -> Tensor")) {
    return PropagateCatShape(node);
  }

  if (auto maybe_complete_types = gatherTensorTypes<CompleteTensorType>(node)) {
    if (PropagateCompleteShapeOnNode(node, insert_expands, std::move(*maybe_complete_types))) {
      return;
    }
  }

  if (auto maybe_tensor_types = gatherTensorTypes<TensorType>(node)) {
    if (PropagateTensorShapeOnNode(node, insert_expands, std::move(*maybe_tensor_types))) {
      return;
    }
  }

  if (PropagateShapeOnNodeByRunningIt(node)) {
    return;
  }
  return setUnshapedType(node);
}

bool PropagateTensorShapeOnNode(Node * node, bool insert_expands,
                                std::vector<TensorTypePtr> tensor_types) {
  static const auto broadcast = [](std::vector<TensorTypePtr>& tensor_types) -> TensorTypePtr {
    JIT_ASSERT(!tensor_types.empty());
    auto any_type = tensor_types[0];
    auto max_dims = any_type->dim();
    for (auto & type : tensor_types) {
      max_dims = std::max(max_dims, type->dim());
    }
    return TensorType::create(any_type->scalarType(), any_type->device(), max_dims);
  };
  const auto getSingleOutputType = [&]() -> TypePtr {
    if (node->matches("aten::add(Tensor self, Tensor other, *, Scalar alpha) -> Tensor") ||
        node->matches("aten::sub(Tensor self, Tensor other, *, Scalar alpha) -> Tensor") ||
        node->matches("aten::mul(Tensor self, Tensor other) -> Tensor") ||
        node->matches("aten::pow(Tensor self, Tensor exponent) -> Tensor") ||
        node->matches("aten::min(Tensor self, Tensor other) -> Tensor") ||
        node->matches("aten::max(Tensor self, Tensor other) -> Tensor") ||
        node->matches("aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta, Scalar alpha) -> Tensor")) {
      return broadcast(tensor_types);
    } else if (node->matches("aten::add(Tensor self, Scalar other, Scalar alpha) -> Tensor") ||
               node->matches("aten::sub(Tensor self, Scalar other, Scalar alpha) -> Tensor") ||
               node->matches("aten::mul(Tensor self, Scalar other) -> Tensor") ||
               node->matches("aten::pow(Tensor self, Scalar exponent) -> Tensor") ||
               node->matches("aten::add(Scalar other, Tensor self) -> Tensor") ||
               node->matches("aten::sub(Scalar other, Tensor self) -> Tensor") ||
               node->matches("aten::mul(Scalar other, Tensor self) -> Tensor") ||
               node->matches("aten::mm(Tensor self, Tensor mat2) -> Tensor")) {
      return tensor_types.at(0);
    } else if (node->matches("aten::lt(Tensor self, Tensor other) -> Tensor") ||
               node->matches("aten::le(Tensor self, Tensor other) -> Tensor") ||
               node->matches("aten::gt(Tensor self, Tensor other) -> Tensor") ||
               node->matches("aten::ge(Tensor self, Tensor other) -> Tensor") ||
               node->matches("aten::eq(Tensor self, Tensor other) -> Tensor") ||
               node->matches("aten::ne(Tensor self, Tensor other) -> Tensor")) {
      return broadcast(tensor_types)->toScalarType(at::kByte);
    } else if (node->matches("aten::lt(Tensor self, Scalar other) -> Tensor") ||
               node->matches("aten::le(Tensor self, Scalar other) -> Tensor") ||
               node->matches("aten::gt(Tensor self, Scalar other) -> Tensor") ||
               node->matches("aten::ge(Tensor self, Scalar other) -> Tensor") ||
               node->matches("aten::eq(Tensor self, Scalar other) -> Tensor") ||
               node->matches("aten::ne(Tensor self, Scalar other) -> Tensor")) {
      return tensor_types.at(0)->toScalarType(at::kByte);
    } else if (node->matches("aten::neg(Tensor self) -> Tensor") ||
               node->matches("aten::t(Tensor self) -> Tensor") ||
               node->matches("aten::sigmoid(Tensor self) -> Tensor") ||
               node->matches("aten::tanh(Tensor self) -> Tensor") ||
               node->matches("aten::exp(Tensor self) -> Tensor") ||
               node->matches("aten::relu(Tensor self) -> Tensor") ||
               node->matches("aten::mm(Tensor self, Tensor mat2) -> Tensor") ||
               node->matches("aten::narrow(Tensor self, int dim, int start, int length) -> Tensor") ||
               node->matches("aten::index_select(Tensor self, int dim, Tensor index) -> Tensor")) {
      return tensor_types.at(0);
    } else if (node->matches("aten::type_as(Tensor self, Tensor other) -> Tensor")) {
      return tensor_types.at(0)->toScalarType(tensor_types.at(1)->scalarType());
    } else if (node->matches("aten::sum(Tensor self) -> Tensor")) {
      // TODO: this depends on the dtype argument. why don't we have access to it in here?
      // TODO: integral types are upcast
      auto & t = tensor_types.at(0);
      return TensorType::create(t->scalarType(), t->device(), 0);
    } else if (node->matches("aten::sum(Tensor self, int[] dim, int keepdim) -> Tensor",
              /*with_const=*/attr::keepdim)) {
      auto & t = tensor_types.at(0);
      bool keepdim = node->get<int64_t>(attr::keepdim).value();
      if (!keepdim) {
        if (auto dims = node->get<std::vector<int64_t>>(attr::dim)) {
          // TODO: do we need to account for duplicates in dim here?
          return t->withDim(t->dim() - dims->size());
        }
        return nullptr;
      } else {
        return t;
      }
      return nullptr;
    } else if (node->matches("aten::unsqueeze(Tensor self, int dim) -> Tensor")) {
      auto & t = tensor_types.at(0);
      return t->withDim(t->dim() + 1);
    } else if (node->matches("aten::view(Tensor self, int[] size) -> Tensor", /*with_const=*/attr::size) ||
               node->matches("aten::expand(Tensor self, int[] size, *, int implicit) -> Tensor", /*with_const=*/attr::size)) {
      return tensor_types.at(0)->withDim(node->get<std::vector<int64_t>>(attr::size)->size());
    }
    return nullptr;
  };
  if (node->outputs().size() == 1) {
    if (auto type = getSingleOutputType()) {
      node->output()->setType(type);
      return true;
    }
  }
  setUnshapedType(node);
  return false;
}

bool PropagateCompleteShapeOnNode(Node * node, bool insert_expands,
                                  std::vector<CompleteTensorTypePtr> tensor_types) {
  // For expensive ops we can directly encode their shape propagation
  // here, otherwise we fallback to running a fake version of the op
  // to get a quick and dirty propagation.
  if (node->matches("aten::add(Tensor self, Tensor other, *, Scalar alpha) -> Tensor") ||
      node->matches("aten::sub(Tensor self, Tensor other, *, Scalar alpha) -> Tensor") ||
      node->matches("aten::mul(Tensor self, Tensor other) -> Tensor")) {
    // These nodes and "div" handle tensors of different shapes internally,
    // so there's no need to insert explicit expand nodes. Note that "div" is
    // handled by the fallthrough because it's not always safe to run it due
    // to integer divide-by-zero.
    return PropagateShapeOnNodeByRunningIt(node);
  } else if (node->matches("aten::add(Tensor self, Scalar other, Scalar alpha) -> Tensor") ||
             node->matches("aten::sub(Tensor self, Scalar other, Scalar alpha) -> Tensor") ||
             node->matches("aten::mul(Tensor self, Scalar other) -> Tensor") ||
             node->matches("aten::pow(Tensor self, Scalar exponent) -> Tensor") ||
             node->matches("aten::add(Scalar other, Tensor self) -> Tensor") ||
             node->matches("aten::sub(Scalar other, Tensor self) -> Tensor") ||
             node->matches("aten::mul(Scalar other, Tensor self) -> Tensor")) {
    node->output()->setType(tensor_types.at(0));
    return true;
  } else if (insert_expands && (
      node->matches("aten::pow(Tensor self, Tensor exponent) -> Tensor") ||
      node->matches("aten::min(Tensor self, Tensor other) -> Tensor") ||
      node->matches("aten::max(Tensor self, Tensor other) -> Tensor") ||
      node->matches("aten::lt(Tensor self, Tensor other) -> Tensor") ||
      node->matches("aten::le(Tensor self, Tensor other) -> Tensor") ||
      node->matches("aten::gt(Tensor self, Tensor other) -> Tensor") ||
      node->matches("aten::ge(Tensor self, Tensor other) -> Tensor") ||
      node->matches("aten::eq(Tensor self, Tensor other) -> Tensor") ||
      node->matches("aten::ne(Tensor self, Tensor other) -> Tensor"))) {
    // Binary broadcasting ops
    // NB: we don't handle the nodes in any other way (note the lack of return!),
    // because the type casting logic in scalar cases is non-trivial.
    // It's better to just run them.
    broadcastBinary(node, tensor_types, 0, 1);
    return PropagateShapeOnNodeByRunningIt(node);
  } else if (node->matches("aten::neg(Tensor self) -> Tensor") ||
             node->matches("aten::sigmoid(Tensor self) -> Tensor") ||
             node->matches("aten::tanh(Tensor self) -> Tensor")) {
    node->output()->setType(tensor_types.at(0)->contiguous());
    return true;
  } else if (node->matches("aten::mm(Tensor self, Tensor mat2) -> Tensor")) {
    auto lhs_type = tensor_types.at(0);
    auto rhs_type = tensor_types.at(1);
    SHAPE_ASSERT(lhs_type->sizes().size() == 2 && rhs_type->sizes().size() == 2);
    node->output()->setType(CompleteTensorType::create(
      lhs_type->scalarType(), lhs_type->device(),
      at::IntList{lhs_type->sizes().at(0), rhs_type->sizes().at(1)}));
    return true;
  } else if (node->matches("aten::t(Tensor self) -> Tensor")) {
    auto tp = tensor_types.at(0);
    auto sizes = tp->sizes();
    auto strides = tp->strides();
    SHAPE_ASSERT(sizes.size() == 2);
    std::swap(sizes.at(0), sizes.at(1));
    std::swap(strides.at(0), strides.at(1));
    node->output()->setType(tp->withSizesStrides(sizes, strides));
    return true;
  } else if (node->matches("aten::narrow(Tensor self, int dim, int start, int length) -> Tensor",
                           /*with_const=*/{attr::dim, attr::length})) {
    auto tp = tensor_types.at(0);
    auto sizes = tp->sizes();
    int64_t dim = node->get<int64_t>(attr::dim).value();
    int64_t length = node->get<int64_t>(attr::length).value();
    SHAPE_ASSERT(dim >= 0 && static_cast<size_t>(dim) < sizes.size());
    sizes.at(dim) = length;
    node->output()->setType(tp->withSizesStrides(sizes, tp->strides()));
    return true;
  } else if (node->matches("aten::sum(Tensor self) -> Tensor")) {
    node->output()->setType(tensor_types.at(0)->withSizes({}));
    return true;
  } else if (node->matches("aten::sum(Tensor self, int[] dim, bool keepdim) -> Tensor",
             /*with_const=*/{attr::dim, attr::keepdim})) {
    auto & tp = tensor_types.at(0);
    auto sizes = tp->sizes();
    auto dims = node->get<std::vector<int64_t>>(attr::dim).value();
    bool keepdim = node->get<int64_t>(attr::keepdim).value();
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
  } else if (node->matches("aten::squeeze(Tensor self, int dim) -> Tensor", /*with_const=*/attr::dim)) {
    auto & tp = tensor_types.at(0);
    auto sizes = tp->sizes();
    auto strides = tp->strides();
    int64_t dim = wrapDim(node->get<int64_t>(attr::dim).value(), sizes);
    SHAPE_ASSERT(dim >= 0 && static_cast<size_t>(dim) < sizes.size());
    if (sizes.at(dim) == 1) {
      sizes.erase(sizes.begin() + dim);
      strides.erase(strides.begin() + dim);
    }
    node->output()->setType(tp->withSizesStrides(sizes, strides));
    return true;
  } else if (node->matches("aten::unsqueeze(Tensor self, int dim) -> Tensor", /*with_const=*/attr::dim)) {
    auto & tp = tensor_types.at(0);
    auto sizes = tp->sizes();
    auto strides = tp->strides();
    int64_t dim = wrapDim(node->get<int64_t>(attr::dim).value(), sizes);
    SHAPE_ASSERT(dim >= 0 && static_cast<size_t>(dim) <= sizes.size());
    int64_t new_stride = dim >= static_cast<int64_t>(sizes.size()) ? 1 : sizes.at(dim) * strides.at(dim);
    sizes.insert(sizes.begin() + dim, 1);
    strides.insert(strides.begin() + dim, new_stride);
    node->output()->setType(tp->withSizesStrides(sizes, strides));
    return true;
  } else if (node->matches("aten::view(Tensor self, int[] size) -> Tensor", /*with_const=*/attr::size)) {
    auto sizes = node->get<std::vector<int64_t>>(attr::size).value();
    bool inferred = false;
    size_t inferred_idx;
    int64_t size_product = 1;
    for (size_t i = 0; i < sizes.size(); ++i) {
      if (sizes[i] == -1) {
        if (inferred) throw propagation_error();
        inferred = true;
        inferred_idx = i;
      } else {
        size_product *= sizes[i];
      }
    }

    if (inferred) {
      SHAPE_ASSERT(size_product != 0);
      size_t numel = 1;
      for (int64_t s : tensor_types.at(0)->sizes())
        numel *= s;
      int64_t inferred_size = numel / size_product;
      sizes[inferred_idx] = inferred_size;
    }
    node->output()->setType(tensor_types.at(0)->withSizes(sizes));
    return true;
  } else if (node->matches("aten::type_as(Tensor self, Tensor other) -> Tensor")) {
    if (tensor_types.at(0)->scalarType() == tensor_types.at(1)->scalarType()) {
      node->output()->setType(node->namedInput(attr::self)->type());
    } else {
      // This will be a copy, so the result will be contiguous
      node->output()->setType(tensor_types.at(1)->withSizes(tensor_types.at(0)->sizes()));
    }
    return true;
  } else if (node->matches("aten::expand(Tensor self, int[] size, *, bool implicit) -> Tensor",
             /*with_const=*/attr::size)) {
    auto tp = tensor_types.at(0);
    std::vector<int64_t> sizes, strides;
    std::tie(sizes, strides) = at::inferExpandGeometry(
        tp->sizes(), tp->strides(), node->get<std::vector<int64_t>>(attr::size).value());
    node->output()->setType(tp->withSizesStrides(sizes, strides));
    return true;
  } else if (node->matches("aten::index_select(Tensor self, int dim, Tensor index) -> Tensor",
             /*with_const=*/attr::dim)) {
    auto ten = tensor_types.at(0);
    auto index = tensor_types.at(1);
    int64_t dim = node->get<int64_t>(attr::dim).value();
    SHAPE_ASSERT(index->sizes().size() == 1);
    SHAPE_ASSERT(dim >= 0 && static_cast<size_t>(dim) < ten->sizes().size());
    std::vector<int64_t> sizes = ten->sizes();
    sizes[dim] = index->sizes()[0];
    node->output()->setType(ten->withSizes(sizes));
    return true;
  } else if (node->matches("aten::chunk(Tensor self, int chunks, int dim) -> Tensor[]",
                           /*with_const=*/{attr::chunks, attr::dim})) {
    auto input_type = tensor_types.at(0);
    auto sizes = input_type->sizes();
    const auto & strides = input_type->strides();
    int64_t dim = node->get<int64_t>(attr::dim).value();
    int64_t chunks = node->get<int64_t>(attr::chunks).value();
    sizes[dim] /= chunks;
    for (Value * output : node->outputs()) {
      output->setType(input_type->withSizesStrides(sizes, strides));
    }
    if (input_type->sizes().at(dim) % chunks != 0) {
      sizes[dim] = input_type->sizes().at(dim) % chunks;
      node->outputs().back()->setType(input_type->withSizesStrides(sizes, strides));
    }
    return true;
  } else if (node->kind() == onnx::Shape) {
    SHAPE_ASSERT(node->inputs().size() == 1 && node->outputs().size() == 1);
    std::vector<int64_t> dim_vec = {(int64_t)tensor_types.at(0)->sizes().size()};
    at::IntList dims(dim_vec);
    node->output()->setType(
        CompleteTensorType::create(at::kLong, -1, dims));
    return true;
  } else if (node->kind() == onnx::Reshape) {
    setUnshapedType(node);
    return true;
  }
  setUnshapedType(node);
  return false;
}

void PropagateShapeOnBlock(Block * block, bool insert_expands) {
  for (Node * node : block->nodes()) {
    try {
      PropagateShapeOnNode(node, insert_expands);
    } catch(propagation_error& e) {
      setUnshapedType(node);
    } catch(std::exception & e) {
      if(auto sl = node->getSourceLocation()) {
        sl->wrapAndRethrowException(e, "operation failed shape propagation");
      } else {
        throw;
      }
    }
  }
}

} // anonymous namespace

void PropagateInputShapes(Graph & graph, const CompleteArgumentSpec & spec) {
  JIT_ASSERT(graph.inputs().size() == spec.size());
  for(size_t i = 0; i < spec.size(); ++i) {
    auto argspec = spec.at(i);
    if (!argspec.isTensor()) continue;
    graph.inputs()[i]->setType(argspec);
  }
  PropagateShapeOnBlock(graph.block());
}

void PropagateInputShapes(Graph & graph, const ArgumentSpec & spec) {
  JIT_ASSERT(graph.inputs().size() == spec.size());
  for(size_t i = 0; i < spec.size(); ++i) {
    const auto & argspec = spec.at(i);
    if (!argspec.isTensor()) continue;
    graph.inputs()[i]->setType(argspec);
  }
  PropagateShapeOnBlock(graph.block());
}

namespace {

void EraseShapeInformation(at::ArrayRef<Value*> vals) {
  for (Value * v : vals) {
    v->setType(unshapedType(v->type()));
  }
}

void EraseShapeInformation(Block * b) {
  EraseShapeInformation(b->inputs());
  EraseShapeInformation(b->outputs());
  for (Node * n : b->nodes()) {
    EraseShapeInformation(n->outputs());
    for (Block *sb : n->blocks()) {
      EraseShapeInformation(sb);
    }
  }
}

} // anonymous namespace

void EraseShapeInformation(Graph & graph) {
  EraseShapeInformation(graph.block());
}

}}
