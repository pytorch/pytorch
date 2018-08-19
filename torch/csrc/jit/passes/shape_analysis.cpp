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
  if (TensorTypePtr type = type_->cast<TensorType>()) {
    auto backend = type->device() == -1 ? at::Backend::CPU : at::Backend::CUDA;
    at::DeviceGuard device_guard(type->device());
    auto& attype = at::getType(backend, type->scalarType());
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

// for each node in the schema with type Tensor, extract the TensorType
// returns at::nullopt if any Tensor in the schema does not have a known shape
// ignores non-tensor in the list of inputs
at::optional<std::vector<TensorTypePtr>> gatherTensorTypes(Node *node) {
  std::vector<TensorTypePtr> tensor_types;

  auto & schema = node->schema();
  auto & args = schema.arguments;
  // can't handle varargs primitives because we don't know what should be a Tensor
  if (schema.is_vararg) {
    return at::nullopt;
  }
  size_t input_i = 0;
  for (auto& arg : args) {
    size_t consume_n; // how many tensors do we check for in the input list
    if (arg.type->isSubtypeOf(ListType::ofTensors())) {
      // we have a list of tensor, there is only ever one list
      // so we calculte how many elements must be in it by how much bigger
      // or smaller the input list is compared to the arguments in the schema
      consume_n = node->inputs().size() + 1 - args.size();
    } else if (arg.type->isSubtypeOf(DynamicType::get())) {
      // a single Tensor for this argument
      consume_n = 1;
    } else {
      // this argument is not a tensor, skip it
      consume_n = 0;
    }
    for(size_t j = 0; j < consume_n; j++) {
      // bail out if a tensor does not have a size
      TensorTypePtr type = node->input(input_i++)->type()->cast<TensorType>();
      if (!type)
        return at::nullopt;
      tensor_types.push_back(type);
    }
  }

  return tensor_types;
}

bool mergeTypes(ArrayRef<Value*> lhs, ArrayRef<Value*> rhs, ArrayRef<Value*> outputs) {
  JIT_ASSERT(lhs.size() == rhs.size() && rhs.size() == outputs.size());
  bool changed = false;
  for(size_t i = 0; i < lhs.size(); ++i) {
    auto old_output_type = outputs[i]->type();
    if(*lhs[i]->type() == *rhs[i]->type()) {
      outputs[i]->setType(lhs[i]->type());
    } else {
      outputs[i]->setType(DynamicType::get());
    }
    if(*old_output_type != *outputs[i]->type())
      changed = true;
  }
  return changed;
}

void PropagateShapeOnNode(Node * node, bool insert_expands=true);

void broadcastBinary(Node *node, std::vector<TensorTypePtr>& types, size_t idx1, size_t idx2) {
  auto expected_size = at::infer_size(types[idx1]->sizes(), types[idx2]->sizes());
  auto broadcast = [&](size_t input_idx) {
    TensorTypePtr input_type = types.at(input_idx);
    if (input_type->sizes() == expected_size)
      return;
    auto graph = node->owningGraph();
    WithInsertPoint point_guard { node };
    Node *expand = graph->create(aten::expand,
                                 {node->inputs().at(input_idx),
                                  graph->insertConstant(expected_size),
                                  graph->insertConstant(0)})
                        ->insertBefore(node);
    PropagateShapeOnNode(expand);
    node->replaceInput(input_idx, expand->output());
  };
  broadcast(idx1);
  broadcast(idx2);
  types[0] = node->inputs().at(idx1)->type()->expect<TensorType>();
  types[1] = node->inputs().at(idx2)->type()->expect<TensorType>();
}

void PropagateShapeOnNodeByRunningIt(Node* node) {
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

bool isValidArgumentForRunning(Value* v) {
  // allow constants
  if(toIValue(v))
    return true;
  if(TensorTypePtr tt = v->type()->cast<TensorType>()) {
    return !at::isIntegralType(tt->scalarType());
  }
  return v->type()->isSubtypeOf(FloatType::get());
}
bool isValidReturnForRunning(Value* v) {
  return v->type()->isSubtypeOf(DynamicType::get()) ||
      v->type()->isSubtypeOf(NumberType::get());
}

bool canPropagateShapeByRunningIt(Node* node) {
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
    case prim::TensorToNum:
    case prim::NumToTensor:
      return; // correct num type is already set
    case prim::Constant: {
      if(node->output()->type()->isSubtypeOf(DynamicType::get())) {
        node->output()->inferTypeFrom(node->t(attr::value));
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
  if (node->matches("aten::cat(Tensor[] tensors, int dim) -> Tensor", /*with_const=*/attr::dim)) {
    auto list_node = node->namedInput(attr::tensors)->node();
    JIT_ASSERT(list_node->kind() == prim::ListConstruct);
    auto tensors = list_node->inputs();
    if (tensors.size() > 0) {
      auto input_types = fmap(tensors, [](Value *v) { return v->type()->cast<TensorType>(); });
      if (std::all_of(input_types.begin(), input_types.end(),
          [](const TensorTypePtr& tp) { return tp != nullptr; })) {
        std::vector<int64_t> sizes = input_types[0]->sizes();
        const int64_t dim = wrapDim(node->get<int64_t>(attr::dim).value(), sizes);
        const int64_t ndim = sizes.size();

        if (dim < 0 || dim >= ndim)
          goto cat_fail;

        sizes[dim] = 0;
        for (auto & tp : input_types) {
          auto & tp_sizes = tp->sizes();
          if (sizes.size() != tp_sizes.size())
            goto cat_fail;
          for (int64_t i = 0; i < ndim; ++i) {
            if (sizes[i] != tp_sizes[i] && i != dim) {
              goto cat_fail;
            }
          }
          sizes[dim] += tp_sizes[dim];
        }
        node->output()->setType(input_types[0]->withSizes(sizes));
        return;
      }
    }
  }
cat_fail:

  bool can_propagate_by_running = canPropagateShapeByRunningIt(node);
  auto maybe_tensor_types = gatherTensorTypes(node);
  if (!maybe_tensor_types) {
    if(can_propagate_by_running)
      return PropagateShapeOnNodeByRunningIt(node);
    return setUnshapedType(node);
  }
  auto & tensor_types = *maybe_tensor_types;

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
    PropagateShapeOnNodeByRunningIt(node);
    return;
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
  } else if (node->matches("aten::neg(Tensor self) -> Tensor")) {
    node->output()->setType(tensor_types.at(0)->contiguous());
    return;
  } else if (node->matches("aten::mm(Tensor self, Tensor mat2) -> Tensor")) {
    auto lhs_type = tensor_types.at(0);
    auto rhs_type = tensor_types.at(1);
    SHAPE_ASSERT(lhs_type->sizes().size() == 2 && rhs_type->sizes().size() == 2);
    node->output()->setType(TensorType::create(
      lhs_type->scalarType(), lhs_type->device(),
      at::IntList{lhs_type->sizes().at(0), rhs_type->sizes().at(1)}));
    return;
  } else if (node->matches("aten::t(Tensor self) -> Tensor")) {
    auto tp = tensor_types.at(0);
    auto sizes = tp->sizes();
    auto strides = tp->strides();
    SHAPE_ASSERT(sizes.size() == 2);
    std::swap(sizes.at(0), sizes.at(1));
    std::swap(strides.at(0), strides.at(1));
    node->output()->setType(tp->withSizesStrides(sizes, strides));
    return;
  } else if (node->matches("aten::narrow(Tensor self, int dim, int start, int length) -> Tensor",
                           /*with_const=*/{attr::dim, attr::length})) {
    auto tp = tensor_types.at(0);
    auto sizes = tp->sizes();
    int64_t dim = node->get<int64_t>(attr::dim).value();
    int64_t length = node->get<int64_t>(attr::length).value();
    SHAPE_ASSERT(dim >= 0 && static_cast<size_t>(dim) < sizes.size());
    sizes.at(dim) = length;
    node->output()->setType(tp->withSizesStrides(sizes, tp->strides()));
    return;
  } else if (node->matches("aten::sum(Tensor self) -> Tensor")) {
    node->output()->setType(tensor_types.at(0)->withSizes({}));
    return;
  } else if (node->matches("aten::sum(Tensor self, int[] dim, int keepdim) -> Tensor",
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
    return;
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
    return;
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
    return;
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
    return;
  } else if (node->matches("aten::type_as(Tensor self, Tensor other) -> Tensor")) {
    if (tensor_types.at(0)->scalarType() == tensor_types.at(1)->scalarType()) {
      node->output()->setType(node->namedInput(attr::self)->type());
    } else {
      // This will be a copy, so the result will be contiguous
      node->output()->setType(tensor_types.at(1)->withSizes(tensor_types.at(0)->sizes()));
    }
    return;
  } else if (node->matches("aten::expand(Tensor self, int[] size, *, int implicit) -> Tensor",
             /*with_const=*/attr::size)) {
    auto tp = tensor_types.at(0);
    std::vector<int64_t> sizes, strides;
    std::tie(sizes, strides) = at::inferExpandGeometry(
        tp->sizes(), tp->strides(), node->get<std::vector<int64_t>>(attr::size).value());
    node->output()->setType(tp->withSizesStrides(sizes, strides));
    return;
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
    return;
  } else if (node->kind() == onnx::Shape) {
    SHAPE_ASSERT(node->inputs().size() == 1 && node->outputs().size() == 1);
    std::vector<int64_t> dim_vec = {(int64_t)tensor_types.at(0)->sizes().size()};
    at::IntList dims(dim_vec);
    node->output()->setType(
        TensorType::create(at::kLong, -1, dims));
    return;
  } else if (node->kind() == onnx::Reshape) {
    setUnshapedType(node);
    return;
  }

  // If we haven't managed to handle the op so far, we fall back to inferring the
  // shapes by doing an example run of the op (if we can).
  if (can_propagate_by_running) {
    PropagateShapeOnNodeByRunningIt(node);
  } else {
    setUnshapedType(node);
  }
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

void PropagateInputShapes(Graph & graph, const ArgumentSpec & spec) {
  JIT_ASSERT(graph.inputs().size() == spec.size());
  for(size_t i = 0; i < spec.size(); ++i) {
    auto argspec = spec.at(i);
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
