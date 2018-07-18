#include "torch/csrc/jit/passes/shape_analysis.h"

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/argument_spec.h"
#include "torch/csrc/jit/operator.h"

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

void setDynamicType(Node * node) {
  for(auto o : node->outputs()) {
    o->setType(DynamicType::get());
  }
}

int64_t wrapDim(int64_t dim, at::IntList sizes) {
  if (dim < 0) {
    dim += sizes.size();
  }
  return dim;
}

at::Tensor representativeTensor(const TensorType * type) {
  auto backend = type->device() == -1 ? at::kCPU : at::kCUDA;
  at::DeviceGuard device_guard(type->device());
  auto & attype = at::getType(backend, type->scalarType());
  return attype.tensor(type->sizes(), type->strides()).zero_();
}

void PropagateShapeOnBlock(Block * block, bool insert_expands=true);

at::optional<std::vector<TensorType*>> gatherTensorTypes(Node *node) {
  std::vector<TensorType*> tensor_types;
  tensor_types.reserve(node->inputs().size());
  // TODO (apaszke): Remove once we stop using attributes
  // XXX: we also make the exception for cat, because we need shape prop to work for it
  // (we have tests). We'll have to remove the special case once we stop flattening lists into inputs.
  if (node->hasAttributes() || node->kind() == aten::cat) {
    std::vector<Value*> inputs = node->inputs();
    if (node->kind() == aten::cat && inputs.back()->type()->isSubtypeOf(*IntType::get())) {
      inputs.pop_back();
    }
    for (Value *v : inputs) {
      TensorType* type = v->type()->cast<TensorType>();
      if(!type) return at::nullopt;
      tensor_types.push_back(type);
    }
  } else {
    auto & schema = node->schema();
    auto & args = schema.arguments;
    // XXX: This gets triggered for nodes that have Tensor[] as arguments.
    // Those are currently very annoying to handle, because the lists are simply
    // inlined into the node inputs, so we bail out from shape propagation for now.
    if (schema.is_vararg || args.size() != node->inputs().size()) {
      return at::nullopt;
    }
    for (size_t i = 0; i < node->inputs().size(); ++i) {
      if (!args[i].type->isSubtypeOf(*DynamicType::get())) continue;
      TensorType *type = node->input(i)->type()->cast<TensorType>();
      if (!type) return at::nullopt;
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

void broadcastBinary(Node *node, std::vector<TensorType*>& types, size_t idx1, size_t idx2) {
  auto expected_size = at::infer_size(types[idx1]->sizes(), types[idx2]->sizes());
  auto broadcast = [&](size_t input_idx) {
    TensorType* input_type = types.at(input_idx);
    if (input_type->sizes() == expected_size)
      return;
    auto graph = node->owningGraph();
    Node *expand = graph->create(aten::expand, {node->inputs().at(input_idx)})
                        ->is_(attr::size, expected_size)
                        ->i_(attr::implicit, 0)
                        ->insertBefore(node);
    PropagateShapeOnNode(expand);
    node->replaceInput(input_idx, expand->output());
  };
  broadcast(idx1);
  broadcast(idx2);
  types[0] = node->inputs().at(idx1)->type()->expect<TensorType>();
  types[1] = node->inputs().at(idx2)->type()->expect<TensorType>();
}

void PropagateShapeOnNodeByRunningIt(Node* node, const std::vector<TensorType*>& types) {
  auto op = getOperation(node);
  Stack stack;

  size_t types_i = 0;
  // TODO (apaszke): remove once we stop using attributes
  if (node->hasAttributes()) {
    for (auto & type : types) {
      stack.push_back(representativeTensor(type));
    }
  // TODO (apaszke): remove once aten::cat is saner (see first XXX in gatherTensorTypes)
  } else if (node->kind() == aten::cat) {
    for (auto & type : types) {
      stack.push_back(representativeTensor(type));
    }
    stack.push_back(node->get(attr::dim).value());
  } else {
    JIT_ASSERT(node->schema().arguments.size() == node->inputs().size());
    for (const auto & arg : node->schema().arguments) {
      if (arg.type->isSubtypeOf(*DynamicType::get())) {
        stack.emplace_back(representativeTensor(types[types_i++]));
      } else {
        auto maybe_val = node->get(Symbol::attr(arg.name));
        if (!maybe_val) {
          setDynamicType(node);
          return;
        }
        stack.push_back(std::move(*maybe_val));
      }
    }
  }

  // XXX: we're not catching any exceptions from the op for now. This
  // is to uncover any mistakes we could make when editing this code,
  // and eventually it shouldn't matter, because this phase should be
  // preceded by schema checking.
  op(stack);

  JIT_ASSERT(stack.size() == node->outputs().size());
  for (size_t i = 0; i < stack.size(); ++i) {
    node->outputs()[i]->inferTypeFrom(stack[i].toTensor());
  }
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
    case prim::NumToTensor:
    case prim::TensorToNum: {
      node->output()->setType(node->inputs()[0]->type());
      return;
    }
    case prim::Constant: {
      node->output()->inferTypeFrom(node->t(attr::value));
      return;
    }
    case prim::PythonOp:
    case prim::Print:
    case prim::Undefined: {
      setDynamicType(node);
      return;
    }
    default:
      break; // fall-through
  }

  auto maybe_tensor_types = gatherTensorTypes(node);
  if (!maybe_tensor_types) {
    return setDynamicType(node);
  }
  auto & tensor_types = *maybe_tensor_types;

  // For expensive ops we can directly encode their shape propagation
  // here, otherwise we fallback to running a fake version of the op
  // to get a quick and dirty propagation.
  if (insert_expands && (
      node->matches("aten::add(Tensor self, Tensor other, *, Scalar alpha) -> Tensor") ||
      node->matches("aten::sub(Tensor self, Tensor other, *, Scalar alpha) -> Tensor") ||
      node->matches("aten::mul(Tensor self, Tensor other) -> Tensor") ||
      node->matches("aten::div(Tensor self, Tensor other) -> Tensor") ||
      node->matches("aten::pow(Tensor self, Tensor exponent) -> Tensor") ||
      node->matches("aten::min(Tensor self, Tensor other) -> Tensor") ||
      node->matches("aten::max(Tensor self, Tensor other) -> Tensor") ||
      node->matches("aten::lt(Tensor self, Tensor other) -> Tensor") ||
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
    node->output()->setType(std::make_shared<TensorType>(
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
  } else if (node->matches("aten::narrow(Tensor self, int dim, int start, int length) -> Tensor") &&
             node->knows(attr::dim) && node->knows(attr::length)) {
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
  } else if (node->matches("aten::sum(Tensor self, int[] dim, int keepdim) -> Tensor") &&
             node->knows(attr::dim) && node->knows(attr::keepdim)) {
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
  } else if (node->matches("aten::squeeze(Tensor self, int dim) -> Tensor") &&
             node->knows(attr::dim)) {
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
  } else if (node->matches("aten::unsqueeze(Tensor self, int dim) -> Tensor") &&
             node->knows(attr::dim)) {
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
  } else if (node->matches("aten::view(Tensor self, int[] size) -> Tensor") &&
             node->knows(attr::size)) {
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
      node->output()->setType(node->getValue(attr::self)->type());
    } else {
      // This will be a copy, so the result will be contiguous
      node->output()->setType(tensor_types.at(1)->withSizes(tensor_types.at(0)->sizes()));
    }
    return;
  } else if (node->matches("aten::expand(Tensor self, int[] size, *, int implicit) -> Tensor") &&
             node->knows(attr::size)) {
    auto tp = tensor_types.at(0);
    std::vector<int64_t> sizes, strides;
    std::tie(sizes, strides) = at::inferExpandGeometry(
        tp->sizes(), tp->strides(), node->get<std::vector<int64_t>>(attr::size).value());
    node->output()->setType(tp->withSizesStrides(sizes, strides));
    return;
  } else if (node->matches("aten::index_select(Tensor self, int dim, Tensor index) -> Tensor") &&
             node->knows(attr::dim)) {
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
        std::make_shared<TensorType>(at::kLong, -1, dims));
    return;
  } else if (node->kind() == onnx::Reshape) {
    setDynamicType(node);
    return;
  }

  // If we haven't managed to handle the op so far, we fall back to inferring the
  // shapes by doing an example run of the op (if we can).
  // Integral typed inputs are often an indicator that we're indexing into
  // a tensor, so we should special-case these ops in the shape propagation.
  // Additionally, passing in a zero representative tensor into an integer
  // division op causes divide-by-zero errors
  bool shape_inferenceable = !std::any_of(tensor_types.begin(), tensor_types.end(), [](TensorType* t){
    return at::isIntegralType(t->scalarType());
  });
  if (shape_inferenceable) {
    PropagateShapeOnNodeByRunningIt(node, tensor_types);
  } else {
    setDynamicType(node);
  }
}

void PropagateShapeOnBlock(Block * block, bool insert_expands) {
  for (Node * node : block->nodes()) {
    try {
      PropagateShapeOnNode(node, insert_expands);
    } catch(propagation_error& e) {
      setDynamicType(node);
    } catch(std::exception & e) {
      if(auto sl = node->getSourceLocation()) {
        sl->wrapAndRethrowException(e, "operation failed shape propagation");
      } else {
        throw;
      }
    }
  }
}

}
void PropagateInputShapes(Graph & graph, const ArgumentSpec & spec) {
  JIT_ASSERT(graph.inputs().size() == spec.size());
  for(size_t i = 0; i < spec.size(); ++i) {
    graph.inputs()[i]->setType(spec.tensorInfo(i));
  }
  PropagateShapeOnBlock(graph.block());
}

}}
