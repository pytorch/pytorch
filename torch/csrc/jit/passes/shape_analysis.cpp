#include "torch/csrc/jit/passes/shape_analysis.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/argument_spec.h"
#include "torch/csrc/jit/generated/aten_dispatch.h"

#include <iostream>

namespace torch { namespace jit {

struct propagation_error : std::exception {};

#define SHAPE_ASSERT(cond) if (!(cond)) throw propagation_error()

namespace {
void setDynamicType(Node * node) {
  for(auto o : node->outputs()) {
    o->setType(DynamicType::get());
  }
}

at::Tensor representativeTensor(const TensorType * type) {
  auto backend = type->device() == -1 ? at::kCPU : at::kCUDA;
  auto & attype = at::getType(backend, type->scalarType());
  return attype.tensor(type->sizes(), type->strides()).zero_();
}

void PropagateShapeOnBlock(Block * block);

std::pair<std::vector<TensorType*>, bool> gatherTypes(at::ArrayRef<Value*> values) {
  std::vector<TensorType*> types;
  bool present = true;
  for(auto v : values) {
    TensorType* type = v->type()->cast<TensorType>();
    if(!type)
      present = false;
    types.push_back(type);
  }
  return std::make_pair(std::move(types), present);
}

bool mergeTypes(ArrayRef<Value*> lhs, ArrayRef<Value*> rhs, ArrayRef<Value*> outputs) {
  JIT_ASSERT(lhs.size() == rhs.size() && rhs.size() == outputs.size());
  bool changed = false;
  for(size_t i = 0; i < lhs.size(); ++i) {
    if(*lhs[i]->type() == *rhs[i]->type()) {
      outputs[i]->setType(lhs[i]->type());
    } else {
      outputs[i]->setType(DynamicType::get());
      changed = true;
    }
  }
  return changed;
}

void PropagateShapeOnNode(Node * node) {
  using AKind = AttributeKind;
  // These don't require the types and present flag. Return early after we
  // process them
  switch(node->kind()) {
    case kIf: {
      auto then_block = node->blocks().at(0);
      auto else_block = node->blocks().at(1);
      PropagateShapeOnBlock(then_block);
      PropagateShapeOnBlock(else_block);
      mergeTypes(then_block->outputs(), else_block->outputs(), node->outputs());
      return;
    }
    case kLoop: {
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
        PropagateShapeOnBlock(body_block);
      } while(mergeTypes(loop_carried_block, loop_carried_outputs, loop_carried_block));

      for(size_t i = 0; i < loop_carried_inputs.size(); ++i) {
        node->outputs()[i]->setType(loop_carried_block[i]->type());
      }
      return;
    }
    default: ; // fall-through
  }

  std::vector<TensorType*> types;
  bool present;
  std::tie(types, present) = gatherTypes(node->inputs());
  if(!present) {
    return setDynamicType(node);
  }

  bool handled = false;
  // XXX: real attributes of node can be a superset of attrs
  // XXX: if this returns true then you are obliged to set the types
  auto check_overload = [&](size_t num_inputs, size_t num_outputs,
                            std::vector<std::pair<AttributeKind,Symbol>> attrs = {}) -> bool {
    JIT_ASSERT(!handled);
    if (node->inputs().size() != num_inputs) return false;
    if (node->outputs().size() != num_outputs) return false;
    for (auto & attr : attrs) {
      if (!node->hasAttribute(attr.second)) return false;
      if (node->kindOf(attr.second) != attr.first) return false;
    }
    handled = true;
    return true;
  };

  switch(node->kind()) {
    //TODO: for expensive ops we can directly encode their shape propagation
    // here, otherwise we fallback to running a fake version of the op
    // to get a quick and dirty propagation
    case kneg: {
      if (!check_overload(/*num_inputs=*/1, /*num_outputs=*/1)) break;
      node->output()->setType(types.at(0)->contiguous());
    } break;
    case kmm: {
      if (!check_overload(/*num_inputs=*/2, /*num_outputs=*/1)) break;
      auto lhs_type = types.at(0);
      auto rhs_type = types.at(1);
      SHAPE_ASSERT(lhs_type->sizes().size() == 2 && rhs_type->sizes().size() == 2);
      node->output()->setType(std::make_shared<TensorType>(
        lhs_type->scalarType(), lhs_type->device(),
        at::IntList{lhs_type->sizes().at(0), rhs_type->sizes().at(1)}));
    } break;
    case kt: {
      if (!check_overload(/*num_inputs=*/1, /*num_outputs=*/1)) break;
      auto tp = types.at(0);
      auto sizes = tp->sizes();
      auto strides = tp->strides();
      SHAPE_ASSERT(sizes.size() == 2);
      std::swap(sizes.at(0), sizes.at(1));
      std::swap(strides.at(0), strides.at(1));
      node->output()->setType(tp->withSizesStrides(sizes, strides));
    } break;
    case knarrow: {
      if (check_overload(/*num_inputs=*/1, /*num_outputs=*/1,
                         {{AKind::i, kdim},
                          {AKind::i, klength}})) {
        auto tp = types.at(0);
        auto sizes = tp->sizes();
        int64_t dim = node->i(kdim);
        int64_t length = node->i(klength);
        SHAPE_ASSERT(dim >= 0 && static_cast<size_t>(dim) < sizes.size());
        sizes.at(dim) = length;
        node->output()->setType(tp->withSizesStrides(sizes, tp->strides()));
      }
    } break;
    case ksum: {
      if (check_overload(/*num_inputs=*/1, /*num_outputs=*/1,
                         {{AKind::i, kdim},
                          {AKind::i, kkeepdim}})) {
        auto tp = types.at(0);
        auto sizes = tp->sizes();
        int64_t dim = node->i(kdim);
        SHAPE_ASSERT(dim >= 0 && static_cast<size_t>(dim) < sizes.size());
        if (node->i(kkeepdim)) {
          sizes.at(dim) = 1;
        } else {
          sizes.erase(sizes.begin() + dim);
        }
        node->output()->setType(tp->withSizes(sizes));
      } else if (check_overload(/*num_inputs=*/1, /*num_outputs=*/1)) {
        node->output()->setType(types.at(0)->withSizes({}));
      }
    } break;
    case ksqueeze: {
      if (check_overload(/*num_inputs=*/1, /*num_outputs=*/1,
                         {{AKind::i, kdim}})) {
        auto tp = types.at(0);
        auto sizes = tp->sizes();
        auto strides = tp->strides();
        int64_t dim = node->i(kdim);
        SHAPE_ASSERT(dim >= 0 && static_cast<size_t>(dim) < sizes.size());
        if (sizes.at(dim) == 1) {
          sizes.erase(sizes.begin() + dim);
          strides.erase(strides.begin() + dim);
        }
        node->output()->setType(tp->withSizesStrides(sizes, strides));
      }
    } break;
    case kunsqueeze: {
      if (check_overload(/*num_inputs=*/1, /*num_outputs=*/1,
                         {{AKind::i, kdim}})) {
        auto tp = types.at(0);
        auto sizes = tp->sizes();
        auto strides = tp->strides();
        int64_t dim = node->i(kdim);
        SHAPE_ASSERT(dim >= 0 && static_cast<size_t>(dim) <= sizes.size());
        sizes.insert(sizes.begin() + dim, 1);
        strides.insert(strides.begin() + dim, 1);
        node->output()->setType(tp->withSizesStrides(sizes, strides));
      }
    } break;
    case kview: {
      if (check_overload(/*num_inputs=*/1, /*num_outputs=*/1,
                         {{AKind::is, ksize}})) {
        auto sizes = node->is(ksize);
        bool inferred = false;
        size_t inferred_idx;
        int64_t size_product = 1;
        for (size_t i=0; i<sizes.size(); ++i) {
          if (sizes[i] == -1) {
            if (inferred) throw propagation_error();
            inferred = true;
            inferred_idx = i;
          } else {
            size_product *= sizes[i];
          }
        }

        if (inferred) {
          auto rep_ten = representativeTensor(types[0]);
          SHAPE_ASSERT(size_product != 0);
          int64_t inferred_size = rep_ten.numel() / size_product;
          sizes[inferred_idx] = inferred_size;
        }
        node->output()->setType(types.at(0)->withSizes(sizes));
      }
    } break;
    case kReplaceIfUndef: {
      // If types[0] has a type, then it is not defined, and the type will
      // get set to types[0] because that will be the value propagated.
      // If its type is not defined, then unification is an undefined type.
      SHAPE_ASSERT(types.size() == 1);
      node->output()->setType(types.at(0)->shared_from_this());
      handled = true;
    } break;
    case kConstant: {
      node->output()->inferTypeFrom(node->t(kvalue));
      handled = true;
    } break;
    case kUndefined: {
      node->output()->setType(DynamicType::get());
      handled = true;
    } break;
    case kPythonOp: {
      setDynamicType(node);
      handled = true;
    } break;
    case kPrint: {
      setDynamicType(node);
      handled = true;
    } break;
    default: {
    } break;
  }

  // If we haven't manage to handle the op so far, we fall back to inferring the
  // shapes by doing an example run of the op (if we can).
  if (!handled) {
    auto op_info = getTensorOp(node);
    std::vector<at::Tensor> stack;
    bool shape_inferenceable = true;
    // Integral typed inputs are often an indicator that we're indexing into
    // a tensor, so we should special-case these ops in the shape propagation.
    // Additionally, passing in a zero representative tensor into an integer
    // division op causes divide-by-zero errors
    for(auto & type : types) {
      if (at::isIntegralType(type->scalarType())) {
        shape_inferenceable = false;
        break;
      }
      stack.push_back(representativeTensor(type));
    }
    if (!shape_inferenceable) {
      setDynamicType(node);
    } else {
      // XXX: we're not catching any exceptions from the op for now. This
      // is to uncover any mistakes we could make when editing this code,
      // and eventually it shouldn't matter, because this phase should be
      // preceded by schema checking.
      op_info.op(stack);
      JIT_ASSERT(stack.size() == node->outputs().size());
      for(size_t i = 0; i < stack.size(); ++i) {
        node->outputs()[i]->inferTypeFrom(stack[i]);
      }
    }
  }
}

void PropagateShapeOnBlock(Block * block) {
  for (Node * node : block->nodes()) {
    try {
      PropagateShapeOnNode(node);
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
