#include <Python.h>
#include "torch/csrc/jit/passes/shape_analysis.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/argument_spec.h"
#include "torch/csrc/jit/generated/aten_dispatch.h"

#include <iostream>

namespace torch { namespace jit {

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
  std::vector<TensorType*> types;
  bool present;
  std::tie(types, present) = gatherTypes(node->inputs());
  if(!present) {
    return setDynamicType(node);
  }

  switch(node->kind()) {
    //TODO: for expensive ops we can directly encode their shape propagation
    // here, otherwise we fallback to running a fake version of the op
    // to get a quick and dirty propagation
    case kneg: {
      node->output()->setType(types.at(0)->contiguous());
    } break;
    case kmm: {
      auto lhs_type = types.at(0);
      auto rhs_type = types.at(1);
      node->output()->setType(std::make_shared<TensorType>(
        lhs_type->scalarType(), lhs_type->device(),
        at::IntList{lhs_type->sizes().at(0), rhs_type->sizes().at(1)}));
    } break;
    case kt: {
      auto tp = types.at(0);
      auto sizes = tp->sizes();
      auto strides = tp->strides();
      std::swap(sizes.at(0), sizes.at(1));
      std::swap(strides.at(0), strides.at(1));
      node->output()->setType(tp->withSizesStrides(sizes, strides));
    } break;
    case knarrow: {
      auto tp = types.at(0);
      auto sizes = tp->sizes();
      int64_t dim = node->i(kdim);
      int64_t length = node->i(klength);
      sizes.at(dim) = length;
      node->output()->setType(tp->withSizesStrides(sizes, tp->strides()));
    } break;
    case ksum: {
      if (node->hasAttribute(kdim)) {
        auto tp = types.at(0);
        auto sizes = tp->sizes();
        int64_t dim = node->i(kdim);
        JIT_ASSERT(dim >= 0 && static_cast<size_t>(dim) < sizes.size());
        if (node->i(kkeepdim)) {
          sizes[dim] = 1;
        } else {
          sizes.erase(sizes.begin() + dim);
        }
        node->output()->setType(tp->withSizes(sizes));
      } else {
        node->output()->setType(types.at(0)->withSizes({}));
      }
    } break;
    case ksqueeze: {
      auto tp = types.at(0);
      auto sizes = tp->sizes();
      auto strides = tp->strides();
      int64_t dim = node->i(kdim);
      JIT_ASSERT(dim >= 0 && static_cast<size_t>(dim) < sizes.size());
      if (sizes[dim] == 1) {
        sizes.erase(sizes.begin() + dim);
        strides.erase(strides.begin() + dim);
      }
      node->output()->setType(tp->withSizesStrides(sizes, strides));
    } break;
    case kunsqueeze: {
      auto tp = types.at(0);
      auto sizes = tp->sizes();
      auto strides = tp->strides();
      int64_t dim = node->i(kdim);
      JIT_ASSERT(dim >= 0 && static_cast<size_t>(dim) <= sizes.size());
      sizes.insert(sizes.begin() + dim, 1);
      strides.insert(strides.begin() + dim, 1);
      node->output()->setType(tp->withSizesStrides(sizes, strides));
    } break;
    case kview: {
      node->output()->setType(types.at(0)->withSizes(node->is(ksizes)));
    } break;
    case kReplaceIfUndef: {
      // If types[0] has a type, then it is not defined, and the type will
      // get set to types[0] because that will be the value propagated.
      // If its type is not defined, then unification is an undefined type.
      node->output()->setType(types.at(0)->shared_from_this());
    } break;
    case kConstant: {
      node->output()->inferTypeFrom(node->t(kvalue));
    } break;
    case kUndefined: {
      node->output()->setType(DynamicType::get());
    } break;
    case kIf: {
      auto then_block = node->blocks().at(0);
      auto else_block = node->blocks().at(1);
      PropagateShapeOnBlock(then_block);
      PropagateShapeOnBlock(else_block);
      mergeTypes(then_block->outputs(), else_block->outputs(), node->outputs());
    } break;
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
    } break;
    default: {
      auto op_info = getTensorOp(node);
      std::vector<at::Tensor> stack;
      for(auto & type : types) {
        stack.push_back(representativeTensor(type));
      }
      op_info.op(stack);
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
