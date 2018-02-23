#include <Python.h>
#include "torch/csrc/jit/passes/shape_analysis.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/argument_spec.h"
#include "torch/csrc/jit/generated/aten_dispatch.h"

#include <iostream>

namespace torch { namespace jit {

namespace {
void SetUnknownType(Node * node) {
  for(auto o : node->outputs()) {
    o->setType(nullptr);
  }
}

at::Tensor representativeTensor(const TensorType * type) {
  auto backend = type->device() == -1 ? at::kCPU : at::kCUDA;
  auto & attype = at::getType(backend, type->scalarType());
  return attype.tensor(type->sizes(), type->strides()).zero_();
}

std::vector<at::Tensor> runNode(Node * n, ArrayRef<at::Tensor> inputs) {
  // this is verbose because ATen dispatch works with raw retainable pointers
  // if we end up no longer needing retainable versions after the JIT changes
  // this code could be simplified
  list_of_retainable rinputs;
  list_of_retainable routputs;
  for(auto i : inputs) {
    rinputs.push_back(i.get());
  }
  getTensorOp(n).op(rinputs, routputs);
  std::vector<at::Tensor> outputs;
  for(auto & i : routputs) {
    outputs.push_back(unsafeToTensorSteal(std::move(i)));
  }
  return outputs;
}

void PropagateShapeOnNode(Node * node) {
  std::vector<TensorType*> types;
  // get all the input types, propagate unknown types if we don't have
  // valid tensor types for the inputs
  for(auto input : node->inputs()) {
    if(!input->hasType()) {
      return SetUnknownType(node);
    }
    if(TensorType * t = input->type()->cast<TensorType>()) {
      types.push_back(t);
    } else {
      return SetUnknownType(node);
    }
  }

  switch(node->kind()) {
    //TODO: for expensive ops we can directly encode their shape propagation
    // here, otherwise we fallback to running a fake version of the op
    // to get a quick and dirty propagation
    case kneg: {
      node->output()->setType(types[0]->contiguous());
    } break;
    case kmm: {
      auto lhs_type = types.at(0);
      auto rhs_type = types.at(1);
      node->output()->setType(std::make_shared<TensorType>(
        lhs_type->scalarType(), lhs_type->device(),
        at::IntList{lhs_type->sizes()[0], rhs_type->sizes()[1]}));
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
      node->output()->setType(tp->withSizes(sizes));
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
      node->output()->setType(types[0]->shared_from_this());
    } break;
    case kConstant: {
      node->output()->inferTypeFrom(node->t(kvalue));
    } break;
    case kUndefined: {
      node->output()->setType(nullptr);
    } break;
    default: {
      auto op = getTensorOp(node);
      std::vector<at::Tensor> inputs;
      for(auto & type : types) {
        inputs.push_back(representativeTensor(type));
      }
      auto outputs = runNode(node, inputs);
      for(size_t i = 0; i < outputs.size(); ++i) {
        node->outputs()[i]->inferTypeFrom(outputs[i]);
      }
    }
  }

}

void PropagateShapeOnBlock(Block * block) {
  for (Node * node : block->nodes()) {
    PropagateShapeOnNode(node);
    for (Block * sub_block : node->blocks()) {
      PropagateShapeOnBlock(sub_block);
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
