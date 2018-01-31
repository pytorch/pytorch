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
  std::vector<const TensorType*> types;
  // get all the input types, propagate unknown types if we don't have
  // valid tensor types for the inputs
  for(auto input : node->inputs()) {
    if(!input->hasType()) {
      return SetUnknownType(node);
    }
    if(TensorType * t = input->type()->expect<TensorType>()) {
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
      break;
    } default: {
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

}

void PropagateInputShapes(Graph & graph, const ArgumentSpec & spec) {
  JIT_ASSERT(graph.inputs().size() == spec.size());
  for(size_t i = 0; i < spec.size(); ++i) {
    graph.inputs()[i]->setType(spec.tensorInfo(i));
  }
  for(auto n : graph.nodes()) {
    PropagateShapeOnNode(n);
  }
}

}}
