#include "torch/csrc/autograd/functions/batch_normalization.h"
#include <sstream>

#include "torch/csrc/jit/ir.h"

namespace torch {
namespace autograd {

jit::node_list BatchNormForward::primspec(PrimSpecContext* ctx, jit::node_list inputs) {
  auto & g = ctx->graph;
  auto bn = g->appendNode(g->create(jit::kSpatialBN,{inputs.at(0),inputs.at(1)}));
  // X, Scale, Bias
  // TODO: Factor this logic into a helper, and make sure it gets applied
  // consistently.  See also convolution.cpp
  if (inputs[2]->kind() != jit::kConstant || inputs.at(2)->t(jit::kValue).defined()) {
    bn->addInput(inputs[2]);
  }

  bn->i_(jit::kis_test, !this->training);
  bn->f_(jit::kepsilon, eps);
  bn->s_(jit::korder,"NCHW");
  bn->f_(jit::kmomentum,1-momentum);

  std::vector<int64_t> inplace_outputs;
  auto orig_output = g->appendNode(g->createSelect(bn, 0));
  inplace_outputs.push_back(-1);

  auto typ = inputs.at(1)->type()->cast<torch::jit::TensorType>();
  int64_t the_size = typ->sizes()[0];
  std::stringstream ss;
  ss << the_size << "_" << ctx->batch_norm_count;
  std::string suffix = ss.str();
  auto sm = g->addInput()->setDebugName("saved_mean_" + suffix);
  auto sv = g->addInput()->setDebugName("saved_var_" + suffix);

  int64_t sm_start = bn->inputs().size();
  bn->addInput(sm);
  bn->addInput(sv);

  if(this->training) {
    g->appendNode(g->createSelect(bn, 1));
    inplace_outputs.push_back(sm_start);
    g->appendNode(g->createSelect(bn, 2));
    inplace_outputs.push_back(sm_start+1);
    // dummy output
    for(int i = 3; i < 5; i++) {
      g->appendNode(g->createSelect(bn, i)->setDebugName("batch_norm_dead_output"));
      inplace_outputs.push_back(-1);
    }
  }
  bn->is_(jit::kInPlaceOutputs,std::move(inplace_outputs));
  ctx->batch_norm_count++;
  return {orig_output,  g->create(jit::kUnused)};
}


} // torch::autograd
} // torch
