#include "torch/csrc/autograd/functions/batch_normalization.h"
#include <sstream>

#include "torch/csrc/jit/ir.h"

namespace torch {
namespace autograd {

jit::node_list BatchNormForward::primspec(PrimSpecContext* ctx, jit::node_list inputs) {
  auto & g = ctx->graph;
  // X, Scale, Bias
  auto bn = g->appendNode(g->create(jit::kSpatialBN,{inputs.at(0),inputs.at(1),inputs.at(2)}));
  bn->addInput(jit::tracer::getBufferTrace(*ctx->buffer_map, running_mean));
  bn->addInput(jit::tracer::getBufferTrace(*ctx->buffer_map, running_var));
  bn->i_(jit::kis_test, !this->training);
  bn->f_(jit::kepsilon, eps);
  bn->s_(jit::korder, "NCHW");
  bn->f_(jit::kmomentum, 1 - momentum);

  auto orig_output = g->appendNode(g->createSelect(bn, 0));

  if(this->training) {
    g->appendNode(g->createSelect(bn, 1)->setType(bn->inputs().at(3)->type()));
    g->appendNode(g->createSelect(bn, 2)->setType(bn->inputs().at(4)->type()));
    // dummy output
    for(int i = 3; i < 5; i++) {
      g->appendNode(g->createSelect(bn, i)->setDebugName("batch_norm_dead_output"));
    }
  }
  bn->is_(jit::kconsumed_inputs,{0,0,0,1,1});

  ctx->batch_norm_count++;
  return {orig_output};
}


} // torch::autograd
} // torch
