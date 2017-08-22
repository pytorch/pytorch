#include "torch/csrc/autograd/functions/batch_normalization.h"
#include <sstream>

namespace torch {
namespace autograd {

void BatchNormForward::primspec(PrimSpecContext* ctx, jit::node_list inputs, jit::node_list outputs) {
  toffee::NodeProto* p_n = ctx->graph->add_node();
  p_n->set_op_type("SpatialBN");

  // X, Scale, Bias
  p_n->add_input(ctx->node(inputs[0]));
  p_n->add_input(ctx->node(inputs[1]));
  // TODO: Factor this logic into a helper, and make sure it gets applied
  // consistently.  See also convolution.cpp
  if (inputs[2]->kind() != jit::kConstant || inputs[2]->t(jit::kValue).defined()) {
    p_n->add_input(ctx->node(inputs[2]));
  }

  p_n->add_output(ctx->node(outputs[0]));
  JIT_ASSERT(outputs.at(1)->type()->kind() == jit::TypeKind::HandleType);

  toffee::AttributeProto* attr;

  #define ADD_ATTR(name,format,value) \
    attr = p_n->add_attribute(); \
    attr->set_name(name); \
    attr->set_##format(value);

  ADD_ATTR("is_test",i,0);
  ADD_ATTR("epsilon",f,eps);
  ADD_ATTR("order",s,"NCHW");
  ADD_ATTR("momentum",f,1-momentum);

  auto typ = inputs.at(1)->type()->cast<torch::jit::TensorType>();
  int64_t the_size = typ->sizes()[0];
  std::stringstream ss;
  ss << the_size << "_" << ctx->batch_norm_count;
  std::string suffix = ss.str();
  auto sm = "saved_mean_" + suffix;
  auto sv = "saved_var_" + suffix;
  ctx->graph->add_input(sm);
  ctx->graph->add_input(sv);
  p_n->add_input(sm);
  p_n->add_output(sm);
  p_n->add_input(sv);
  p_n->add_output(sv);

  // dummy output
  for(int i = 3; i < 5; i++) {
    p_n->add_output(
        "batch_norm_dead_output_" + std::to_string(i)+std::to_string(ctx->batch_norm_count)
    );
  }

  ctx->batch_norm_count++;
}

} // torch::autograd
} // torch
