#include "torch/csrc/autograd/functions/batch_normalization.h"

namespace torch {
namespace autograd {

void BatchNormForward::primspec(PrimSpecContext* ctx, jit::node_list inputs, jit::node_list outputs) {
  toffee::NodeProto* p_n = ctx->graph->add_node();
  p_n->set_op_type("SpatialBN");

  // X, Scale, Bias
  for (auto n : inputs) {
    p_n->add_input(ctx->node(n));
  }
  for (auto n : outputs) {
    p_n->add_output(ctx->node(n));
  }

  toffee::AttributeProto* attr;

  #define ADD_ATTR(name,format,value) \
    attr = p_n->add_attribute(); \
    attr->set_name(name); \
    attr->set_##format(value);

  ADD_ATTR("is_test",i,0);
  ADD_ATTR("epsilon",f,eps);
  ADD_ATTR("order",s,"NCHW");
  ADD_ATTR("momentum",f,momentum);

  auto sm = "saved_mean"+std::to_string(ctx->batch_norm_count);
  auto sv = "saved_var"+std::to_string(ctx->batch_norm_count);
  ctx->graph->add_input(sm);
  ctx->graph->add_input(sv);
  p_n->add_input(sm);
  p_n->add_input(sv);
  ctx->batch_norm_count++;
}

} // torch::autograd
} // torch
