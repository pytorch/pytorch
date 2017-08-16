#include "torch/csrc/autograd/convolution.h"

void ConvForward::primspec(PrimSpecContext* ctx, jit::node_list inputs, jit::node_list outputs) {
  toffee::NodeProto* p_n = ctx->graph->add_node();
  p_n->set_op_type("Conv");

  // Basic (TODO: factor me out into helper on PrimSpecContext... maybe; this
  // is predicated on us not making a better builder API)
  for (auto n : inputs) {
    p_n->add_input(ctx->node(n));
  }
  for (auto n : outputs) {
    p_n->add_output(ctx->node(n));
  }

  toffee::AttributeProto* attr;
  // Irritatingly, Caffe2 requires us to specify kernels,
  // but we don't actually have that information directly
  // recorded in ConvForward.  So we have to reverse
  // engineer it from the input types...
  // TODO: dynamic_cast ew
  auto weight_type = inputs.at(1)->type()->cast<jit::TensorType>();
  JIT_ASSERT(weight_type);
  auto weight_size = weight_type->sizes();
  std::vector<int64_t> kernel_size(weight_size.begin() + 2, weight_size.end());
  attr = p_n->add_attribute();
  attr->set_name("kernels");
  for (int kernel : kernel_size) {
    attr->add_ints(kernel);
  }

  attr = p_n->add_attribute();
  attr->set_name("strides");
  for (int s : stride) {
    attr->add_ints(s);
  }
  attr = p_n->add_attribute();
  attr->set_name("pads");
  for (int p : padding) {
    attr->add_ints(p);
  }
  // NB: Caffe2 let's specifying top and bottom pads separately;
  // PyTorch assumes it's symmetric
  for (int p : padding) {
    attr->add_ints(p);
  }
  attr = p_n->add_attribute();
  attr->set_name("dilations");
  for (int d : dilation) {
    attr->add_ints(d);
  }
  // Not in Toffee?
  JIT_ASSERT(transposed == false);
  for (int p : output_padding) {
    JIT_ASSERT(p == 0);
  }
  attr = p_n->add_attribute();
  attr->set_name("group");
  attr->set_i(groups);
  // ignore benchmark/cudnn_enabled
}

