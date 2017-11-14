#include "torch/csrc/autograd/functions/convolution.h"

namespace torch { namespace autograd {

// Note [Caffe2ConvTranspose]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// ConvTranspose in Caffe2 is a bit silly: bias is mandatory.  But ONNX
// has removed bias input from official ConvTranspose.  How can the Caffe2
// backend do the translation?  It can't!  It's impossible!  So as a temporary
// hack while we wait for Caffe2 to make bias optional, we are using a
// Caffe2ConvTranspose experimental ONNX op which has a mandatory bias.
// PyTorch has no trouble making the zero-filled tensor.
//
// For code simplicity, even if PyTorch was given a bias tensor, it is NOT
// passed here; it's done as an external addition.  This is less efficient
// but this code should be temporary anyway.

jit::value_list ConvForward::symbolic(SymbolicContext* ctx, jit::value_list inputs) {
  auto & g = ctx->graph;
  // See Note [Caffe2ConvTranspose]
  auto n = g->create(!transposed ? jit::kConv : jit::kConvTranspose,
                                   {inputs.at(0), inputs.at(1)});

  // Irritatingly, Caffe2 requires us to specify kernels,
  // but we don't actually have that information directly
  // recorded in ConvForward.  So we have to reverse
  // engineer it from the input types...
  // TODO: dynamic_cast ew
  auto weight_type = inputs.at(1)->type()->cast<jit::TensorType>();
  JIT_ASSERT(weight_type);
  auto weight_size = weight_type->sizes();

  // See Note [Caffe2ConvTranspose]
  if(transposed) {
    auto tn = g->appendNode(g->createConstant(at::CPU(at::kFloat).zeros({weight_size[1]})));
    n->addInput(tn->output());
  }

  g->appendNode(n);

  std::vector<int64_t> kernel_size(weight_size.begin() + 2, weight_size.end());
  n->is_(jit::kkernel_shape, std::move(kernel_size));
  std::vector<int64_t> kernel_stride(stride.begin(),stride.end());
  n->is_(jit::kstrides, std::move(kernel_stride));

  std::vector<int64_t> kernel_pads(padding.begin(),padding.end());
  // NB: Caffe2 let's specifying top and bottom pads separately;
  // PyTorch assumes it's symmetric
  for (int p : padding) {
    kernel_pads.push_back(p);
  }
  n->is_(jit::kpads,std::move(kernel_pads));

  std::vector<int64_t> kernel_dilations(dilation.begin(),dilation.end());
  n->is_(jit::kdilations,std::move(kernel_dilations));
  n->i_(jit::kgroup,groups);

  // Not in ONNX?
  // TODO: implement it once ConvTranspose in ONNX gets `adj` argument instead
  // of providing `output_shape`
  for (int p : output_padding) {
    JIT_EXPECTM(p == 0, "output padding is not supported.");
  }

  // ignore benchmark/cudnn_enabled

  if (inputs.at(2)->node()->kind() != jit::kUndefined) {
    // TODO: Set type here based on RETURN type (not available atm)
    auto a_n = g->create(jit::kAdd, {n->output(), inputs.at(2)});
    a_n->i_(jit::kbroadcast, 1);
    a_n->i_(jit::kaxis, 1);
    g->appendNode(a_n);
    return {a_n->output()};
  } else {
    return {n->output()};
  }
}

}}
