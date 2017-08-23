#include "torch/csrc/autograd/functions/convolution.h"

namespace torch { namespace autograd {

template<typename T>
static T all_equal(at::ArrayRef<T> ts, const char * name) {
  JIT_ASSERT(ts.size() > 0);
  auto v = ts[0];
  for(auto t : ts) {
    JIT_ASSERTM(v == t, "all elements of %s must be the same for transposed", name);
  }
  return v;
}

jit::node_list ConvForward::primspec(PrimSpecContext* ctx, jit::node_list inputs) {
  auto & g = ctx->graph;
  auto n = g->appendNode(g->create(!transposed ? jit::kConv : jit::kConvTranspose, {inputs.at(0),inputs.at(1)}));

  // TODO: Factor this logic into a helper, and make sure it gets applied
  // consistently. See also batch_normalization.cpp
  if (inputs.at(2)->kind() != jit::kConstant || inputs.at(2)->t(jit::kValue).defined()) {
    n->addInput(inputs.at(2));
  }

  // Irritatingly, Caffe2 requires us to specify kernels,
  // but we don't actually have that information directly
  // recorded in ConvForward.  So we have to reverse
  // engineer it from the input types...
  // TODO: dynamic_cast ew
  auto weight_type = inputs.at(1)->type()->cast<jit::TensorType>();
  JIT_ASSERT(weight_type);
  auto weight_size = weight_type->sizes();

  if(!transposed) {
    std::vector<int64_t> kernel_size(weight_size.begin() + 2, weight_size.end());
    n->is_(jit::kkernels,std::move(kernel_size));
    std::vector<int64_t> kernel_stride(stride.begin(),stride.end());
    n->is_(jit::kstrides,std::move(kernel_stride));

    std::vector<int64_t> kernel_pads(padding.begin(),padding.end());
    // NB: Caffe2 let's specifying top and bottom pads separately;
    // PyTorch assumes it's symmetric
    for (int p : padding) {
      kernel_pads.push_back(p);
    }
    n->is_(jit::kpads,std::move(kernel_pads));

    std::vector<int64_t> kernel_dilations(dilation.begin(),dilation.end());
    n->is_(jit::kdilations,std::move(kernel_dilations));
    // Not in Toffee?
    for (int p : output_padding) {
      JIT_ASSERT(p == 0);
    }
    n->i_(jit::kgroup,groups);
  } else {
    n->i_(jit::kkernel, all_equal<int64_t>(weight_size, "kernels"));
    n->i_(jit::kstride, all_equal<int>(stride, "strides"));
    n->i_(jit::kpad, all_equal<int>(padding,"padding"));
    JIT_ASSERT(1 == all_equal<int>(dilation,"dialations"));
  }
  // ignore benchmark/cudnn_enabled
  return {n, g->create(jit::kUnused) };
}

}}
