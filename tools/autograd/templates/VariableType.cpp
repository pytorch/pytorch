#include "torch/csrc/autograd/VariableTypeUtils.h"

#include <ATen/TypeDefault.h>
#include <torch/library.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/csrc/autograd/anomaly_mode.h>

// ${generated_comment}

// NOTE [Sharded File]: on this file's split-into-shards state
//
// Back in the good old days, VariableType.cpp was generated as one
// file with every function in it, and everything was great and
// simple.
//
// However, this file was also very large (over 36,000 lines), and
// compiling it was very slow, and in fact was a significant
// bottleneck for incremental rebuilds. To address this, we now
// generate the file split across multiple shards, named
// VariableType_0.cpp and so on, which can be compiled in parallel.
//
// For ease of inspection and debugging, so that it's not necessary to
// go rooting around in multiple files, we also generate all the
// functions together in VariableTypeEverything.cpp. This generated
// file is only for convenience; it's not actually used in the
// build. If the file you're looking at now is one of the shards, you
// may want to switch over to the Everything variant to make you
// grepping smoother.

using namespace at;
using namespace torch::autograd::generated;

namespace torch { namespace autograd {

namespace VariableType {
namespace{
  void reset_grad_accumulator(Variable & self) {
    AutogradMeta* meta = torch::autograd::impl::get_autograd_meta(self);
    if (meta != nullptr) {
      meta->grad_accumulator_.reset();
    }
  }
  Tensor maybe_multiply(const Tensor & t, const Scalar & s) {
    bool is_one = false;
    if (s.isFloatingPoint()) {
      is_one = s.toDouble() == 1;
    } else if(s.isIntegral(true)) {
      is_one = s.toLong() == 1;
    }

    if (is_one) {
      return t;
    } else {
      return t * s;
    }
  }
}

namespace {
${type_derived_method_definitions}
}
}

namespace {

TORCH_LIBRARY_IMPL(aten, Autograd, m) {
  ${wrapper_registrations}
}

}

}} // namespace torch::autograd
