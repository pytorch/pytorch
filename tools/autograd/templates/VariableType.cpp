// start of bad stuff copy pasted from Functions.cpp

// NB: Must be at the top of file to avoid including the deprecated "math.h".
// https://stackoverflow.com/questions/6563810/m-pi-works-with-math-h-but-not-with-cmath-in-visual-studio
#ifdef _MSC_VER
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#endif

#include "torch/csrc/autograd/generated/Functions.h"
#include <ATen/Utils.h>
#include <c10/core/TensorOptions.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/ExpandUtils.h>
#include <ATen/core/Reduction.h>
#include <ATen/Dispatch.h>

#include <ciso646>
#include <algorithm>
#include <numeric>
#include <functional>

// end of bad stuff

#include "torch/csrc/autograd/VariableTypeUtils.h"

#include <ATen/TypeDefault.h>
#include <torch/library.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/csrc/autograd/anomaly_mode.h>
#include <ATen/Utils.h>

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

namespace {
  
${manual_backward_functions}

}

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
