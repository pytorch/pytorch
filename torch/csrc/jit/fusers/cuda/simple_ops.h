#if defined USE_CUDA && !(defined _WIN32) && !(defined USE_ROCM)
#pragma once

#include "torch/csrc/jit/fusers/cuda/cuda_fuser.h"
#include <unordered_map>

namespace torch { namespace jit {

// What is a simple mappable operator?  It is:
//    - Has an output with the same sizes as its input
//    - Single output
//    - Can handle non-contiguous input
//    - Produces contiguous output
// Some of these restrictions may be relaxable, but you should
// carefully read the code first, as we rely on these assumptions.
std::unordered_set<NodeKind> simple_mappable = {
  aten::__and__,
  aten::__lshift__,
  aten::__or__,
  aten::__rshift__,
  aten::__xor__,
  aten::abs,
  aten::acos,
  aten::add,
  aten::asin,
  aten::atan,
  aten::atan2,
  aten::ceil,
  aten::cos,
  aten::cosh,
  aten::div,
  aten::eq,
  aten::exp,
  aten::expm1,
  aten::floor,
  aten::fmod,
  aten::frac,
  aten::ge,
  aten::gt,
  aten::le,
  aten::lgamma,
  aten::log,
  aten::log10,
  aten::log1p,
  aten::log2,
  aten::lt,
  aten::max,
  aten::min,
  aten::mul,
  aten::ne,
  aten::neg,
  aten::pow,
  aten::reciprocal,
  aten::relu,
  aten::remainder,
  aten::round,
  aten::rsqrt,
  aten::sigmoid,
  aten::sin,
  aten::sinh,
  aten::sqrt,
  aten::sub,
  aten::tan,
  aten::tanh,
  aten::trunc,
  aten::type_as,
  aten::_sigmoid_backward,
  aten::_tanh_backward,
  // TODO support those
  //aten::clamp,
  //aten::lerp,
  aten::rand_like,
};

bool isSimpleMap(Node* node) {
  // TODO: use signature matching
  if (simple_mappable.count(node->kind()) == 0) return false;
  if ((node->kind() == aten::min || node->kind() == aten::max) 
      && node->inputs().size() == 1)
    return false;
  return true;
}

 
} //namespace jit
} //namespace torch

#endif // defined USE_CUDA && !(defined _WIN32) && !(defined USE_ROCM)
