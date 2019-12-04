#pragma once

#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/types.h>
#include <torch/nn/options/normalization.h>

using namespace torch::autograd;

namespace torch {
namespace nn {
namespace functions {

class CrossMapLRN2d : public torch::autograd::Function<CrossMapLRN2d> {
 public:
  static Variable forward(
      AutogradContext *ctx, 
      const Variable& input,
      const CrossMapLRN2dOptions& options);
  
  static variable_list backward(AutogradContext *ctx, variable_list grad_output);
};

}
}
}
