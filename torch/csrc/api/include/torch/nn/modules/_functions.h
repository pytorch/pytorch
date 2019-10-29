#pragma once

#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/types.h>

using namespace torch::autograd;

namespace torch{
namespace nn{

class cross_map_lrn2d : public torch::autograd::Function<cross_map_lrn2d> {
 public:
  static Variable forward(
      AutogradContext *ctx, 
      Variable input, 
      const int64_t size, 
      const double alpha=1e-4, 
      const double beta=0.75, 
      const int64_t k=1);
  
  static variable_list backward(AutogradContext *ctx, variable_list grad_output);
};

}
}
