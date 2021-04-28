#pragma once

#include <torch/optim/optimizer.h>

#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch {
namespace optim {

class TORCH_API LRScheduler {
public:

  //This class needs to take a reference of an optimizer from outside such that it can
  //modify its learning rates; due to this the lifetime of said optimizer must be maintained
  LRScheduler(torch::optim::Optimizer& optimizer);

  virtual ~LRScheduler() = default;

  void step();

protected:
  //A vector of learning rates is calculated and returned from the specific subclass.
  //A vector is returned with each element being a separate learning rate for each
  //param group - although the normal use case would be to return a vector of identical
  //elements.
  virtual std::vector<double> get_lrs() = 0;

  //Get current learning rates from the optimizer
  std::vector<double> get_current_lrs() const;

  unsigned step_count_;

private:
  void set_optimizer_lrs(const std::vector<double>& learning_rates);

  torch::optim::Optimizer& optimizer_;

};
} // namespace optim
} // namspace torch
