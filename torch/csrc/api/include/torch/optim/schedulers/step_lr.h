#pragma once

/*
 * Decays the learning rate of each parameter group by gamma every step_size epochs.
 */

#include <torch/optim/schedulers/lr_scheduler.h>

namespace torch {
namespace optim {

class TORCH_API StepLR : public LRScheduler {
public:

  StepLR(torch::optim::Optimizer& optimizer,
         const unsigned step_size,
         const double gamma = 0.1,
         const bool verbose = false);

private:
  std::vector<double> get_lrs() override;

  const unsigned step_size_;
  const double gamma_;

};
} // namespace optim
} // namespace torch
