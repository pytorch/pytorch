#include <torch/optim/schedulers/step_lr.h>

namespace torch {
namespace optim {

StepLR::StepLR(torch::optim::Optimizer& optimizer,
               const unsigned step_size,
               const double gamma) :
  LRScheduler(optimizer),
  step_size_(step_size),
  gamma_(gamma) {}

std::vector<double> StepLR::get_lrs() {
  if(step_count_ == 0 || step_count_ % step_size_ != 0)
    return get_current_lrs();
  else {
    std::vector<double> lrs = get_current_lrs();
    std::transform(lrs.begin(), lrs.end(), lrs.begin(),
                   [this](const double& v){ return this->gamma_ * v; });
    return lrs;
  }
}

} // namespace optim
} // namespace torch
