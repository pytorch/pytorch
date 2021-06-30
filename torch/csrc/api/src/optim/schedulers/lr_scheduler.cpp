#include <torch/optim/schedulers/lr_scheduler.h>

namespace torch {
namespace optim {

LRScheduler::LRScheduler(torch::optim::Optimizer& optimizer) :
  step_count_(0),
  optimizer_(optimizer) {}

void LRScheduler::step() {
  std::vector<double> learning_rates = get_lrs();
  set_optimizer_lrs(learning_rates);
  step_count_++;
}

void LRScheduler::set_optimizer_lrs(const std::vector<double>& learning_rates) {
  //Check the number of learning rates is equal to the number of parameters groups in the
  //optimizer
  TORCH_CHECK(learning_rates.size() == optimizer_.param_groups().size(),
              "Number of learning rates not equal to the number of param groups\n",
              "Number of learning rates given: ", learning_rates.size(),
              "\nNumber of param groups: ", optimizer_.param_groups().size());

  for(std::size_t i = 0; i < optimizer_.param_groups().size(); i++)
    optimizer_.param_groups()[i].options().set_lr(learning_rates[i]);
}

std::vector<double> LRScheduler::get_current_lrs() const {
  std::vector<double> learnings_rates(optimizer_.param_groups().size());
  for(std::size_t i = 0; i < optimizer_.param_groups().size(); i++)
      learnings_rates[i] = optimizer_.param_groups()[i].options().get_lr();
  return learnings_rates;
}

} // namespace optim
} // namespace torch
