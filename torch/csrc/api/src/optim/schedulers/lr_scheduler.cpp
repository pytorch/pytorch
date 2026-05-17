#include <c10/util/irange.h>
#include <torch/optim/schedulers/lr_scheduler.h>

namespace torch::optim {

LRScheduler::LRScheduler(torch::optim::Optimizer& optimizer)
    : optimizer_(optimizer) {}

void LRScheduler::step() {
  std::vector<double> learning_rates = get_lrs();
  set_optimizer_lrs(learning_rates);
  step_count_++;
}

void LRScheduler::set_optimizer_lrs(const std::vector<double>& learning_rates) {
  // Check the number of learning rates is equal to the number of parameters
  // groups in the optimizer
  TORCH_CHECK(
      learning_rates.size() == optimizer_.param_groups().size(),
      "Number of learning rates not equal to the number of param groups\n",
      "Number of learning rates given: ",
      learning_rates.size(),
      "\nNumber of param groups: ",
      optimizer_.param_groups().size());

  for (const auto i : c10::irange(optimizer_.param_groups().size())) {
    optimizer_.param_groups()[i].options().set_lr(learning_rates[i]);
  }
}

std::vector<double> LRScheduler::get_current_lrs() const {
  std::vector<double> learnings_rates(optimizer_.param_groups().size());
  if (!learnings_rates.empty()) {
    for (const auto i : c10::irange(optimizer_.param_groups().size())) {
      learnings_rates[i] = optimizer_.param_groups()[i].options().get_lr();
    }
  }
  return learnings_rates;
}

} // namespace torch::optim
