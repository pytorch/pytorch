#include <torch/optim/schedulers/lr_scheduler.h>

namespace torch {
namespace optim {

LRScheduler::LRScheduler(torch::optim::Optimizer& optimizer, const bool verbose) :
  optimizer_(optimizer),
  step_count_(0),
  verbose_(verbose) {}

void LRScheduler::step() {
  //Get learning rates from subclass
  std::vector<double> learning_rates = get_lrs();
  if(verbose_)
    print_lrs(learning_rates);
  set_optimizer_lrs(learning_rates);
  step_count_++;
}

void LRScheduler::set_optimizer_lrs(const std::vector<double>& learning_rates) {
  //Check the number of learning rates is equal to the number of parameters groups in
  //the optimizer
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

unsigned LRScheduler::get_step_count() const {
  return step_count_;
}

void LRScheduler::print_lrs(const std::vector<double>& learning_rates) const {
  //Store current cout state
  std::ios previous_cout_state(nullptr);
  previous_cout_state.copyfmt(std::cout);
  //Modify cout to format floats in scientific notation with 4dp precision
  std::cout << std::scientific;
  std::cout.precision(4);

  for(std::size_t i = 0; i < learning_rates.size(); i++)
    std::cout << "Adjusting learning rate of group " << i << " to " <<
      learning_rates[i] << std::endl;

  //Restore previous cout state
  std::cout.copyfmt(previous_cout_state);
}

} // namespace optim
} // namespace torch
