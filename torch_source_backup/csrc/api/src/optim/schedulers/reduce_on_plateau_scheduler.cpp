#include <torch/optim/schedulers/reduce_on_plateau_scheduler.h>

#include <iomanip>

namespace torch::optim {

ReduceLROnPlateauScheduler::ReduceLROnPlateauScheduler(
    Optimizer& optimizer,
    SchedulerMode mode,
    float factor,
    int patience,
    double threshold,
    ThresholdMode threshold_mode,
    int cooldown,
    const std::vector<float>& min_lr,
    double eps,
    bool verbose)
    : optimizer(optimizer) {
  if (min_lr.empty()) {
    this->min_lrs = std::vector<float>(optimizer.param_groups().size());
  } else {
    // Check if number of learning rates is equal to the number of parameters
    // groups in the optimizer
    TORCH_CHECK(
        min_lr.size() == optimizer.param_groups().size(),
        "Number of learning rates not equal to the number of param groups\n",
        "Number of learning rates given: ",
        min_lr.size(),
        "\nNumber of param groups: ",
        optimizer.param_groups().size());
    this->min_lrs = min_lr;
  }

  TORCH_CHECK(factor < 1.0, "Factor should be < 1.0.");
  this->factor = factor;
  this->patience = patience;
  this->cooldown = cooldown;
  this->eps = eps;
  this->verbose = verbose;

  init_is_better(mode, threshold, threshold_mode);
  reset();
}

void ReduceLROnPlateauScheduler::step(float metrics) {
  last_epoch++;

  if (is_better(metrics)) {
    best = metrics;
    num_bad_epochs = 0;
  } else {
    num_bad_epochs++;
  }

  if (in_cooldown()) {
    cooldown_counter--;
    num_bad_epochs = 0;
  }

  if (num_bad_epochs > patience) {
    reduce_lr(last_epoch);
    cooldown_counter = cooldown;
    num_bad_epochs = 0;
  }
}

void ReduceLROnPlateauScheduler::reduce_lr(int epoch) {
  for (std::size_t i = 0; i < optimizer.param_groups().size(); i++) {
    auto old_lr = optimizer.param_groups()[i].options().get_lr();
    auto new_lr = std::fmax(old_lr * factor, min_lrs[i]);
    if (old_lr - new_lr > eps) {
      optimizer.param_groups()[i].options().set_lr(new_lr);
      if (verbose) {
        std::cout << std::setprecision(4) << "Epoch " << epoch
                  << ": reducing learning rate of group " << i << " to "
                  << new_lr << '\n';
      }
    }
  }
}

void ReduceLROnPlateauScheduler::reset() {
  this->cooldown_counter = 0;
  this->num_bad_epochs = 0;
  this->last_epoch = 0;
  this->best = mode_worse;
}

bool ReduceLROnPlateauScheduler::in_cooldown() const {
  return cooldown_counter > 0;
}

bool ReduceLROnPlateauScheduler::is_better(float a) {
  if (mode == min && threshold_mode == rel) {
    auto rel_epsilon = 1.0 - threshold;
    return a < best * rel_epsilon;
  } else if (mode == min && threshold_mode == abs) {
    return a < best - threshold;
  } else if (mode == max && threshold_mode == rel) {
    auto rel_epsilon = 1.0 + threshold;
    return a > best * rel_epsilon;
  } else {
    return a > best * threshold;
  }
}

void ReduceLROnPlateauScheduler::init_is_better(
    SchedulerMode mode,
    double threshold,
    ThresholdMode threshold_mode) {
  if (mode == min) {
    mode_worse = std::numeric_limits<float>::max();
  } else {
    mode_worse = std::numeric_limits<float>::min();
  }

  this->mode = mode;
  this->threshold_mode = threshold_mode;
  this->threshold = threshold;
}
} // namespace torch::optim
