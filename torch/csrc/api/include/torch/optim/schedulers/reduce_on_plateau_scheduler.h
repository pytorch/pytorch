#pragma once

#include <torch/optim/optimizer.h>
#include <torch/optim/schedulers/lr_scheduler.h>

#include <torch/csrc/Export.h>

#include <string>

#include <cmath>

#include <iostream>

namespace torch {
namespace optim {

class TORCH_API ReduceLROnPlateauScheduler {
 public:
  enum SchedulerMode { min, max };
  enum ThresholdMode { rel, abs };
  ReduceLROnPlateauScheduler(
      Optimizer& optimizer,
      SchedulerMode mode = min,
      float factor = 0.1,
      int patience = 10,
      double threshold = 1e-4,
      ThresholdMode threshold_mode = rel,
      int cooldown = 0,
      const std::vector<float>& min_lr = std::vector<float>(),
      double eps = 1e-8,
      bool verbose = false);

  virtual ~ReduceLROnPlateauScheduler() = default;

  void step(float metric);

 private:
  void reset();
  void reduce_lr(int epoch);
  bool in_cooldown();
  bool is_better(float a);
  void init_is_better(
      SchedulerMode mode,
      double threshold,
      ThresholdMode threshold_mode);

  Optimizer& optimizer;
  SchedulerMode mode;
  float mode_worse;
  float factor;
  int patience;
  double threshold;
  ThresholdMode threshold_mode;
  int cooldown;
  int cooldown_counter;
  std::vector<float> min_lrs;
  double eps;
  float best;
  bool verbose;
  int last_epoch;
  int num_bad_epochs;
};
} // namespace optim
} // namespace torch
