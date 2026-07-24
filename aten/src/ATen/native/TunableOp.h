// Original TunableOp is from onnxruntime.
// https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/framework/tunable.h
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// Adapting TunableOp into PyTorch
// Copyright (c) Advanced Micro Devices, Inc.
//
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace at::native::tunable {

struct Stats {
  void sample_value(double value) {
    if (_n == 0) {
      _min = value;
      _max = value;
    } else {
      _min = std::min(_min, value);
      _max = std::max(_max, value);
    }
    ++_n;
    const double delta = value - _mean;
    _mean += delta / static_cast<double>(_n);
    _M2 += delta * (value - _mean);
  }

  double variance() const {
    return _n > 1 ? _M2 / static_cast<double>(_n - 1) : 0.0;
  }

  double stddev() const {
    return std::sqrt(variance());
  }

  size_t _n = 0;
  double _mean = 0.0;
  double _M2 = 0.0;
  double _min = 0.0;
  double _max = 0.0;
};

struct TuningPolicy {
  int first_profile_iterations = 3;
  double first_prune_ratio = 1.5;
  int second_profile_iterations = 10;
  double second_prune_ratio = 1.15;
  int default_tuning_iterations = 100;
  double max_tuning_duration_ms = 30.0;
  int max_tuning_iterations = 100;
  double max_warmup_duration_ms = 0.0;
  int max_warmup_iterations = 0;
  // Keep the fallback candidate unless a challenger beats it by this factor;
  // the fallback is typically a heuristic that should not lose to noise.
  double min_improvement_ratio = 1.0;
};

enum class CandidateStatus {
  Unsupported,
  PrunedFirst,
  PrunedSecond,
  NumericalFailure,
  Profiled,
};

struct CandidateResult {
  CandidateStatus status = CandidateStatus::Unsupported;
  Stats stats;
  size_t samples = 0;
};

struct TuningResult {
  size_t candidate_index = 0;
  double time_ms = std::numeric_limits<double>::infinity();
  std::vector<CandidateResult> candidates;
};

inline int tuning_iterations(const TuningPolicy& policy, double approximate_ms) {
  int iterations = policy.default_tuning_iterations;
  if (policy.max_tuning_duration_ms > 0.0) {
    const int duration_iterations = static_cast<int>(policy.max_tuning_duration_ms / approximate_ms);
    iterations = policy.max_tuning_iterations > 0 ? std::min(policy.max_tuning_iterations, duration_iterations)
                                                  : duration_iterations;
  } else if (policy.max_tuning_iterations > 0) {
    iterations = policy.max_tuning_iterations;
  }
  return std::max(1, iterations);
}

inline int warmup_iterations(const TuningPolicy& policy, double approximate_ms) {
  if (policy.max_warmup_duration_ms > 0.0) {
    const int duration_iterations = static_cast<int>(policy.max_warmup_duration_ms / approximate_ms);
    return policy.max_warmup_iterations > 0 ? std::min(policy.max_warmup_iterations, duration_iterations)
                                            : duration_iterations;
  }
  return policy.max_warmup_iterations;
}

template <typename IsSupported, typename Profile, typename NumericallyValid, typename WarmUp>
TuningResult findFastest(size_t candidate_count,
                         size_t fallback_index,
                         const TuningPolicy& policy,
                         IsSupported&& is_supported,
                         Profile&& profile,
                         NumericallyValid&& numerically_valid,
                         WarmUp&& warm_up) {
  TuningResult result;
  result.candidate_index = fallback_index;
  result.candidates.resize(candidate_count);

  for (size_t i = 0; i < candidate_count; ++i) {
    auto& candidate = result.candidates[i];
    if (!is_supported(i)) {
      continue;
    }

    auto stats = profile(i, policy.first_profile_iterations);
    candidate.stats = stats;
    candidate.samples += stats._n;
    if (stats._n == 0 || !std::isfinite(stats._mean) || stats._mean <= 0.0) {
      continue;
    }
    if (stats._mean > policy.first_prune_ratio * result.time_ms) {
      candidate.status = CandidateStatus::PrunedFirst;
      continue;
    }

    stats = profile(i, policy.second_profile_iterations);
    candidate.stats = stats;
    candidate.samples += stats._n;
    if (stats._n == 0 || !std::isfinite(stats._mean) || stats._mean <= 0.0) {
      continue;
    }
    if (stats._mean > policy.second_prune_ratio * result.time_ms) {
      candidate.status = CandidateStatus::PrunedSecond;
      continue;
    }
    if (!numerically_valid(i)) {
      candidate.status = CandidateStatus::NumericalFailure;
      continue;
    }

    warm_up(i, warmup_iterations(policy, stats._mean));
    stats = profile(i, tuning_iterations(policy, stats._mean));
    candidate.stats = stats;
    candidate.samples += stats._n;
    if (stats._n == 0 || !std::isfinite(stats._mean) || stats._mean <= 0.0) {
      continue;
    }
    candidate.status = CandidateStatus::Profiled;
    if (stats._mean < result.time_ms) {
      result.candidate_index = i;
      result.time_ms = stats._mean;
    }
  }
  if (result.candidate_index != fallback_index && policy.min_improvement_ratio > 1.0) {
    auto& fallback = result.candidates[fallback_index];
    if (fallback.status == CandidateStatus::Profiled) {
      // GPU clocks ramp over the tuning session, so the first-profiled
      // fallback ran at the lowest frequency; re-profile it hot before
      // deciding whether the winner truly beats it.
      auto stats = profile(fallback_index, tuning_iterations(policy, fallback.stats._mean));
      fallback.samples += stats._n;
      if (stats._n > 0 && std::isfinite(stats._mean) && stats._mean > 0.0 &&
          stats._mean < fallback.stats._mean) {
        fallback.stats = stats;
      }
      if (fallback.stats._mean <= result.time_ms * policy.min_improvement_ratio) {
        result.candidate_index = fallback_index;
        result.time_ms = fallback.stats._mean;
      }
    }
  }
  return result;
}

} // namespace at::native::tunable
