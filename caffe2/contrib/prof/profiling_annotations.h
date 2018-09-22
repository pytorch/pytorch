// This file defines classes that hold profiling information for
// NeuralNetOperator and NeuralNetData.
#pragma once

#include "caffe2/contrib/prof/prof_dag_net.h"
#include "caffe2/proto/prof_dag.pb.h"

namespace caffe2 {
namespace contrib {
namespace prof {

// Accumulates data points and generates two point summary: mean, stddev.
class TwoNumberStats {
 public:
  TwoNumberStats() : sum_(0), squareSum_(0), count_(0) {}
  // To prepopulate state of the TwoNumberStats accumulator.
  TwoNumberStats(float mean, float stddev, int count)
      : sum_(mean * count),
        squareSum_((stddev * stddev + mean * mean) * count),
        count_(count) {}
  // This is a small structure and so it's OK to copy (and move).
  TwoNumberStats(const TwoNumberStats& other) = default;
  TwoNumberStats(TwoNumberStats&& other) = default;
  void addPoint(float point) {
    sum_ += point;
    squareSum_ += point * point;
    count_++;
  }
  float getMean() const {
    if (count_ == 0) {
      return 0;
    }
    return sum_ / count_;
  }
  // Returns population stddev.
  float getStddev() const {
    if (count_ == 0) {
      return 0;
    }
    return sqrt((count_ * squareSum_ - sum_ * sum_) / (count_ * count_));
  }
  // Serializes the internal state.
  TwoNumberStatsProto ToProto() const {
    TwoNumberStatsProto proto;
    proto.set_mean(getMean());
    proto.set_stddev(getStddev());
    proto.set_count(count_);
    return proto;
  }
  // Merges another stat accumulator into this one.
  void Merge(const TwoNumberStats& other) {
    sum_ += other.sum_;
    squareSum_ += other.squareSum_;
    count_ += other.count_;
  }

 private:
  // Sum of data points.
  float sum_;
  // Sum of square of data points.
  float squareSum_;
  // Sample count.
  int count_;
};

// Annotations used when profiling a NeuralNetOperator.
class ProfilingOperatorAnnotation {
 public:
  ProfilingOperatorAnnotation() {}
  explicit ProfilingOperatorAnnotation(const ProfDAGProto& stats_proto)
      : execution_time_ms_(
            stats_proto.execution_time().mean(),
            stats_proto.execution_time().stddev(),
            stats_proto.execution_time().count()) {}
  ProfilingOperatorAnnotation(ProfilingOperatorAnnotation&&) = default;
  // Accessors
  const TwoNumberStats& getExecutionTimeMs() const {
    return execution_time_ms_;
  }
  TwoNumberStats* getMutableExecutionTimeMs() {
    return &execution_time_ms_;
  }

 private:
  // Statistics for how long this op took to execute.
  TwoNumberStats execution_time_ms_;
};

// Annotations used when profiling a NeuralNetData. Data this class
// stores is translatable to/from BlobProfile. Note: translation
// may be lossy due to use of floating point arithmetic.
class ProfilingDataAnnotation {
 public:
  ProfilingDataAnnotation() {}
  explicit ProfilingDataAnnotation(const BlobProfile& profile)
      : used_bytes_(
            profile.bytes_used().mean(),
            profile.bytes_used().stddev(),
            profile.bytes_used().count()) {}
  ProfilingDataAnnotation(ProfilingDataAnnotation&&) = default;
  // Accessors
  const TwoNumberStats& getUsedBytes() const {
    return used_bytes_;
  }
  TwoNumberStats* getMutableUsedBytes() {
    return &used_bytes_;
  }

 private:
  // Statistics for how much data this tensor/parameter used (per invocation of
  // the op that generated the data).
  TwoNumberStats used_bytes_;
};

} // namespace prof
} // namespace contrib
} // namespace caffe2
