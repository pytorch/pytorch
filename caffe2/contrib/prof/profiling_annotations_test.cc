// Unit tests for profiling_annotations.h.
#include "caffe2/contrib/prof/profiling_annotations.h"

#include <gtest/gtest.h>

namespace caffe2 {
namespace contrib {
namespace prof {
namespace {

TEST(TwoNumberStatsTest, ComputeAndGetOpStatsSummary) {
  // e.g., 2 and 3
  TwoNumberStats stats;
  stats.addPoint(2);
  stats.addPoint(3);
  EXPECT_FLOAT_EQ(2.5, stats.getMean());
  // Population standard deviation.
  EXPECT_FLOAT_EQ(0.5, stats.getStddev());
}

TEST(TwoNumberStatsTest, TestRestore) {
  // Expect that restore&recompute is still the same.
  // E.g., 2 and 3 (above).
  TwoNumberStats stats(2.5, 0.5, 2);
  // Expect that restore&recompute is still the same.
  EXPECT_FLOAT_EQ(2.5, stats.getMean());
  // Population standard deviation.
  EXPECT_FLOAT_EQ(0.5, stats.getStddev());
}

TEST(ProfilingAnnotationsTest, BasicAccessToActiveData) {
  ProfilingOperatorAnnotation op_annotation;
  op_annotation.getMutableExecutionTimeMs()->addPoint(5);
  EXPECT_EQ(5, op_annotation.getExecutionTimeMs().getMean());

  ProfilingDataAnnotation data_annotation;
  data_annotation.getMutableUsedBytes()->addPoint(7);
  EXPECT_EQ(7, data_annotation.getUsedBytes().getMean());
}

} // namespace
} // namespace prof
} // namespace contrib
} // namespace caffe2
