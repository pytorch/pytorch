// Unit tests for profiling_annotations.h.
#include "caffe2/contrib/prof/profiling_annotations.h"

#include <gtest/gtest.h>

namespace caffe2 {
namespace {

TEST(ProfilingAnnotationsTest, BasicAccess) {
  ProfilingOperatorAnnotation op_annotation;
  op_annotation.mutable_execution_time_ms()->sum = 5;
  ProfilingDataAnnotation data_annotation;
  data_annotation.mutable_used_bytes()->sum = 7;
  auto* op_annotation_ptr =
      dyn_cast<ProfilingOperatorAnnotation>(&op_annotation);
  ASSERT_NE(nullptr, op_annotation_ptr);
  EXPECT_EQ(5, op_annotation_ptr->execution_time_ms().sum);
  auto* data_annotation_ptr =
      dyn_cast<ProfilingDataAnnotation>(&data_annotation);
  ASSERT_NE(nullptr, data_annotation_ptr);
  EXPECT_EQ(7, data_annotation_ptr->used_bytes().sum);
  EXPECT_EQ(nullptr, dyn_cast<ProfilingOperatorAnnotation>(&data_annotation));
  EXPECT_EQ(nullptr, dyn_cast<ProfilingDataAnnotation>(&op_annotation));
}

} // namespace
} // namespace caffe2
