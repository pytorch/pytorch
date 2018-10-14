#include <gtest/gtest.h>
#include <sstream>
#include "ATen/core/TensorTypeId.h"
#include "ATen/core/TensorTypeIdRegistration.h"

TEST(TensorTypeIdTest, Printing) {
  std::ostringstream ss;
  ss << at::UndefinedTensorId();
  EXPECT_EQ(ss.str(), "1");
}
