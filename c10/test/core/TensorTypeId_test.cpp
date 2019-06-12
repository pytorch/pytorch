#include <gtest/gtest.h>
#include <sstream>
#include <c10/core/TensorTypeId.h>
#include <c10/core/TensorTypeIdRegistration.h>

TEST(TensorTypeIdTest, Printing) {
  std::ostringstream ss;
  ss << at::UndefinedTensorId();
  EXPECT_EQ(ss.str(), "1");
}
