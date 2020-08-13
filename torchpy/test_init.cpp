#include <gtest/gtest.h>

#include <torchpy.h>

TEST(TorchpyTest, Init) {
  torchpy::init();
}

TEST(TorchpyTest, HelloPy) {
  auto h = torchpy::hello();
  GTEST_ASSERT_EQ(h, "Hello Py");
}