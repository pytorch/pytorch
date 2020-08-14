#include <gtest/gtest.h>

#include <torchpy.h>

TEST(TorchpyTest, InitFinalize) {
  torchpy::init();
  torchpy::finalize();
}

TEST(TorchpyTest, HelloPy) {
  torchpy::init();
  auto h = torchpy::hello();
  GTEST_ASSERT_EQ(h, "Hello Py");
  torchpy::finalize();
}