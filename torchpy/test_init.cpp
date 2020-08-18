#include <gtest/gtest.h>

#include <torchpy.h>

TEST(TorchpyTest, InitFinalize) {
  torchpy::init();
  torchpy::finalize();
  torchpy::init();
  torchpy::finalize();
}

TEST(TorchpyTest, TestGetLoad) {
  torchpy::init();
  torchpy::test_get_load();
  torchpy::test_get_load();
  torchpy::finalize();
}

TEST(TorchpyTest, HelloPy) {
  torchpy::init();
  auto h = torchpy::hello();
  GTEST_ASSERT_EQ(h, "Hello Py");
  torchpy::finalize();
  torchpy::init();
  h = torchpy::hello();
  GTEST_ASSERT_EQ(h, "Hello Py");
  torchpy::finalize();
}