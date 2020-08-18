#include <gtest/gtest.h>

#include <torchpy.h>

TEST(TorchpyTest, InitFinalize) {
  torchpy::init();
  torchpy::finalize();
  torchpy::init();
  torchpy::finalize();
}
