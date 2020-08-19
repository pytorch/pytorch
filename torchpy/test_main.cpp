#include <gtest/gtest.h>
#include <torchpy.h>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);

  // Ideally, init/finalize inside each test for more isolation,
  // but, since importing torch has side effects, I couldn't make this work
  torchpy::init();

  int rc = RUN_ALL_TESTS();

  torchpy::finalize();

  return rc;
}
