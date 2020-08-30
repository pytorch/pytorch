#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include "interpreter.h"

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);

  int rc = RUN_ALL_TESTS();

  return rc;
}

TEST(Interpreter, Hello) {
  Interpreter interp;
  interp.run_some_python("print('hello from first interpeter!')");

  Interpreter interp2;
  interp2.run_some_python("print('hello from second interpeter!')");
}