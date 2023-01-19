#include <iostream>

#include "examples/test_cc_shared_library/foo.h"

int main() {
  std::cout << "hello " << foo() << std::endl;
  return 0;
}
