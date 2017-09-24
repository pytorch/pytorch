#include "ATen/ATen.h"

// // for TH compat test only...
// struct THFloatTensor;
// extern "C" THFloatTensor * THFloatTensor_newWithSize2d(size_t a, size_t b);
// extern "C" void THFloatTensor_fill(THFloatTensor *, float v);

// #include <iostream>
// #include <chrono>
// #include <string.h>
// #include <sstream>
#include "test_assert.h"

using namespace at;


static void test(Type & type) {
  {
    std::cout << "dlconvertor: convert ATen to DLTensor" << std::endl;
    auto a = type.tensor();
    a.resize_({3,4});
    std::cout << a.numel() << std::endl;
    ASSERT(a.numel() == 12);
    a.resize_({5, 7});
    std::cout << a.numel() << std::endl;
    ASSERT(a.numel() == 35);

  }

}

int main(int argc, char ** argv)
{
  std::cout << "=========================== CPU ===========================" << std::endl;
  test(CPU(kFloat));
  // if(at::hasCUDA()) {
  //   if(argc == 2 && 0 == strcmp(argv[1],"-n")) {
  //     std::cout << "skipping cuda...\n";
  //   } else {
  //     std::cout << "=========================== GPU ===========================" << std::endl;
  //     test(CUDA(kFloat));
  //   }
  // }
  return 0;
}
