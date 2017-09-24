#include "ATen/ATen.h"
#include "ATen/dlconvertor.h"

#include <iostream>
#include <string.h>
#include <sstream>
#include "test_assert.h"

using namespace at;

static void test() {
  {
    std::cout << "dlconvertor: convert ATen to DLTensor" << std::endl;
    Tensor a = CPU(at::kFloat).zeros({3,4});
    std::cout << a.numel() << std::endl;
    ASSERT(a.numel() == 12);
    dlpack::DLConvertor convertor(a);
    convertor.convertToDLTensor(a);
  }

}

int main(int argc, char ** argv)
{
  std::cout << "=========================== CPU ===========================" << std::endl;
  // kFloat is the ScalarType defined in doc/Type.h line 75
  // CPU is inline function defined in ATen/Context.h like 79
  test();
  return 0;
}
