#include "ATen/ATen.h"
#include "ATen/DLConvertor.h"

#include <iostream>
#include <string.h>
#include <sstream>
#include "test_assert.h"

using namespace at;

static void test() {
  {
    std::cout << "dlconvertor: convert ATen to DLTensor" << std::endl;
    Tensor a = CPU(at::kFloat).rand({3,4});
    std::cout << a.numel() << std::endl;
    DLManagedTensor* dlMTensor = toDLPack(a);
    std::cout << "dlconvertor: convert DLTensor to ATen" << std::endl;
    Tensor b = fromDLPack(dlMTensor);
    ASSERT(a.equal(b));
    std::cout << "conversion was fine" << std::endl;
  }

}

int main(int argc, char ** argv)
{
  std::cout << "======================= CPU =====================" << std::endl;
  test();
  return 0;
}
