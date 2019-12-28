#include <test/cpp/jit/test_base.h>
#include <iostream>

#include <torch/csrc/jit/fuser/common/ir_printer.h>
#include <torch/csrc/jit/fuser/common/ir.h>
#include <torch/csrc/jit/fuser/common/types.h>
#include <torch/csrc/jit/fuser/common/expr.h>

// Tests go in torch::jit
using namespace torch::jit::fuser;
namespace torch {
namespace jit {

// 1. Test cases are void() functions.
// 2. They start with the prefix `test`
void testCPUFusion() {
    // ...
}

void testGPUFusion() {
  std::cout<<"Hello world from testGPUFusion"<<std::endl;
  auto test = For::make(Variable::make("i", DType(CType::kInt32)), IntImm::make(0), IntImm::make(1), EmptyExpr::make());
  //std::cout<<test<<std::endl;
  IRPrinter p(std::cout);
  p.print(test);
}

void testGPUHelloFusion(){
  std::cout<<"Hello world from testGPUHelloFusion"<<std::endl;
}
}} // torch::jit
