#include <test/cpp/jit/test_base.h>

// #include <torch/csrc/jit/fuser/interface.h>
#include <torch/csrc/jit/fuser/common/ir.h>

#include <iostream>

// Tests go in torch::jit
namespace torch {
namespace jit {

using namespace torch::jit::fuser;

// 1. Test cases are void() functions.
// 2. They start with the prefix `test`
void testCPUFusion() {
  Val v{0, ValType::Scalar};
  const auto val_type = v.type();
  std::cout << "val type: " << val_type << std::endl;

  SampleValHandler handler;
  const auto result = v.dispatch(handler);
  std::cout << "dispatch result: " << result << std::endl;

  Val v_expr{1, ValType::Expr};
  const auto result_expr = v_expr.dispatch(handler);
  std::cout << "expr dispatch result: " << result_expr << std::endl;
}

void testGPUFusion() {
  // std::cout << "Hello world from testGPUFusion" << std::endl;
  // auto test = For::make(Variable::make("i", DType(CType::kInt32)), IntImm::make(0), IntImm::make(1), EmptyExpr::make());
  // //std::cout<<test<<std::endl;
  // IRPrinter p{std::cout};
  // p.print(test);
}

void testGPUHelloFusion(){
  // std::cout << "Hello world from testGPUHelloFusion" << std::endl;
}

}} // torch::jit
