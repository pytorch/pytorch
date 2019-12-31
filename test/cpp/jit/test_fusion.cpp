#include <test/cpp/jit/test_base.h>

// #include <torch/csrc/jit/fuser/interface.h>
#include <torch/csrc/jit/fuser/common/ir.h>
#include <torch/csrc/jit/fuser/common/fusion.h>
#include <torch/csrc/jit/fuser/common/visitor.h>

#include <iostream>

// Tests go in torch::jit
namespace torch {
namespace jit {

using namespace torch::jit::fuser;

// 1. Test cases are void() functions.
// 2. They start with the prefix `test`
void testCPUFusion() {
  torch::jit::fuser::Float* f = new Float{2.f};
  const auto val_type = f->type();
  std::cout << "val type: " << val_type << std::endl;

  SimpleHandler* handler = new SimpleHandler{};
  const auto result = f->dispatch(handler);
  std::cout << "dispatch result: " << result << std::endl;

  Val* v = static_cast<Val*>(f);
  const auto v_result = v->dispatch(handler);
  std::cout << "dispatch result: " << v_result << std::endl;

  Float* f1 = static_cast<Float*>(v);

  Fusion fusion;
  Add* add = new Add{};
  fusion.appendExpr(add);

  IRPrinter* printer = new IRPrinter{};
  for (auto* e : fusion.list()) {
    e->dispatch(printer);
  }
}

void testGPUFusion() {
  // std::cout << "Hello world from testGPUFusion" << std::endl;
  // auto test = For::make(Variable::make("i", DType(CType::kInt32)), IntImm::make(0), IntImm::make(1), EmptyExpr::make());
  // //std::cout<<test<<std::endl;
  // IRPrinter p{std::cout};
  // p.print(test);t
}

void testGPUHelloFusion(){
  // std::cout << "Hello world from testGPUHelloFusion" << std::endl;
}

}} // torch::jit
