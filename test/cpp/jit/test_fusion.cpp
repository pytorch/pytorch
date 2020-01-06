#include <test/cpp/jit/test_base.h>

// #include <torch/csrc/jit/fuser/interface.h>
#include <torch/csrc/jit/fuser/common/ir.h>
#include <torch/csrc/jit/fuser/common/fusion.h>
#include <torch/csrc/jit/fuser/common/visitor.h>
#include <torch/csrc/jit/fuser/common/arith.h>

#include <iostream>

// Tests go in torch::jit
namespace torch {
namespace jit {

using namespace torch::jit::fuser;

// 1. Test cases are void() functions.
// 2. They start with the prefix `test`
void testCPUFusion() {

  Fusion fusion;
  Manager m(&fusion);
  assert(m.fusion() == &fusion);
  
  Float* f = new Float{2.f};
  
  const auto val_type = f->type();
  
  SimpleHandler* handler = new SimpleHandler{};
  const auto result = f->dispatch(handler);
  std::cout << "Dispatch Float result: " << result << std::endl;

  Val* v = static_cast<Val*>(f);
  const auto v_result = v->dispatch(handler);
  std::cout << "Dispatch Val result: " << v_result << std::endl;

  Statement* s = static_cast<Statement*>(f);
  const auto s_result = s->dispatch(handler);
  std::cout << "Dispatch Statement result: " << s_result << std::endl;
  
  Float* f1 = static_cast<Float*>(v);
  Float* f3 = new Float();
  Float* f2 = new Float{3.f};

  Add* an_add = new Add(f3, f1, f2);
  std::cout<<"Explicit add construction: "<<fusion<<std::endl;

  Fusion fusion2;
  Manager m2(&fusion2);
  
  Float* f4 = new Float{4.f};
  Int* i1 = new Int{3};
  auto f5 = add(f4, i1);
  std::cout<<"Implicit add construction (f + i): "<<fusion2<<std::endl;

  assert(m.fusion() == &fusion2);

}

void testFusionContainer(){

}

void testGPUFusion() {
  /*
  Fusion fusion;
  Manager m(&fusion);
  Tensor* T1 = new Tensor{};
  Tensor* T2 = new Tensor{};
  Float* F1 = new Float{1.0};
  Val* T3 = add(T2, F1);

  fusion.addInput(T1);
  fusion.addInput(T2);
  fusion.addOutput(T3);

  std::cout << fusion << std::endl;
*/
}

void testGPUHelloFusion(){
  // std::cout << "Hello world from testGPUHelloFusion" << std::endl;
}

}} // torch::jit
