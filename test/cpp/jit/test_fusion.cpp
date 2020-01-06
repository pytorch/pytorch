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
void testFusionCPU(){}

void testFusionDispatch(){

  Fusion fusion;
  Manager m(&fusion);
  
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

}

void testFusionSimpleArith(){
  Fusion fusion;
  Manager m(&fusion);
 
  Float* f1 = new Float(1.f);
  Float* f2 = new Float{2.f};
  Float* f3 = new Float();
  
  Add* an_add = new Add(f3, f1, f2);
  std::cout<<"Explicit add construction: "<<fusion<<std::endl;

}

void testFusionContainer(){
  Fusion fusion1;
  Manager m(&fusion1);
  Manager *inst = &(m.instance());
  
  Float* f1 = new Float(1.f);
  Float* f2 = new Float(2.f);
  auto f3 = add(f1, f2);
  std::cout<<"Implicit add construction: "<<fusion1<<std::endl;

  Fusion fusion2;
  {
    Manager m2(&fusion2);
    Float* f3 = new Float(1.f);
    Float* f4 = new Float(2.f);
    auto f5 = add(f3, f4);
    TORCH_CHECK(m.fusion() == &fusion2);
    TORCH_CHECK(&(m2.instance()) == inst);
  }

  TORCH_CHECK(m.fusion() == &fusion1);
  TORCH_CHECK(&(m.instance()) == inst);
}

void testFusionSimpleTypePromote(){
  Fusion fusion;
  Manager m(&fusion);
  Manager *inst = &(m.instance());
  
  Float* f4 = new Float{4.f};
  Int* i1 = new Int{3};
  auto f5 = add(f4, i1);

  TORCH_CHECK(f5->getValType() == ValType::Float);
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
