#include <test/cpp/jit/test_base.h>

// #include <torch/csrc/jit/fuser/interface.h>
#include <torch/csrc/jit/fuser/common/ir.h>
#include <torch/csrc/jit/fuser/common/fusion.h>
#include <torch/csrc/jit/fuser/common/visitor.h>
#include <torch/csrc/jit/fuser/common/mutator.h>
#include <torch/csrc/jit/fuser/common/arith.h>

#include <iostream>

// Tests go in torch::jit
namespace torch {
namespace jit {

using namespace torch::jit::fuser;

// 1. Test cases are void() functions.
// 2. They start with the prefix `test`

void testGPU_FusionDispatch(){

  Fusion fusion;
  FusionGuard fg(&fusion);
  
  Float* f = new Float{2.f};
  
  const auto val_type = f->type();
  
  SimpleHandler* handler = new SimpleHandler{};
  const auto result = f->dispatch(handler);
  std::cout << "Dispatch 2.f by Float reference: " << result << std::endl;

  Val* v = static_cast<Val*>(f);
  const auto v_result = v->dispatch(handler);
  std::cout << "Dispatch 2.f by Val reference: " << v_result << std::endl;

  Statement* s = static_cast<Statement*>(f);
  const auto s_result = s->dispatch(handler);
  std::cout << "Dispatch 2.f by Statement reference: " << s_result << std::endl;

}

void testGPU_FusionSimpleArith(){
  Fusion fusion;
  FusionGuard fg(&fusion);
  
 
  Float* f1 = new Float(1.f);
  Float* f2 = new Float{2.f};
  Float* f3 = new Float();
  
  Add* an_add = new Add(f3, f1, f2);
  std::cout<<"Explicit add construction of 1.f + 2.f: "<<fusion<<std::endl;

}

void testGPU_FusionContainer(){
  Fusion fusion1;
  FusionGuard fg(&fusion1);
  
  
  Float* f1 = new Float(1.f);
  Float* f2 = new Float(2.f);
  auto f3 = add(f1, f2);
  std::cout<<"Implicit add construction of 1.f + 2.f : "<<fusion1<<std::endl;

  Fusion fusion2;
  {
    FusionGuard fg2(&fusion2);
    Float* f3 = new Float(1.f);
    Float* f4 = new Float(2.f);
    auto f5 = add(f3, f4);
    TORCH_CHECK(FusionGuard::getCurFusion() == &fusion2);
  }

  TORCH_CHECK(FusionGuard::getCurFusion() == &fusion1);
  
}

void testGPU_FusionSimpleTypePromote(){
  Fusion fusion;
  FusionGuard fg(&fusion);
  
  Float* f4 = new Float{4.f};
  Int* i1 = new Int{3};
  auto f5 = add(f4, i1);

  TORCH_CHECK(f5->getValType() == ValType::Float);
}

void testGPU_FusionMutator(){
  Fusion fusion;
  FusionGuard fg(&fusion);
  
  Float* f4 = new Float{1.f};
  Int* i1 = new Int{3};
  Val* f5 = add(f4, i1);
  std::cout<<"Replacing floats of val 1 with 0 in: "<<fusion<<std::endl;
  BaseMutator mutator;
  mutator.mutate(&fusion);
  std::cout<<"Replaced: "<<fusion<<std::endl;

}

void testGPU_FusionRegister() {
  Fusion fusion;
  FusionGuard fg(&fusion);
  Float* v1 = new Float{1.f};
  Float* v2 = new Float{2.f};
  Val* v3 = add(v1, v2);
  Val* v4 = add(v1, v2);
  TORCH_CHECK(v1->name()+1 == v2->name());
  TORCH_CHECK(v2->name()+1 == v3->name());
  TORCH_CHECK(v3->name()+1 == v4->name());
  TORCH_CHECK(fusion.origin(v3)->name()+1 == fusion.origin(v4)->name());
}

void testGPU_FusionTopoSort() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  Float* v0 = new Float{1.f};
  Float* v1 = new Float{2.f};
  Val* v2 = add(v0, v1);
  Float* v3 = new Float{3.f};
  Val* v4 = add(v2, v3);
  TORCH_CHECK(fusion.origin(v2)->name() == 0);
  TORCH_CHECK(fusion.origin(v4)->name() == 1);
  std::vector<const Expr*> exprs = fusion.exprs();
  TORCH_CHECK(exprs[0] == fusion.origin(v2));
  TORCH_CHECK(exprs[1] == fusion.origin(v4));
  
  // TODO: test exprs with multiple output when we have nodes with multiple outputs
  // case:
  //   %1, %2 = op0(%0)
  //   %3, %4 = op1(%1)
  //   %5 = op2(%2, %4)
  //   output (%4, %5)
}

void testGPU_Fusion() {}

}} // torch::jit
