#include <test/cpp/jit/test_base.h>

// #include <torch/csrc/jit/fuser/interface.h>
#include <torch/csrc/jit/fuser/common/ir.h>
#include <torch/csrc/jit/fuser/common/fusion.h>
#include <torch/csrc/jit/fuser/common/mutator.h>
#include <torch/csrc/jit/fuser/common/arith.h>
#include <torch/csrc/jit/fuser/common/iriostream.h>

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


//dummy expr with 2 outputs only for toposort test.
struct TORCH_API DummyExpr : public Expr {
  ~DummyExpr () = default;
  DummyExpr (
    const Val* _outlhs
  , const Val* _outrhs
  , const Val* _lhs
  , const Val* _rhs):Expr(ExprType::Add) //Not terribly safe...
  {
    addOutput(_outlhs);
    addOutput(_outrhs);
    addInput(_lhs);
    addInput(_rhs);
    this->name_ = FusionGuard::getCurFusion()->registerExpr(this);
  }
  DummyExpr(const DummyExpr& other) = delete;
  DummyExpr& operator=(const DummyExpr& other) = delete;
  DummyExpr(DummyExpr&& other) = delete;
  DummyExpr& operator=(DummyExpr&& other) = delete;
};

void testGPU_FusionTopoSort() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  //e0: v3, v2 = dummy(v1, v0)
  //e1: v4     =   add(v3, v2)
  //e2: v5     =   add(v2, v4)
  Float* v0 = new Float{1.f};
  Float* v1 = new Float{2.f};
  Float* v2 = new Float();
  Float* v3 = new Float();
  Float* v4 = new Float();
  Float* v5 = new Float();

  Expr* e0 = new DummyExpr(v3, v2, v1, v0);
  Expr* e1 = new Add(v4, v3, v2);
  Expr* e2 = new Add(v5, v2, v4);
  
  std::vector<const Expr*> exprs = fusion.exprs();

  TORCH_CHECK(exprs.size() == 3);
  TORCH_CHECK(exprs[0] == e0);
  TORCH_CHECK(exprs[1] == e1);
  TORCH_CHECK(exprs[2] == e2);
  
  fusion.addOutput(v2);
  exprs = fusion.exprs(true);
  TORCH_CHECK(exprs.size() == 1);
  TORCH_CHECK(exprs[0] == e0);

  fusion.addOutput(v5);
  exprs = fusion.exprs(true);
  TORCH_CHECK(exprs[0] == e0);
  TORCH_CHECK(exprs[1] == e1);
  TORCH_CHECK(exprs[2] == e2);

  fusion.addOutput(v4);
  exprs = fusion.exprs(true);
  TORCH_CHECK(exprs[0] == e0);
  TORCH_CHECK(exprs[1] == e1);
  TORCH_CHECK(exprs[2] == e2);

  fusion.addOutput(v3);
  exprs = fusion.exprs(true);
  TORCH_CHECK(exprs[0] == e0);
  TORCH_CHECK(exprs[1] == e1);
  TORCH_CHECK(exprs[2] == e2);


  TORCH_CHECK(fusion.origin(v2)->name() == 0);
  TORCH_CHECK(fusion.origin(v3)->name() == 0);
  TORCH_CHECK(fusion.origin(v4)->name() == 1);
  TORCH_CHECK(fusion.origin(v5)->name() == 2);

}

void testGPU_Fusion() {}

}} // torch::jit
