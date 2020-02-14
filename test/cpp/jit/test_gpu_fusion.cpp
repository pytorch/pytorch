#include <test/cpp/jit/test_base.h>

// #include <torch/csrc/jit/fuser/interface.h>
#include <torch/csrc/jit/fuser/common/ir.h>
#include <torch/csrc/jit/fuser/common/fusion.h>
#include <torch/csrc/jit/fuser/common/mutator.h>
#include <torch/csrc/jit/fuser/common/arith.h>
#include <torch/csrc/jit/fuser/common/iriostream.h>
#include <torch/csrc/jit/fuser/common/tensor.h>
#include <torch/csrc/jit/fuser/common/tensor_meta.h>

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
    
  std::cout << "Dispatch 2.f by Float reference: " << f << std::endl;

  std::cout << "Dispatch 2.f by Val reference: " << static_cast<Val*>(f) << std::endl;

  std::cout << "Dispatch 2.f by Statement reference: " << static_cast<Statement*>(f) << std::endl;
}

void testGPU_FusionSimpleArith(){
  Fusion fusion;
  FusionGuard fg(&fusion);
  
 
  Float* f1 = new Float(1.f);
  Float* f2 = new Float{2.f};
  Float* f3 = new Float();
  
  BinaryOp* an_add = new BinaryOp(BinaryOpType::Add, f3, f1, f2);
  std::cout<<"Explicit add construction of 1.f + 2.f: "<<fusion<<std::endl;

}

void testGPU_FusionContainer(){
  Fusion fusion1;
  FusionGuard fg(&fusion1);
  
  
  Float* f1 = new Float(1.f);
  Float* f2 = new Float(2.f);
  auto f3 = binary_op(BinaryOpType::Add, f1, f2);
  std::cout<<"Implicit add construction of 1.f + 2.f : "<<fusion1<<std::endl;

  Fusion fusion2;
  {
    FusionGuard fg2(&fusion2);
    Float* f3 = new Float(1.f);
    Float* f4 = new Float(2.f);
    auto f5 = binary_op(BinaryOpType::Add, f3, f4);
    TORCH_CHECK(FusionGuard::getCurFusion() == &fusion2);
  }

  TORCH_CHECK(FusionGuard::getCurFusion() == &fusion1);
  
}


void testGPU_FusionSimpleTypePromote(){
  Fusion fusion;
  FusionGuard fg(&fusion);
  
  Float* f4 = new Float{4.f};
  Int* i1 = new Int{3};
  auto f5 = binary_op(BinaryOpType::Add, f4, i1);

  TORCH_CHECK(f5->getDataType() == DataType::Float);
}

void testGPU_FusionCastOp(){
  Fusion fusion;
  FusionGuard fg(&fusion);
  
  Float* f3_test = new Float{3.f};
  Int* i3 = new Int{3};
  auto f3 = cast_op(DataType::Float, i3);

  TORCH_CHECK(f3->getDataType().value() == f3_test->getDataType().value());
}

void testGPU_FusionMutator(){
  Fusion fusion;
  FusionGuard fg(&fusion);
  
  Float* f4 = new Float{1.f};
  Int* i1 = new Int{3};
  const Val* f5 = binary_op(BinaryOpType::Add, f4, i1);
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
  const Val* v3 = binary_op(BinaryOpType::Add, v1, v2);
  const Val* v4 = binary_op(BinaryOpType::Add, v1, v2);
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
  , const Val* _rhs):Expr(ExprType::BinaryOp) //Not terribly safe...
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
  Expr* e1 = new BinaryOp(BinaryOpType::Add, v4, v3, v2);
  Expr* e2 = new BinaryOp(BinaryOpType::Add, v5, v2, v4);
  
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

void testGPU_FusionTensor() {
  auto tensor = at::randn({2, 3, 4, 5}, at::kCUDA);
  auto sizes = tensor.sizes().vec();
  auto tensor_type = TensorType::create(tensor);

  Fusion fusion;
  FusionGuard fg(&fusion);
  auto fuser_tensor  = new Tensor(tensor_type);
  TORCH_CHECK(fuser_tensor->hasContiguityInfo() == 1);
  TORCH_CHECK(fuser_tensor->getDataType().value() == DataType::Float); 
  TORCH_CHECK(fuser_tensor->domain() == nullptr);
  
  auto fuser_null_tensor  = new Tensor(DataType::Int);
  TORCH_CHECK(fuser_null_tensor->hasContiguityInfo() == 0);
  TORCH_CHECK(fuser_null_tensor->getDataType().value() == DataType::Int); 
  TORCH_CHECK(fuser_null_tensor->domain() == nullptr);
}

void testGPU_FusionTensorContiguity() {
  {
    // NCHW memory layout
    auto tensor = at::randn({2, 3, 4, 5});
    auto sizes = tensor.sizes().vec();
    auto strides = tensor.strides().vec();
    TensorContiguity t_c(sizes, strides);
    TORCH_CHECK(t_c.rank() == 4);
    TORCH_CHECK(t_c.getBroadcastDims().size() == 0);
    for (int i = 0; i < 4; i++) {
      TORCH_CHECK(!t_c.isBroadcastDim(i));
      if (i < 3) {
        TORCH_CHECK(t_c.canCollapseToHigher(i));
      }
    }
  }

  {
    // NHWC memory layout
    TensorContiguity t_c({2, 3, 4, 5}, {60, 1, 15, 3});
    TORCH_CHECK(t_c.rank() == 4);
    TORCH_CHECK(t_c.getBroadcastDims().size() == 0);
    for (int i = 0; i < 4; i++) {
      TORCH_CHECK(!t_c.isBroadcastDim(i));
      if (i < 3) {
        TORCH_CHECK((t_c.canCollapseToHigher(i) ^ (i!=2)));
      }
    }
  }
  
  {
    // NHWC memory layout with broadcast
    TensorContiguity t_c({2, 3, 4, 5}, {120, 0, 30, 3});
    TORCH_CHECK(t_c.rank() == 4);
    auto b_dims = t_c.getBroadcastDims();
    TORCH_CHECK(b_dims.size() == 1 && b_dims[0] == 1);
    for (int i = 0; i < 4; i++) {
      TORCH_CHECK(!(t_c.isBroadcastDim(i)) ^ (i==1));
      if (i < 3) {
        TORCH_CHECK(!(t_c.canCollapseToHigher(i)));
      }
    }
  }
}

void testGPU_FusionTVSplit() {

  Fusion fusion;
  FusionGuard fg(&fusion);

  const Tensor *t = Tensor::MakeDummyTensor(3);

  const TensorView *tv = split(t, 2, 2);

  std::cout<<"Split fusion output: "<<fusion<<std::endl;
  
}

void testGPU_FusionTVMerge() {

  Fusion fusion;
  FusionGuard fg(&fusion);

  const Tensor *t = Tensor::MakeDummyTensor(3);

  const TensorView *tv = merge(t, 1);

  std::cout<<"Merge fusion output: "<<fusion<<std::endl;
  
}

void testGPU_FusionTVReorder() {

  Fusion fusion;
  FusionGuard fg(&fusion);

  const Tensor *t = Tensor::MakeDummyTensor(3);

  std::unordered_map<int, int> shift_right{
    {-1, 0}
  };
  std::unordered_map<int, int> shift_left{
    {0, -1}
  };

  std::unordered_map<int, int> shift_left_2{
    {0, -1},
    {1, 0},
    {2, 1}
  };

  std::unordered_map<int, int> swap{
    {0, 2},
    {2, 0}
  };

  const TensorView *s_leftl = reorder(t, shift_left);
  for(int i = 0; i < t->domain()->size(); i++)
    TORCH_CHECK(t->domain()->axis(i) == s_leftl->domain()->axis(i-1));
  
  const TensorView *s_left2 = reorder(t, shift_left);
  for(int i = 0; i < t->domain()->size(); i++)
    TORCH_CHECK(t->domain()->axis(i) == s_left2->domain()->axis(i-1));

  const TensorView *s_right = reorder(t, shift_right);
  for(int i = 0; i < t->domain()->size(); i++)
    TORCH_CHECK(t->domain()->axis(i-1) == s_right->domain()->axis(i));

  const TensorView *rswap = reorder(t, swap);
  TORCH_CHECK(t->domain()->axis(0) == rswap->domain()->axis(2));
  TORCH_CHECK(t->domain()->axis(2) == rswap->domain()->axis(0));
  TORCH_CHECK(t->domain()->axis(1) == rswap->domain()->axis(1));

}

void testGPU_FusionEquality(){
  Fusion fusion;
  FusionGuard fg(&fusion);

  Float* fval1 = new Float();
  const Float* fval1_copy = fval1;
  Float* fval2 = new Float();
  Float* fone = new Float(1.0);

  TORCH_CHECK(fval1->same_as(fval1_copy));
  TORCH_CHECK(!fval1->same_as(fval2));
  TORCH_CHECK(!fone->same_as(fval1));
  TORCH_CHECK(fone->same_as(new Float(1.0)));

  Int* ival1 = new Int();
  const Int* ival1_copy = ival1;
  Int* ival2 = new Int();
  Int* ione = new Int(1);

  TORCH_CHECK(ival1->same_as(ival1_copy));
  TORCH_CHECK(!ival1->same_as(ival2));
  TORCH_CHECK(!ione->same_as(ival1));
  TORCH_CHECK(ione->same_as(new Int(1)));

  const BinaryOp* add1 = new BinaryOp(BinaryOpType::Add, new Float(), fval1, ival1);
  const BinaryOp* add1_copy = new BinaryOp(BinaryOpType::Add, new Float(), fval1, ival1);
  const BinaryOp* sub1 = new BinaryOp(BinaryOpType::Sub, new Float(), fval1, ival1);

  const UnaryOp* neg1 = new UnaryOp(UnaryOpType::Neg, new Float(), fval1);
  const UnaryOp* neg2 = new UnaryOp(UnaryOpType::Neg, new Float(), fval2);
  const UnaryOp* neg1_copy = new UnaryOp(UnaryOpType::Neg, new Float(), fval1);

  TORCH_CHECK(add1->same_as(add1_copy));
  TORCH_CHECK(!add1->same_as(sub1));

  TORCH_CHECK(neg1->same_as(neg1_copy));
  TORCH_CHECK(!static_cast<const Expr*>(neg1)->same_as(add1));
  TORCH_CHECK(!neg1->same_as(neg2));

}

void testGPU_Fusion() {}

}} // torch::jit
