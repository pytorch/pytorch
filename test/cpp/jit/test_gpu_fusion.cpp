#include <test/cpp/jit/test_base.h>

// #include <torch/csrc/jit/fuser/interface.h>
#include <torch/csrc/jit/fuser/common/arith.h>
#include <torch/csrc/jit/fuser/common/fusion.h>
#include <torch/csrc/jit/fuser/common/ir.h>
#include <torch/csrc/jit/fuser/common/iriostream.h>
#include <torch/csrc/jit/fuser/common/mutator.h>
#include <torch/csrc/jit/fuser/common/tensor.h>
#include <torch/csrc/jit/fuser/common/tensor_meta.h>
#include <torch/csrc/jit/fuser/common/transform_replay.h>

// fuser and IR parser
#include <torch/csrc/jit/fuser/cuda/parser.h>
#include "torch/csrc/jit/irparser.h"

#include <iostream>

// Tests go in torch::jit
namespace torch {
namespace jit {

using namespace torch::jit::fuser;

// 1. Test cases are void() functions.
// 2. They start with the prefix `test`

void testGPU_FusionDispatch() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  Float* f = new Float{2.f};

  std::cout << "Dispatch 2.f by Float reference: " << f << std::endl;

  std::cout << "Dispatch 2.f by Val reference: " << static_cast<Val*>(f)
            << std::endl;

  std::cout << "Dispatch 2.f by Statement reference: "
            << static_cast<Statement*>(f) << std::endl;
}

void testGPU_FusionSimpleArith() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  Float* f1 = new Float(1.f);
  Float* f2 = new Float{2.f};
  Float* f3 = new Float();

  BinaryOp* an_add = new BinaryOp(BinaryOpType::Add, f3, f1, f2);
  std::cout << "Explicit add construction of 1.f + 2.f: " << fusion
            << std::endl;
}

void testGPU_FusionContainer() {
  Fusion fusion1;
  FusionGuard fg(&fusion1);

  Float* f1 = new Float(1.f);
  Float* f2 = new Float(2.f);
  auto f3 = add(f1, f2);
  std::cout << "Implicit add construction of 1.f + 2.f : " << fusion1
            << std::endl;

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

void testGPU_FusionSimpleTypePromote() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  Float* f4 = new Float{4.f};
  Int* i1 = new Int{3};
  auto f5 = binary_op(BinaryOpType::Add, f4, i1);

  TORCH_CHECK(f5->getDataType() == DataType::Float);
}

void testGPU_FusionCastOp() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  Float* f3_test = new Float{3.f};
  Int* i3 = new Int{3};
  auto f3 = cast_op(DataType::Float, i3);

  TORCH_CHECK(f3->getDataType().value() == f3_test->getDataType().value());
}

class ZeroMutator : public BaseMutator {
 public:
  Statement* mutate(Float* f) {
    if (f->isConst() && *(f->value()) == 1.0)
      return new Float(0.0);
    return f;
  }
};

void testGPU_FusionMutator() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  Float* f4 = new Float{1.f};
  Int* i1 = new Int{3};
  Val* f5 = binary_op(BinaryOpType::Add, f4, i1);
  std::cout << "Replacing floats of val 1 with 0 in: " << fusion << std::endl;
  ZeroMutator mutator;
  BaseMutator* base_mutator = &mutator;
  base_mutator->mutate(&fusion);
  std::cout << "Replaced: " << fusion << std::endl;
}

void testGPU_FusionRegister() {
  Fusion fusion;
  FusionGuard fg(&fusion);
  Float* v1 = new Float{1.f};
  Float* v2 = new Float{2.f};
  Val* v3 = binary_op(BinaryOpType::Add, v1, v2);
  Val* v4 = binary_op(BinaryOpType::Add, v1, v2);
  TORCH_CHECK(v1->name() + 1 == v2->name());
  TORCH_CHECK(v2->name() + 1 == v3->name());
  TORCH_CHECK(v3->name() + 1 == v4->name());
  TORCH_CHECK(fusion.origin(v3)->name() + 1 == fusion.origin(v4)->name());
}

// dummy expr with 2 outputs only for toposort test.
struct TORCH_API DummyExpr : public Expr {
  ~DummyExpr() = default;
  DummyExpr(Val* _outlhs, Val* _outrhs, Val* _lhs, Val* _rhs)
      : Expr(ExprType::BinaryOp) // Not terribly safe...
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

  // e0: v3, v2 = dummy(v1, v0)
  // e1: v4     =   add(v3, v2)
  // e2: v5     =   add(v2, v4)
  // e3: v6     =   add(v5, v5)
  Float* v0 = new Float{1.f};
  Float* v1 = new Float{2.f};
  Float* v2 = new Float();
  Float* v3 = new Float();
  Float* v4 = new Float();
  Float* v5 = new Float();
  Float* v6 = new Float();

  Expr* e0 = new DummyExpr(v3, v2, v1, v0);
  Expr* e1 = new BinaryOp(BinaryOpType::Add, v4, v3, v2);
  Expr* e2 = new BinaryOp(BinaryOpType::Add, v5, v2, v4);
  Expr* e3 = new BinaryOp(BinaryOpType::Add, v6, v5, v5);

  std::vector<Expr*> exprs = fusion.exprs();

  TORCH_CHECK(exprs.size() == 4);
  TORCH_CHECK(exprs[0] == e0);
  TORCH_CHECK(exprs[1] == e1);
  TORCH_CHECK(exprs[2] == e2);
  TORCH_CHECK(exprs[3] == e3);

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

  fusion.addOutput(v6);
  exprs = fusion.exprs(true);
  TORCH_CHECK(exprs.size() == 4);
  TORCH_CHECK(exprs[0] == e0);
  TORCH_CHECK(exprs[1] == e1);
  TORCH_CHECK(exprs[2] == e2);
  TORCH_CHECK(exprs[3] == e3);

  TORCH_CHECK(fusion.origin(v2)->name() == 0);
  TORCH_CHECK(fusion.origin(v3)->name() == 0);
  TORCH_CHECK(fusion.origin(v4)->name() == 1);
  TORCH_CHECK(fusion.origin(v5)->name() == 2);
  TORCH_CHECK(fusion.origin(v6)->name() == 3);
}

void testGPU_FusionTensor() {
  auto tensor = at::randn({2, 3, 4, 5}, at::kCUDA);
  auto sizes = tensor.sizes().vec();
  auto tensor_type = TensorType::create(tensor);

  Fusion fusion;
  FusionGuard fg(&fusion);
  auto fuser_tensor = new Tensor(tensor_type);
  TORCH_CHECK(fuser_tensor->hasContiguityInfo() == 1);
  TORCH_CHECK(fuser_tensor->getDataType().value() == DataType::Float);
  TORCH_CHECK(fuser_tensor->domain() == nullptr);

  auto fuser_null_tensor = new Tensor(DataType::Int);
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
        TORCH_CHECK((t_c.canCollapseToHigher(i) ^ (i != 2)));
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
      TORCH_CHECK(!(t_c.isBroadcastDim(i)) ^ (i == 1));
      if (i < 3) {
        TORCH_CHECK(!(t_c.canCollapseToHigher(i)));
      }
    }
  }

  {
    // contiguity across size-1 dimension
    auto tensor = at::randn({4, 1, 4});
    auto sizes = tensor.sizes().vec();
    auto strides = tensor.strides().vec();
    auto dim = sizes.size();
    TensorContiguity t_c(sizes, strides);
    TORCH_CHECK(t_c.rank() == sizes.size());
    auto b_dims = t_c.getBroadcastDims();
    TORCH_CHECK(b_dims.size() == 0);
    TORCH_CHECK(t_c.getFCD() == 2);
    TORCH_CHECK(t_c.hasContiguousFCD());
    for (int i = 0; i < dim; i++) {
      TORCH_CHECK(!t_c.isBroadcastDim(i));
      if (i < dim - 1) {
        TORCH_CHECK(t_c.canCollapseToHigher(i));
      }
    }
  }

  {
    // no contiguity across size-1 dimension
    auto tensor = at::randn({4, 4, 4}).split(1, 1)[0];
    auto sizes = tensor.sizes().vec();
    auto strides = tensor.strides().vec();
    TensorContiguity t_c(sizes, strides);
    TORCH_CHECK(!(t_c.canCollapseToHigher(0)));
    TORCH_CHECK((t_c.canCollapseToHigher(1)));
  }

  {
    // no contiguity across size-1 dimension
    auto tensor = at::randn({4, 1, 8}).split(4, 2)[0];
    auto sizes = tensor.sizes().vec();
    auto strides = tensor.strides().vec();
    TensorContiguity t_c(sizes, strides);
    TORCH_CHECK((t_c.canCollapseToHigher(0)));
    TORCH_CHECK((!t_c.canCollapseToHigher(1)));
  }

  {
    // no contiguity across size-1 dimension
    auto tensor = at::randn({8, 1, 4}).split(4, 0)[0];
    auto sizes = tensor.sizes().vec();
    auto strides = tensor.strides().vec();
    TensorContiguity t_c(sizes, strides);
    TORCH_CHECK((t_c.canCollapseToHigher(0)));
    TORCH_CHECK((t_c.canCollapseToHigher(1)));
  }

  {
    // test merge
    TensorContiguity t_c_l({4, 4, 4}, {16, 4, 1});
    TensorContiguity t_c_r({4, 4, 4}, {16, 4, 1});
    t_c_l.merge(t_c_r);
    TORCH_CHECK((t_c_l.isIdentical(t_c_r)));
  }

  {
    TensorContiguity t_c_l({4, 4, 4, 4}, {16, 0, 4, 1});
    TensorContiguity t_c_r({4, 4, 4, 4}, {64, 16, 4, 1});
    t_c_l.merge(t_c_r);
    TORCH_CHECK(t_c_l.getFCD() == 3);
    TORCH_CHECK(t_c_l.getAxisByStride(0) == 0);
  }

  {
    // NHWC + NCHW
    TensorContiguity t_c_l({4, 4, 4, 4}, {64, 16, 4, 1});
    TensorContiguity t_c_r({4, 4, 4, 4}, {64, 1, 16, 4});
    t_c_l.merge(t_c_r);
    TORCH_CHECK(!t_c_l.hasContiguousFCD());
    TORCH_CHECK(t_c_l.getFCD() == -1);
    TORCH_CHECK(t_c_l.getAxisByStride(0) == 0);
    TORCH_CHECK(t_c_l.getAxisByStride(1) == -1);
    TORCH_CHECK(t_c_l.getAxisByStride(2) == -1);
    TORCH_CHECK(t_c_l.getAxisByStride(3) == -1);
  }

  {
    // NCHW + NCHW with broadcasting
    TensorContiguity t_c_l({4, 4, 4, 4}, {4, 1, 4, 0});
    TensorContiguity t_c_r({4, 4, 4, 4}, {64, 1, 16, 4});
    t_c_l.merge(t_c_r);
    TORCH_CHECK(t_c_l.getFCD() == 1);
    TORCH_CHECK(t_c_l.getAxisByStride(0) == 0);
  }
}

void testGPU_FusionTVSplit() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv = new TensorView(Tensor::MakeDummyTensor(3));

  tv = split(tv, 2, 2);
  std::cout << "Split: " << tv << std::endl;

  std::cout << "Split fusion output: " << fusion << std::endl;
}

void testGPU_FusionTVMerge() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv = new TensorView(Tensor::MakeDummyTensor(3));

  tv = merge(tv, 1);

  std::cout << "Merge fusion output: " << fusion << std::endl;
}

void testGPU_FusionTVReorder() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  Tensor* dummyTensor = Tensor::MakeDummyTensor(3);

  std::unordered_map<int, int> shift_right{{-1, 0}};

  std::unordered_map<int, int> shift_left{{0, -1}};

  std::unordered_map<int, int> shift_left_2{{0, -1}, {1, 0}, {2, 1}};

  std::unordered_map<int, int> swap{{0, 2}, {2, 0}};
  TensorView* ref = new TensorView(dummyTensor);
  TensorView* tv = new TensorView(dummyTensor);

  TensorView* s_leftl = reorder(tv, shift_left);
  for (int i = 0; i < tv->domain()->size(); i++)
    TORCH_CHECK(ref->domain()->axis(i) == s_leftl->domain()->axis(i - 1));

  tv = new TensorView(dummyTensor);
  TensorView* s_left2 = reorder(tv, shift_left);
  for (int i = 0; i < tv->domain()->size(); i++)
    TORCH_CHECK(ref->domain()->axis(i) == s_left2->domain()->axis(i - 1));

  tv = new TensorView(dummyTensor);
  TensorView* s_right = reorder(tv, shift_right);
  for (int i = 0; i < tv->domain()->size(); i++)
    TORCH_CHECK(ref->domain()->axis(i - 1) == s_right->domain()->axis(i));

  tv = new TensorView(dummyTensor);
  TensorView* rswap = reorder(tv, swap);
  TORCH_CHECK(ref->domain()->axis(0) == rswap->domain()->axis(2));
  TORCH_CHECK(ref->domain()->axis(2) == rswap->domain()->axis(0));
  TORCH_CHECK(ref->domain()->axis(1) == rswap->domain()->axis(1));
}

void testGPU_FusionEquality() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  Float* fval1 = new Float();
  Float* fval1_copy = fval1;
  Float* fval2 = new Float();
  Float* fone = new Float(1.0);

  TORCH_CHECK(fval1->same_as(fval1_copy));
  TORCH_CHECK(!fval1->same_as(fval2));
  TORCH_CHECK(!fone->same_as(fval1));
  TORCH_CHECK(fone->same_as(new Float(1.0)));

  Int* ival1 = new Int();
  Int* ival1_copy = ival1;
  Int* ival2 = new Int();
  Int* ione = new Int(1);

  TORCH_CHECK(ival1->same_as(ival1_copy));
  TORCH_CHECK(!ival1->same_as(ival2));
  TORCH_CHECK(!ione->same_as(ival1));
  TORCH_CHECK(ione->same_as(new Int(1)));

  BinaryOp* add1 = new BinaryOp(BinaryOpType::Add, new Float(), fval1, ival1);
  BinaryOp* add1_copy =
      new BinaryOp(BinaryOpType::Add, new Float(), fval1, ival1);
  BinaryOp* sub1 = new BinaryOp(BinaryOpType::Sub, new Float(), fval1, ival1);

  UnaryOp* neg1 = new UnaryOp(UnaryOpType::Neg, new Float(), fval1);
  UnaryOp* neg2 = new UnaryOp(UnaryOpType::Neg, new Float(), fval2);
  UnaryOp* neg1_copy = new UnaryOp(UnaryOpType::Neg, new Float(), fval1);

  TORCH_CHECK(add1->same_as(add1_copy));
  TORCH_CHECK(!add1->same_as(sub1));

  TORCH_CHECK(neg1->same_as(neg1_copy));
  TORCH_CHECK(!static_cast<Expr*>(neg1)->same_as(add1));
  TORCH_CHECK(!neg1->same_as(neg2));
}

void testGPU_FusionReplaceAll() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  Float* f0 = new Float();
  Float* f1 = new Float{1.f};
  Float* f2 = new Float{2.f};
  Float* f3 = new Float();
  Float* f4 = static_cast<Float*>(add(f1, f0));

  // replace the output f4 with f3
  ReplaceAll::instancesOf(f4, f3);

  // f3 should now have an origin function
  TORCH_CHECK(fusion.origin(f3) != nullptr);

  // Should have removed f4 completely so we shouldn't have any other expr than
  // f3 construction
  TORCH_CHECK(fusion.exprs().size() == 1);

  // Replace constant Float's of value 1.f with 2.f
  ReplaceAll::instancesOf(f1, f2);
  BinaryOp* bop = static_cast<BinaryOp*>(fusion.origin(f3));
  // make sure the binary op (origin of f3) actually changed to 2.f
  TORCH_CHECK(static_cast<Float*>(bop->lhs())->same_as(new Float{2.f}));
}

void testGPU_FusionComputeAt() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<IterDomain*> dom;
  dom.push_back(new IterDomain(new Int()));
  dom.push_back(new IterDomain(new Int()));
  dom.push_back(new IterDomain(new Int(), ParallelType::Serial, true));
  dom.push_back(new IterDomain(new Int()));

  TensorDomain* td = new TensorDomain(dom);
  TensorView* tv0 = new TensorView(new Tensor(DataType::Float, td));
  TensorView* tv2 = new TensorView(new Tensor(DataType::Float, td));
  TensorView* tv3 = new TensorView(new Tensor(DataType::Float, td));

  // tv2 = tv3 + 1.0
  BinaryOp* add_node =
      new BinaryOp(BinaryOpType::Add, tv2, tv3, new Float(1.0));
  // tv0 = tv2 + 1.0
  BinaryOp* add_node2 =
      new BinaryOp(BinaryOpType::Add, tv0, tv2, new Float(1.0));

  //[I0, I1, R0, I2]
  tv0 = split(tv0, 0, 4);
  //[I0o, I0i{4}, I1, R0, I2]
  tv0 = merge(tv0, 1);
  //[I0o, I0i{4}*I1, R0, I2]
  tv0 = split(tv0, -1, 2);
  //[I0o, I0i{4}*I1, R0, I2o, I2i{2}]
  tv0 = reorder(tv0, {{0, 2}, {2, 0}, {3, 4}});
  //[R0, I0i{4}*I1, I0o, I2i, I2o{2}]
  std::cout << "Replaying: " << td << "\n-> " << tv0 << "\n on " << tv2
            << " and " << tv3 << "\nwith \'compute_at(2)\' produces:\n"

            << tv3->computeAt(tv0, 2)

            << "\nWhich should be unchanged, however\n"
            << tv2 << " should be along the lines of: "
            << "\n[R0, I0i{4}*I1, I0o, I2]" << std::endl;
 
}

void testGPU_FusionComputeAt2() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<IterDomain*> dom;
  dom.push_back(new IterDomain(new Int()));
  dom.push_back(new IterDomain(new Int()));
  dom.push_back(new IterDomain(new Int(), ParallelType::Serial, true));
  dom.push_back(new IterDomain(new Int()));

  TensorDomain* td = new TensorDomain(dom);
  TensorView* tv = new TensorView(new Tensor(DataType::Float, td));
  TensorView* tv2 = new TensorView(new Tensor(DataType::Float, td));

  new BinaryOp(BinaryOpType::Add, tv2, new Float(0.0), new Float(1.0));
  new BinaryOp(BinaryOpType::Add, tv, tv2, new Float(1.0));

  //[I0, I1, R0, I2]
  tv = split(tv, -1, 4);
  //[I0, I1, R0, I2o, I2i{4}]
  tv = reorder(tv, {{3, 0}, {0, 3}, {1, 4}, {4, 1}});
  //[I2o, I2i{4}, R0, I0, I1]
  tv = split(tv, 3, 2);
  //[I2o, I2i{4}, R0, I0o, I0i{2}, I1]
  tv = reorder(
      tv,
      {
          {3, 0}, {4, 1}, {5, 2}, {2, 3}
          //{0, 4} //doesn't need to be specified
          //{1, 5} //doesn't need to be specified
      });
  //[I0o, I0i{2}, I1, R0, I2o, I2i{4}]

  std::cout << "Replaying: " << td << "\n -> " << tv << "\n on " << tv2
            << "\n with \'compute_at(2)\' produces: \n"
            << tv2->computeAt(tv, 2) << std::endl;
  std::cout << "Which should be along the lines of:";
  std::cout << "[I0o, I0i{2}, I1, R0, I2]" << std::endl;
}

void testGPU_FusionComputeAt3() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<IterDomain*> dom;
  dom.push_back(new IterDomain(new Int()));
  dom.push_back(new IterDomain(new Int()));

  TensorDomain* td = new TensorDomain(dom);
  TensorView* tv0 = new TensorView(new Tensor(DataType::Float, td));
  TensorView* tv1 = new TensorView(new Tensor(DataType::Float, td));
  TensorView* tv2 = new TensorView(new Tensor(DataType::Float, td));
  TensorView* tv3 = new TensorView(new Tensor(DataType::Float, td));

  new BinaryOp(BinaryOpType::Add, tv1, new Float(0.0), new Float(1.0));
  new BinaryOp(BinaryOpType::Add, tv0, tv1, new Float(2.0));
  new BinaryOp(BinaryOpType::Add, tv2, tv1, new Float(3.0));
  new BinaryOp(BinaryOpType::Add, tv3, tv2, new Float(4.0));

  //tv1 =   0 + 1
  //tv0 = tv1 + 2
  //tv2 = tv1 + 2
  //tv3 = tv2 + 2
  std::cout << "Replaying " << tv3 << "->";
  //[I0, I1]
  tv3 = split(tv3, 0, 4);
  //[I0o, I0i{4}, I1]
  tv3 = reorder(tv3, {{2, 0}});
  //[I1, I0o, I0i{4}]
  tv3 = split(tv3, 0, 2);
  //[I1o, I1i{2} I0o, I0i{4}]
  tv3 = reorder(
      tv3,
      { {0, 2}, {1, 3} });
  //[I0o, I0i{4}, I1o, I1i{2}]

  std::cout << tv3 <<std::endl;
  tv1->computeAt(tv3, 1);

  std::cout << "on to:\n" << tv1 << "\n" << tv2 << "\nand\n" << tv0 << std::endl;
  std::cout << "These domains should approximately be: [I0o, I0i{4}, I1o, I1i{2}]" << std::endl;
}

void testGPU_FusionParser() {
  auto g = std::make_shared<Graph>();
  const auto graph0_string = R"IR(
    graph(%0 : Float(2, 3, 4),
          %1 : Float(2, 3, 4)):
      %c0 : Float(2, 3, 4) = aten::mul(%0, %1)
      %d0 : Float(2, 3, 4) = aten::mul(%c0, %0)
      return (%d0))IR";
  torch::jit::script::parseIR(graph0_string, g.get());

  // strides are not yet supported in the irparser.
  for (auto val : g->block()->inputs()) {
    if (val->isCompleteTensor())
      val->setType(val->type()->cast<TensorType>()->contiguous());
  }
  for (auto node : g->block()->nodes()) {
    for (auto val : node->outputs()) {
      if (val->isCompleteTensor())
        val->setType(val->type()->cast<TensorType>()->contiguous());
    }
  }

  Fusion fusion;
  fuser::cuda::parseJitIR(g, fusion);

  FusionGuard fg(&fusion);
  TORCH_CHECK(fusion.exprs().size() == 2);
}

void testGPU_FusionDependency() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  Float* f0 = new Float(0.f);
  Float* f1 = new Float(1.f);
  auto f2 = add(f0, f1);

  auto f3 = add(f2, f2);

  Float* f4 = new Float(4.f);
  Float* f5 = new Float(5.f);
  auto f6 = add(f4, f5);

  Float* f7 = new Float(7.f);
  Float* f8 = new Float(8.f);
  auto f9 = add(f7, f8);

  auto f10 = add(f6, f9);

  auto f11 = add(f3, f10);

  TORCH_CHECK(DependencyCheck::isDependencyOf(f0, f11));
  TORCH_CHECK(DependencyCheck::isDependencyOf(f1, f11));
  TORCH_CHECK(DependencyCheck::isDependencyOf(f2, f11));
  TORCH_CHECK(DependencyCheck::isDependencyOf(f3, f11));
  TORCH_CHECK(DependencyCheck::isDependencyOf(f6, f11));
  TORCH_CHECK(DependencyCheck::isDependencyOf(f9, f11));
  TORCH_CHECK(DependencyCheck::isDependencyOf(f0, f2));
  TORCH_CHECK(DependencyCheck::isDependencyOf(f2, f3));
  TORCH_CHECK(DependencyCheck::isDependencyOf(f4, f6));
  TORCH_CHECK(DependencyCheck::isDependencyOf(f8, f10));

  TORCH_CHECK(!DependencyCheck::isDependencyOf(f11, f0));
  TORCH_CHECK(!DependencyCheck::isDependencyOf(f11, f1));
  TORCH_CHECK(!DependencyCheck::isDependencyOf(f11, f2));
  TORCH_CHECK(!DependencyCheck::isDependencyOf(f11, f3));
  TORCH_CHECK(!DependencyCheck::isDependencyOf(f11, f4));
  TORCH_CHECK(!DependencyCheck::isDependencyOf(f11, f5));
  TORCH_CHECK(!DependencyCheck::isDependencyOf(f2, f0));
  TORCH_CHECK(!DependencyCheck::isDependencyOf(f3, f2));
  TORCH_CHECK(!DependencyCheck::isDependencyOf(f6, f4));
  TORCH_CHECK(!DependencyCheck::isDependencyOf(f10, f8));

  std::stack<Val*> dep_chain = DependencyCheck::getDependencyChain(f0, f11);
  TORCH_CHECK(dep_chain.top() == f11);
  dep_chain.pop();
  TORCH_CHECK(dep_chain.top() == f3);
  dep_chain.pop();
  TORCH_CHECK(dep_chain.top() == f2);
  dep_chain.pop();

  dep_chain = DependencyCheck::getDependencyChain(f6, f11);
  TORCH_CHECK(dep_chain.top() == f11);
  dep_chain.pop();
  TORCH_CHECK(dep_chain.top() == f10);
  dep_chain.pop();

  dep_chain = DependencyCheck::getDependencyChain(f4, f11);
  TORCH_CHECK(dep_chain.top() == f11);
  dep_chain.pop();
  TORCH_CHECK(dep_chain.top() == f10);
  dep_chain.pop();
  TORCH_CHECK(dep_chain.top() == f6);
  dep_chain.pop();

  dep_chain = DependencyCheck::getDependencyChain(f11, f2);
  TORCH_CHECK(dep_chain.empty());
}

void testGPU_Fusion() {}

} // namespace jit
} // namespace torch
