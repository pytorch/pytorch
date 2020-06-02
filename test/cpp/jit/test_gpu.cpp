#if defined(USE_CUDA)
#include <test/cpp/jit/test_base.h>

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/kernel.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/mutator.h>
#include <torch/csrc/jit/codegen/cuda/tensor_meta.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>
#include <torch/csrc/jit/codegen/cuda/transform_rfactor.h>

// fuser and IR parser
#include <torch/csrc/jit/codegen/cuda/parser.h>
#include "torch/csrc/jit/ir/irparser.h"

#include <iostream>

// Tests go in torch::jit
namespace torch {
namespace jit {

using namespace torch::jit::fuser;

TensorView* makeDummyTensor(int nDims, DataType dtype = DataType::Float) {
  std::vector<IterDomain*> dom;
  for (int i = 0; i < nDims; i++)
    dom.push_back(new IterDomain(new Int(0), new Int()));

  return new TensorView(new TensorDomain(dom), dtype);
}

// 1. Test cases are void() functions.
// 2. They start with the prefix `test`

void testGPU_FusionDispatch() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  Float* f = new Float{2.f};
  std::stringstream ss1, ss2, ss3;
  ss1 << f;
  ss2 << static_cast<Val*>(f);
  ss3 << static_cast<Statement*>(f);
  TORCH_CHECK(
      ss1.str().compare(ss2.str()) == 0 && ss1.str().compare(ss3.str()) == 0,
      "Error with dispatch system where results differ by passing Float* vs Val* vs Statement*.");
}

void testGPU_FusionSimpleArith() {
  std::stringstream ss1, ss2;

  Fusion fusion;
  FusionGuard fg(&fusion);

  Float* f1 = new Float(1.f);
  Float* f2 = new Float{2.f};
  Float* f3 = new Float();

  // Disrupt the fusion to make sure guard works well
  {
    Fusion fusion2;
    FusionGuard fg(&fusion2);

    Float* f1 = new Float(1.f);
    Float* f2 = new Float(2.f);
    add(f1, f2);
    ss2 << fusion2;
  }

  new BinaryOp(BinaryOpType::Add, f3, f1, f2);
  ss1 << fusion;

  TORCH_CHECK(
      ss1.str().compare(ss2.str()) == 0,
      "Error where explicit add nodes don't match implicit add nodes.");
}

void testGPU_FusionSimpleTypePromote() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  Float* f4 = new Float{4.f};
  Int* i1 = new Int{3};
  auto f5 = add(f4, i1);

  TORCH_CHECK(f5->getDataType() == DataType::Float);
}

class ZeroMutator : public OptOutMutator {
 public:
  Statement* mutate(Float* f) {
    if (f->isConst() && *(f->value()) == 1.0)
      return new Float(0.0);
    return f;
  }
  void mutate(Fusion* f) {
    OptOutMutator::mutate(f);
  }
};

void testGPU_FusionMutator() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  Float* f4 = new Float{1.f};
  Int* i1 = new Int{3};
  Val* f5 = add(f4, i1);
  ZeroMutator mutator;
  mutator.mutate(&fusion);
  Val* lhs = static_cast<BinaryOp*>(fusion.origin(f5))->lhs();
  TORCH_CHECK(
      lhs->getValType().value() == ValType::Scalar &&
      lhs->getDataType().value() == DataType::Float);
  Float* flhs = static_cast<Float*>(lhs);

  TORCH_CHECK(flhs->value().value() == 0.f);
}

void testGPU_FusionRegister() {
  Fusion fusion;
  FusionGuard fg(&fusion);
  Float* v1 = new Float{1.f};
  Float* v2 = new Float{2.f};
  Val* v3 = binaryOp(BinaryOpType::Add, v1, v2);
  Val* v4 = binaryOp(BinaryOpType::Add, v1, v2);
  TORCH_CHECK(v1->name() + 1 == v2->name());
  TORCH_CHECK(v2->name() + 1 == v3->name());
  TORCH_CHECK(v3->name() + 1 == v4->name());
  TORCH_CHECK(fusion.origin(v3)->name() + 1 == fusion.origin(v4)->name());
}

// dummy expr with 2 outputs only for toposort test.
struct DummyExpr : public Expr {
  ~DummyExpr() = default;
  DummyExpr(Val* _outlhs, Val* _outrhs, Val* _lhs, Val* _rhs)
      : Expr(ExprType::UnaryOp) // Not terribly safe...
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
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto tensor = at::randn({2, 3, 4, 5}, options);
  auto sizes = tensor.sizes().vec();
  auto tensor_type = TensorType::create(tensor);

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto fuser_tensor = new TensorView(tensor_type);
  TORCH_CHECK(fuser_tensor->getDataType().value() == DataType::Float);
  TORCH_CHECK(fuser_tensor->domain() != nullptr);
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
    TORCH_CHECK(t_c.rank() == (int)sizes.size());
    auto b_dims = t_c.getBroadcastDims();
    TORCH_CHECK(b_dims.size() == 0);
    TORCH_CHECK(t_c.getFCD() == 2);
    TORCH_CHECK(t_c.hasContiguousFCD());
    for (decltype(dim) i = 0; i < dim; i++) {
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

  TensorView* tv = makeDummyTensor(3);

  tv = tv->split(2, 2);
  TORCH_CHECK(tv->nDims() == 4);
  Expr* outer = tv->axis(2)->extent()->getOrigin();

  TORCH_CHECK(
      outer->getExprType().value() == ExprType::BinaryOp &&
      static_cast<BinaryOp*>(outer)->getBinaryOpType() ==
          BinaryOpType::CeilDiv &&
      static_cast<BinaryOp*>(outer)->lhs()->sameAs(
          tv->getRootDomain()->axis(2)->extent()) &&
      static_cast<Int*>(static_cast<BinaryOp*>(outer)->rhs())
          ->sameAs(new Int(2)));

  IterDomain* inner = static_cast<IterDomain*>(tv->axis(3));
  TORCH_CHECK(
      inner->extent()->isScalar() &&
      static_cast<Int*>(inner->extent())->isConst() &&
      static_cast<Int*>(inner->extent())->value().value() == 2);
}

void testGPU_FusionTVMerge() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv = makeDummyTensor(3);

  tv = tv->merge(1);
  Expr* axisOp = tv->axis(1)->extent()->getOrigin();

  TORCH_CHECK(
      tv->nDims() == 2 && axisOp->getExprType() == ExprType::BinaryOp &&
      static_cast<BinaryOp*>(axisOp)->getBinaryOpType() == BinaryOpType::Mul &&
      static_cast<BinaryOp*>(axisOp)->lhs() ==
          tv->getRootDomain()->axis(1)->extent() &&
      static_cast<BinaryOp*>(axisOp)->rhs() ==
          tv->getRootDomain()->axis(2)->extent());
}

void testGPU_FusionTVReorder() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* dummyTensor = makeDummyTensor(3);

  std::unordered_map<int, int> shift_right{{-1, 0}};

  std::unordered_map<int, int> shift_left{{0, -1}};

  std::unordered_map<int, int> shift_left_2{{0, -1}, {1, 0}, {2, 1}};

  std::unordered_map<int, int> swap{{0, 2}, {2, 0}};
  TensorView* ref = dummyTensor->clone();
  TensorView* tv = dummyTensor->clone();

  TensorView* s_leftl = tv->reorder(shift_left);
  for (int i = 0; i < (int)tv->nDims(); i++)
    TORCH_CHECK(ref->axis(i) == s_leftl->axis(i - 1));

  tv = dummyTensor->clone();
  TensorView* s_left2 = tv->reorder(shift_left);
  for (int i = 0; i < (int)tv->nDims(); i++)
    TORCH_CHECK(ref->axis(i) == s_left2->axis(i - 1));

  tv = dummyTensor->clone();
  TensorView* s_right = tv->reorder(shift_right);
  for (int i = 0; i < (int)tv->nDims(); i++)
    TORCH_CHECK(ref->axis(i - 1) == s_right->axis(i));

  tv = dummyTensor->clone();
  TensorView* rswap = tv->reorder(swap);
  TORCH_CHECK(ref->axis(0) == rswap->axis(2));
  TORCH_CHECK(ref->axis(2) == rswap->axis(0));
  TORCH_CHECK(ref->axis(1) == rswap->axis(1));
}

void testGPU_FusionEquality() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  Float* fval1 = new Float();
  Float* fval1_copy = fval1;
  Float* fval2 = new Float();
  Float* fone = new Float(1.0);

  TORCH_CHECK(fval1->sameAs(fval1_copy));
  TORCH_CHECK(!fval1->sameAs(fval2));
  TORCH_CHECK(!fone->sameAs(fval1));
  TORCH_CHECK(fone->sameAs(new Float(1.0)));

  Int* ival1 = new Int();
  Int* ival1_copy = ival1;
  Int* ival2 = new Int();
  Int* ione = new Int(1);

  TORCH_CHECK(ival1->sameAs(ival1_copy));
  TORCH_CHECK(!ival1->sameAs(ival2));
  TORCH_CHECK(!ione->sameAs(ival1));
  TORCH_CHECK(ione->sameAs(new Int(1)));

  BinaryOp* add1 = new BinaryOp(BinaryOpType::Add, new Float(), fval1, ival1);
  BinaryOp* add1_copy =
      new BinaryOp(BinaryOpType::Add, new Float(), fval1, ival1);
  BinaryOp* sub1 = new BinaryOp(BinaryOpType::Sub, new Float(), fval1, ival1);

  UnaryOp* neg1 = new UnaryOp(UnaryOpType::Neg, new Float(), fval1);
  UnaryOp* neg2 = new UnaryOp(UnaryOpType::Neg, new Float(), fval2);
  UnaryOp* neg1_copy = new UnaryOp(UnaryOpType::Neg, new Float(), fval1);

  TORCH_CHECK(add1->sameAs(add1_copy));
  TORCH_CHECK(!add1->sameAs(sub1));

  TORCH_CHECK(neg1->sameAs(neg1_copy));
  TORCH_CHECK(!static_cast<Expr*>(neg1)->sameAs(add1));
  TORCH_CHECK(!neg1->sameAs(neg2));
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
  TORCH_CHECK(static_cast<Float*>(bop->lhs())->sameAs(new Float{2.f}));
}

void testGPU_FusionParser() {
  auto g = std::make_shared<Graph>();
  const auto graph0_string = R"IR(
    graph(%0 : Float(2:1),
          %1 : Float(2:1)):
      %c0 : Float(2:1) = aten::mul(%0, %1)
      %d0 : Float(2:1) = aten::mul(%c0, %0)
      return (%d0))IR";
  torch::jit::parseIR(graph0_string, g.get());

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
  FusionGuard fg(&fusion);
  torch::jit::fuser::cuda::CudaKernel prog;
  // These can be set to anything as there are no bindings!
  // All CTAS and threads execute the same thing.
  prog.grid(4);
  prog.block(32);
  prog.device_ = 0;
  fuser::cuda::parseJitIR(g, fusion, &prog);

  std::stringstream ref;
  ref << "__global__ void CUDAGeneratedKernel(Tensor<float, 1> T0, Tensor<float, 1> T1, Tensor<float, 1> T3){\n"
      << "  float T2[4];\n"
      << "  if ( ( ( ( ( ( blockIdx.x * 4 ) + ( 4 - 1 ) ) * 128 ) + threadIdx.x ) < T1.size[0] ) ) { \n"
      << "    for(size_t i108 = 0; i108 < 4; ++i108 ) {\n"
      << "      T2[ i108 ]\n"
      << "         = T0[ ( ( ( ( ( blockIdx.x * 4 ) + i108 ) * 128 ) + threadIdx.x ) * T0.stride[0] ) ]\n"
      << "         * T1[ ( ( ( ( ( blockIdx.x * 4 ) + i108 ) * 128 ) + threadIdx.x ) * T1.stride[0] ) ];\n"
      << "    }\n"
      << "  } else { \n"
      << "    for(size_t i108 = 0; i108 < 4; ++i108 ) {\n"
      << "      if ( ( ( ( ( ( blockIdx.x * 4 ) + i108 ) * 128 ) + threadIdx.x ) < T1.size[0] ) ) { \n"
      << "        T2[ i108 ]\n"
      << "           = T0[ ( ( ( ( ( blockIdx.x * 4 ) + i108 ) * 128 ) + threadIdx.x ) * T0.stride[0] ) ]\n"
      << "           * T1[ ( ( ( ( ( blockIdx.x * 4 ) + i108 ) * 128 ) + threadIdx.x ) * T1.stride[0] ) ];\n"
      << "      }\n"
      << "    }\n"
      << "  }\n"
      << "  if ( ( ( ( ( ( blockIdx.x * 4 ) + ( 4 - 1 ) ) * 128 ) + threadIdx.x ) < T3.size[0] ) ) { \n"
      << "    for(size_t i109 = 0; i109 < 4; ++i109 ) {\n"
      << "      T3[ ( ( ( ( ( blockIdx.x * 4 ) + i109 ) * 128 ) + threadIdx.x ) * T3.stride[0] ) ]\n"
      << "         = T2[ i109 ]\n"
      << "         * T0[ ( ( ( ( ( blockIdx.x * 4 ) + i109 ) * 128 ) + threadIdx.x ) * T0.stride[0] ) ];\n"
      << "    }\n"
      << "  } else { \n"
      << "    for(size_t i109 = 0; i109 < 4; ++i109 ) {\n"
      << "      if ( ( ( ( ( ( blockIdx.x * 4 ) + i109 ) * 128 ) + threadIdx.x ) < T3.size[0] ) ) { \n"
      << "        T3[ ( ( ( ( ( blockIdx.x * 4 ) + i109 ) * 128 ) + threadIdx.x ) * T3.stride[0] ) ]\n"
      << "           = T2[ i109 ]\n"
      << "           * T0[ ( ( ( ( ( blockIdx.x * 4 ) + i109 ) * 128 ) + threadIdx.x ) * T0.stride[0] ) ];\n"
      << "      }\n"
      << "    }\n"
      << "  }\n"
      << "}\n";

  GPULower gpulw(&fusion);
  std::stringstream cdg;
  gpulw.printKernel(cdg);
  if (ref.str().size() != cdg.str().size() ||
      ref.str().compare(cdg.str()) != 0) {
    std::cerr
        << " Codegen mismatch, codegen possibly changed, or is incorrect. "
        << " \n ========= REF ========= \n"
        << ref.str() << "\n========= RESULT ========== \n"
        << cdg.str() << "\n=================" << std::endl;
    TORCH_CHECK(false);
  }
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

  auto dep_chain = DependencyCheck::getSingleDependencyChain(f0, f11);
  TORCH_CHECK(dep_chain.back() == f11);
  dep_chain.pop_back();
  TORCH_CHECK(dep_chain.back() == f3);
  dep_chain.pop_back();
  TORCH_CHECK(dep_chain.back() == f2);
  dep_chain.pop_back();

  dep_chain = DependencyCheck::getSingleDependencyChain(f6, f11);
  TORCH_CHECK(dep_chain.back() == f11);
  dep_chain.pop_back();
  TORCH_CHECK(dep_chain.back() == f10);
  dep_chain.pop_back();

  dep_chain = DependencyCheck::getSingleDependencyChain(f4, f11);
  TORCH_CHECK(dep_chain.back() == f11);
  dep_chain.pop_back();
  TORCH_CHECK(dep_chain.back() == f10);
  dep_chain.pop_back();
  TORCH_CHECK(dep_chain.back() == f6);
  dep_chain.pop_back();

  dep_chain = DependencyCheck::getSingleDependencyChain(f11, f2);
  TORCH_CHECK(dep_chain.empty());
}

void testGPU_FusionCodeGen() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeDummyTensor(3);

  new BinaryOp(BinaryOpType::Add, tv0, new Float(0.0), new Float(1.0));
  TensorView* tv1 = static_cast<TensorView*>(add(tv0, new Float(2.0)));
  TensorView* tv2 = static_cast<TensorView*>(add(tv1, new Float(3.0)));
  fusion.addOutput(tv2);

  //[I0, I1, I2]
  tv2 = tv2->split(0, 4);
  //[I0o, I0i{4}, I1, I2]
  tv2 = tv2->merge(1);
  //[I0o, I0i{4}*I1, I2]
  tv2 = tv2->split(-1, 2);
  //[I0o, I0i{4}*I1, I2o, I2i{2}]
  tv2 = tv2->reorder({{0, 1}, {1, 0}, {3, 2}});
  //[I0i{4}*I1, I0o, I2i{2}, I2o]

  tv0->computeAt(tv2, -1);

  torch::jit::fuser::cuda::CudaKernel prog;
  prog.device_ = 0;
  // These can be set to anything as there are no bindings!
  // All CTAS and threads execute the same thing.
  prog.grid(4);
  prog.block(32);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor output = at::empty({16, 8, 8}, options);

  torch::jit::fuser::cuda::compileKernel(fusion, &prog);
  torch::jit::fuser::cuda::runTestKernel(&prog, {}, {output});

  at::Tensor output_ref = at::zeros_like(output, options);
  output_ref = output_ref + 0.0 + 1.0 + 2.0 + 3.0;

  TORCH_CHECK(output_ref.equal(output));
}

void testGPU_FusionCodeGen2() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeDummyTensor(3);
  TensorView* tv1 = makeDummyTensor(3);
  TensorView* tv2 = static_cast<TensorView*>(add(tv1, new Float(2.0)));
  TensorView* tv3 = static_cast<TensorView*>(add(tv0, tv2));

  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addOutput(tv3);

  //[I0, I1, I2]
  tv3->reorder({{0, 2}, {2, 0}});
  //[I2, I1, I0]
  tv3->split(-1, 4);
  //[I2, I1, I0o, I0i{4}]
  tv3->reorder({{2, 0}, {3, 1}, {0, 3}});
  // I0o, I0i{4}, I1, I2]

  tv0->computeAt(tv3, -1);
  tv1->computeAt(tv3, -1);

  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);

  torch::jit::fuser::cuda::CudaKernel prog;
  prog.device_ = 0;
  prog.grid(4);
  prog.block(8);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor input1 = at::randn({16, 8, 8}, options);
  at::Tensor input2 = at::randn_like(input1);
  ;
  at::Tensor output = at::empty_like(input1);

  torch::jit::fuser::cuda::compileKernel(fusion, &prog);
  torch::jit::fuser::cuda::runTestKernel(&prog, {input1, input2}, {output});

  at::Tensor tv2_ref = input2 + 2.0;
  at::Tensor output_ref = input1 + tv2_ref;

  TORCH_CHECK(output_ref.equal(output));
}

void testGPU_FusionSimplePWise() {
  Fusion fusion;
  FusionGuard fg(&fusion);
  // dimensionality of the problem
  int nDims = 3;

  // Set up your input tensor views
  TensorView* tv0 = makeDummyTensor(nDims);
  TensorView* tv1 = makeDummyTensor(nDims);

  // Register your inputs
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // Do math with it, it returns a `Val*` but can be static_casted back to
  // TensorView
  TensorView* tv2 = static_cast<TensorView*>(add(tv1, new Float(2.0)));
  TensorView* tv3 = static_cast<TensorView*>(add(tv0, tv2));

  // Register your outputs
  fusion.addOutput(tv3);

  // Do transformations, remember, transformations are outputs to inputs
  // This doesn't have to be in this order
  tv3->merge(1);
  tv3->merge(0);

  // Split by n_threads
  tv3->split(-1, 128 * 2);
  tv3->split(-1, 128);

  // For all inputs, computeAt the output inline, temporaries should be squeezed
  // between them
  tv0->computeAt(tv3, -1);
  tv1->computeAt(tv3, -1);

  // Parallelize TV3
  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(-2)->parallelize(ParallelType::TIDy);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);

  torch::jit::fuser::cuda::CudaKernel prog;
  prog.device_ = 0;
  prog.grid(64); //   1 CTA
  prog.block(128, 2); // 256 Threads

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor input1 = at::randn({64, 2, 128}, options);
  at::Tensor input2 = at::rand_like(input1);
  at::Tensor output = at::empty_like(input1);

  torch::jit::fuser::cuda::compileKernel(fusion, &prog);
  torch::jit::fuser::cuda::runTestKernel(&prog, {input1, input2}, {output});

  at::Tensor tv2_ref = input2 + 2.0;
  at::Tensor output_ref = input1 + tv2_ref;

  TORCH_CHECK(output_ref.equal(output));
}

void testGPU_FusionExecKernel() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeDummyTensor(2);
  TensorView* tv1 = makeDummyTensor(2);

  // Register your inputs
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // Do math with it, it returns a `Val*` but can be static_casted back to
  // TensorView
  TensorView* tv2 = static_cast<TensorView*>(add(tv1, new Float(2.0)));
  TensorView* tv3 = static_cast<TensorView*>(add(tv0, tv2));

  // Register your outputs
  fusion.addOutput(tv3);

  tv3->merge(0);
  tv3->split(0, 128);
  tv3->split(0, 4);

  // For all inputs, computeAt the output inline, temporaries should be squeezed
  // between them
  tv0->computeAt(tv3, 1);
  tv1->computeAt(tv3, 1);

  // Parallelize TV3
  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::Unroll);
  tv3->axis(1)->parallelize(ParallelType::Unroll);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);

  torch::jit::fuser::cuda::CudaKernel prog;
  prog.device_ = 0;
  prog.grid(1); // 1 CTA
  prog.block(128); // 128 Threads

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor input1 = at::ones({1, 128}, options);
  at::Tensor input2 = at::ones_like(input1);

  at::Tensor output = at::empty_like(input1);

  torch::jit::fuser::cuda::compileKernel(fusion, &prog);
  torch::jit::fuser::cuda::runTestKernel(&prog, {input1, input2}, {output});

  at::Tensor check = at::full({1, 128}, 4, options);
  ;
  TORCH_CHECK(output.equal(check));
}

int ceilDiv_(int a, int b) {
  return (a + b - 1) / b;
}

void testGPU_FusionAdvancedComputeAt() {
  // Case 1
  /*
   * tv1 = tv0 * -1
   * tv2 = tv0 + 3
   * tv3 = tv0 * 2
   * tv4 = tv2 + tv1
   * tv5 = tv4 + tv3
   * tv6 = tv0 + tv3
   */
  {
    Fusion fusion;
    FusionGuard fg(&fusion);

    TensorView* tv0 = makeDummyTensor(2);
    fusion.addInput(tv0);

    TensorView* tv1 = static_cast<TensorView*>(mul(tv0, new Float(-1.0)));
    TensorView* tv2 = static_cast<TensorView*>(add(tv0, new Float(3.0)));
    TensorView* tv3 = static_cast<TensorView*>(mul(tv0, new Float(2.0)));
    TensorView* tv4 = static_cast<TensorView*>(add(tv2, tv1));

    TensorView* tv5 = static_cast<TensorView*>(add(tv4, tv3));
    TensorView* tv6 = static_cast<TensorView*>(add(tv0, tv3));

    fusion.addOutput(tv5);
    fusion.addOutput(tv6);

    tv0->computeAt(tv3, 1);

    // // Check propagation of this computeAt.
    TORCH_CHECK(tv0->getComputeAtView() == tv3);
    TORCH_CHECK(tv1->getComputeAtView() == tv4);
    TORCH_CHECK(tv2->getComputeAtView() == tv4);
    TORCH_CHECK(tv3->getComputeAtView() == tv6);
    TORCH_CHECK(tv4->getComputeAtView() == tv5);
    TORCH_CHECK(tv5->getComputeAtView() == tv6);
    TORCH_CHECK(!tv6->hasComputeAt());

    // Lets setup to actually run
    tv6->merge(0);
    tv6->split(0, 128);
    tv6->split(0, 4);

    tv6->axis(0)->parallelize(ParallelType::BIDx);

    tv0->computeAt(tv6, 1);

    TORCH_CHECK(tv0->getComputeAtView() == tv3 && tv0->nDims() == 3);
    TORCH_CHECK(tv1->getComputeAtView() == tv4 && tv1->nDims() == 3);
    TORCH_CHECK(tv2->getComputeAtView() == tv4 && tv2->nDims() == 3);
    TORCH_CHECK(tv3->getComputeAtView() == tv6 && tv3->nDims() == 3);
    TORCH_CHECK(tv4->getComputeAtView() == tv5 && tv4->nDims() == 3);
    TORCH_CHECK(tv5->getComputeAtView() == tv6 && tv5->nDims() == 3);
    TORCH_CHECK(!tv6->hasComputeAt());

    for (Val* val : fusion.vals()) {
      if (!fusion.hasInput(val) &&
          val->getValType().value() == ValType::TensorView) {
        TensorView* tv = static_cast<TensorView*>(val);
        tv->axis(1)->parallelize(ParallelType::Unroll);
        tv->axis(-1)->parallelize(ParallelType::TIDx);
      }
    }

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

    at::Tensor t0 = at::randn({129, 127}, options);

    auto t1 = t0.mul({-1.0});
    auto t2 = t0.add({3.0});
    auto t3 = t0.mul({2.0});
    auto t4 = t2.add(t1);
    auto t5 = t4.add(t3);
    auto t6 = t0.add(t3);

    at::Tensor kernel_tv5 = at::empty_like(t0, options);
    at::Tensor kernel_tv6 = at::empty_like(t0, options);

    torch::jit::fuser::cuda::CudaKernel prog;
    prog.device_ = 0;

    int blocks = ceilDiv_(
        ceilDiv_(t0.numel(), 128), 4); // numel / unroll factor / threads
    prog.grid(blocks);
    prog.block(128);
    torch::jit::fuser::cuda::compileKernel(fusion, &prog);
    torch::jit::fuser::cuda::runTestKernel(
        &prog, {t0}, {kernel_tv5, kernel_tv6});

    TORCH_CHECK(at::allclose(kernel_tv5, t5));
    TORCH_CHECK(at::allclose(kernel_tv6, t6));
  }

  // Case 2
  /*
   * tv1 = tv0 * -1
   * tv2 = tv0 + 3
   * tv3 = tv0 * 2
   * tv4 = tv2 + tv1
   * tv5 = tv4 + tv3
   * tv6 = tv5 + tv3
   */
  {
    Fusion fusion;
    FusionGuard fg(&fusion);

    TensorView* tv0 = makeDummyTensor(2);
    fusion.addInput(tv0);

    TensorView* tv1 = static_cast<TensorView*>(mul(tv0, new Float(-1.0)));
    TensorView* tv2 = static_cast<TensorView*>(add(tv0, new Float(3.0)));
    TensorView* tv3 = static_cast<TensorView*>(mul(tv0, new Float(2.0)));
    TensorView* tv4 = static_cast<TensorView*>(add(tv2, tv1));

    TensorView* tv5 = static_cast<TensorView*>(add(tv4, tv3));
    TensorView* tv6 = static_cast<TensorView*>(add(tv5, tv3));

    fusion.addOutput(tv5);
    fusion.addOutput(tv6);

    tv2->computeAt(tv4, 1);
    TORCH_CHECK(!tv0->hasComputeAt());
    TORCH_CHECK(!tv1->hasComputeAt());
    TORCH_CHECK(tv2->getComputeAtView() == tv4);
    TORCH_CHECK(!tv3->hasComputeAt());
    TORCH_CHECK(!tv4->hasComputeAt());
    TORCH_CHECK(!tv5->hasComputeAt());
    TORCH_CHECK(!tv6->hasComputeAt());

    // Lets setup to actually run
    tv6->merge(0);
    tv6->split(0, 128);
    tv6->split(0, 4);

    tv6->axis(0)->parallelize(ParallelType::BIDx);

    tv0->computeAt(tv6, 1);

    for (Val* val : fusion.vals()) {
      if (!fusion.hasInput(val) &&
          val->getValType().value() == ValType::TensorView) {
        TensorView* tv = static_cast<TensorView*>(val);

        tv->axis(1)->parallelize(ParallelType::Unroll);
        tv->axis(-1)->parallelize(ParallelType::TIDx);
      }
    }

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor t0 = at::randn({129, 127}, options);

    auto t1 = t0.mul({-1.0});
    auto t2 = t0.add({3.0});
    auto t3 = t0.mul({2.0});
    auto t4 = t2.add(t1);
    auto t5 = t4.add(t3);
    auto t6 = t5.add(t3);

    at::Tensor kernel_tv5 = at::empty_like(t0, options);
    at::Tensor kernel_tv6 = at::empty_like(t0, options);

    torch::jit::fuser::cuda::CudaKernel prog;
    prog.device_ = 0;

    int blocks = ceilDiv_(
        ceilDiv_(t0.numel(), 128), 4); // numel / unroll factor / threads
    prog.grid(blocks);
    prog.block(128);
    torch::jit::fuser::cuda::compileKernel(fusion, &prog);
    torch::jit::fuser::cuda::runTestKernel(
        &prog, {t0}, {kernel_tv5, kernel_tv6});

    GPULower gpulw(&fusion);
    std::stringstream cdg;
    gpulw.printKernel(cdg);

    TORCH_CHECK(at::allclose(kernel_tv5, t5), cdg.str());
    TORCH_CHECK(at::allclose(kernel_tv6, t6));
  }

  // Case 3
  // T2 = T1 * 0.979361
  // T3 = T2 * T0
  {
    Fusion fusion;
    FusionGuard fg(&fusion);

    TensorView* tv0 = makeDummyTensor(4);
    fusion.addInput(tv0);

    TensorView* tv1 = makeDummyTensor(4);
    fusion.addInput(tv1);

    TensorView* tv2 = static_cast<TensorView*>(mul(tv1, new Float(.979361)));
    TensorView* tv3 = static_cast<TensorView*>(mul(tv2, tv0));

    fusion.addOutput(tv3);

    // Lets setup to actually run
    while (tv3->nDims() > 1)
      tv3->merge(0);
    tv3->split(0, 128);
    tv3->split(0, 4);

    tv0->computeAt(tv3, 1);
    tv1->computeAt(tv3, 1);

    tv3->axis(0)->parallelize(ParallelType::BIDx);

    for (Val* val : fusion.vals()) {
      if (!fusion.hasInput(val) &&
          val->getValType().value() == ValType::TensorView) {
        TensorView* tv = static_cast<TensorView*>(val);

        tv->axis(1)->parallelize(ParallelType::Unroll);
        tv->axis(-1)->parallelize(ParallelType::TIDx);
      }
    }

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor t0 = at::randn({129, 127, 63, 65}, options);
    at::Tensor t1 = at::rand_like(t0, options);

    auto t2 = t1.mul({0.979361});
    auto t3 = t2.mul(t0);

    at::Tensor kernel_tv3 = at::empty_like(t0, options);

    torch::jit::fuser::cuda::CudaKernel prog;
    prog.device_ = 0;

    int blocks = ceilDiv_(
        ceilDiv_(t0.numel(), 128), 4); // numel / unroll factor / threads

    prog.grid(blocks);
    prog.block(128);
    torch::jit::fuser::cuda::compileKernel(fusion, &prog);
    torch::jit::fuser::cuda::runTestKernel(&prog, {t0, t1}, {kernel_tv3});

    GPULower gpulw(&fusion);
    std::stringstream cdg;
    gpulw.printKernel(cdg);

    TORCH_CHECK(at::allclose(kernel_tv3, t3), cdg.str());
  }

  // Case 4
  // T4 = T2 - T3
  // T5 = T1 + T4
  // T6 = T5 - T0
  {
    Fusion fusion;
    FusionGuard fg(&fusion);

    TensorView* tv0 = makeDummyTensor(4);
    fusion.addInput(tv0);

    TensorView* tv1 = makeDummyTensor(4);
    fusion.addInput(tv1);

    TensorView* tv2 = makeDummyTensor(4);
    fusion.addInput(tv2);

    TensorView* tv3 = makeDummyTensor(4);
    fusion.addInput(tv3);

    TensorView* tv4 = static_cast<TensorView*>(sub(tv2, tv3));
    TensorView* tv5 = static_cast<TensorView*>(add(tv1, tv4));
    TensorView* tv6 = static_cast<TensorView*>(sub(tv5, tv0));

    fusion.addOutput(tv6);

    // Lets setup to actually run
    while (tv6->nDims() > 1)
      tv6->merge(0);
    tv6->split(0, 128);
    tv6->split(0, 4);

    tv0->computeAt(tv6, 1);
    tv1->computeAt(tv6, 1);
    tv2->computeAt(tv6, 1);
    tv3->computeAt(tv6, 1);

    tv6->axis(0)->parallelize(ParallelType::BIDx);

    for (Val* val : fusion.vals()) {
      if (!fusion.hasInput(val) &&
          val->getValType().value() == ValType::TensorView) {
        TensorView* tv = static_cast<TensorView*>(val);

        tv->axis(1)->parallelize(ParallelType::Unroll);
        tv->axis(-1)->parallelize(ParallelType::TIDx);
      }
    }

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor t0 = at::randn({129, 127, 63, 65}, options);
    at::Tensor t1 = at::rand_like(t0, options);
    at::Tensor t2 = at::rand_like(t0, options);
    at::Tensor t3 = at::rand_like(t0, options);

    auto t4 = t2.sub(t3);
    auto t5 = t1.add(t4);
    auto t6 = t5.sub(t0);

    at::Tensor kernel_tv6 = at::empty_like(t0, options);

    torch::jit::fuser::cuda::CudaKernel prog;
    prog.device_ = 0;

    int blocks = ceilDiv_(
        ceilDiv_(t0.numel(), 128), 4); // numel / unroll factor / threads

    prog.grid(blocks);
    prog.block(128);
    torch::jit::fuser::cuda::compileKernel(fusion, &prog);
    torch::jit::fuser::cuda::runTestKernel(
        &prog, {t0, t1, t2, t3}, {kernel_tv6});

    GPULower gpulw(&fusion);
    std::stringstream cdg;
    gpulw.printKernel(cdg);

    TORCH_CHECK(at::allclose(kernel_tv6, t6), cdg.str());
  }
}

void testGPU_FusionScalarInputs() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeDummyTensor(2);
  fusion.addInput(tv0);
  TensorView* tv1 = makeDummyTensor(2);
  fusion.addInput(tv1);

  Float* f0 = new Float();
  fusion.addInput(f0);
  Float* f1 = new Float();
  fusion.addInput(f1);
  Float* f2 = new Float();
  fusion.addInput(f2);
  Float* f3 = new Float();
  fusion.addInput(f3);
  Val* f4 = mul(f0, f1);
  Val* f5 = sub(f2, f3);

  TensorView* tv2 = static_cast<TensorView*>(sub(tv1, f4));
  TensorView* tv3 = static_cast<TensorView*>(add(tv0, f5));
  TensorView* tv4 = static_cast<TensorView*>(mul(tv3, tv2));

  fusion.addOutput(tv4);

  // Lets setup to actually run
  while (tv4->nDims() > 1)
    tv4->merge(0);
  tv4->split(0, 128);
  tv4->split(0, 4);

  tv0->computeAt(tv4, 1);
  tv1->computeAt(tv4, 1);

  tv4->axis(0)->parallelize(ParallelType::BIDx);

  for (Val* val : fusion.vals()) {
    if (!fusion.hasInput(val) &&
        val->getValType().value() == ValType::TensorView) {
      TensorView* tv = static_cast<TensorView*>(val);

      tv->axis(1)->parallelize(ParallelType::Unroll);
      tv->axis(-1)->parallelize(ParallelType::TIDx);
    }
  }

  // f4 = f0 * f1
  // f5 = f2 - f3
  // t2 = t1 - f4
  // t3 = t0 + f5
  // t4 = t3 * t2

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  float fl0 = 0.1;
  float fl1 = -0.2;
  float fl2 = 0.3;
  float fl3 = -0.4;
  float fl4 = fl0 * fl1;
  float fl5 = fl2 - fl3;

  at::Tensor t0 = at::randn({129, 127}, options);
  at::Tensor t1 = at::rand_like(t0, options);

  auto t2 = t1.sub(fl4);
  auto t3 = t0.add(fl5);
  auto t4 = t3.mul(t2);

  at::Tensor kernel_tv4 = at::empty_like(t0, options);

  torch::jit::fuser::cuda::CudaKernel prog;
  prog.device_ = 0;

  int blocks =
      ceilDiv_(ceilDiv_(t0.numel(), 128), 4); // numel / unroll factor / threads

  prog.grid(blocks);
  prog.block(128);
  torch::jit::fuser::cuda::compileKernel(fusion, &prog);
  at::Scalar test(fl0);

  torch::jit::fuser::cuda::runTestKernel(
      &prog,
      {t0,
       t1,
       at::Scalar(fl0),
       at::Scalar(fl1),
       at::Scalar(fl2),
       at::Scalar(fl3)},
      {kernel_tv4});

  GPULower gpulw(&fusion);
  std::stringstream cdg;
  gpulw.printKernel(cdg);

  TORCH_CHECK(at::allclose(kernel_tv4, t4), cdg.str());
}

void testGPU_FusionLoopUnroll() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeDummyTensor(1);
  TensorView* tv1 = makeDummyTensor(1);

  // Register your inputs
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // Do math with it, it returns a `Val*` but can be static_casted back to
  // TensorView
  TensorView* tv2 = static_cast<TensorView*>(add(tv1, new Float(2.0)));
  TensorView* tv3 = static_cast<TensorView*>(add(tv0, tv2));

  // Register your outputs
  fusion.addOutput(tv3);

  int block_size = 16;

  tv3->split(0, block_size);
  tv3->split(0, 4);

  // For all inputs, computeAt the output inline, temporaries should be squeezed
  // between them
  tv0->computeAt(tv3, 1);
  tv1->computeAt(tv3, 1);

  // Parallelize
  tv2->axis(1)->parallelize(ParallelType::Unroll);
  tv3->axis(1)->parallelize(ParallelType::Unroll);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(0)->parallelize(ParallelType::BIDx);

  int inp_size = 129;

  torch::jit::fuser::cuda::CudaKernel prog;
  prog.device_ = 0;
  prog.grid((inp_size + 63) / 64);
  prog.block(block_size);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor input1 = at::ones({inp_size}, options);
  at::Tensor input2 = at::ones_like(input1);

  at::Tensor output = at::empty_like(input1);

  torch::jit::fuser::cuda::compileKernel(fusion, &prog);
  torch::jit::fuser::cuda::runTestKernel(&prog, {input1, input2}, {output});

  at::Tensor check = at::full({inp_size}, 4, options);

  TORCH_CHECK(output.equal(check));
}

void testGPU_FusionForLoop() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const auto TV0 = new TensorView(
      new TensorDomain({new IterDomain(new Int(0), new Int(16))}),
      DataType::Float);
  const auto TV1 = new TensorView(
      new TensorDomain({new IterDomain(new Int(0), new Int(16))}),
      DataType::Float);

  fusion.addInput(TV0);
  fusion.addInput(TV1);

  auto ID0 = new IterDomain(new Int(0), new Int(8));

  TensorView* TV2 = static_cast<TensorView*>(add(TV0, TV1));
  BinaryOp* op = static_cast<BinaryOp*>(TV2->getOrigin());
  fusion.addOutput(TV2);

  ForLoop* fl = new ForLoop(new Int(), ID0, {op});
  std::stringstream result;
  std::stringstream ref;
  result << fl;
  ref << "for(size_t i3{0}; i3 < iS{8}; ++i3 ) {\nT2[ iS{16} ] = T0[ iS{16} ] + T1[ iS{16} ]\n}";

  if (result.str().compare(ref.str()) == 0) {
    std::stringstream err_msg;
    err_msg << "ForLoop printing has changed or something has gone wrong. "
            << result.str() << "\n does not match reference: " << ref.str()
            << std::endl;
    TORCH_CHECK(false, err_msg.str());
  }
}

/*
 * Helper function for single op testing that generates a codegen operand
 */

Val* gen_jit_operand(std::pair<ValType, DataType> desc) {
  if (desc.first == ValType::TensorView) {
    return makeDummyTensor(2, desc.second);
  } else if (desc.first == ValType::Scalar) {
    if (desc.second == DataType::Float)
      return new Float();
    else if (desc.second == DataType::Int)
      return new Int();
    else
      TORCH_CHECK("Not currently supported type", desc.first);
  } else {
    TORCH_CHECK("Not currently supported type", desc.first);
  }
  return nullptr;
}

/*
 * Helper function for single op testing that generates an ATen operand
 */

IValue gen_aten_operand(
    std::pair<ValType, DataType> desc,
    int blocks,
    int threads,
    bool rand) {
  if (desc.first == ValType::TensorView) {
    if (desc.second == DataType::Float) {
      auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
      if (rand)
        return IValue(at::rand({blocks, threads}, options));
      else
        return IValue(at::empty({blocks, threads}, options));
    } else if (desc.second == DataType::Half) {
      auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
      if (rand)
        return IValue(at::rand({blocks, threads}, options));
      else
        return IValue(at::empty({blocks, threads}, options));
    } else if (desc.second == DataType::Bool) {
      if (rand) {
        auto options =
            at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
        return IValue(at::rand({blocks, threads}, options).to(at::kBool));
      } else {
        auto options =
            at::TensorOptions().dtype(at::kBool).device(at::kCUDA, 0);
        return IValue(at::empty({blocks, threads}, options));
      }
    } else {
      TORCH_CHECK("Not currently supported type", desc.second)
    }
  } else if (desc.first == ValType::Scalar) {
    if (desc.second == DataType::Float)
      return IValue(at::Scalar(1.f));
    else if (desc.second == DataType::Int)
      return IValue(at::Scalar(1));
    else
      TORCH_CHECK("Not currently supported type", desc.first);
  } else {
    TORCH_CHECK("Not currently supported type", desc.first);
  }
  return nullptr;
}

/*
 * Templatized Helper Function To generate single Op comparison between the
 * JIT codegen for Cuda and the ATen Library.
 */

using OutputPair = std::pair<ValType, DataType>;
template <
    typename AtenFunc,
    typename JitFunc,
    typename InputTuple,
    size_t... NumInputs>
void test_op(
    int blocks,
    int threads,
    std::string op_str,
    AtenFunc af,
    JitFunc jf,
    OutputPair op,
    InputTuple it,
    std::index_sequence<NumInputs...>) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Generate Input JIT function Inputs and add them as Inputs to the Fusion
  // Graph
  std::array<Val*, sizeof...(NumInputs)> jit_inputs = {
      gen_jit_operand(std::get<NumInputs>(it))...};
  std::for_each(jit_inputs.begin(), jit_inputs.end(), [&fusion](Val* v) {
    fusion.addInput(v);
  });
  TensorView* out =
      static_cast<TensorView*>(jf(std::get<NumInputs>(jit_inputs)...));
  fusion.addOutput(out);

  std::for_each(jit_inputs.begin(), jit_inputs.end(), [out](Val* v) {
    if (v->getValType() == ValType::TensorView)
      static_cast<TensorView*>(v)->computeAt(out, -1);
  });
  out->axis(0)->parallelize(ParallelType::BIDx);
  out->axis(-1)->parallelize(ParallelType::TIDx);

  torch::jit::fuser::cuda::CudaKernel prog;
  prog.device_ = 0;
  prog.grid(blocks);
  prog.block(threads);
  torch::jit::fuser::cuda::compileKernel(fusion, &prog);

  std::array<IValue, sizeof...(NumInputs)> aten_inputs = {gen_aten_operand(
      std::get<NumInputs>(it), blocks, threads, /*rand*/ true)...};
  const at::ArrayRef<IValue> aten_inputs_ivalues(aten_inputs);

  at::Tensor output =
      gen_aten_operand(op, blocks, threads, /*rand*/ false).toTensor();
  std::vector<at::Tensor> output_vect = {output};
  cudaDeviceSynchronize();
  if (fusion.hasRNG())
    at::manual_seed(0);
  torch::jit::fuser::cuda::runTestKernel(
      &prog, aten_inputs_ivalues, output_vect);
  cudaDeviceSynchronize();

  if (fusion.hasRNG())
    at::manual_seed(0);
  at::Tensor ref_output = af(aten_inputs);
  cudaDeviceSynchronize(); // This sync shouldn't be necessary;

  std::function<std::string()> aten_inputs_to_str =
      [&aten_inputs]() -> std::string {
    int input_cnt = 1;
    std::stringstream ss;
    std::for_each(
        aten_inputs.begin(), aten_inputs.end(), [&input_cnt, &ss](IValue& iv) {
          ss << "\nINPUT" << input_cnt++ << ": " << iv.toTensor();
        });
    return ss.str();
  };

  at::Tensor diff;
  if (output.scalar_type() == at::kBool) {
    diff = at::eq(output, ref_output);
  } else {
    diff = at::sub(output, ref_output);
  }

  TORCH_CHECK(
      (output.scalar_type() == at::kBool
           ? output.equal(ref_output)
           :
           // The absolute Tolerance was raised to 1e-07 from 1e-08 to allow
           // allow for the remainder function to pass.
           output.allclose(ref_output, /*rtol*/ 1e-05, /*atol*/ 1e-07)),
      "\nOp Type: -- ",
      op_str,
      " -- had a mismatch.",
      aten_inputs_to_str(),
      "\nJIT: ",
      output,
      "\nREF: ",
      ref_output,
      "\nDIFF: ",
      diff,
      "\n");
}

/*
 *  Templatized Helper Function that uses variadic templates to
 *  process a variable length Input Tuple of different Operand Type.
 */
template <typename AtenFunc, typename JitFunc, typename InputTuple>
void test_op(
    int blocks,
    int threads,
    std::string op_str,
    AtenFunc af,
    JitFunc jf,
    OutputPair op,
    InputTuple it) {
  static constexpr auto size = std::tuple_size<InputTuple>::value;
  test_op(
      blocks,
      threads,
      op_str,
      af,
      jf,
      op,
      it,
      std::make_index_sequence<size>{});
}

void testGPU_FusionUnaryOps() {
  using OpTuple =
      std::tuple<at::Tensor (*)(const at::Tensor&), UnaryOpType, std::string>;

  // [Note: explicit tuple type for uniform initialization list]
  // Tuple type must be explicitly specified for each uniform initialization
  // list within the vector to make this code compatible with some old env
  // which we still need to support. eg. gcc 5.4 + cuda 9.2.
  std::vector<OpTuple> ops{
      OpTuple{at::abs, UnaryOpType::Abs, "abs"},
      OpTuple{at::acos, UnaryOpType::Acos, "acos"},
      OpTuple{at::asin, UnaryOpType::Asin, "asin"},
      OpTuple{at::atan, UnaryOpType::Atan, "atan"},
      // There does not appear to be an appropriate ATen function for atanh
      // OpTuple{at::atanh,      UnaryOpType::Atanh,      "atanh"      },
      OpTuple{at::ceil, UnaryOpType::Ceil, "ceil"},
      OpTuple{at::cos, UnaryOpType::Cos, "cos"},
      OpTuple{at::cosh, UnaryOpType::Cosh, "cosh"},
      OpTuple{at::erf, UnaryOpType::Erf, "erf"},
      OpTuple{at::erfc, UnaryOpType::Erfc, "erfc"},
      OpTuple{at::exp, UnaryOpType::Exp, "exp"},
      OpTuple{at::expm1, UnaryOpType::Expm1, "expm1"},
      OpTuple{at::floor, UnaryOpType::Floor, "floor"},
      OpTuple{at::frac, UnaryOpType::Frac, "frac"},
      OpTuple{at::gelu, UnaryOpType::Gelu, "gelu"},
      OpTuple{at::lgamma, UnaryOpType::Lgamma, "lgamma"},
      OpTuple{at::log, UnaryOpType::Log, "log"},
      OpTuple{at::log10, UnaryOpType::Log10, "log10"},
      OpTuple{at::log1p, UnaryOpType::Log1p, "log1p"},
      OpTuple{at::log2, UnaryOpType::Log2, "log2"},
      OpTuple{at::neg, UnaryOpType::Neg, "neg"},
      OpTuple{at::reciprocal, UnaryOpType::Reciprocal, "reciprocal"},
      OpTuple{at::relu, UnaryOpType::Relu, "relu"},
      OpTuple{at::round, UnaryOpType::Round, "round"},
      OpTuple{at::rsqrt, UnaryOpType::Rsqrt, "rsqrt"},
      OpTuple{at::sigmoid, UnaryOpType::Sigmoid, "sigmoid"},
      OpTuple{at::sin, UnaryOpType::Sin, "sin"},
      OpTuple{at::sinh, UnaryOpType::Sinh, "sinh"},
      OpTuple{at::sqrt, UnaryOpType::Sqrt, "sqrt"},
      OpTuple{at::tan, UnaryOpType::Tan, "tan"},
      OpTuple{at::tanh, UnaryOpType::Tanh, "tanh"},
      OpTuple{at::trunc, UnaryOpType::Trunc, "trunc"}};

  std::for_each(ops.begin(), ops.end(), [](OpTuple& op) {
    test_op(
        /*blocks*/ 640,
        /*threads*/ 64,
        /*name*/ std::get<2>(op),
        /*Aten Func   */
        [&op](std::array<IValue, 1>& vals) {
          return std::get<0>(op)(vals[0].toTensor());
        },
        /*JIT  Func   */
        [&op](Val* in1) -> Val* { return unaryOp(std::get<1>(op), in1); },
        /*Output      */ std::make_pair(ValType::TensorView, DataType::Float),
        /*Inputs Tuple*/
        std::make_tuple(std::make_pair(ValType::TensorView, DataType::Float)));
  });

  test_op(
      /*blocks*/ 128,
      /*threads*/ 64,
      /*name*/ "rand_like",
      /*Aten Func   */
      [](std::array<IValue, 1>& vals) {
        return at::rand_like(vals[0].toTensor());
      },
      /*JIT  Func   */
      [](Val* in1) -> Val* { return unaryOp(UnaryOpType::RandLike, in1); },
      /*Output      */ std::make_pair(ValType::TensorView, DataType::Float),
      /*Inputs Tuple*/
      std::make_tuple(std::make_pair(ValType::TensorView, DataType::Float)));
}

void testGPU_FusionBinaryOps() {
  using AtenFuncSig = at::Tensor (*)(const at::Tensor&, const at::Tensor&);
  using OpTuple = std::tuple<AtenFuncSig, BinaryOpType, std::string>;

  // see [Note: explicit tuple type for uniform initialization list]
  std::vector<OpTuple> logic_ops{OpTuple{at::eq, BinaryOpType::Eq, "eq"},
                                 OpTuple{at::ge, BinaryOpType::GE, "ge"},
                                 OpTuple{at::gt, BinaryOpType::GT, "gt"},
                                 OpTuple{at::le, BinaryOpType::LE, "le"},
                                 OpTuple{at::lt, BinaryOpType::LT, "lt"},
                                 OpTuple{at::ne, BinaryOpType::NE, "ne"}};

  std::for_each(logic_ops.begin(), logic_ops.end(), [](OpTuple& op) {
    test_op(
        /*blocks*/ 640,
        /*threads*/ 64,
        /*name*/ std::get<2>(op),
        /*Aten Func   */
        [&op](std::array<IValue, 2>& vals) {
          return std::get<0>(op)(vals[0].toTensor(), vals[1].toTensor());
        },
        /*JIT  Func   */
        [&op](Val* in1, Val* in2) -> Val* {
          return binaryOp(std::get<1>(op), in1, in2);
        },
        /*Output      */ std::make_pair(ValType::TensorView, DataType::Bool),
        /*Inputs Tuple*/
        std::make_tuple(
            std::make_pair(ValType::TensorView, DataType::Float),
            std::make_pair(ValType::TensorView, DataType::Float)));
  });

  // see [Note: explicit tuple type for uniform initialization list]
  std::vector<OpTuple> math_ops{
      OpTuple{at::atan2, BinaryOpType::Atan2, "atan2"},
      OpTuple{at::div, BinaryOpType::Div, "div"},
      OpTuple{at::fmod, BinaryOpType::Fmod, "fmod"},
      OpTuple{at::max, BinaryOpType::Max, "max"},
      OpTuple{at::min, BinaryOpType::Min, "min"},
      OpTuple{at::mul, BinaryOpType::Mul, "mul"},
      OpTuple{at::pow, BinaryOpType::Pow, "pow"},
      // NOTE: Remainder does not match the Aten impl exactly
      // despite using an identical function.
      OpTuple{at::remainder, BinaryOpType::Remainder, "remainder"},
  };

  std::for_each(math_ops.begin(), math_ops.end(), [](OpTuple& op) {
    test_op(
        /*blocks*/ 640,
        /*threads*/ 64,
        /*name*/ std::get<2>(op),
        /*Aten Func   */
        [&op](std::array<IValue, 2>& vals) {
          return std::get<0>(op)(vals[0].toTensor(), vals[1].toTensor());
        },
        /*JIT  Func   */
        [&op](Val* in1, Val* in2) -> Val* {
          return binaryOp(std::get<1>(op), in1, in2);
        },
        /*Output      */ std::make_pair(ValType::TensorView, DataType::Float),
        /*Inputs Tuple*/
        std::make_tuple(
            std::make_pair(ValType::TensorView, DataType::Float),
            std::make_pair(ValType::TensorView, DataType::Float)));
  });

  test_op(
      /*blocks*/ 640,
      /*threads*/ 64,
      /*name*/ "add_alpha",
      /*Aten Func   */
      [](std::array<IValue, 3>& vals) {
        return at::add(
            vals[0].toTensor(), vals[1].toTensor(), vals[2].toScalar());
      },
      /*JIT  Func   */ add_alpha,
      /*Output      */ std::make_pair(ValType::TensorView, DataType::Float),
      /*Inputs Tuple*/
      std::make_tuple(
          std::make_pair(ValType::TensorView, DataType::Float),
          std::make_pair(ValType::TensorView, DataType::Float),
          std::make_pair(ValType::Scalar, DataType::Float)));
  test_op(
      /*blocks*/ 640,
      /*threads*/ 64,
      /*name*/ "sub_alpha",
      /*Aten Func   */
      [](std::array<IValue, 3>& vals) {
        return at::sub(
            vals[0].toTensor(), vals[1].toTensor(), vals[2].toScalar());
      },
      /*JIT  Func   */ sub_alpha,
      /*Output      */ std::make_pair(ValType::TensorView, DataType::Float),
      /*Inputs Tuple*/
      std::make_tuple(
          std::make_pair(ValType::TensorView, DataType::Float),
          std::make_pair(ValType::TensorView, DataType::Float),
          std::make_pair(ValType::Scalar, DataType::Float)));
}

void testGPU_FusionTernaryOps() {
  test_op(
      /*blocks*/ 640,
      /*threads*/ 64,
      /*name*/ "clamp",
      /*Aten Func   */
      [](std::array<IValue, 1>& vals) {
        return at::clamp(vals[0].toTensor(), 0.f, 1.f);
      },
      /*JIT  Func   */
      [](Val* in1) -> Val* {
        return clamp(in1, new Float(0.f), new Float(1.f));
      },
      /*Output      */ std::make_pair(ValType::TensorView, DataType::Float),
      /*Inputs Tuple*/
      std::make_tuple(std::make_pair(ValType::TensorView, DataType::Float)));
  test_op(
      /*blocks*/ 640,
      /*threads*/ 64,
      /*name*/ "threshold",
      /*Aten Func   */
      [](std::array<IValue, 1>& vals) {
        return at::threshold(vals[0].toTensor(), 0.f, 1.f);
      },
      /*JIT  Func   */
      [](Val* in1) -> Val* {
        return threshold(in1, new Float(0.f), new Float(1.f));
      },
      /*Output      */ std::make_pair(ValType::TensorView, DataType::Float),
      /*Inputs Tuple*/
      std::make_tuple(std::make_pair(ValType::TensorView, DataType::Float)));
  test_op(
      /*blocks*/ 640,
      /*threads*/ 64,
      /*name*/ "where",
      /*Aten Func   */
      [](std::array<IValue, 3>& vals) {
        return at::where(
            vals[0].toTensor(), vals[1].toTensor(), vals[2].toTensor());
      },
      /*JIT  Func   */ where,
      /*Output      */ std::make_pair(ValType::TensorView, DataType::Float),
      /*Inputs Tuple*/
      std::make_tuple(
          std::make_pair(ValType::TensorView, DataType::Bool),
          std::make_pair(ValType::TensorView, DataType::Float),
          std::make_pair(ValType::TensorView, DataType::Float)));
}

void testGPU_FusionCompoundOps() {
  test_op(
      /*blocks*/ 640,
      /*threads*/ 64,
      /*name*/ "lerp",
      /*Aten Func   */
      [](std::array<IValue, 3>& vals) {
        return at::lerp(
            vals[0].toTensor(), vals[1].toTensor(), vals[2].toTensor());
      },
      /*JIT  Func   */ lerp,
      /*Output      */ std::make_pair(ValType::TensorView, DataType::Float),
      /*Inputs Tuple*/
      std::make_tuple(
          std::make_pair(ValType::TensorView, DataType::Float),
          std::make_pair(ValType::TensorView, DataType::Float),
          std::make_pair(ValType::TensorView, DataType::Float)));
  test_op(
      /*blocks*/ 640,
      /*threads*/ 64,
      /*name*/ "addcmul",
      /*Aten Func   */
      [](std::array<IValue, 4>& vals) {
        return at::addcmul(
            vals[0].toTensor(),
            vals[1].toTensor(),
            vals[2].toTensor(),
            vals[3].toScalar());
      },
      /*JIT  Func   */ addcmul,
      /*Output      */ std::make_pair(ValType::TensorView, DataType::Float),
      /*Inputs Tuple*/
      std::make_tuple(
          std::make_pair(ValType::TensorView, DataType::Float),
          std::make_pair(ValType::TensorView, DataType::Float),
          std::make_pair(ValType::TensorView, DataType::Float),
          std::make_pair(ValType::Scalar, DataType::Float)));
}

void testGPU_FusionCastOps() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeDummyTensor(2, DataType::Half);

  Val* intrm1 = castOp(DataType::Float, tv0);
  TensorView* out = static_cast<TensorView*>(castOp(DataType::Half, intrm1));

  fusion.addInput(tv0);
  fusion.addOutput(out);
  tv0->computeAt(out, -1);

  out->axis(0)->parallelize(ParallelType::BIDx);
  out->axis(-1)->parallelize(ParallelType::TIDx);

  torch::jit::fuser::cuda::CudaKernel prog;
  prog.device_ = 0;
  prog.grid(1);
  prog.block(4);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);

  at::Tensor input1 = at::rand({1, 4}, options);
  at::Tensor output = at::empty_like(input1);
  at::Tensor ref_output = at::empty_like(input1);

  std::array<IValue, 1> inputs = {input1};
  const at::ArrayRef<IValue> input_ivalues(inputs);
  std::vector<at::Tensor> outputs{{output}};

  torch::jit::fuser::cuda::compileKernel(fusion, &prog);
  torch::jit::fuser::cuda::runTestKernel(&prog, input_ivalues, outputs);

  ref_output = at::_cast_Half(at::_cast_Float(input1));

  TORCH_CHECK(
      output.equal(ref_output),
      "\nOp Type: -- ",
      "cast FP16->FP32->FP16",
      " -- had a mismatch.\n",
      "IN1 : ",
      input1,
      "\n",
      "JIT: ",
      output,
      "\n",
      "REF: ",
      ref_output,
      "\n");
}

// We want split/merge/reorder all tested both on and off rfactor domains, also
// want compute at into the rfactor domain, and into its consumer
void testGPU_FusionRFactorReplay() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeDummyTensor(2);

  // Register your inputs
  fusion.addInput(tv0);

  // Do math with it, it returns a `Val*` but can be static_casted back to
  // TensorView
  TensorView* tv1 = static_cast<TensorView*>(sum(tv0, {1}));
  // tv1[I0, R1]
  tv1->split(0, 32);
  // tv1[I0o, I0i{32}, R1]
  tv1->split(0, 16);
  // tv1[I0oo, I0oi{16}, I0i{32}, R1]
  tv1->split(-1, 8);
  // tv1[I0oo, I0oi{16}, I0i{32}, R1o, R1i{8}]
  tv1->split(-2, 4);
  // tv1[I0oo, I0oi{16}, I0i{32}, R1oo, R1oi{4}, R1i{8}]

  tv1->reorder({{0, -2}, {2, -1}, {-3, 0}, {-1, 1}});
  // tv1[R1oo, R1i{8}, I0oi{16}, R1oi{4}, I0oo, I0i{32}]

  tv1->merge(0);
  tv1->merge(-2);

  // tv1[R1oo*R1i{8}, I0oi{16}, R1oi{4}, I0oo*I0i{32}]
  TensorDomain* new_domain = TransformRFactor::runReplay(tv1->domain(), {0});
  TensorDomain* new_domain2 = TransformRFactor::runReplay2(tv1->domain(), {0});
  // new_domain[R(R1oo*R1i{8})rf, I0oi{16}, ir1oi{4}rf, I0oo*I0i{32}]
  // new_domain2[                 I0oi{16},           , I0oo*I0i{32}, R1oi{4}]

  // Move rfactor axis to end, keep iter rfactor axis
  auto reordered_new_domain = new_domain->reorder({{0, -1}, {2, 2}});
  // reordered_new_domain[I0oi{16}, I0oo*I0i{32}, ir1oi{4}rf, R(R1oo*R1i{8})rf]

  TensorDomain* casp =
      TransformReplay::replayCasP(new_domain2, reordered_new_domain, 2);
  // new_domain[I0oi{16}, I0oo*I0i{32}, ir1oi{4}rf, R(R1oo*R1i{8})rf]
  //      casp[I0oi{16}, I0oo*I0i{32},  R1oi{4}]

  casp = casp->split(1, 2);
  // casp      [I0oi{16}, (I0oo*I0i{32})o, I(Ioo*I0i)i{2}, ir1oi{4}]
  // new_domain[I0oi{16},  I0oo*I0i{32}  ,                 ir1oi{4}rf,
  // R(R1oo*R1i{8})rf]
  TensorDomain* pasc = TransformReplay::replayPasC(new_domain, casp, 2);
  // pasc      [I0oi{16}, (I0oo*I0i{32})o, I(Ioo*I0i)i{2}, ir1oi{4}rf,
  // R(R1oo*R1i{8})rf]

  TORCH_CHECK(
      new_domain->nDims() - 1 == new_domain2->nDims(),
      casp->nDims() == new_domain2->nDims() + 1,
      pasc->nDims() == new_domain->nDims() + 1,
      "Error in rfactor, number of dimensions is not correct.");

  TORCH_CHECK(
      !casp->sameAs(new_domain2) && !pasc->sameAs(new_domain) &&
          !new_domain->sameAs(new_domain2) &&
          !tv1->domain()->sameAs(new_domain) &&
          !tv1->domain()->sameAs(new_domain2),
      "Error in rfactor, number of dimensions is not correct.");

  auto dom = new_domain->rootDomain()->domain();
  TORCH_CHECK(
      !new_domain->rootDomain()->axis(0)->isReduction() &&
          std::any_of(
              dom.begin(),
              dom.end(),
              [](IterDomain* id) { return id->isReduction(); }) &&
          std::any_of(
              dom.begin(),
              dom.end(),
              [](IterDomain* id) { return id->isRFactorProduct(); }),
      "Error in rFactor, there seems to be something wrong in root domain.");

  auto dom2 = new_domain2->rootDomain()->domain();
  TORCH_CHECK(
      !new_domain2->rootDomain()->axis(0)->isReduction() &&
          std::any_of(
              dom2.begin(),
              dom2.end(),
              [](IterDomain* id) { return id->isReduction(); }),
      "Error in rFactor, there seems to be something wrong in root domain.");
}

// Start off simple, block on the outer dim
// block stride + thread all reduce + unrolling on inner dim
void testGPU_FusionReduction() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeDummyTensor(2);
  fusion.addInput(tv0);

  // tv1[I0, R1] = tv0[I0, I1]
  TensorView* tv1 = static_cast<TensorView*>(
      reductionOp(BinaryOpType::Add, {1}, new Float(0), tv0));
  fusion.addOutput(tv1);

  TORCH_CHECK(fusion.hasReduction(), "Could not detect reduction in fusion.");

  tv1->split(1, 128);
  // tv1[I0, R1o, R1i{128}] = tv0[I0, I1]
  tv1->split(1, 4);
  // tv1[I0, R1oo, R1oi{4}, R1i{128}] = tv0[I0, I1]

  TensorView* tv2 = tv1->rFactor({1});
  // tv2[I0, R1oo, Ir1oi{4}, Ir1i{128}] = tv0[I0, I1]
  // tv1[I0,        R1oi{4},  R1i{128}] = tv2[I0, R1oo, Ir1oi{4}, Ir1i{128}]

  TensorView* tv3 = tv1->rFactor({1});
  // tv2[I0, R1oo, Ir1oi{4}, Ir1i{128}] = tv0[I0, I1]
  // tv3[I0,        R1oi{4}, Ir1i{128}] = tv2[I0, R1oo, Ir1oi{4}, Ir1i{128}]
  // tv1[I0,                  R1i{128}] = tv3[I0,        R1oi{4}, Ir1i{128}]

  // Incrementally, can print in between for debugging
  tv0->computeAt(tv2, 1);
  tv2->computeAt(tv3, 1);
  tv3->computeAt(tv1, 1);

  // Re do it all at once, because why not.
  tv0->computeAt(tv1, 1);

  tv2->axis(2)->parallelize(ParallelType::Unroll);
  tv3->axis(0)->parallelize(ParallelType::BIDx);

  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);

  // for(auto expr : fusion.exprs(true))
  // std::cout<<expr<<std::endl;
  // GPULower lower(&fusion);
  // lower.printKernel(std::cout);

  int numel_x = 65000;
  int numel_y = 1025;

  torch::jit::fuser::cuda::CudaKernel prog;
  prog.device_ = 0;
  prog.grid(numel_x);
  prog.block(128);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::rand({numel_x, numel_y}, options);
  at::Tensor cg_output = at::empty({numel_x}, options);

  torch::jit::fuser::cuda::compileKernel(fusion, &prog);
  torch::jit::fuser::cuda::runTestKernel(&prog, {input}, {cg_output});

  auto aten_output = input.sum({1});
  TORCH_CHECK(aten_output.allclose(cg_output));
}

void testGPU_FusionReduction2() {
  {
    Fusion fusion;
    FusionGuard fg(&fusion);

    // Set up your input tensor views
    TensorView* tv0 = makeDummyTensor(2);
    fusion.addInput(tv0);

    // tv1[I0, R1] = tv0[I0, I1]
    TensorView* tv1 = static_cast<TensorView*>(
        reductionOp(BinaryOpType::Add, {1}, new Float(0), tv0));

    fusion.addOutput(tv1);

    bool bind_bidx = false;
    bool bind_tidx = true;
    bool bind_tidy = true;
    bool bind_unroll = false;

    int numel_x = 1025; // Cannot exceed block dim max size / tidy
    int numel_y = 129;
    int tidx = 16;
    int tidy = 8;
    int unroll_factor = 4;

    int bidx = bind_tidy ? ceilDiv_(numel_x, tidy) : numel_x;

    tv1->split(1, tidx);
    // tv1[I0, R1o, R1i{tidx}] = tv0[I0, I1]

    tv1->split(1, unroll_factor);
    // tv1[I0, R1oo, R1oi{unroll}, R1i{tidx}] = tv0[I0, I1]

    tv1->split(0, tidy);

    TensorView* tv2 = tv1->rFactor({-3});
    // tv2[I0,             >R1oo<, Ir1oi{unroll}, Ir1i{tidx}]
    // tv1[I0o, I0i{tidy},          R1oi{unroll},  R1i{tidx}]

    TensorView* tv3 = tv1->rFactor({-2});
    // tv2[I0,             >R1oo<, Ir1oi{unroll}, Ir1i{tidx}]
    // tv3[I0,                      R1oi{unroll}, Ir1i{tidx}]
    // tv1[I0o, I0i{tidy},                         R1i{tidx}]

    tv0->computeAt(tv1, -2);

    if (bind_unroll)
      tv2->axis(-2)->parallelize(ParallelType::Unroll);
    if (bind_bidx)
      tv1->axis(0)->parallelize(ParallelType::BIDx);
    if (bind_tidy)
      tv1->axis(1)->parallelize(ParallelType::TIDy);

    if (bind_tidx) {
      tv2->axis(-1)->parallelize(ParallelType::TIDx);
      tv3->axis(-1)->parallelize(ParallelType::TIDx);
      tv1->axis(-1)->parallelize(ParallelType::TIDx);
    }

    torch::jit::fuser::cuda::CudaKernel prog;
    prog.device_ = 0;
    prog.grid(bind_bidx ? bidx : 1);
    prog.block(bind_tidx ? tidx : 1, bind_tidy ? tidy : 1);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor input = at::rand({numel_x, numel_y}, options);
    at::Tensor cg_output = at::empty({numel_x}, options);

    torch::jit::fuser::cuda::compileKernel(fusion, &prog);
    torch::jit::fuser::cuda::runTestKernel(&prog, {input}, {cg_output});

    c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));

    auto aten_output = input.sum({1});
    TORCH_CHECK(aten_output.allclose(cg_output));
  }

  {
    // What if Z participates in the reduction with X?
    Fusion fusion;
    FusionGuard fg(&fusion);

    // Set up your input tensor views
    TensorView* tv0 = makeDummyTensor(2);
    fusion.addInput(tv0);

    // tv1[I0, R1] = tv0[I0, I1]
    TensorView* tv1 = static_cast<TensorView*>(
        reductionOp(BinaryOpType::Add, {1}, new Float(0), tv0));

    fusion.addOutput(tv1);

    int numel_x = 1025; // Cannot exceed block dim max size / tidy
    int numel_y = 129;
    int tidx = 16;
    int tidz = 8;

    tv1->split(1, tidz);
    // tv1[I0, R1o, R1i{tidz}] = tv0[I0, I1]

    tv1->split(1, tidx);
    // tv1[I0, R1oo, R1oi{tidx}, R1i{tidz}] = tv0[I0, I1]

    TensorView* tv2 = tv1->rFactor({-3});
    // tv2[I0,  >R1oo<, Ir1oi{tidx}, Ir1i{tidz}]
    // tv1[I0o,          R1oi{tidx},  R1i{tidz}]

    tv0->computeAt(tv1, -3);

    tv1->axis(0)->parallelize(ParallelType::BIDx);
    tv1->axis(-2)->parallelize(ParallelType::TIDx);
    tv1->axis(-1)->parallelize(ParallelType::TIDz);

    tv2->axis(-2)->parallelize(ParallelType::TIDx);
    tv2->axis(-1)->parallelize(ParallelType::TIDz);

    torch::jit::fuser::cuda::CudaKernel prog;
    prog.device_ = 0;
    prog.grid(numel_x);
    prog.block(tidx, 1, tidz);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor input = at::rand({numel_x, numel_y}, options);
    at::Tensor cg_output = at::empty({numel_x}, options);

    torch::jit::fuser::cuda::compileKernel(fusion, &prog);
    torch::jit::fuser::cuda::runTestKernel(&prog, {input}, {cg_output});

    c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));

    auto aten_output = input.sum({1});
    TORCH_CHECK(aten_output.allclose(cg_output));
  }
}

} // namespace jit
} // namespace torch
#endif // #if defined(USE_CUDA)
