#include <stdexcept>
#include "test/cpp/tensorexpr/test_base.h"

#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_deserializer.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_serializer.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>
#include <torch/csrc/jit/testing/file_check.h>

#include <torch/csrc/jit/json.hpp>
#include <sstream>

namespace torch {
namespace jit {
using json = nlohmann::json;
using namespace torch::jit::tensorexpr;

void checkRoundTrip(ExprHandle expr) {
  std::stringstream original;
  original << *expr.node();
  std::stringstream round_trip;
  round_trip << *deserializeExpr(
      torch::jit::tensorexpr::serialize(expr.node()));
  ASSERT_EQ(original.str(), round_trip.str());
}

void checkRoundTrip(Stmt* stmt) {
  std::stringstream original;
  original << *stmt;
  std::stringstream round_trip;
  round_trip << *deserializeStmt(torch::jit::tensorexpr::serialize(stmt));
  ASSERT_EQ(original.str(), round_trip.str());
}

void testIRSerializationBasicValueTest() {
  KernelScope kernel_scope;
  ExprHandle a = IntImm::make(2), b = IntImm::make(3);
  ExprHandle c = Add::make(a, b);

  checkRoundTrip(c);
}

void testIRSerializationLetStoreTest() {
  KernelScope kernel_scope;
  Placeholder a_buf("a", kFloat, {1});
  Placeholder b_buf("b", kFloat, {1});

  ExprHandle load_a = a_buf.load(0);
  VarHandle var = VarHandle("v", kFloat);
  Stmt* let_store = Let::make(var, load_a);
  Stmt* store_b = b_buf.store({0}, var);

  Block* block = Block::make({let_store, store_b});
  checkRoundTrip(block);
}

void testIRSerializationIMMTest() {
  KernelScope kernel_scope;

  ExprHandle body;
#define IMM_SERIALIZE_TEST(Type, Name)                                      \
  body = ExprHandle((Type)2) + (ExprHandle((Type)3) + ExprHandle((Type)4)); \
  checkRoundTrip(body);
  AT_FORALL_SCALAR_TYPES_AND(Half, IMM_SERIALIZE_TEST);

  // bool doesnt support arithmetic
  VarHandle x("x", kBool);
  VarHandle y("y", kBool);
  checkRoundTrip(x == y);
}

void testIRSerializationBinOp() {
  KernelScope kernel_scope;
  VarHandle a("x", kFloat);
  VarHandle b("y", kFloat);

  checkRoundTrip(Mul::make(a.node(), b.node()));
  checkRoundTrip(Sub::make(a.node(), b.node()));
  checkRoundTrip(Mul::make(a.node(), b.node()));
  checkRoundTrip(Div::make(a.node(), b.node()));
  checkRoundTrip(Mod::make(a.node(), b.node()));
  checkRoundTrip(Max::make(a.node(), b.node(), true));
  checkRoundTrip(Min::make(a.node(), b.node(), true));
  checkRoundTrip(And::make(a.node(), b.node()));
  checkRoundTrip(Or::make(a.node(), b.node()));
  checkRoundTrip(Xor::make(a.node(), b.node()));
  checkRoundTrip(Lshift::make(a.node(), b.node()));
  checkRoundTrip(Rshift::make(a.node(), b.node()));
  checkRoundTrip(RoundOff::make(a.node(), b.node()));
}

void testIRSerializationCompareSelect() {
  KernelScope kernel_scope;
  constexpr int N = 1024;
  Placeholder a(BufHandle("A", {N}, kInt));
  Placeholder b(BufHandle("B", {N}, kInt));

  VarHandle i("i", kInt);
  checkRoundTrip(
      CompareSelect::make(a.load(i), b.load(i), CompareSelectOperation::kEQ));
}

void testIRSerializationCastTest() {
  KernelScope kernel_scope;
  VarHandle x("x", kHalf);
  VarHandle y("y", kFloat);
  ExprHandle body = ExprHandle(2.f) +
      (Cast::make(kFloat, x) * ExprHandle(3.f) + ExprHandle(4.f) * y);

  checkRoundTrip(body);
}

void testIRSerializationRampLoadBroadcast() {
  KernelScope kernel_scope;
  const int kVectorSize = 8;
  const int kVectorCount = 128;
  const int kTotalSize = kVectorSize * kVectorCount;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kFloat));
  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.loadWithMask(
      {Ramp::make(index * kVectorSize, 1, kVectorSize)},
      Broadcast::make(1, kVectorSize));
  checkRoundTrip(load_a);
}

void testIRSerializationForBlock() {
  KernelScope kernel_scope;
  const int N = 16;

  Placeholder a_buf("a", kInt, {N});
  VarHandle index = VarHandle("index", kInt);
  Stmt* body = a_buf.store({index}, 5);
  Stmt* loop = For::make(index, 0, N, body);
  checkRoundTrip(loop);
}

void testIRSerializationIfThenElse() {
  KernelScope kernel_scope;
  ExprHandle v = ifThenElse(ExprHandle(1), ExprHandle(1.0f), ExprHandle(2.0f));
  checkRoundTrip(v);
}

void testIRSerializationInstrinsic() {
  KernelScope kernel_scope;
  VarHandle var = VarHandle("var", kInt);
  checkRoundTrip(Intrinsics::make(IntrinsicsOp::kCeil, var));
}

void testIRSerializationAlloc() {
  KernelScope kernel_scope;

  const int N = 64;
  std::vector<Stmt*> block;
  VarHandle c_var("c", kHandle);
  std::vector<const Expr*> dims;
  dims.push_back(ExprHandle(N).node());
  BufHandle c{new Buf(c_var.node(), dims, kFloat)};
  Allocate* alloc = Allocate::make(c_var, kFloat, {N});
  checkRoundTrip(alloc);
}

void testIRSerializationFree() {
  KernelScope kernel_scope;
  VarHandle c_var("c", kHandle);
  Free* free_stmt = Free::make(c_var);
  checkRoundTrip(free_stmt);
}

void testIRSerializationCond() {
  KernelScope kernel_scope;
  const int N = 16;
  Placeholder a_buf("a", kFloat, {N});
  VarHandle index = VarHandle("index", kInt);
  Stmt* assign_x2 = a_buf.store({index}, cast<float>(index) * 2);
  Stmt* assign_x3 = a_buf.store({index}, cast<float>(index) * 3);
  ExprHandle even_cond = CompareSelect::make(Mod::make(index, 2), 0, kEQ);
  Stmt* assign = Cond::make(even_cond, assign_x2, assign_x3);
  checkRoundTrip(assign);
}

void testIRSerializationFunction() {
  KernelScope kernel_scope;
  int M = 4;
  int N = 20;

  Tensor* producer = Compute(
      "producer",
      {{M, "m"}, {N, "n"}},
      [&](const ExprHandle& m, const ExprHandle& n) { return m * n; });

  Tensor* chunk_0 = Compute(
      "chunk",
      {{M, "m"}, {N / 2, "n"}},
      [&](const ExprHandle& m, const ExprHandle& n) {
        return producer->call(m, n);
      });

  LoopNest l({producer, chunk_0});
  auto* body = l.root_stmt();
  checkRoundTrip(body);
}

void testIRSerializationAtomicAdd() {
  KernelScope kernel_scope;

  VarHandle index = VarHandle("index", kInt);
  int M = 4;
  int N = 20;

  std::vector<const Expr*> indices = {index.node()};
  Tensor* producer = Compute(
      "producer",
      {{M, "m"}, {N, "n"}},
      [&](const ExprHandle& m, const ExprHandle& n) { return m * n; });
  checkRoundTrip(
      new AtomicAdd(producer->buf(), indices, IntImm::make(2).node()));
}

void testIRSerializationLet() {
  KernelScope scope;
  checkRoundTrip(new SyncThreads());
}

} // namespace jit
} // namespace torch
