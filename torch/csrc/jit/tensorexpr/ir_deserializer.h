#pragma once

#include <torch/csrc/jit/json.hpp>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

namespace torch {
namespace jit {
namespace tensorexpr {

using json = nlohmann::json;

#define JSON_CACHE_GUARD()  \
  if (cachedKernelObj(v)) { \
    return;                 \
  }

/* Expression hasher providing comparable values representing sub-exprs.
 * Uses memoization to avoid excessive recursion. */
class TORCH_API IRDeserializer : public IRVisitor {
 public:
  bool cachedKernelObj(const json& json) {
    size_t id = json["json_id"];
    return JSONToObj_.count(id) != 0;
  }

  const Expr* getCachedExpr(const json& json) {
    if (json.find("name") != json.end()) {
      std::cout << " GETTING CACHED : " << json["name"];
    }

    auto it = JSONToObj_.find(json["json_id"]);
    if (it != JSONToObj_.end()) {
      const Expr* obj = dynamic_cast<const Expr*>(it->second);
      assert(obj);
      return obj;
    }
    assert(false);
  }

  const Stmt* getCachedStmt(const json& json) {
    auto it = JSONToObj_.find(json["json_id"]);
    if (it != JSONToObj_.end()) {
      const Stmt* obj = dynamic_cast<const Stmt*>(it->second);
      assert(obj);
      return obj;
    }
    assert(false);
  }

  const Expr* dsExprImpl(const json& json);
  const Expr* dsExpr(const json& json);
  const std::vector<const Expr*> dsVecExpr(const json& json);

  const Cast* deserializeCast(const json& json);
  const Var* deserializeVar(const json& json);
  const Ramp* deserializeRamp(const json& json);
  const Broadcast* deserializeBroadcast(const json& json);
  const IfThenElse* deserializeIfThenElse(const json& json);
  // const Term * deserializeTerm(const json& json);
  // const Polynomial * deserializePolynomial(const json& json);
  const MaxTerm* deserializeMaxTerm(const json& json);
  const MinTerm* deserializeMinTerm(const json& json);
  const FunctionCall* deserializeFunctionCall(const json& json);
  const Buf* deserializeBuf(const json& json);
  const Intrinsics* deserializeIntrinsics(const json& json);

  AtomicAdd* deserializeAtomicAdd(const json& json);
  SyncThreads* deserializeSyncThreads(const json& json);
  Let* deserializeLet(const json& json);
  Store* deserializeStore(const json& json);
  Block* deserializeBlock(const json& json);
  For* deserializeFor(const json& json);
  Free* deserializeFree(const json& json);
  Cond* deserializeCond(const json& json);

  Stmt* dsStmt(const json& v);
  std::vector<Stmt*> dsVecStmt(const json& v);

  const LongImm* deserializeLongImm(const json& json);
  const ShortImm* deserializeShortImm(const json& json);
  const BoolImm* deserializeBoolImm(const json& json);
  const FloatImm* deserializeFloatImm(const json& json);
  const DoubleImm* deserializeDoubleImm(const json& json);
  const CharImm* deserializeCharImm(const json& json);
  const HalfImm* deserializeHalfImm(const json& json);
  const IntImm* deserializeIntImm(const json& json);
  const ByteImm* deserializeByteImm(const json& json);

  // void visit(const Load* v) override;
  // void visit(const Store* v) override;
  // void visit(const Block* v) override;
  // void visit(const For* v) override;
  // void visit(const Allocate* v) override;
  // void visit(const Free* v) override;
  // void visit(const Cond* v) override;
  // void visit(const Term* v) override;
  // void visit(const Polynomial* v) override;
  // void visit(const MaxTerm* v) override;
  // void visit(const MinTerm* v) override;

  // void visit(const FunctionCall* v) override;
  // void visit(const ReduceOp* v) override;

  // void visit(const AtomicAdd* v) override;
  // void visit(const SyncThreads* v) override;
  // void visit(const Let* v) override;

  // void visit(const Buf* v) override;
  // const Add * deserializeAdd(const json& json);
  // const Sub * deserializeSub(const json& json);
  // const Mul * deserializeMul(const json& json);
  // const Div * deserializeDiv(const json& json);
  // const Mod * deserializeMod(const json& json);
  // const Max * deserializeMax(const json& json);
  // const And * deserializeAnd(const json& json);
  // const Or * deserializeOr(const json& json);
  // const Xor * deserializeXor(const json& json);
  // const Lshift * deserializeLshift(const json& json);
  // const Rshift * deserializeRshift(const json& json);
  // const CompareSelect * deserializeCompareSelect(const json& json);

  // const Expr * deserializeSub(const json& json);
  // const Expr * deserializeMul(const json& json);
  // const Expr * deserializeDiv(const json& json);
  // const Expr * deserializeMod(const json& json);
  // const Expr * deserializeMax(const json& json);
  // const Expr * deserializeAnd(const json& json);
  // const Expr * deserializeOr(const json& json);
  // const Expr * deserializeXor(const json& json);
  // const Expr * deserializeLshift(const json& json);
  // const Expr * deserializeRshift(const json& json);
  // const Expr * deserializeCompareSelect(const json& json);

  const Expr* deserializeBinaryOp(const json& json, IRNodeType type);

  //   void visit(const Add* v) override;
  //   void visit(const Sub* v) override;
  //   void visit(const Mul* v) override;
  //   void visit(const Div* v) override;
  //   void visit(const Mod* v) override;
  //   void visit(const Max* v) override;
  //   void visit(const Min* v) override;
  //   void visit(const And* v) override;
  //   void visit(const Or* v) override;
  //   void visit(const Xor* v) override;
  //   void visit(const Lshift* v) override;
  //   void visit(const Rshift* v) override;
  //   void visit(const CompareSelect* v) override;

  // #define IMM_SERIALIZE_VISIT(Type, Name) void visit(const Name##Imm* v)
  // override;
  //   AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, IMM_SERIALIZE_VISIT);
  // #undef IMM_SERIALIZE_VISIT

  //   void visit(const Cast* v) override;
  //   void visit(const Var* v) override;
  //   void visit(const Ramp* v) override;
  //   void visit(const Load* v) override;
  //   void visit(const Store* v) override;
  //   void visit(const Block* v) override;
  //   void visit(const For* v) override;
  //   void visit(const Broadcast* v) override;
  //   void visit(const IfThenElse* v) override;
  //   void visit(const BaseCallNode* v) override;
  //   void visit(const Intrinsics* v) override;
  //   void visit(const Allocate* v) override;
  //   void visit(const Free* v) override;
  //   void visit(const Cond* v) override;
  //   void visit(const Term* v) override;
  //   void visit(const Polynomial* v) override;
  //   void visit(const MaxTerm* v) override;
  //   void visit(const MinTerm* v) override;

  //   void visit(const FunctionCall* v) override;
  //   void visit(const RoundOff* v) override;
  //   void visit(const ReduceOp* v) override;

  //   void visit(const AtomicAdd* v) override;
  //   void visit(const SyncThreads* v) override;
  //   void visit(const Let* v) override;

  //   void visit(const Buf* v) override;

  //   // BaseCallNode is the base class for all call nodes.
  //   // For any visitors that only needs the common behavior, only override
  //   this
  //   // function is enough. This is because all derived class handlers will
  //   call
  //   // this function by default.
  //   // Override the derived class handler only if the logic is more specific
  //   to
  //   // that.

  //   // json JSONOf(const Expr* e) {
  //   //   auto it = exprToJSON_.find(e);
  //   //   if (it != exprToJSON_.end()) {
  //   //     return it->second;
  //   //   }
  //   //   assert(false);
  //   // }

  //   // json JSONOf(const Stmt* s) {
  //   //   auto it = exprToJSON_.find(s);
  //   //   if (it != exprToJSON_.end()) {
  //   //     return it->second;
  //   //   }
  //   //   assert(false);
  //   // }

  void putObj(json h, const KernelScopedObject* e) {
    size_t id = h["json_id"];
    auto res = JSONToObj_.emplace(id, e);
    if (res.second == false) {
      // This is always a logic bug since we should check the cache first.
      throw std::runtime_error("hash collision");
    }
  }

 private:
  std::unordered_map<size_t, const KernelScopedObject*> JSONToObj_;
  HashProvider h_;
};

TORCH_API const Stmt* deserializeStmt(const json& json);
TORCH_API const Expr* deserializeExpr(const json& json);
TORCH_API const KernelScopedObject* deserialize(const json& json);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
