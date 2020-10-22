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
  bool cachedKernelObj(const json& v) {
    size_t id = v["json_id"];
    return JSONToObj_.count(id) != 0;
  }

  const Expr* getCachedExpr(const json& v) {
    auto it = JSONToObj_.find(v["json_id"]);
    if (it != JSONToObj_.end()) {
      return it->second;
    }
    assert(false);
  }

  const Expr* dsExprImpl(const json& v);
  const Expr* dsExpr(const json& v);
  const std::vector<const Expr*> dsVecExpr(const json& v);

  const Cast* deserializeCast(const json& v);
  const Var* deserializeVar(const json& v);
  const Ramp* deserializeRamp(const json& v);
  const Broadcast* deserializeBroadcast(const json& v);
  const IfThenElse* deserializeIfThenElse(const json& v);
  const FunctionCall* deserializeFunctionCall(const json& v);
  const Buf* deserializeBuf(const json& v);
  const Intrinsics* deserializeIntrinsics(const json& v);
  const Load* deserializeLoad(const json& v);
  const CompareSelect* deserializeCompareSelect(const json& v);

  AtomicAdd* deserializeAtomicAdd(const json& v);
  SyncThreads* deserializeSyncThreads(const json& v);
  Let* deserializeLet(const json& v);
  Allocate* deserializeAllocate(const json& v);

  Store* deserializeStore(const json& v);
  Block* deserializeBlock(const json& v);
  For* deserializeFor(const json& v);
  Free* deserializeFree(const json& v);
  Cond* deserializeCond(const json& v);

  Stmt* dsStmt(const json& v);
  std::vector<Stmt*> dsVecStmt(const json& v);

  const LongImm* deserializeLongImm(const json& v);
  const ShortImm* deserializeShortImm(const json& v);
  const BoolImm* deserializeBoolImm(const json& v);
  const FloatImm* deserializeFloatImm(const json& v);
  const DoubleImm* deserializeDoubleImm(const json& v);
  const CharImm* deserializeCharImm(const json& v);
  const HalfImm* deserializeHalfImm(const json& v);
  const IntImm* deserializeIntImm(const json& v);
  const ByteImm* deserializeByteImm(const json& v);

  const Expr* deserializeBinaryOp(const json& v, IRNodeType type);

  void putObj(json h, const Expr* e) {
    size_t id = h["json_id"];
    auto res = JSONToObj_.emplace(id, e);
    if (res.second == false) {
      // This is always a logic bug since we should check the cache first.
      throw std::runtime_error("hash collision");
    }
  }

 private:
  std::unordered_map<size_t, const Expr*> JSONToObj_;
};

TORCH_API const Stmt* deserializeStmt(const json& v);
TORCH_API const Expr* deserializeExpr(const json& v);
TORCH_API const KernelScopedObject* deserialize(const json& v);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
