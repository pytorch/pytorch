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

class TORCH_API IRSerializer : public IRVisitor {
 public:
  bool cachedJSON(const KernelScopedObject* e) {
    return exprToJSON_.find(e) != exprToJSON_.end();
  }

  json serialize(ExprHandle);
  json serialize(const Expr&);
  json serialize(const Stmt&);

  json visitAndSerialize(const Expr* v);
  json visitAndSerialize(const std::vector<const Expr*>& v);
  json visitAndSerialize(const Stmt* v);
  json visitAndSerialize(const std::vector<const Stmt*>& v);

  void visit(const Add* v) override;
  void visit(const Sub* v) override;
  void visit(const Mul* v) override;
  void visit(const Div* v) override;
  void visit(const Mod* v) override;
  void visit(const Max* v) override;
  void visit(const Min* v) override;
  void visit(const And* v) override;
  void visit(const Or* v) override;
  void visit(const Xor* v) override;
  void visit(const Lshift* v) override;
  void visit(const Rshift* v) override;
  void visit(const CompareSelect* v) override;
  void visit(const RoundOff* v) override;

#define IMM_SERIALIZE_VISIT(Type, Name) void visit(const Name##Imm* v) override;
  AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, IMM_SERIALIZE_VISIT);
#undef IMM_SERIALIZE_VISIT

  void visit(const Cast* v) override;
  void visit(const Var* v) override;
  void visit(const Ramp* v) override;
  void visit(const Load* v) override;
  void visit(const Store* v) override;
  void visit(const Block* v) override;
  void visit(const For* v) override;
  void visit(const Broadcast* v) override;
  void visit(const IfThenElse* v) override;
  void visit(const BaseCallNode* v) override;
  void visit(const Intrinsics* v) override;
  void visit(const Allocate* v) override;
  void visit(const Free* v) override;
  void visit(const Cond* v) override;
  void visit(const Term* v) override;
  void visit(const Polynomial* v) override;
  void visit(const MaxTerm* v) override;
  void visit(const MinTerm* v) override;

  void visit(const FunctionCall* v) override;
  void visit(const ReduceOp* v) override;

  void visit(const AtomicAdd* v) override;
  void visit(const SyncThreads* v) override;
  void visit(const Let* v) override;

  void visit(const Buf* v) override;

  json JSONOf(const KernelScopedObject* s) {
    // TODO: we could return just the json id here, since when we deserialize
    // we cache on ID. this would require always visiting fields in exactly the
    // same order in deserializer as in the serializer, which is a little
    // brittle
    auto it = exprToJSON_.find(s);
    if (it != exprToJSON_.end()) {
      return it->second;
    }
    throw std::runtime_error("no cached json");
  }

  void putJSON(const KernelScopedObject* e, json h) {
    // JSON has no built-in way to make an object serialized twice resolve to
    // the same reference, so we cache each object and assign a unique ID
    // that on deserialization will resolve to one object

    h["json_id"] = json_id_++;
    auto res = exprToJSON_.emplace(e, h);
    if (res.second == false) {
      // This is always a logic bug since we should check the cache first.
      throw std::runtime_error("hash collision");
    }
  }

 private:
  size_t json_id_ = 0;
  std::unordered_map<const KernelScopedObject*, json> exprToJSON_;
  UniqueNameManager name_manager_;
};

TORCH_API json serialize(const Expr* expr);
TORCH_API json serialize(const Stmt* expr);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
