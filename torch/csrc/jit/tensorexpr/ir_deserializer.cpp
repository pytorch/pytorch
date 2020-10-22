#include <torch/csrc/jit/tensorexpr/ir_deserializer.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>

using json = nlohmann::json;

namespace torch {
namespace jit {
namespace tensorexpr {

const Expr* IRDeserializer::deserializeBinaryOp(
    const json& v,
    IRNodeType type) {
  bool option = false;
  if (v.find("propagate_nans") != v.end()) {
    option = v["propagate_nans"];
  }
  return newBinaryOpOfType(type, dsExpr(v["lhs"]), dsExpr(v["rhs"]), option);
}

const LongImm* IRDeserializer::deserializeLongImm(const json& v) {
  return new LongImm(v["value"]);
}

const ShortImm* IRDeserializer::deserializeShortImm(const json& v) {
  return new ShortImm(v["value"]);
}
const BoolImm* IRDeserializer::deserializeBoolImm(const json& v) {
  return new BoolImm(v["value"]);
}
const FloatImm* IRDeserializer::deserializeFloatImm(const json& v) {
  return new FloatImm(v["value"]);
}
const DoubleImm* IRDeserializer::deserializeDoubleImm(const json& v) {
  return new DoubleImm(v["value"]);
}
const CharImm* IRDeserializer::deserializeCharImm(const json& v) {
  return new CharImm(v["value"]);
  ;
}
const HalfImm* IRDeserializer::deserializeHalfImm(const json& v) {
  float val = v["value"];
  return new HalfImm(at::Half(val));
}
const IntImm* IRDeserializer::deserializeIntImm(const json& v) {
  return new IntImm(v["value"]);
}

const ByteImm* IRDeserializer::deserializeByteImm(const json& v) {
  return new ByteImm(v["value"]);
}

Dtype deserializeDtype(const json& v) {
  std::string dtype = v["dtype"];
  return Dtype(toDtype(dtype));
}

const Cast* IRDeserializer::deserializeCast(const json& v) {
  return new Cast(deserializeDtype(v), dsExpr(v["src_value"]));
}

const Ramp* IRDeserializer::deserializeRamp(const json& v) {
  return new Ramp(dsExpr(v["base"]), dsExpr(v["stride"]), v["lanes"]);
}
const Broadcast* IRDeserializer::deserializeBroadcast(const json& v) {
  return new Broadcast(dsExpr(v["value"]), v["lanes"]);
}
const Var* IRDeserializer::deserializeVar(const json& v) {
  return new Var(v["name"], deserializeDtype(v));
}
const IfThenElse* IRDeserializer::deserializeIfThenElse(const json& v) {
  return new IfThenElse(
      dsExpr(v["condition"]),
      dsExpr(v["true_value"]),
      dsExpr(v["false_value"]));
}

const Load* IRDeserializer::deserializeLoad(const json& v) {
  const Buf* buf = dynamic_cast<const Buf*>(dsExpr(v["buf"]));
  return new Load(buf, dsVecExpr(v["indices"]), dsExpr(v["mask"]));
}

const FunctionCall* IRDeserializer::deserializeFunctionCall(const json& v) {
  const Expr* body = dsExpr(v["body"]);
  std::vector<const Expr*> args = dsVecExpr(v["args"]);
  std::vector<const Expr*> dims = dsVecExpr(v["dims"]);
  std::vector<const Var*> arg_vars;
  for (auto arg : args) {
    auto var = dynamic_cast<const Var*>(arg);
    arg_vars.push_back(var);
  }
  const Buf* buf = dynamic_cast<const Buf*>(dsExpr(v["buf"]));
  const std::string name = v["function_name"];
  Function* func =
      new Function(name, const_cast<Buf*>(buf), dims, arg_vars, body);
  Tensor* tensor = new Tensor(func, v["output_index"]);
  return new FunctionCall(tensor, dsVecExpr(v["params"]));
}

const Buf* IRDeserializer::deserializeBuf(const json& v) {
  const Var* var = dynamic_cast<const Var*>(dsExpr(v["base_handle"]));
  return new Buf(var, dsVecExpr(v["dims"]), deserializeDtype(v));
}

const Intrinsics* IRDeserializer::deserializeIntrinsics(const json& v) {
  return new Intrinsics(
      stringToIntrinsics(v["func_name"]), dsVecExpr(v["params"]));
}

const std::vector<const Expr*> IRDeserializer::dsVecExpr(const json& v) {
  std::vector<const Expr*> vec;
  for (const auto& obj : v.items()) {
    vec.push_back(dsExpr(obj.value()));
  }
  return vec;
}

AtomicAdd* IRDeserializer::deserializeAtomicAdd(const json& v) {
  const Buf* buf = dynamic_cast<const Buf*>(dsExpr(v["buf"]));
  return new AtomicAdd(buf, dsVecExpr(v["indices"]), dsExpr(v["value"]));
}

SyncThreads* IRDeserializer::deserializeSyncThreads(const json& v) {
  return new SyncThreads();
}

Let* IRDeserializer::deserializeLet(const json& v) {
  const Var* var = dynamic_cast<const Var*>(dsExpr(v["var"]));
  return new Let(var, dsExpr(v["val"]));
}

Store* IRDeserializer::deserializeStore(const json& v) {
  const Buf* buf = dynamic_cast<const Buf*>(dsExpr(v["buf"]));
  return new Store(
      buf, dsVecExpr(v["indices"]), dsExpr(v["value"]), dsExpr(v["mask"]));
}

Block* IRDeserializer::deserializeBlock(const json& v) {
  return new Block(dsVecStmt(v["stmts"]));
}

For* IRDeserializer::deserializeFor(const json& v) {
  const Var* var = static_cast<const Var*>(dsExpr(v["var"]));
  const json& loop_options = v["loop_options"];
  const json& buffer_mapping_json = loop_options["buffer_mapping"];
  std::unordered_map<std::string, const Buf*> buffer_mapping;
  for (const auto& obj : buffer_mapping_json.items()) {
    std::string key = obj.key();
    const Buf* buf = dynamic_cast<const Buf*>(dsExpr(obj.value()));
    buffer_mapping[key] = buf;
  }
  LoopOptions lo;
  lo.set_buffer_mapping(buffer_mapping);
  lo.set_gpu_thread_index(loop_options["gpu_thread_index"]);
  lo.set_gpu_block_index(loop_options["gpu_block_index"]);
  return new For(
      var, dsExpr(v["start"]), dsExpr(v["stop"]), dsStmt(v["body"]), lo);
}

Free* IRDeserializer::deserializeFree(const json& v) {
  const Var* var = static_cast<const Var*>(dsExpr(v["buffer_var"]));
  return new Free(var);
}

const CompareSelect* IRDeserializer::deserializeCompareSelect(const json& v) {
  CompareSelectOperation op = v["compare_select_op"];
  return new CompareSelect(
      dsExpr(v["lhs"]),
      dsExpr(v["rhs"]),
      dsExpr(v["ret_val1"]),
      dsExpr(v["ret_val2"]),
      op);
}

Cond* IRDeserializer::deserializeCond(const json& v) {
  return new Cond(
      dsExpr(v["condition"]), dsStmt(v["true_stmt"]), dsStmt(v["false_stmt"]));
}

Allocate* IRDeserializer::deserializeAllocate(const json& v) {
  const Var* buffer_var = dynamic_cast<const Var*>(dsExpr(v["buffer_var"]));
  return new Allocate(buffer_var, deserializeDtype(v), dsVecExpr(v["dims"]));
}

const Expr* IRDeserializer::dsExprImpl(const json& v) {
  std::string expr_type = v["expr_type"];

#define CHECK_BINARY_OP(cond)                           \
  if (expr_type == #cond) {                             \
    return deserializeBinaryOp(v, IRNodeType::k##cond); \
  }

  CHECK_BINARY_OP(Xor)
  CHECK_BINARY_OP(Add)
  CHECK_BINARY_OP(Sub)
  CHECK_BINARY_OP(Mul)
  CHECK_BINARY_OP(Div)
  CHECK_BINARY_OP(Mod)
  CHECK_BINARY_OP(Max)
  CHECK_BINARY_OP(Min)
  CHECK_BINARY_OP(And)
  CHECK_BINARY_OP(Or)
  CHECK_BINARY_OP(Xor)
  CHECK_BINARY_OP(Lshift)
  CHECK_BINARY_OP(Rshift)
  CHECK_BINARY_OP(RoundOff)
#undef CHECK_BINARY_OP

#define CHECK_EXPR_TYPE(expr_name)    \
  if (expr_type == #expr_name) {      \
    return deserialize##expr_name(v); \
  }
  CHECK_EXPR_TYPE(Cast);
  CHECK_EXPR_TYPE(Var);
  CHECK_EXPR_TYPE(Ramp);
  CHECK_EXPR_TYPE(Broadcast);
  CHECK_EXPR_TYPE(IfThenElse);
  CHECK_EXPR_TYPE(Load);
  CHECK_EXPR_TYPE(FunctionCall);
  CHECK_EXPR_TYPE(Buf);
  CHECK_EXPR_TYPE(Intrinsics);
  CHECK_EXPR_TYPE(CompareSelect);

#define IMM_CHECK(Type, Name) CHECK_EXPR_TYPE(Name##Imm);
  AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, IMM_CHECK);
#undef IMM_CHECK

#undef CHECK_EXPR_TYPE

  assert(false);
}

const Expr* IRDeserializer::dsExpr(const json& v) {
  if (cachedKernelObj(v)) {
    return getCachedExpr(v);
  }
  const Expr* expr = dsExprImpl(v);
  putObj(v, expr);
  return expr;
}

std::vector<Stmt*> IRDeserializer::dsVecStmt(const json& v) {
  std::vector<Stmt*> vec;
  for (const auto& obj : v.items()) {
    vec.push_back(dsStmt(obj.value()));
  }
  return vec;
}

Stmt* IRDeserializer::dsStmtImpl(const json& v) {
  std::string stmt_type = v["stmt_type"];

#define CHECK_STMT_TYPE(stmt_name)    \
  if (stmt_type == #stmt_name) {      \
    return deserialize##stmt_name(v); \
  }

  CHECK_STMT_TYPE(AtomicAdd)
  CHECK_STMT_TYPE(SyncThreads)
  CHECK_STMT_TYPE(Let)
  CHECK_STMT_TYPE(Store)
  CHECK_STMT_TYPE(Block)
  CHECK_STMT_TYPE(For)
  CHECK_STMT_TYPE(Free)
  CHECK_STMT_TYPE(Cond)
  CHECK_STMT_TYPE(Allocate)

  assert(false);
}

Stmt* IRDeserializer::dsStmt(const json& v) {
  if (cachedKernelObj(v)) {
    return getCachedStmt(v);
  }
  Stmt* stmt = dsStmtImpl(v);
  putObj(v, stmt);
  return stmt;
}

const Expr* deserializeExpr(const json& v) {
  IRDeserializer d;
  return d.dsExpr(v);
}

const Stmt* deserializeStmt(const json& v) {
  IRDeserializer d;
  return d.dsStmt(v);
}

const KernelScopedObject* deserialize(const json& v) {
  if (v.find("expr_type") != v.end()) {
    return deserializeExpr(v);
  } else {
    return deserializeStmt(v);
  }
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
