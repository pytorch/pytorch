#include <torch/csrc/jit/tensorexpr/ir_deserializer.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>

using json = nlohmann::json;

namespace torch {
namespace jit {
namespace tensorexpr {

const Expr* IRDeserializer::deserializeBinaryOp(
    const json& json,
    IRNodeType type) {
  bool option = false;
  if (json.find("propagate_nans") != json.end()) {
    option = json["propagate_nans"];
  }
  const Expr* binop =
      newBinaryOpOfType(type, dsExpr(json["lhs"]), dsExpr(json["rhs"]), option);
  return binop;
}

const LongImm* IRDeserializer::deserializeLongImm(const json& json) {
  const LongImm* longimm = new LongImm(json["value"]);
  return longimm;
}

const ShortImm* IRDeserializer::deserializeShortImm(const json& json) {
  const ShortImm* longimm = new ShortImm(json["value"]);
  return longimm;
}
const BoolImm* IRDeserializer::deserializeBoolImm(const json& json) {
  const BoolImm* boolimm = new BoolImm(json["value"]);
  return boolimm;
}
const FloatImm* IRDeserializer::deserializeFloatImm(const json& json) {
  const FloatImm* floatimm = new FloatImm(json["value"]);
  return floatimm;
}
const DoubleImm* IRDeserializer::deserializeDoubleImm(const json& json) {
  const DoubleImm* doublimm = new DoubleImm(json["value"]);
  return doublimm;
}
const CharImm* IRDeserializer::deserializeCharImm(const json& json) {
  const CharImm* charimm = new CharImm(json["value"]);
  return charimm;
}
const HalfImm* IRDeserializer::deserializeHalfImm(const json& json) {
  float val = json["value"];
  const HalfImm* charimm = new HalfImm(at::Half(val));
  return charimm;
}
const IntImm* IRDeserializer::deserializeIntImm(const json& json) {
  const IntImm* intimm = new IntImm(json["value"]);
  return intimm;
}

const ByteImm* IRDeserializer::deserializeByteImm(const json& json) {
  const ByteImm* byteimm = new ByteImm(json["value"]);
  return byteimm;
}

Dtype getDtype(const json& v) {
  std::string dtype = v["dtype"];
  return Dtype(toDtype(dtype));
}

const Cast* IRDeserializer::deserializeCast(const json& v) {
  auto cast = new Cast(getDtype(v), dsExpr(v["src_value"]));
  return cast;
}

const Ramp* IRDeserializer::deserializeRamp(const json& v) {
  auto ramp = new Ramp(dsExpr(v["base"]), dsExpr(v["stride"]), v["lanes"]);
  return ramp;
}
const Broadcast* IRDeserializer::deserializeBroadcast(const json& v) {
  auto broadcast = new Broadcast(dsExpr(v["value"]), v["lanes"]);
  return broadcast;
}
const Var* IRDeserializer::deserializeVar(const json& v) {
  auto var = new Var(v["name"], getDtype(v));
  if (v["name"] == "producer") {
    std::cout << "here we go\n";
  }
  return var;
}
const IfThenElse* IRDeserializer::deserializeIfThenElse(const json& v) {
  auto ifthenelse = new IfThenElse(
      dsExpr(v["condition"]),
      dsExpr(v["true_value"]),
      dsExpr(v["false_value"]));
  return ifthenelse;
}
const MaxTerm* IRDeserializer::deserializeMaxTerm(const json& v) {
  auto maxterm = new MaxTerm(
      h_, dsExpr(v["scalar"]), v["propagate_nans"], dsVecExpr(v["variables"]));
  return maxterm;
}

const MinTerm* IRDeserializer::deserializeMinTerm(const json& v) {
  auto minterm = new MinTerm(
      h_, dsExpr(v["scalar"]), v["propagate_nans"], dsVecExpr(v["variables"]));
  return minterm;
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
  const std::string name = v["function_name"];
  // Todo: can Function or Tensor appear multiple times, should I be caching ?
  // Dependency/ownership of Function/Tensor confusing, leave as is for initial
  // implementation No visitor for Function or Tensor is defined
  Function* func = new Function(name, dims, arg_vars, body);
  Tensor* tensor = new Tensor(func, v["output_index"]);
  FunctionCall* func_call = new FunctionCall(tensor, dsVecExpr(v["params"]));
  return func_call;
}

const Buf* IRDeserializer::deserializeBuf(const json& v) {
  // TODO: need to add caching to the direct invocations
  const json& j = v["base_handle"];
  if (j["name"] == "producer") {
    std::cout << " hi elias \n";
  }

  const Var* var = dynamic_cast<const Var*>(dsExpr(v["base_handle"]));
  const std::vector<const Expr*> dims = dsVecExpr(v["dims"]);
  Dtype dtype = getDtype(v);
  auto buf = new Buf(var, dims, dtype);
  return buf;
}

const Intrinsics* IRDeserializer::deserializeIntrinsics(const json& v) {
  const Intrinsics* intrinsics = new Intrinsics(
      stringToIntrinsics(v["func_name"]), dsVecExpr(v["params"]));
  return intrinsics;
}

const std::vector<const Expr*> IRDeserializer::dsVecExpr(const json& v) {
  std::vector<const Expr*> vec;
  for (const auto& obj : v.items()) {
    vec.push_back(dsExpr(obj.value()));
  }
  return vec;
}

AtomicAdd* IRDeserializer::deserializeAtomicAdd(const json& json) {
  const Buf* buf = dynamic_cast<const Buf*>(dsExpr(json["buf"]));
  AtomicAdd* atomicadd =
      new AtomicAdd(buf, dsVecExpr(json["indices"]), dsExpr(json["value"]));
  return atomicadd;
}

SyncThreads* IRDeserializer::deserializeSyncThreads(const json& json) {
  SyncThreads* syncthreads = new SyncThreads();
  return syncthreads;
}

Let* IRDeserializer::deserializeLet(const json& json) {
  const Var* var = dynamic_cast<const Var*>(dsExpr(json["var"]));
  Let* let = new Let(var, dsExpr(json["value"]));
  return let;
}

Store* IRDeserializer::deserializeStore(const json& json) {
  const Buf* buf = dynamic_cast<const Buf*>(dsExpr(json["buf"]));
  Store* store = new Store(
      buf,
      dsVecExpr(json["indices"]),
      dsExpr(json["value"]),
      dsExpr(json["mask"]));
  return store;
}

Block* IRDeserializer::deserializeBlock(const json& json) {
  Block* block = new Block(dsVecStmt(json["stmts"]));
  return block;
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
  For* for_var = new For(
      var, dsExpr(v["start"]), dsExpr(v["stop"]), dsStmt(v["body"]), lo);
  return for_var;
}

Free* IRDeserializer::deserializeFree(const json& json) {
  const Var* var = static_cast<const Var*>(dsExpr(json["buffer_var"]));
  Free* free = new Free(var);
  return free;
}

Cond* IRDeserializer::deserializeCond(const json& json) {
  Cond* cond = new Cond(
      dsExpr(json["condition"]),
      dsStmt(json["true_stmt"]),
      dsStmt(json["false_stmt"]));
  return cond;
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
  CHECK_BINARY_OP(CompareSelect)
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
  // CHECK_EXPR_TYPE(Term);
  // CHECK_EXPR_TYPE(Polynomial);
  CHECK_EXPR_TYPE(MaxTerm);
  CHECK_EXPR_TYPE(MinTerm);
  CHECK_EXPR_TYPE(FunctionCall);
  CHECK_EXPR_TYPE(Buf);
  CHECK_EXPR_TYPE(Intrinsics);

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

Stmt* IRDeserializer::dsStmt(const json& v) {
  if (v["json_id"] == 8) {
    std::cout << "Hello\n";
  }

  // std::cout << " DS STMT:: \n";
  // std::cout << v;
  // std::cout << "\n";

  if (cachedKernelObj(v)) {
    // TODO: separate map for stmts
    return const_cast<Stmt*>(getCachedStmt(v));
  }

  // std::cout << " DS STMT:: \n";
  // std::cout << v;
  // std::cout << "\n";

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

  assert(false);
}

const Expr* deserializeExpr(const json& json) {
  IRDeserializer d;
  return d.dsExpr(json);
}

const Stmt* deserializeStmt(const json& json) {
  IRDeserializer d;
  return d.dsStmt(json);
}

const KernelScopedObject* deserialize(const json& json) {
  if (json.find("expr_type") != json.end()) {
    return deserializeExpr(json);
  } else {
    return deserializeStmt(json);
  }
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
