#include <torch/csrc/jit/tensorexpr/ir_serializer.h>

using json = nlohmann::json;

namespace torch {
namespace jit {
namespace tensorexpr {

#define JSON_CACHE_GUARD() \
  if (cachedJSON(v)) {     \
    return;                \
  }

json IRSerializer::serialize(ExprHandle expr) {
  expr.node()->accept(this);
  return JSONOf(expr.node());
}

json IRSerializer::serialize(const Expr& expr) {
  expr.accept(this);
  return JSONOf(&expr);
}

json IRSerializer::serialize(const Stmt& stmt) {
  stmt.accept(this);
  return JSONOf(&stmt);
}

json serializeDtype(Dtype dtype) {
  return toString(dtype.scalar_type());
}

json IRSerializer::visitAndSerialize(const Expr* v) {
  if (cachedJSON(v)) {
    return JSONOf(v);
  }
  v->accept(this);
  return JSONOf(v);
}

json IRSerializer::visitAndSerialize(const std::vector<const Expr*>& v) {
  std::vector<json> json_vec;
  for (const auto expr : v) {
    json_vec.push_back(visitAndSerialize(expr));
  }
  return json_vec;
}

json IRSerializer::visitAndSerialize(const Stmt* v) {
  if (cachedJSON(v)) {
    return JSONOf(v);
  }
  v->accept(this);
  return JSONOf(v);
}

json IRSerializer::visitAndSerialize(const std::vector<const Stmt*>& v) {
  std::vector<json> json_vec;
  for (const auto stmt : v) {
    json_vec.push_back(visitAndSerialize(stmt));
  }
  return json_vec;
}

void IRSerializer::visit(LongImm const* v) {
  JSON_CACHE_GUARD();
  putJSON(
      v,
      {
          {"expr_type", "LongImm"},
          {"value", v->value()},
          {"dtype", serializeDtype(v->dtype())},
      });
}

void IRSerializer::visit(ShortImm const* v) {
  JSON_CACHE_GUARD();
  putJSON(
      v,
      {
          {"expr_type", "ShortImm"},
          {"value", v->value()},
          {"dtype", serializeDtype(v->dtype())},
      });
}

void IRSerializer::visit(BoolImm const* v) {
  JSON_CACHE_GUARD();
  putJSON(
      v,
      {
          {"expr_type", "BoolImm"},
          {"value", v->value()},
          {"dtype", serializeDtype(v->dtype())},
      });
}

void IRSerializer::visit(FloatImm const* v) {
  JSON_CACHE_GUARD();
  putJSON(
      v,
      {
          {"expr_type", "FloatImm"},
          {"value", v->value()},
          {"dtype", serializeDtype(v->dtype())},
      });
}

void IRSerializer::visit(DoubleImm const* v) {
  JSON_CACHE_GUARD();
  putJSON(
      v,
      {
          {"expr_type", "DoubleImm"},
          {"value", v->value()},
          {"dtype", serializeDtype(v->dtype())},
      });
}

void IRSerializer::visit(CharImm const* v) {
  JSON_CACHE_GUARD();
  putJSON(
      v,
      {
          {"expr_type", "CharImm"},
          {"value", v->value()},
          {"dtype", serializeDtype(v->dtype())},
      });
}

void IRSerializer::visit(HalfImm const* v) {
  JSON_CACHE_GUARD();
  putJSON(
      v,
      {
          {"expr_type", "HalfImm"},
          {"value", static_cast<float>(v->value())},
          {"dtype", serializeDtype(v->dtype())},
      });
}

void IRSerializer::visit(ByteImm const* v) {
  JSON_CACHE_GUARD();
  putJSON(
      v,
      {
          {"expr_type", "ByteImm"},
          {"value", v->value()},
          {"dtype", serializeDtype(v->dtype())},
      });
}

void IRSerializer::visit(IntImm const* v) {
  JSON_CACHE_GUARD();
  putJSON(
      v,
      {
          {"expr_type", "IntImm"},
          {"value", v->value()},
          {"dtype", serializeDtype(v->dtype())},
      });
}

template <typename Op>
void visitBinaryOp(
    const BinaryOpNode<Op>* v,
    const std::string& op_str,
    IRSerializer* serializer) {
  if (serializer->cachedJSON(v)) {
    return;
  }
  json value = {
      {"lhs", serializer->visitAndSerialize(v->lhs())},
      {"rhs", serializer->visitAndSerialize(v->rhs())},
      {"dtype", serializeDtype(v->dtype())},
      {"expr_type", op_str},
  };
  serializer->putJSON(v, value);
}

void IRSerializer::visit(const Add* v) {
  visitBinaryOp(v, "Add", this);
}

void IRSerializer::visit(const Sub* v) {
  visitBinaryOp(v, "Sub", this);
}

void IRSerializer::visit(const Mul* v) {
  visitBinaryOp(v, "Mul", this);
}

void IRSerializer::visit(const Div* v) {
  visitBinaryOp(v, "Div", this);
}

void IRSerializer::visit(const Mod* v) {
  visitBinaryOp(v, "Mod", this);
}

void IRSerializer::visit(const Max* v) {
  JSON_CACHE_GUARD();
  json value = {{"lhs", visitAndSerialize(v->lhs())},
                {"rhs", visitAndSerialize(v->rhs())},
                {"dtype", serializeDtype(v->dtype())},
                {"expr_type", "Max"},
                {"propagate_nans", v->propagate_nans()}};
  putJSON(v, value);
}

void IRSerializer::visit(const Min* v) {
  JSON_CACHE_GUARD();
  json value = {{"lhs", visitAndSerialize(v->lhs())},
                {"rhs", visitAndSerialize(v->rhs())},
                {"dtype", serializeDtype(v->dtype())},
                {"expr_type", "Min"},
                {"propagate_nans", v->propagate_nans()}};
  putJSON(v, value);
}

void IRSerializer::visit(const And* v) {
  visitBinaryOp(v, "And", this);
}

void IRSerializer::visit(const Or* v) {
  visitBinaryOp(v, "Or", this);
}

void IRSerializer::visit(const Xor* v) {
  visitBinaryOp(v, "Xor", this);
}

void IRSerializer::visit(const Lshift* v) {
  visitBinaryOp(v, "Lshift", this);
}

void IRSerializer::visit(const Rshift* v) {
  visitBinaryOp(v, "Rshift", this);
}

void IRSerializer::visit(const CompareSelect* v) {
  JSON_CACHE_GUARD();
  putJSON(
      v,
      {
          {"expr_type", "CompareSelect"},
          {"dtype", serializeDtype(v->dtype())},
          {"lhs", visitAndSerialize(v->lhs())},
          {"rhs", visitAndSerialize(v->rhs())},
          {"ret_val1", visitAndSerialize(v->ret_val1())},
          {"ret_val2", visitAndSerialize(v->ret_val2())},
          {"compare_select_op", v->compare_select_op()},
      });
}

void IRSerializer::visit(const Cast* v) {
  JSON_CACHE_GUARD();
  putJSON(
      v,
      {{"expr_type", "Cast"},
       {"dtype", serializeDtype(v->dtype())},
       {"src_value", visitAndSerialize(v->src_value())}});
}

void IRSerializer::visit(const Var* v) {
  JSON_CACHE_GUARD();
  putJSON(
      v,
      {{"expr_type", "Var"},
       {"dtype", serializeDtype(v->dtype())},
       {"name", name_manager_.get_unique_name(v)}});
}

void IRSerializer::visit(const Ramp* v) {
  JSON_CACHE_GUARD();
  putJSON(
      v,
      {
          {"expr_type", "Ramp"},
          {"dtype", serializeDtype(v->dtype())},
          {"base", visitAndSerialize(v->base())},
          {"stride", visitAndSerialize(v->stride())},
          {"lanes", v->lanes()},
      });
}

void IRSerializer::visit(const Load* v) {
  JSON_CACHE_GUARD();
  putJSON(
      v,
      {
          {"buf", visitAndSerialize(v->buf())},
          {"indices", visitAndSerialize(v->indices())},
          {"mask", visitAndSerialize(v->mask())},
          {"expr_type", "Load"},
      });
}

void IRSerializer::visit(const Store* v) {
  JSON_CACHE_GUARD();
  putJSON(
      v,
      {
          {"stmt_type", "Store"},
          {"indices", visitAndSerialize(v->indices())},
          {"value", visitAndSerialize(v->value())},
          {"mask", visitAndSerialize(v->mask())},
          {"buf", visitAndSerialize(v->buf())},
      });
}

void IRSerializer::visit(const Block* v) {
  JSON_CACHE_GUARD();
  std::vector<json> stmts;
  for (Stmt* s : *v) {
    s->accept(this);
    stmts.push_back(JSONOf(s));
  }
  putJSON(
      v,
      {
          {"stmt_type", "Block"},
          {"stmts", stmts},
      });
}

void IRSerializer::visit(const For* v) {
  JSON_CACHE_GUARD();
  auto buffer_mapping = v->loop_options().get_buffer_mapping();
  json buffer_mapping_json;
  for (const auto& mapping : buffer_mapping) {
    buffer_mapping_json[mapping.first] = visitAndSerialize(mapping.second);
  }
  json loop_options_json = {
      {"buffer_mapping", buffer_mapping_json},
      {"gpu_block_index", v->loop_options().gpu_block_index()},
      {"gpu_thread_index", v->loop_options().gpu_thread_index()},
  };
  json for_json = {
      {"stmt_type", "For"},
      {"var", visitAndSerialize(v->var())},
      {"start", visitAndSerialize(v->start())},
      {"stop", visitAndSerialize(v->stop())},
      {"loop_options", loop_options_json},
  };
  if (v->body()) {
    for_json["body"] = visitAndSerialize(v->body());
  }
  putJSON(v, for_json);
}

void IRSerializer::visit(const Broadcast* v) {
  JSON_CACHE_GUARD();
  putJSON(
      v,
      {
          {"expr_type", "Broadcast"},
          {"value", visitAndSerialize(v->value())},
          {"lanes", v->lanes()},
          {"dtype", serializeDtype(v->dtype())},
      });
}

void IRSerializer::visit(const IfThenElse* v) {
  JSON_CACHE_GUARD();
  putJSON(
      v,
      {
          {"expr_type", "IfThenElse"},
          {"true_value", visitAndSerialize(v->true_value())},
          {"false_value", visitAndSerialize(v->false_value())},
          {"condition", visitAndSerialize(v->condition())},
          {"dtype", serializeDtype(v->dtype())},
      });
}

void IRSerializer::visit(const BaseCallNode* v) {
  // should have visited the inherited class
  assert(false);
}

void IRSerializer::visit(const Intrinsics* v) {
  putJSON(
      v,
      {{"expr_type", "Intrinsics"},
       {"dtype", serializeDtype(v->dtype())},
       {"func_name", v->func_name()},
       {"params", visitAndSerialize(v->params())}});
}

void IRSerializer::visit(const Allocate* v) {
  JSON_CACHE_GUARD();
  putJSON(
      v,
      {
          {"stmt_type", "Allocate"},
          {"dtype", serializeDtype(v->dtype())},
          {"buffer_var", visitAndSerialize(v->buffer_var())},
          {"dims", visitAndSerialize(v->dims())},
      });
}

void IRSerializer::visit(const Free* v) {
  JSON_CACHE_GUARD();
  putJSON(
      v,
      {
          {"stmt_type", "Free"},
          {"buffer_var", visitAndSerialize(v->buffer_var())},
      });
}

void IRSerializer::visit(const Buf* v) {
  JSON_CACHE_GUARD();
  putJSON(
      v,
      {{"expr_type", "Buf"},
       {"dtype", serializeDtype(v->dtype())},
       {"base_handle", visitAndSerialize(v->base_handle())},
       {"dims", visitAndSerialize(v->dims())}});
}

void IRSerializer::visit(const Cond* v) {
  JSON_CACHE_GUARD();
  putJSON(
      v,
      {{"stmt_type", "Cond"},
       {"condition", visitAndSerialize(v->condition())},
       {"true_stmt", visitAndSerialize(v->true_stmt())},
       {"false_stmt", visitAndSerialize(v->false_stmt())}});
}

// Following terms except only temporarily in IR simplification
// So we do not need to serialize them
void IRSerializer::visit(const Term* v) {
  assert(false);
}

void IRSerializer::visit(const Polynomial* v) {
  assert(false);
}

void IRSerializer::visit(const MaxTerm* v) {
  assert(false);
}

void IRSerializer::visit(const MinTerm* v) {
  assert(false);
}

void IRSerializer::visit(const FunctionCall* v) {
  JSON_CACHE_GUARD();
  std::vector<json> args;
  for (const auto arg : v->tensor()->args()) {
    args.push_back(visitAndSerialize(arg));
  }
  putJSON(
      v,
      {
          {"expr_type", "FunctionCall"},
          {"params", visitAndSerialize(v->params())},
          {"output_index", v->tensor()->output_index()},
          {"dtype", serializeDtype(v->dtype())},
          {"args", args},
          {"function_name", v->tensor()->buf()->name_hint()},
          {"dims", visitAndSerialize(v->tensor()->dims())},
          {"buf", visitAndSerialize(v->tensor()->buf())},
          {"body", visitAndSerialize(v->tensor()->body())},
      });
}

void IRSerializer::visit(const RoundOff* v) {
  JSON_CACHE_GUARD();
  visitBinaryOp(v, "RoundOff", this);
};

void IRSerializer::visit(const ReduceOp* v) {
  assert(false);
  // can't serialize lambda reduction interaction
  // TOOD: reduction interaction should be a Function Node or something else
};

void IRSerializer::visit(const AtomicAdd* v) {
  JSON_CACHE_GUARD();
  putJSON(
      v,
      {
          {"stmt_type", "AtomicAdd"},
          {"indices", visitAndSerialize(v->indices())},
          {"buf", visitAndSerialize(v->buf())},
          {"value", visitAndSerialize(v->value())},
      });
}
void IRSerializer::visit(const SyncThreads* v) {
  JSON_CACHE_GUARD();
  putJSON(v, {{"stmt_type", "SyncThreads"}});
}
void IRSerializer::visit(const Let* v) {
  putJSON(
      v,
      {
          {"stmt_type", "Let"},
          {"dtype", serializeDtype(v->dtype())},
          {"var", visitAndSerialize(v->var())},
          {"val", visitAndSerialize(v->value())},
      });
}

#undef JSON_CACHE_GUARD

json serialize(const Expr* expr) {
  assert(expr);
  IRSerializer s;
  return s.serialize(*expr);
}

json serialize(const Stmt* stmt) {
  assert(stmt);
  IRSerializer s;
  return s.serialize(*stmt);
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
