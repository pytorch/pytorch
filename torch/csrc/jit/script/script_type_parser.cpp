#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/script/script_type_parser.h>

namespace torch {
namespace jit {
namespace script {

namespace {
const std::unordered_map<std::string, TypePtr>& ident_to_type_lut() {
  static std::unordered_map<std::string, TypePtr> map = {
      {"Tensor", TensorType::get()},
      {"int", IntType::get()},
      {"float", FloatType::get()},
      {"bool", BoolType::get()},
      {"str", StringType::get()},
      {"Device", DeviceObjType::get()},
      // technically this is not a python type but we need it when
      // parsing serialized methods that use implicit converions to Scalar
      {"number", NumberType::get()},
      {"None", NoneType::get()},
  };
  return map;
}

bool isTorch(const Expr& expr) {
  return expr.kind() == TK_VAR && Var(expr).name().name() == "torch";
}

std::string collectQualname(const Select& select) {
  Expr base = select.value();
  if (base.kind() == TK_VAR) {
    return Var(base).name().name() + "." + select.selector().name();
  }
  std::string basename = collectQualname(Select(base));
  return basename + "." + select.selector().name();
}
} // namespace

TypePtr ScriptTypeParser::subscriptToType(
    const std::string& typeName,
    const Subscript& subscript) const {
  if (typeName == "Tuple") {
    std::vector<TypePtr> subscript_expr_types;
    for (auto expr : subscript.subscript_exprs()) {
      subscript_expr_types.push_back(parseTypeFromExpr(expr));
    }
    return TupleType::create(subscript_expr_types);
  } else if (typeName == "List") {
    if (subscript.subscript_exprs().size() != 1) {
      throw ErrorReport(subscript)
          << " expected exactly one element type but found "
          << subscript.subscript_exprs().size();
    }
    auto elem_type = parseTypeFromExpr(*subscript.subscript_exprs().begin());
    return ListType::create(elem_type);

  } else if (typeName == "Optional") {
    if (subscript.subscript_exprs().size() != 1) {
      throw ErrorReport(subscript)
          << " expected exactly one element type but found "
          << subscript.subscript_exprs().size();
    }
    auto elem_type = parseTypeFromExpr(*subscript.subscript_exprs().begin());
    return OptionalType::create(elem_type);

  } else if (typeName == "Future") {
    if (subscript.subscript_exprs().size() != 1) {
      throw ErrorReport(subscript)
          << " expected exactly one element type but found "
          << subscript.subscript_exprs().size();
    }
    auto elem_type = parseTypeFromExpr(*subscript.subscript_exprs().begin());
    return FutureType::create(elem_type);
  } else if (typeName == "Dict") {
    if (subscript.subscript_exprs().size() != 2) {
      throw ErrorReport(subscript)
          << " expected exactly 2 element types but found "
          << subscript.subscript_exprs().size();
    }
    auto key_type = parseTypeFromExpr(subscript.subscript_exprs()[0]);
    auto value_type = parseTypeFromExpr(subscript.subscript_exprs()[1]);
    return DictType::create(key_type, value_type);
  } else {
    throw ErrorReport(subscript.range())
        << "Unknown type constructor " << typeName;
  }
}

c10::optional<std::pair<TypePtr, int32_t>> ScriptTypeParser::parseBroadcastList(
    const Expr& expr) const {
  if (expr.kind() != TK_SUBSCRIPT)
    return c10::nullopt;
  auto subscript = Subscript(expr);
  if (subscript.value().kind() != TK_VAR)
    return c10::nullopt;
  auto var = Var(subscript.value());
  auto subscript_exprs = subscript.subscript_exprs();

  // handle the case where the BroadcastingList is wrapped in a Optional type
  if (var.name().name() == "Optional") {
    auto broadcast_list = parseBroadcastList(subscript_exprs[0]);
    if (broadcast_list) {
      TypePtr opt_type = OptionalType::create(broadcast_list->first);
      return std::pair<TypePtr, int32_t>(opt_type, broadcast_list->second);
    } else {
      return c10::nullopt;
    }
  } else if (var.name().name().find("BroadcastingList") != 0) {
    return c10::nullopt;
  }

  if (subscript_exprs.size() != 1)
    throw ErrorReport(subscript.subscript_exprs().range())
        << "BroadcastingList/Optional[BroadcastingList] "
           "must be subscripted with a type";

  auto typ = subscript_exprs[0];
  auto len = var.name().name().substr(strlen("BroadcastingList"));

  if (typ.kind() != TK_VAR)
    throw ErrorReport(subscript.value().range())
        << "Subscripted type must be a type identifier";

  auto value_name = Var(typ).name().name();
  if (value_name != "float" && value_name != "int")
    throw ErrorReport(subscript.value().range())
        << "Broadcastable lists only supported for int or float";

  auto elem_ptr = ident_to_type_lut().find(value_name);
  AT_ASSERT(elem_ptr != ident_to_type_lut().end());
  TypePtr list_ptr = ListType::create(elem_ptr->second);

  const char* len_c = len.c_str();
  char* end;
  size_t len_v = strtoull(len_c, &end, 10);
  if (end != len_c + len.size()) {
    throw ErrorReport(subscript.subscript_exprs().range())
        << "subscript of Broadcastable list must be a positive integer";
  }
  return std::pair<TypePtr, int32_t>(list_ptr, len_v);
}

// gets the base type name given namespaces where the types live
// turns torch.Tensor -> Tensor, X -> X
c10::optional<std::string> ScriptTypeParser::parseBaseTypeName(
    const Expr& expr) const {
  switch (expr.kind()) {
    case TK_VAR: {
      return Var(expr).name().name();
    }
    case TK_NONE: {
      return "None";
    }
    case '.': {
      auto select = Select(expr);
      const std::string& name = select.selector().name();
      // Special case for torch.Tensor
      if (isTorch(select.value()) && name == "Tensor") {
        return "Tensor";
      } else {
        // Otherwise, it's a fully qualified class name
        return collectQualname(select);
      }
    } break;
  }
  return at::nullopt;
}

TypePtr ScriptTypeParser::parseTypeFromExpr(const Expr& expr) const {
  if (expr.kind() == TK_SUBSCRIPT) {
    auto subscript = Subscript(expr);
    auto value_name = parseBaseTypeName(subscript.value());
    if (!value_name) {
      throw ErrorReport(subscript.value().range())
          << "Subscripted type must be a type identifier";
    }
    return subscriptToType(*value_name, subscript);
  } else if (auto name = parseBaseTypeName(expr)) {
    auto itr = ident_to_type_lut().find(*name);
    if (itr != ident_to_type_lut().end()) {
      return itr->second;
    }
    if (resolver_) {
      if (auto typePtr = resolver_->resolveType(*name)) {
        return typePtr;
      }
    }
    throw ErrorReport(expr) << "Unknown type name " << *name;
  }
  throw ErrorReport(expr.range())
      << "Expression of type " << kindToString(expr.kind())
      << " cannot be used in a type expression";
}

TypePtr ScriptTypeParser::parseType(const std::string& str) {
  Parser p(str);
  return parseTypeFromExpr(p.parseExp());
}
} // namespace script
} // namespace jit
} // namespace torch
