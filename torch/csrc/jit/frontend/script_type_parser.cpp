#include <torch/csrc/jit/frontend/script_type_parser.h>

#include <ATen/core/type_factory.h>
#include <torch/csrc/jit/frontend/parser.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/custom_class.h>

namespace torch::jit {
namespace {

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

const std::unordered_map<std::string, c10::TypePtr>& string_to_type_lut() {
  return c10::DefaultTypeFactory::basePythonTypes();
}

} // namespace

TypePtr ScriptTypeParser::subscriptToType(
    const std::string& typeName,
    const Subscript& subscript) const {
  if (typeName == "Tuple" || typeName == "tuple") {
    if (subscript.subscript_exprs().size() == 1 &&
        subscript.subscript_exprs()[0].kind() == TK_TUPLE_LITERAL) {
      // `typing.Tuple` special cases syntax for empty tuple annotations,
      // i.e. `typing.Tuple[()]`. Allow for parsing an empty tuple literal
      // here. See https://docs.python.org/3/library/typing.html#typing.Tuple
      auto tup_literal = TupleLiteral(subscript.subscript_exprs()[0]);
      if (!tup_literal.inputs().empty()) {
        throw(
            ErrorReport(tup_literal.range())
            << "Tuple literal in Tuple type annotation must not "
            << "have any elements!");
      }
      return TupleType::create({});
    }
    std::vector<TypePtr> subscript_expr_types;
    for (auto expr : subscript.subscript_exprs()) {
      subscript_expr_types.emplace_back(parseTypeFromExprImpl(expr));
    }
    return TupleType::create(subscript_expr_types);
  } else if (typeName == "List" || typeName == "list") {
    if (subscript.subscript_exprs().size() != 1) {
      throw ErrorReport(subscript)
          << " expected exactly one element type but found "
          << subscript.subscript_exprs().size();
    }
    auto elem_type =
        parseTypeFromExprImpl(*subscript.subscript_exprs().begin());
    return ListType::create(elem_type);

  } else if (typeName == "Optional") {
    if (subscript.subscript_exprs().size() != 1) {
      throw ErrorReport(subscript)
          << " expected exactly one element type but found "
          << subscript.subscript_exprs().size();
    }
    auto elem_type =
        parseTypeFromExprImpl(*subscript.subscript_exprs().begin());
    return OptionalType::create(elem_type);

  } else if (typeName == "Union") {
    std::vector<TypePtr> subscript_expr_types;
    subscript_expr_types.reserve(subscript.subscript_exprs().size());
    for (auto expr : subscript.subscript_exprs()) {
      subscript_expr_types.emplace_back(parseTypeFromExprImpl(expr));
    }
    return UnionType::create(subscript_expr_types);
  } else if (typeName == "Future" || typeName == "torch.jit.Future") {
    if (subscript.subscript_exprs().size() != 1) {
      throw ErrorReport(subscript)
          << " expected exactly one element type but found "
          << subscript.subscript_exprs().size();
    }
    auto elem_type =
        parseTypeFromExprImpl(*subscript.subscript_exprs().begin());
    return FutureType::create(elem_type);
  } else if (typeName == "Await" || typeName == "torch.jit._Await") {
    if (subscript.subscript_exprs().size() != 1) {
      throw ErrorReport(subscript)
          << " expected exactly one element type but found "
          << subscript.subscript_exprs().size();
    }
    auto elem_type =
        parseTypeFromExprImpl(*subscript.subscript_exprs().begin());
    return AwaitType::create(elem_type);
  } else if (typeName == "RRef") {
    if (subscript.subscript_exprs().size() != 1) {
      throw ErrorReport(subscript)
          << " expected exactly one element type but found "
          << subscript.subscript_exprs().size();
    }
    auto elem_type =
        parseTypeFromExprImpl(*subscript.subscript_exprs().begin());
    return RRefType::create(elem_type);
  } else if (typeName == "Dict" || typeName == "dict") {
    if (subscript.subscript_exprs().size() != 2) {
      throw ErrorReport(subscript)
          << " expected exactly 2 element types but found "
          << subscript.subscript_exprs().size();
    }
    auto key_type = parseTypeFromExprImpl(subscript.subscript_exprs()[0]);
    auto value_type = parseTypeFromExprImpl(subscript.subscript_exprs()[1]);
    return DictType::create(key_type, value_type);
  } else {
    throw ErrorReport(subscript.range())
        << "Unknown type constructor " << typeName;
  }
}

std::optional<std::pair<TypePtr, int32_t>> ScriptTypeParser::parseBroadcastList(
    const Expr& expr) const {
  // Alias torch.nn._common_types._size_?_t to BroadcastingList?[int]
  if (expr.kind() == TK_VAR) {
    auto var = Var(expr);
    auto& name = var.name().name();
    constexpr auto _size_prefix = "_size_";
    constexpr auto _size_suffix = "_t";
    constexpr auto _size_n_len = 9; // strlen("_size_X_t")
    constexpr auto _size_prefix_len = 6; // strlen("_size_");
    if (name.find(_size_prefix) == 0 && name.length() == _size_n_len &&
        name.find(_size_suffix) == _size_prefix_len + 1 &&
        ::isdigit(name[_size_prefix_len])) {
      int n = name[_size_prefix_len] - '0';
      return std::pair<TypePtr, int32_t>(ListType::create(IntType::get()), n);
    }
  }

  if (expr.kind() != TK_SUBSCRIPT)
    return std::nullopt;
  auto subscript = Subscript(expr);
  if (subscript.value().kind() != TK_VAR)
    return std::nullopt;
  auto var = Var(subscript.value());
  auto subscript_exprs = subscript.subscript_exprs();

  // handle the case where the BroadcastingList is wrapped in a Optional type
  if (var.name().name() == "Optional") {
    auto broadcast_list = parseBroadcastList(subscript_exprs[0]);
    if (broadcast_list) {
      TypePtr opt_type = OptionalType::create(broadcast_list->first);
      return std::pair<TypePtr, int32_t>(opt_type, broadcast_list->second);
    } else {
      return std::nullopt;
    }
  } else if (var.name().name().find("BroadcastingList") != 0) {
    return std::nullopt;
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

  auto elem_ptr = string_to_type_lut().find(value_name);
  AT_ASSERT(elem_ptr != string_to_type_lut().end());
  TypePtr list_ptr = ListType::create(elem_ptr->second);

  const char* len_c = len.c_str();
  char* end = nullptr;
  size_t len_v = strtoull(len_c, &end, 10);
  if (end != len_c + len.size()) {
    throw(
        ErrorReport(subscript.subscript_exprs().range())
        << "subscript of Broadcastable list must be a positive integer");
  }
  return std::pair<TypePtr, int32_t>(list_ptr, len_v);
}

// gets the base type name given namespaces where the types live
// turns torch.Tensor -> Tensor, X -> X
std::optional<std::string> ScriptTypeParser::parseBaseTypeName(
    const Expr& expr) const {
  switch (expr.kind()) {
    case TK_VAR: {
      return Var(expr).name().name();
    }
    case TK_NONE: {
      return "None";
    }
    case TK_NONE_TYPE: {
      return "NoneType";
    }
    case '.': {
      auto select = Select(expr);
      const std::string& name = select.selector().name();
      // Special case for torch.Tensor and its' subclasses
      const std::unordered_set<std::string> tensor_subtypes = {
          "Tensor",
          "LongTensor",
          "FloatTensor",
          "DoubleTensor",
          "IntTensor",
          "ShortTensor",
          "HalfTensor",
          "CharTensor",
          "ByteTensor",
          "BoolTensor"};
      if (isTorch(select.value()) && tensor_subtypes.count(name) == 1) {
        return name;
      } else {
        // Otherwise, it's a fully qualified class name
        return collectQualname(select);
      }
    } break;
  }
  return std::nullopt;
}

TypePtr ScriptTypeParser::parseTypeFromExpr(const Expr& expr) const {
  // the resolver needs to recursively resolve the expression, so to avoid
  // resolving all type expr subtrees we only use it for the top level
  // expression and base type names.
  if (expr.kind() == '|') {
    auto converted = pep604union_to_union(expr);
    return parseTypeFromExpr(converted);
  }
  if (resolver_) {
    if (auto typePtr =
            resolver_->resolveType(expr.range().text().str(), expr.range())) {
      return typePtr;
    }
  }
  return parseTypeFromExprImpl(expr);
}

TypePtr ScriptTypeParser::parseTypeFromExprImpl(const Expr& expr) const {
  if (expr.kind() == '|') {
    auto converted = pep604union_to_union(expr);
    return parseTypeFromExprImpl(converted);
  }
  if (expr.kind() == TK_SUBSCRIPT) {
    auto subscript = Subscript(expr);
    auto value_name = parseBaseTypeName(subscript.value());
    if (!value_name) {
      throw ErrorReport(subscript.value().range())
          << "Subscripted type must be a type identifier";
    }
    return subscriptToType(*value_name, subscript);

  } else if (expr.kind() == TK_STRINGLITERAL) {
    const auto& type_name = StringLiteral(expr).text();

    // Check if the type is a custom class. This is done by checking
    // if type_name starts with "torch.classes."
    if (type_name.find("torch.classes.") == 0) {
      auto custom_class_type = getCustomClass("__torch__." + type_name);
      return custom_class_type;
    }

    // `torch.cuda.Stream` and `torch.cuda.Event` are aliased as
    // custom classes of type torch.classes.cuda.Stream and
    // torch.classes.cuda.Event respectively. Return the respective
    // custom class types for these two cases.
    if (type_name.find("torch.cuda.Stream") == 0) {
      auto custom_class_type =
          getCustomClass("__torch__.torch.classes.cuda.Stream");
      return custom_class_type;
    }

    if (type_name.find("torch.cuda.Event") == 0) {
      auto custom_class_type =
          getCustomClass("__torch__.torch.classes.cuda.Event");
      return custom_class_type;
    }

    if (resolver_) {
      if (auto typePtr = resolver_->resolveType(type_name, expr.range())) {
        return typePtr;
      }
    }

    throw ErrorReport(expr) << "Unknown type name '" << type_name << "'";
  } else if (auto name = parseBaseTypeName(expr)) {
    auto itr = string_to_type_lut().find(*name);
    if (itr != string_to_type_lut().end()) {
      return itr->second;
    }
    if (resolver_) {
      if (auto typePtr = resolver_->resolveType(*name, expr.range())) {
        return typePtr;
      }
    }

    if (auto custom_class_type = getCustomClass(*name)) {
      return custom_class_type;
    }

    throw ErrorReport(expr) << "Unknown type name '" << *name << "'";
  }
  throw ErrorReport(expr.range())
      << "Expression of type " << kindToString(expr.kind())
      << " cannot be used in a type expression";
}

TypePtr ScriptTypeParser::parseType(const std::string& str) {
  Parser p(std::make_shared<Source>(str));
  return parseTypeFromExpr(p.parseExp());
}

std::vector<IValue> ScriptTypeParser::evaluateDefaults(
    const SourceRange& r,
    const std::vector<Expr>& default_types,
    const std::vector<Expr>& default_exprs) {
  std::vector<IValue> default_values;
  if (default_exprs.empty())
    return default_values;
  // To evaluate the default expressions, we create a graph with no inputs,
  // and whose returns are the default values we need.
  // We then run constant prop on this graph and check the results are
  // constant. This approach avoids having to have separate handling of
  // default arguments from standard expressions by piecing together existing
  // machinery for graph generation, constant propagation, and constant
  // extraction.
  auto tuple_type = Subscript::create(
      r,
      Var::create(r, Ident::create(r, "Tuple")),
      List<Expr>::create(r, default_types));
  auto blank_decl = Decl::create(
      r, List<Param>::create(r, {}), Maybe<Expr>::create(r, tuple_type));

  auto tuple_expr =
      TupleLiteral::create(r, List<Expr>::create(r, default_exprs));
  auto ret = Return::create(r, tuple_expr);
  auto def = Def::create(
      r,
      Ident::create(r, "defaults"),
      blank_decl,
      List<Stmt>::create(r, {ret}));

  CompilationUnit cu;
  cu.define(
      std::nullopt,
      /*properties=*/{},
      /*propResolvers=*/{},
      {def},
      {resolver_},
      nullptr);
  Stack stack;
  // XXX: We need to turn optimization off here because otherwise we try to
  // recursively initialize stuff in DecomposeOps.
  GraphOptimizerEnabledGuard guard(false);
  auto& f = cu.get_function(def.name().name());
  auto* gf = dynamic_cast<GraphFunction*>(&f);
  TORCH_INTERNAL_ASSERT(gf);
  // 2024.08.14: Since we are starting to deprecate Torchscript usages,
  // we are going to log all the calls for GraphFunction::run. The logging was
  // noisy we also call GraphFunction::run for the default value evaluation
  // which generates a lot of useless log samples. Therefore as a workaround we
  // just directly use the executor API which avoids this placing producing
  // un-necessary log entries.
  gf->get_executor().run(stack);
  return stack.at(0).toTupleRef().elements().vec();
}

std::vector<Argument> ScriptTypeParser::parseArgsFromDecl(
    const Decl& decl,
    bool skip_self) {
  auto params_begin = decl.params().begin();
  auto params_end = decl.params().end();
  if (skip_self) {
    ++params_begin;
  }
  std::vector<Argument> retval;

  std::vector<Expr> default_types;
  std::vector<Expr> default_exprs;
  // gather any non-empty default arguments
  for (auto it = params_begin; it != params_end; ++it) {
    auto param = *it;
    auto def = param.defaultValue();
    if (def.present()) {
      if (!param.type().present()) {
        // We require explicit type-hints for default expressions.
        // If param doesn't have a type, we could default to "Tensor",
        // just like what happens in the Python frontend.
        // However here things are a bit more complicated, because
        // default expressions are evaluated using a custom-built
        // graph, and error messages coming out of that in case
        // the type doesn't match the value are quite obscure.
        throw ErrorReport(param.range())
            << "Keyword arguments with defaults need to be type-hinted (TorchScript C++ frontend)";
      }
      default_types.emplace_back(param.type().get());
      default_exprs.emplace_back(def.get());
    }
  }

  auto default_values =
      evaluateDefaults(decl.range(), default_types, default_exprs);

  auto defaults_it = default_values.begin();
  for (auto it = params_begin; it != params_end; ++it) {
    auto decl_arg = *it;

    TypePtr type;
    std::optional<int32_t> N = std::nullopt;
    if (!decl_arg.type().present()) {
      // If this param doesn't have a type, default to "tensor"
      type = TensorType::getInferred();
    } else {
      // BroadcastList list can only appear at the argument level
      Expr type_expr = decl_arg.type().get();
      if (auto maybe_broad_list = parseBroadcastList(type_expr)) {
        type = maybe_broad_list->first;
        N = maybe_broad_list->second;
      } else {
        type = parseTypeFromExpr(decl_arg.type().get());
      }
    }
    std::optional<IValue> default_value = std::nullopt;
    if (decl_arg.defaultValue().present()) {
      default_value = *defaults_it++;
    }
    auto arg = Argument(
        decl_arg.ident().name(),
        type,
        N,
        default_value,
        decl_arg.kwarg_only(),
        /*alias_info=*/std::nullopt);
    retval.push_back(arg);
  }
  return retval;
}

std::vector<Argument> ScriptTypeParser::parseReturnFromDecl(const Decl& decl) {
  // we represent no annotation on a return type as having no values in the
  // schema's return() list
  // in emitReturn we take the actual return value to be the value of the
  // return statement if no one was provided here
  if (!decl.return_type().present())
    return {};

  if (parseBroadcastList(decl.return_type().get()))
    throw ErrorReport(decl.return_type().range())
        << "Broadcastable lists cannot appear as a return type";

  TypePtr parsed_type;
  Expr type_expr = decl.return_type().get();
  parsed_type = parseTypeFromExpr(type_expr);
  return {Argument(
      "",
      parsed_type,
      /*N =*/std::nullopt,
      /*default_value =*/std::nullopt,
      /*kwarg_only =*/false)};
}
FunctionSchema ScriptTypeParser::parseSchemaFromDef(
    const Def& def,
    bool skip_self) {
  const auto name = def.name().name();
  std::vector<Argument> args = parseArgsFromDecl(def.decl(), skip_self);
  std::vector<Argument> returns = parseReturnFromDecl(def.decl());
  return FunctionSchema(
      name, "", std::move(args), std::move(returns), false, false);
}

c10::IValue ScriptTypeParser::parseClassConstant(const Assign& assign) {
  if (assign.lhs().kind() != TK_VAR) {
    throw ErrorReport(assign.range())
        << "Expected to a variable for class constant";
  }
  if (!assign.type().present()) {
    throw ErrorReport(assign.range())
        << "Expected a type to present for class constant";
  }
  const auto final_type = assign.type().get();
  auto expr = assign.rhs().get();
  if (final_type.kind() != TK_SUBSCRIPT) {
    throw ErrorReport(assign.range())
        << "Expected subscripted type for class constant";
  }
  auto subscript = Subscript(final_type);
  auto value_name = parseBaseTypeName(subscript.value());
  if (!value_name) {
    throw ErrorReport(subscript.value().range())
        << "Subscripted type must be a type identifier";
  }
  if (*value_name != "Final") {
    throw ErrorReport(subscript.range())
        << "Base type must be Final for class constant";
  }
  if (subscript.subscript_exprs().size() != 1) {
    throw ErrorReport(subscript)
        << " expected exactly one element type but found "
        << subscript.subscript_exprs().size();
  }
  auto type = *subscript.subscript_exprs().begin();
  auto default_val = evaluateDefaults(expr.range(), {type}, {expr});
  return *default_val.begin();
}

} // namespace torch::jit
