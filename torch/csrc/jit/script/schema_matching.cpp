#include <torch/csrc/jit/script/schema_matching.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/script/builtin_functions.h>
#include <torch/csrc/jit/script/error_report.h>

namespace torch {
namespace jit {
namespace script {

inline TypePtr unwrapOptional(TypePtr opt_type) {
  if (auto unwrap_list_type = opt_type->cast<OptionalType>()) {
    return unwrap_list_type->getElementType();
  }
  return opt_type;
}

static inline bool isIntOrFloatUsedAsList(
    const Value* value,
    const Argument& arg) {
  // Look for int[N] or float[N]
  const auto& v_type = value->type();
  if (v_type != FloatType::get() && v_type != IntType::get())
    return false;
  auto arg_type = unwrapOptional(arg.type());
  auto list_type = arg_type->cast<ListType>();
  return list_type && list_type->getElementType() == v_type && arg.N();
}

inline bool convertibleToList(const TypePtr& type, const TypePtr& list_type_) {
  auto list_type = list_type_->cast<ListType>();
  if (!list_type) {
    return false;
  }
  if (type->isSubtypeOf(list_type_)) {
    return true;
  }
  if (auto tuple = type->cast<TupleType>()) {
    return std::all_of(
        tuple->elements().begin(),
        tuple->elements().end(),
        [&](const TypePtr& t) {
          return t->isSubtypeOf(list_type->getElementType());
        });
  }
  return false;
}

// applies implict conversion from value trying to turn it into type
// concrete_type it succeeds if the return_value->isSubclassOf(concrete_type)
Value* tryConvertToType(
    const SourceRange& loc,
    Graph& graph,
    const TypePtr& concrete_type,
    Value* value,
    bool allow_conversions) {
  if (auto value_tuple = value->type()->cast<TupleType>()) {
    // Allow homogeneous tuples to be casted implicitly to lists of appropriate
    // types
    if (convertibleToList(value->type(), unwrapOptional(concrete_type))) {
      auto unpacked = createTupleUnpack(value);
      auto elem_type =
          unwrapOptional(concrete_type)->expect<ListType>()->getElementType();
      value = graph.insertNode(graph.createList(elem_type, unpacked))->output();
    }
    // inductively apply implicit conversions to tuples
    if (auto concrete_tuple = concrete_type->cast<TupleType>()) {
      if (!value_tuple->isSubtypeOf(concrete_tuple) &&
          concrete_tuple->elements().size() == value_tuple->elements().size()) {
        auto unpacked = createTupleUnpack(value);
        std::vector<Value*> converted;
        for (size_t i = 0; i < concrete_tuple->elements().size(); ++i) {
          converted.emplace_back(tryConvertToType(
              loc,
              graph,
              concrete_tuple->elements().at(i),
              unpacked.at(i),
              allow_conversions));
        }
        value = graph.insertNode(graph.createTuple(converted))->output();
      }
    }
  }

  if (value->type()->isSubtypeOf(NoneType::get()) &&
      !concrete_type->isSubtypeOf(NoneType::get())) {
    if (auto optional_type = concrete_type->cast<OptionalType>()) {
      value =
          graph.insertNode(graph.createNone(optional_type->getElementType()))
              ->output();
    } else {
      // When try to convert None to non-optional concrete type, create a None
      // node with the return value type of Optional[concrete_type]
      value = graph.insertNode(graph.createNone(concrete_type))->output();
    }
  }

  // implicit conversions
  if (allow_conversions) {
    if (concrete_type->isSubtypeOf(NumberType::get()) &&
        value->type()->isSubtypeOf(TensorType::get())) {
      auto n = graph.createImplicitTensorToNum(concrete_type, value);
      value = graph.insertNode(n)
                  ->setSourceLocation(std::make_shared<SourceRange>(loc))
                  ->output();
    }
    if (value->type()->isSubtypeOf(StringType::get()) &&
        DeviceObjType::get()->isSubtypeOf(concrete_type)) {
      return graph.insert(aten::device, {value}, {}, loc);
    }
  }

  return value;
}

Value* tryMatchArgument(
    const Argument& arg,
    Graph& graph,
    const SourceRange& loc,
    const NamedValue& named_value,
    const std::function<std::ostream&()>& err,
    bool allow_conversions,
    TypeEnv& type_env) {
  Value* value = named_value.value(graph);

  // some functions that take lists of integers or floats for fixed size arrays
  // also allow single ints/floats to be passed in their place.
  // the single int/float is then repeated to the length of the list
  if (isIntOrFloatUsedAsList(value, arg)) {
    std::vector<Value*> repeated(*arg.N(), value);
    value =
        graph.insertNode(graph.createList(value->type(), repeated))->output();
  }

  const MatchTypeReturn matched_type =
      matchTypeVariables(arg.type(), value->type(), type_env);
  if (!matched_type.type) {
    err() << "could not match type " << value->type()->str() << " to "
          << arg.type()->str() << " in argument '" << arg.name()
          << "': " << matched_type.errMsg << "\n"
          << named_value.locOr(loc);
    return nullptr;
  }
  const auto concrete_type = *matched_type.type;

  value = tryConvertToType(loc, graph, concrete_type, value, allow_conversions);

  if (!value->type()->isSubtypeOf(concrete_type)) {
    err() << "expected a value of type " << concrete_type->str()
          << " for argument '" << arg.name() << "' but found "
          << value->type()->str() << "\n"
          << named_value.locOr(loc);
    return nullptr;
  }
  return value;
}

c10::optional<size_t> findInputWithName(
    const std::string& name,
    at::ArrayRef<NamedValue> kwargs) {
  for (size_t i = 0; i < kwargs.size(); ++i) {
    if (kwargs[i].name() == name)
      return i;
  }
  return c10::nullopt;
}

Value* tryCreateList(
    const TypePtr& elem_type,
    Graph& graph,
    const SourceRange& loc,
    at::ArrayRef<NamedValue> varargs,
    const std::function<std::ostream&()>& err,
    bool convert_tensor_to_num,
    TypeEnv& type_env) {
  Argument elem_arg("<varargs>", elem_type);
  std::vector<Value*> list_ctor;
  for (const auto& a : varargs) {
    Value* av = tryMatchArgument(
        elem_arg, graph, loc, a, err, convert_tensor_to_num, type_env);
    if (!av)
      return nullptr;
    list_ctor.push_back(av);
  }
  return graph.insertNode(graph.createList(elem_type, list_ctor))->output();
}

c10::optional<MatchedSchema> tryMatchSchema(
    const FunctionSchema& schema,
    const SourceRange& loc,
    Graph& graph,
    c10::optional<NamedValue> self,
    at::ArrayRef<NamedValue> args,
    at::ArrayRef<NamedValue> kwargs,
    std::ostream& failure_messages,
    bool allow_conversions) {
  auto err = [&]() -> std::ostream& {
    failure_messages << "\nfor operator " << schema << ":\n";
    return failure_messages;
  };

  TypeEnv type_env;
  std::vector<Value*> positional_inputs;
  std::vector<bool> used_kwarg(kwargs.size(), false);

  // if we finish the loop will we have consumed all arguments?
  size_t used_args = 0;
  for (size_t schema_i = 0; schema_i < schema.arguments().size(); ++schema_i) {
    const auto& arg = schema.arguments()[schema_i];
    c10::optional<NamedValue> v;
    if (arg.name() == "self" && self) {
      v = self;
      self = c10::nullopt;
    } else if (!arg.kwarg_only() && used_args < args.size()) {
      // allow zeros(IntArrayRef sizes) to work with zeros(1, 2) or zeros(1)
      if (allow_conversions && arg.type()->kind() ==
              TypeKind::ListType && // the formal must be a list
          !arg.N() && // it must not be a broadcasting list like int[3],
                      // otherwise a single int is a valid input
          (schema_i + 1 == schema.arguments().size() ||
           schema.arguments()[schema_i + 1]
               .kwarg_only())) { // must be the last position argument
        auto actual_type = args[used_args].value(graph)->type();
        if (actual_type->kind() != TypeKind::ListType &&
            !convertibleToList(
                actual_type,
                unwrapOptional(arg.type()))) { // and the actual should not be a
                                               // list already
          auto elem_type =
              unwrapOptional(arg.type())->expect<ListType>()->getElementType();
          Value* list = tryCreateList(
              elem_type,
              graph,
              loc,
              at::ArrayRef<NamedValue>(args).slice(used_args),
              err,
              allow_conversions,
              type_env);
          if (!list)
            return c10::nullopt;
          used_args = args.size();
          positional_inputs.push_back(list);
          continue;
        }
      }

      v = args[used_args];
      used_args++;
    } else if (auto idx = findInputWithName(arg.name(), kwargs)) {
      const NamedValue& nv = kwargs[*idx];
      if (used_kwarg[*idx]) {
        err() << "argument " << nv.name()
              << " specified twice in schema, submit a bug report!\n"
              << nv.locOr(loc);
        return c10::nullopt;
      }
      used_kwarg[*idx] = true;
      v = nv;
    } else if (arg.default_value()) {
      v = NamedValue(*arg.default_value());
    } else {
      err() << "argument " << schema.arguments()[schema_i].name()
            << " not provided.\n"
            << loc;
      return c10::nullopt;
    }
    Value* positional =
        tryMatchArgument(arg, graph, loc, *v, err, allow_conversions, type_env);
    if (!positional)
      return c10::nullopt;
    positional_inputs.push_back(positional);
  }
  // check for unused self argument
  if (self != c10::nullopt) {
    err() << "provided self argument not used in schema\n";
  }

  if (schema.is_vararg()) {
    for (; used_args < args.size(); ++used_args) {
      positional_inputs.push_back(args[used_args].value(graph));
    }
  }

  // check for unused positional arguments
  if (used_args < args.size()) {
    err() << "expected at most " << used_args << " arguments "
          << "but found " << args.size() << " positional arguments.\n"
          << loc << "\n";
    return c10::nullopt;
  }
  // check for unused kwargs
  for (size_t i = 0; i < kwargs.size(); ++i) {
    const auto& nv = kwargs[i];
    if (!used_kwarg[i]) {
      if (!schema.argumentIndexWithName(nv.name())) {
        err() << "keyword argument " << nv.name() << " unknown\n";
      } else {
        err() << "keyword argument " << nv.name() << " specified twice\n";
      }
      return c10::nullopt;
    }
  }

  const auto &returns = schema.returns();
  auto return_types = fmap(returns, [&](const Argument& r) {
    return evalTypeVariables(r.type(), type_env);
  });
  // Codegen does not support return of namedtuples with undefined field names.
  // Therefore, either all or none returns has field names.
  bool return_has_field_names = std::all_of(returns.begin(), returns.end(),
    [&](const Argument& r) { return r.name().length() > 0; });
  c10::OptNameList return_field_names = c10::nullopt;
  if (return_has_field_names) {
    return_field_names = fmap(returns, [&](const Argument& r) {
      return r.name();
    });
  }
  return MatchedSchema{std::move(positional_inputs), std::move(return_types), std::move(return_field_names)};
}

// pack outputs of a function following python rules. If there is a single value
// return a SimpleValue, otherwise pack all the values into a Tuple.
Value* packOutputs(Graph& g, at::ArrayRef<Value*> values, c10::OptNameList field_names) {
  if (values.size() == 1) {
    return values[0];
  }
  return g.insertNode(g.createTuple(values, std::move(field_names)))->output();
}

// Given a successful match between operator schema and symbol, emit a node
// with the appropriate inputs and outputs.
static Value* emitBuiltinNode(
    const MatchedSchema& matched_schema,
    const SourceRange& loc,
    Graph& graph,
    Symbol name) {
  auto n = graph.insertNode(graph.create(name, matched_schema.inputs, 0))
               ->setSourceLocation(std::make_shared<SourceRange>(loc));

  for (auto& ret : matched_schema.return_types) {
    n->addOutput()->setType(ret);
  }

  // assert that we did indeed create an op that has implementation
  // otherwise schema and dispatch are not in sync
  getOperation(n);

  return packOutputs(graph, n->outputs(), matched_schema.return_field_names);
}

static std::string prefixLine(
    const std::string& str,
    const std::string& prefix) {
  std::stringstream ss;
  bool was_newline = true;
  for (auto c : str) {
    if (was_newline)
      ss << prefix;
    ss.put(c);
    was_newline = c == '\n';
  }
  return ss.str();
}

// Search for operators matching the provided symbol name and input types.
// If one is found, emit a node to the graph for that operator.
Value* emitBuiltinCall(
    const SourceRange& loc,
    Graph& graph,
    Symbol name,
    const c10::optional<NamedValue>& self,
    at::ArrayRef<NamedValue> inputs,
    at::ArrayRef<NamedValue> attributes,
    // if true, emitBuiltinCall will throw an exception if this builtin does not
    // exist, otherwise it will return nullptr if the builtin is not found.
    bool required) {
  const auto& variants = getAllOperatorsFor(name);
  const auto& builtin_functions = getAllBuiltinFunctionsFor(name);

  std::stringstream failure_messages;
  // first we try to match the schema without any conversion
  // if no schema matches then insert ImplicitTensorToNum
  for (bool allow_conversions : {false, true}) {
    // clear previous error messages
    failure_messages.str("");
    for (const std::shared_ptr<Operator>& op : variants) {
      const auto matched_schema = tryMatchSchema(
          op->schema(),
          loc,
          graph,
          self,
          inputs,
          attributes,
          failure_messages,
          allow_conversions);
      if (matched_schema) {
        return emitBuiltinNode(*matched_schema, loc, graph, name);
      }
    }
    for (Method* method : builtin_functions) {
      if (auto result = try_emit_call_to(
              graph,
              loc,
              *method,
              self,
              inputs,
              attributes,
              failure_messages,
              nullptr,
              allow_conversions)) {
        return result;
      }
    }
  }

  // none of the options worked
  if (!required) {
    return nullptr;
  }
  // no operators found with the same name, print out similarly named operators
  if (variants.size() == 0) {
    const auto close_symbols = findSimilarOperators(name);
    auto error = ErrorReport(loc);
    const auto& user_function_name = name.toQualString();
    error << "unknown builtin op: " << user_function_name << "\n";
    if (close_symbols.size() == 0) {
      error
          << "Could not find any similar ops to " << user_function_name
          << ". This op may not exist or may not be currently supported in TorchScript\n";
    } else {
      error << "Here are some suggestions: \n";
      for (const auto& sym : close_symbols) {
        error << "\t" << sym.toQualString() << "\n";
      }
    }
    throw error;
  }

  throw ErrorReport(loc) << "arguments for call are not valid:\n"
                         << prefixLine(failure_messages.str(), "  ")
                         << "for call at";
}

} // namespace script
} // namespace jit
} // namespace torch
