#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/script/builtin_functions.h>
#include <torch/csrc/jit/script/error_report.h>
#include <torch/csrc/jit/script/schema_matching.h>

namespace torch {
namespace jit {
namespace script {

static inline TypePtr unwrapOptional(TypePtr opt_type) {
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

/// Returns true if `type` is a Tuple in which all the elements have the
/// same type or if it's a subtype of `list_type_`.
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
          // TODO: resolve VarType if necessary
          return t->isSubtypeOf(list_type->getElementType());
        });
  }
  return false;
}

// Applies implict conversion from value trying to turn it into type
// concrete_type. It succeeds if `return_value->isSubtypeOf(concrete_type)`
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
    // Convert tensor to number
    if (concrete_type->isSubtypeOf(NumberType::get()) &&
        value->type()->isSubtypeOf(TensorType::get())) {
      auto n = graph.createImplicitTensorToNum(concrete_type, value);
      value = graph.insertNode(n)->setSourceRange(loc)->output();
    }

    // Convert strings to device
    if (value->type()->isSubtypeOf(StringType::get()) &&
        DeviceObjType::get()->isSubtypeOf(concrete_type)) {
      return graph.insert(aten::device, {value}, {}, loc);
    }
  }

  return value;
}

// Checks if `named_value` can be used as a value for `arg`. If `arg` is a
// VarType, it will be added to the type_env through `matchTypeVariables` as
// the corresponding actual type. If `allow_conversions` is true, implicit
// conversions to the `arg` type may be performed through `tryConvertToType`.
static Value* tryMatchArgument(
    const Argument& arg,
    Graph& graph,
    const SourceRange& loc,
    const NamedValue& named_value,
    std::ostream* failure_messages,
    const std::function<std::ostream&()>& err,
    bool allow_conversions,
    TypeEnv& type_env) {
  Value* value = named_value.value(graph);

  // Some functions that take lists of integers or floats for fixed size arrays
  // also allow single ints/floats to be passed in their place. The single
  // int/float is then repeated to the length of the list
  if (isIntOrFloatUsedAsList(value, arg)) {
    std::vector<Value*> repeated(*arg.N(), value);
    value =
        graph.insertNode(graph.createList(value->type(), repeated))->output();
  }

  // Resolve VarType variables
  const MatchTypeReturn matched_type =
      matchTypeVariables(arg.type(), value->type(), type_env);
  if (!matched_type.type) {
    if (failure_messages) {
      err() << "Could not match type " << value->type()->python_str() << " to "
            << arg.type()->python_str() << " in argument '" << arg.name()
            << "': " << matched_type.errMsg << ".\n";
    }
    return nullptr;
  }
  const auto concrete_type = *matched_type.type;

  // Check if the value can be matched to the arg through any implicit
  // conversions
  value = tryConvertToType(loc, graph, concrete_type, value, allow_conversions);

  if (!value->type()->isSubtypeOf(concrete_type)) {
    if (failure_messages) {
      auto& ostream = err()
          << arg.formatTypeMismatchMsg(value->type()->python_str());

      if (auto v = value->type()->cast<ListType>()) {
        if (v->getElementType()->isSubtypeOf(TensorType::get())) {
          ostream << "Empty lists default to List[Tensor]. Use torch.jit."
                     "annotate(List[my_type], []) to create an empty list of"
                     " another type.\n";
        }
      }

      if (value->type() == NumberType::get() &&
          value->node()->kind() == aten::item) {
        ostream << "Use int(tensor) or float(tensor) to retrieve item() from a "
                << "tensor with the appropriate type.\n";
      }
    }

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

/// Creates a list with the provided values if each value's type can be matched
/// to an argument with type `elem_type`. If a type in `varargs` does not match
/// `elem_type`, nullptr is returned. This is used for creating lists from
/// varargs so that calls like torch.zeros(1, 2, 3) will be matched to
/// aten::zeros(int[]).
static Value* tryCreateList(
    const TypePtr& elem_type,
    Graph& graph,
    const SourceRange& loc,
    at::ArrayRef<NamedValue> varargs,
    std::ostream* failure_messages,
    const std::function<std::ostream&()>& err,
    bool convert_tensor_to_num,
    TypeEnv& type_env) {
  Argument elem_arg("<varargs>", elem_type);
  std::vector<Value*> list_elements;
  for (const auto& named_value : varargs) {
    // Try to convert named_value to elem_type
    Value* matched_value = tryMatchArgument(
        /*arg=*/elem_arg,
        graph,
        loc,
        named_value,
        failure_messages,
        err,
        /*allow_conversions=*/convert_tensor_to_num,
        type_env);
    if (!matched_value) {
      return nullptr;
    }
    list_elements.push_back(matched_value);
  }

  return graph.insertNode(graph.createList(elem_type, list_elements))->output();
}

// Check if it is possible to convert all the remaining non-kwarg arguments
// to a list. This allows zeros(IntArrayRef sizes) to work with zeros(1, 2) or
// zeros(1)
static bool varargsCanBeUsedAsList(
    const FunctionSchema& schema,
    size_t arg_index,
    const Argument& arg) {
  // The arg must be the last one in the arg list that is not a kwarg
  bool is_last_argument = arg_index + 1 == schema.arguments().size() ||
      schema.arguments()[arg_index + 1].kwarg_only();

  // The formal must be a list
  bool argument_is_list = arg.type()->kind() == TypeKind::ListType;

  // it must not be a broadcasting list like int[3],
  // otherwise a single int is a valid input
  bool arg_is_broadcasting_list = bool(arg.N());

  return is_last_argument && argument_is_list & !arg_is_broadcasting_list;
}

c10::optional<MatchedSchema> tryMatchSchema(
    const FunctionSchema& schema,
    const SourceRange& loc,
    Graph& graph,
    c10::optional<NamedValue> self,
    at::ArrayRef<NamedValue> args,
    at::ArrayRef<NamedValue> kwargs,
    std::ostream* failure_messages,
    bool allow_conversions) {
  auto err = [&]() -> std::ostream& {
    *failure_messages << "\n" << schema << ":\n";
    return *failure_messages;
  };

  // For VarTypes, maps VarType name to actual type as it's used with these
  // args
  TypeEnv type_env;
  std::vector<Value*> positional_inputs;
  std::vector<bool> used_kwarg(kwargs.size(), false);

  // if we finish the loop will we have consumed all arguments?
  size_t used_args = 0;
  for (size_t schema_i = 0; schema_i < schema.arguments().size(); ++schema_i) {
    const auto& arg = schema.arguments()[schema_i];
    c10::optional<NamedValue> actual_named_value;
    if (arg.name() == "self" && self) {
      actual_named_value = self;
      self = c10::nullopt;
    } else if (!arg.kwarg_only() && used_args < args.size()) {
      // Try to convert all the remaining non-kwarg arguments (used_args) to a
      // list. Allow zeros(IntArrayRef sizes) to work with zeros(1, 2) or
      // zeros(1)
      if (allow_conversions && varargsCanBeUsedAsList(schema, schema_i, arg)) {
        auto value = args[used_args].value(graph);
        const auto& actual_type = value->type();
        // The actual cannot already be a list
        if (actual_type->kind() != TypeKind::ListType &&
            !convertibleToList(actual_type, unwrapOptional(arg.type()))) {
          auto formal_type =
              unwrapOptional(arg.type())->expect<ListType>()->getElementType();

          Value* list = tryCreateList(
              formal_type,
              graph,
              loc,
              at::ArrayRef<NamedValue>(args).slice(used_args),
              failure_messages,
              err,
              allow_conversions,
              type_env);
          if (!list) {
            return c10::nullopt;
          }
          used_args = args.size();
          positional_inputs.push_back(list);
          continue;
        }
      }

      // Set actual_named_value to the argument and mark the arg position as
      // used
      actual_named_value = args[used_args];
      used_args++;
    } else if (auto kwarg_idx = findInputWithName(arg.name(), kwargs)) {
      const NamedValue& nv = kwargs[*kwarg_idx];
      if (used_kwarg[*kwarg_idx]) {
        if (failure_messages) {
          err() << "Argument " << nv.name()
                << " specified twice in schema, submit a bug report!\n";
        }
        return c10::nullopt;
      }
      used_kwarg[*kwarg_idx] = true;
      actual_named_value = nv;
    } else if (arg.default_value()) {
      // Argument has a default value and no value was provided, so use the
      // default
      actual_named_value = NamedValue(*arg.default_value());
    } else {
      if (failure_messages) {
        err() << "Argument " << schema.arguments()[schema_i].name()
              << " not provided.\n";
      }
      return c10::nullopt;
    }

    // Make sure the actual_named_value found matches the type of arg
    Value* positional = tryMatchArgument(
        arg,
        graph,
        loc,
        *actual_named_value,
        failure_messages,
        err,
        allow_conversions,
        type_env);
    if (!positional) {
      return c10::nullopt;
    }
    positional_inputs.push_back(positional);
  }
  // check for unused self argument
  if (self != c10::nullopt && failure_messages) {
    err() << "Provided self argument not used in schema.\n";
  }

  if (schema.is_vararg()) {
    for (; used_args < args.size(); ++used_args) {
      positional_inputs.push_back(args[used_args].value(graph));
    }
  }

  // check for unused positional arguments
  if (used_args < args.size()) {
    if (failure_messages) {
      err() << "Expected at most " << used_args << " arguments "
            << "but found " << args.size() << " positional arguments.\n";
    }
    return c10::nullopt;
  }
  // check for unused kwargs
  for (size_t i = 0; i < kwargs.size(); ++i) {
    const auto& nv = kwargs[i];
    if (!used_kwarg[i]) {
      if (failure_messages) {
        if (!schema.argumentIndexWithName(nv.name())) {
          err() << "Keyword argument " << nv.name() << " unknown.\n";
        } else {
          err() << "Keyword argument " << nv.name() << " specified twice.\n";
        }
      }
      return c10::nullopt;
    }
  }

  const auto& returns = schema.returns();
  auto return_types = fmap(returns, [&](const Argument& r) {
    return evalTypeVariables(r.type(), type_env);
  });
  // Codegen does not support return of namedtuples with undefined field names.
  // Therefore, either all or none returns has field names.
  bool return_has_field_names =
      std::all_of(returns.begin(), returns.end(), [&](const Argument& r) {
        return r.name().length() > 0;
      });
  c10::OptNameList return_field_names = c10::nullopt;
  if (return_has_field_names) {
    return_field_names =
        fmap(returns, [&](const Argument& r) { return r.name(); });
  }
  return MatchedSchema{std::move(positional_inputs),
                       std::move(return_types),
                       std::move(return_field_names)};
}

MatchedSchema matchSchema(
    const ::c10::FunctionSchema& schema,
    const SourceRange& loc,
    Graph& graph,
    at::ArrayRef<NamedValue> args,
    at::ArrayRef<NamedValue> kwargs) {
  std::stringstream failure_messages;
  if (auto result = tryMatchSchema(
          schema,
          loc,
          graph,
          c10::nullopt,
          args,
          kwargs,
          &failure_messages,
          /*allow_conversions=*/true)) {
    return *result;
  }
  throw ErrorReport(loc) << failure_messages.str();
}

// pack outputs of a function following python rules. If there is a single value
// return a SimpleValue, otherwise pack all the values into a Tuple.
static Value* packOutputs(
    Graph& g,
    at::ArrayRef<Value*> values,
    c10::OptNameList field_names) {
  if (values.size() == 1) {
    return values[0];
  }
  std::shared_ptr<FunctionSchema> schema;
  if (field_names) {
    schema = TupleType::namedTupleSchemaFromNamesAndTypes(c10::QualifiedName(), field_names.value(), fmap(values, [](Value* v) { return v->type(); }));
  }
  return g
      .insertNode(
          g.createTuple(values, c10::nullopt, std::move(schema)))
      ->output();
}

// Given a successful match between operator schema and symbol, emit a node
// with the appropriate inputs and outputs.
static Value* emitBuiltinNode(
    const MatchedSchema& matched_schema,
    const SourceRange& loc,
    Graph& graph,
    Symbol name) {
  auto n = graph.insertNode(graph.create(name, matched_schema.inputs, 0))
               ->setSourceRange(loc);

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
    bool required,
    bool render_errors) {
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
          render_errors ? &failure_messages : nullptr,
          allow_conversions);
      if (matched_schema) {
        return emitBuiltinNode(*matched_schema, loc, graph, name);
      }
    }
    for (const auto method : builtin_functions) {
      method->ensure_defined();
      if (auto result = tryMatchSchema(
              method->getSchema(),
              loc,
              graph,
              self,
              inputs,
              attributes,
              render_errors ? &failure_messages : nullptr,
              allow_conversions)) {
        // we inline builtin calls because they are normally very small
        // wrappers and are not useful for keeping around to debug
        return insertGraph(graph, *method->graph(), result->inputs).at(0);
      }
    }
  }

  // none of the options worked
  if (!required) {
    return nullptr;
  }

  // If errors were required, but we didn't eagerly render error strings,
  // then replay schema matching with error strings eagerly rendered.
  if (!render_errors) {
    return emitBuiltinCall(
        loc,
        graph,
        name,
        self,
        inputs,
        attributes,
        required,
        /*render_errors=*/true);
  }

  // no operators found with the same name, print out similarly named operators
  if (variants.size() == 0) {
    const auto close_symbols = findSimilarOperators(name);
    auto error = ErrorReport(loc);
    const auto& user_function_name = name.toQualString();
    error << "Unknown builtin op: " << user_function_name << ".\n";
    if (close_symbols.size() == 0) {
      error
          << "Could not find any similar ops to " << user_function_name
          << ". This op may not exist or may not be currently supported in TorchScript.\n";
    } else {
      error << "Here are some suggestions: \n";
      for (const auto& sym : close_symbols) {
        error << "\t" << sym.toQualString() << "\n";
      }
      error << "\nThe original call is";
    }
    throw error;
  }

  throw ErrorReport(loc) << "Arguments for call are not valid.\n"
                         << "The following operator variants are available:\n"
                         << prefixLine(failure_messages.str(), "  ")
                         << "\nThe original call is";
}
} // namespace script
} // namespace jit
} // namespace torch
