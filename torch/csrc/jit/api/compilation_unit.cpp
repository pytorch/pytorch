#include <torch/csrc/jit/api/compilation_unit.h>

namespace torch {
namespace jit {

CompilationUnit::CompilationUnit(const std::string& source)
    : CompilationUnit() {
  // calles the define with native resolver to generate the graph for functions
  define(c10::nullopt, source, nativeResolver(), nullptr);
}

// This pair represents a pair of functions (getter and setter) obtained from
// compiling a Property.
struct CompilationUnit::PropertyPair
    : public std::pair<std::unique_ptr<Function>, std::unique_ptr<Function>> {
  PropertyPair(
      std::unique_ptr<Function> getter,
      std::unique_ptr<Function> setter) {
    TORCH_INTERNAL_ASSERT(getter, "Property pair must have defined getter")
    this->first = std::move(getter);
    this->second = std::move(setter);
  }

  std::unique_ptr<Function>& getGetter() {
    return this->first;
  }

  std::unique_ptr<Function>& getSetter() {
    return this->second;
  }
};

CompilationUnit::PropertyPair CompilationUnit::define_property(
    const c10::optional<c10::QualifiedName>& prefix,
    const Property& prop,
    const ResolverPtr& resolver,
    const Self* self,
    const std::unordered_map<std::string, Function*>& function_table,
    bool shouldMangle) const {
  // self must be defined because properties are features of classes and
  // modules.
  TORCH_INTERNAL_ASSERT(self);

  // Compile the getter function.
  std::unique_ptr<Function> getter_fn = define(
      prefix, prop.getter(), resolver, self, function_table, shouldMangle);

  // Compile the setter function if it exists.
  std::unique_ptr<Function> setter_fn = nullptr;
  if (prop.setter().present()) {
    setter_fn = define(
        prefix,
        prop.setter().get(),
        resolver,
        self,
        function_table,
        shouldMangle);
  }

  // Add the property to the class type definition.
  self->getClassType()->addProperty(
      prop.name().name(), getter_fn.get(), setter_fn.get());

  return PropertyPair(std::move(getter_fn), std::move(setter_fn));
}

std::unique_ptr<Function> CompilationUnit::define(
    const c10::optional<QualifiedName>& prefix,
    const Def& def,
    const ResolverPtr& resolver,
    const Self* self,
    const std::unordered_map<std::string, Function*>& function_table,
    bool shouldMangle,
    CompilationUnit::FunctionType type) const {
  TORCH_INTERNAL_ASSERT(resolver);
  auto _resolver = resolver;
  if (!self) {
    // if self is defined, then these are methods and do not go into the
    // global namespace otherwise, they get defined together so we add them to
    // the function table so the methods can see each other
    _resolver =
        std::make_shared<FunctionResolver>(resolver.get(), function_table);
  }
  auto creator = [def, _resolver, self](Function& method) {
    // Store the function name so that it can be referenced if there is an error
    // while compiling this function
    std::string call_name = method.qualname().name();
    if (self) {
      auto atoms = method.qualname().atoms();
      // There should be at least a ClassName.method_name
      TORCH_INTERNAL_ASSERT(atoms.size() >= 2);
      call_name = atoms.at(atoms.size() - 2) + "." + atoms.at(atoms.size() - 1);
    }
    ErrorReport::CallStack call(call_name, def.range());
    to_ir(def, _resolver, self, method);
  };
  auto name = prefix ? QualifiedName(*prefix, def.name().name())
                     : QualifiedName(def.name().name());
  if (shouldMangle) {
    // If `shouldMangle` is set, we should generate a unique name for this
    // function if there is already an existing one.
    if (auto fn = find_function(name)) {
      name = mangle(name);
    }
  }
  auto fn = torch::make_unique<GraphFunction>(
      std::move(name), std::make_shared<Graph>(), creator);
  if (self) {
    // Register this as a method on `self`'s type
    if (type == CompilationUnit::FunctionType::Hook) {
      self->getClassType()->addForwardHook(fn.get());
    } else if (type == CompilationUnit::FunctionType::PreHook) {
      self->getClassType()->addForwardPreHook(fn.get());
    } else {
      self->getClassType()->addMethod(fn.get());
    }
  }
  return fn;
}

std::vector<Function*> CompilationUnit::define(
    const c10::optional<c10::QualifiedName>& prefix,
    const std::vector<Property>& properties,
    const std::vector<ResolverPtr>& propResolvers,
    const std::vector<Def>& definitions,
    const std::vector<ResolverPtr>& defResolvers,
    const Self* self,
    bool shouldMangle) {
  TORCH_INTERNAL_ASSERT(definitions.size() == defResolvers.size());
  TORCH_INTERNAL_ASSERT(properties.size() == propResolvers.size());
  std::vector<Function*> functions;
  std::unordered_map<std::string, Function*> function_table;

  // Records fn in function_table, functions and with register_function.
  // This is done several times below, so this lambda helps avoid repeating
  // code.
  auto record_function = [&](std::unique_ptr<Function> fn) {
    function_table[fn->name()] = fn.get();
    functions.emplace_back(fn.get());
    this->register_function(std::move(fn));
  };

  for (size_t i = 0; i < properties.size(); i++) {
    PropertyPair property_fns = define_property(
        prefix,
        properties[i],
        propResolvers[i],
        self,
        function_table,
        shouldMangle);

    auto& getter_fn = property_fns.getGetter();
    auto& setter_fn = property_fns.getSetter();

    record_function(std::move(getter_fn));

    if (setter_fn) {
      record_function(std::move(setter_fn));
    }
  }

  for (size_t i = 0; i < definitions.size(); i++) {
    auto fn = define(
        prefix,
        definitions[i],
        defResolvers[i],
        self,
        function_table,
        shouldMangle,
        CompilationUnit::FunctionType::Method);

    record_function(std::move(fn));
  }

  // We need to compile `__init__` first, since it can determine what attributes
  // are available to other methods. So reorder the definitions accordingly.
  for (auto& kv : function_table) {
    if (kv.first == "__init__") {
      kv.second->ensure_defined();
    }
  }

  for (Function* function : functions) {
    function->ensure_defined();
  }

  return functions;
}

void CompilationUnit::define_hooks(
    const c10::optional<c10::QualifiedName>& prefix,
    const std::vector<Def>& hookDefs,
    const std::vector<ResolverPtr>& hookResolvers,
    const std::vector<Def>& preHookDefs,
    const std::vector<ResolverPtr>& preHookResolvers,
    const Self* self,
    bool shouldMangle) {
  TORCH_INTERNAL_ASSERT(hookDefs.size() == hookResolvers.size());
  TORCH_INTERNAL_ASSERT(preHookDefs.size() == preHookResolvers.size());
  std::vector<Function*> functions;
  std::unordered_map<std::string, Function*> function_table;

  // check hook for name collisions and redefinition
  auto check_collisions = [&](const Def& hook) -> Function* {
    auto name = prefix ? QualifiedName(*prefix, hook.name().name()).name()
                       : QualifiedName(hook.name().name()).name();
    // check if hook is already defined for this module
    auto found_hook = function_table.find(name);
    auto existing_hook =
        found_hook != function_table.end() ? found_hook->second : nullptr;
    // check if hook name is already defined on module as method
    if (existing_hook == nullptr) {
      TORCH_CHECK(
          self->getClassType()->findMethod(name) == nullptr &&
              self->getClassType()->findHook(name) == nullptr,
          "Can't define hook: ",
          name,
          " on class: ",
          self->getClassType()->repr_str(),
          " because a method or hook with that name already exists.");
    }
    return existing_hook;
  };

  // build_schema for checking
  auto build_schema = [&](const Def& hook_def,
                          const ResolverPtr& hook_res) -> FunctionSchema {
    ScriptTypeParser typeParser(hook_res);
    FunctionSchema schema =
        typeParser.parseSchemaFromDef(hook_def, true /* skip_self*/);
    // need to add self as the first because we skipped it
    std::vector<Argument> arguments;
    arguments.emplace_back(Argument(
        hook_def.decl().params()[0].ident().name(), self->getClassType()));
    arguments.insert(
        arguments.end(), schema.arguments().begin(), schema.arguments().end());
    return schema.cloneWithArguments(arguments);
  };

  // define hooks
  for (size_t i = 0; i < hookDefs.size(); i++) {
    // check to see if already defined this hook
    auto existing_fn = check_collisions(hookDefs[i]);
    if (existing_fn != nullptr) {
      // add it to class type again so it's called
      self->getClassType()->addForwardHook(existing_fn);
      continue;
    }
    // define hook
    auto fn = define(
        prefix,
        hookDefs[i],
        hookResolvers[i],
        self,
        function_table,
        shouldMangle,
        CompilationUnit::FunctionType::Hook);

    function_table[fn->name()] = fn.get();
    functions.emplace_back(fn.get());
    this->register_function(std::move(fn));
    self->getClassType()->checkForwardHookSchema(
        i, build_schema(hookDefs[i], hookResolvers[i]));
    functions.back()->ensure_defined();
  }

  // define pre_hooks
  for (size_t i = 0; i < preHookDefs.size(); i++) {
    // check to see if already defined this hook
    auto existing_fn = check_collisions(preHookDefs[i]);
    if (existing_fn != nullptr) {
      // add it to class type again so it's called
      self->getClassType()->addForwardPreHook(existing_fn);
      continue;
    }
    // define pre_hook
    auto fn = define(
        prefix,
        preHookDefs[i],
        preHookResolvers[i],
        self,
        function_table,
        shouldMangle,
        CompilationUnit::FunctionType::PreHook);

    function_table[fn->name()] = fn.get();
    functions.emplace_back(fn.get());
    this->register_function(std::move(fn));
    self->getClassType()->checkForwardPreHookSchema(
        i, build_schema(preHookDefs[i], preHookResolvers[i]));
    functions.back()->ensure_defined();
  }
}

std::vector<Function*> CompilationUnit::define(
    const c10::optional<QualifiedName>& prefix,
    const std::string& source,
    const ResolverPtr& resolver,
    const Self* self) {
  Parser p(std::make_shared<Source>(source, "<string>", 1));
  std::vector<Def> definitions;
  std::vector<ResolverPtr> resolvers;
  while (p.lexer().cur().kind != TK_EOF) {
    auto def = Def(p.parseFunction(/*is_method=*/bool(self)));
    definitions.push_back(def);
    resolvers.push_back(resolver);
  }
  return define(
      prefix,
      /*properties=*/{},
      /*propResolvers=*/{},
      definitions,
      resolvers,
      self);
}

void CompilationUnit::define_interface(
    const c10::QualifiedName& qualifiedName,
    const ClassDef& classDef,
    ResolverPtr rcb,
    bool is_module) {
  ScriptTypeParser typeParser(std::move(rcb));
  InterfaceTypePtr iface =
      InterfaceType::create(c10::QualifiedName(qualifiedName), is_module);
  for (const Stmt& stmt : classDef.body()) {
    if (stmt.kind() != TK_DEF) {
      throw ErrorReport(stmt)
          << "interface declartions can only contain method definitions";
    }
    auto method_def = Def(stmt);
    if (!method_def.decl().return_type().present()) {
      throw ErrorReport(method_def)
          << "interface declarations must have a return type annotated.";
    }
    FunctionSchema schema =
        typeParser.parseSchemaFromDef(method_def, /* skip_self*/ true);
    // need to add self as the first because we skipped it
    std::vector<Argument> arguments;
    arguments.emplace_back(
        Argument(method_def.decl().params()[0].ident().name(), iface));
    arguments.insert(
        arguments.end(), schema.arguments().begin(), schema.arguments().end());
    iface->addMethod(schema.cloneWithArguments(std::move(arguments)));
    // we need to make sure everything but the last element is just string
    // literals (aka comments) unless there is "pass" in between
    auto stmts_size = method_def.statements().size();
    for (size_t i = 0; i < stmts_size - 1; i++) {
      auto cur_statement = method_def.statements()[i];
      if (cur_statement.kind() == TK_EXPR_STMT) {
        auto expr = ExprStmt(cur_statement).expr();
        if (expr.kind() != TK_STRINGLITERAL) {
          throw ErrorReport(method_def.range())
              << "interfaces declarations should only contain a single 'pass' statement.";
        }
      }
      // if we see a "pass", we just stop there
      if (cur_statement.kind() == TK_PASS) {
        this->register_type(iface);
        return;
      }
    }

    if (method_def.statements()[stmts_size - 1].kind() != TK_PASS) {
      throw ErrorReport(method_def.range())
          << "interfaces declarations should contain 'pass' statement.";
    }
  }
  this->register_type(iface);
}

} // namespace jit
} // namespace torch
