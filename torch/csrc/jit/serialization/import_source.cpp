#include <torch/csrc/jit/serialization/import_source.h>

#include <ATen/core/ivalue_inl.h>
#include <ATen/core/qualified_name.h>
#include <torch/csrc/jit/frontend/parser.h>
#include <torch/csrc/jit/frontend/resolver.h>
#include <torch/csrc/jit/frontend/script_type_parser.h>
#include <torch/custom_class.h>

#include <regex>

namespace torch::jit {

struct OpsValue : public SugaredValue {
  OpsValue(size_t version) : version_(version) {}
  std::string kind() const override {
    return "ops";
  }
  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      GraphFunction& m,
      const std::string& field) override {
    return std::make_shared<BuiltinModule>(field, version_);
  }
  size_t version_;
};

// Represents nested namespaces, like `foo.bar.Baz`.
// Right now these namespaces can only contain other namespaces or NamedTypes
struct TORCH_API ClassNamespaceValue : public SugaredValue {
  /**
   * @param  name  The fully qualified path, which can resolve either to a
   *               namespace or a NamedType
   * @param  si    The source importer that searches for and loads
   * classes/functions.
   */
  explicit ClassNamespaceValue(
      c10::QualifiedName name,
      std::shared_ptr<SourceImporterImpl> si)
      : basename_(std::move(name)), si_(std::move(si)) {}

  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      GraphFunction& m,
      const std::string& name) override;
  std::string kind() const override {
    return "Class Namespace";
  }

 private:
  c10::QualifiedName basename_;
  std::shared_ptr<SourceImporterImpl> si_;
};

// This value maps attributes CONSTANTS.c0 CONSTANTS.c1 to entries
// in the 'constants' vector. This table is will be stored in a container format
// and given to the import_method when restoring the code.
struct ConstantTableValue : public SugaredValue {
  explicit ConstantTableValue(const std::vector<at::IValue>* constants)
      : constants_(constants) {}
  std::string kind() const override {
    return "CONSTANTS";
  }
  // select an attribute on it, e.g. `this.field`
  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      GraphFunction& m,
      const std::string& field) override {
    const char* field_s = field.c_str();
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    char* end;
    int64_t offset = strtoll(field_s + 1, &end, 10);
    if (field.size() < 2 || *end != 0)
      throw ErrorReport(loc) << "invalid constant specifier: " << field;
    if (offset < 0 || size_t(offset) >= constants_->size()) {
      throw ErrorReport(loc) << "constant index " << offset
                             << " is out of bounds (constant table has "
                             << constants_->size() << " entries)";
    }
    auto ivalue = constants_->at(offset);
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    Value* value;

    // see [Constant Object Weak CompilationUnit Reference]
    if (ivalue.isObject() && !ivalue.toObject()->is_weak_compilation_ref()) {
      auto obj = ivalue.toObject();
      if (!non_holding_object_cache.count(obj)) {
        non_holding_object_cache[obj] = obj->copy_to_weak_compilation_ref();
      }
      value = m.graph()->insertConstant(non_holding_object_cache[obj], loc);
    } else {
      value = m.graph()->insertConstant(constants_->at(offset), loc);
    }

    // specializing tensor type on compilation messes up typing relations
    value->setType(unshapedType(value->type()));

    return std::make_shared<SimpleValue>(value);
  }

 private:
  std::unordered_map<
      c10::intrusive_ptr<at::ivalue::Object>,
      c10::intrusive_ptr<at::ivalue::Object>>
      non_holding_object_cache;
  const std::vector<at::IValue>* constants_;
};

SourceImporterImpl::SourceImporterImpl(
    std::shared_ptr<CompilationUnit> cu,
    const std::vector<at::IValue>* constant_table,
    SourceLoader source_loader,
    size_t version)
    : cu_(std::move(cu)),
      source_loader_(std::move(source_loader)),
      version_(version) {
  env_ = {
      {"torch", std::make_shared<BuiltinModule>("aten", version)},
      {"ops", std::make_shared<OpsValue>(version)},
      // Constants present in the model. Used to resolve "CONSTANTS.n" to the
      // actual value
      {"CONSTANTS", std::make_shared<ConstantTableValue>(constant_table)},
      {"fork", SpecialFormValue::create(prim::fork)},
      {"awaitable", SpecialFormValue::create(prim::awaitable)},
      {"annotate", SpecialFormValue::create(prim::annotate)},
      {"unchecked_cast", SpecialFormValue::create(prim::unchecked_cast)},
      {"uninitialized", SpecialFormValue::create(prim::Uninitialized)},
  };
}

TypePtr SourceImporterImpl::findNamedType(const QualifiedName& name) {
  if (auto custom_class = getCustomClass(name.qualifiedName())) {
    return custom_class;
  }
  parseSourceIfNeeded(name.prefix());
  auto it = to_be_defined_.find(name);
  if (it != to_be_defined_.end() && it->second->kind() == TK_CLASS_DEF) {
    ClassDef cd(std::move(it->second));
    to_be_defined_.erase(it);
    importNamedType(name.prefix(), cd);
  }
  return cu_->get_type(name);
}

Function* SourceImporterImpl::findFunction(const QualifiedName& name) {
  parseSourceIfNeeded(name.prefix());
  auto it = to_be_defined_.find(name);
  if (it != to_be_defined_.end() && it->second->kind() == TK_DEF) {
    Def d(it->second);
    to_be_defined_.erase(it);
    importFunction(name.prefix(), d);
  }
  return cu_->find_function(name);
}

void SourceImporterImpl::parseSourceIfNeeded(const std::string& qualifier) {
  // qualifier may be blank, for instance checking if __torch__ is a class.
  if (qualifier.empty() || loaded_sources_.count(qualifier)) {
    return;
  }
  loaded_sources_.insert(qualifier);
  std::shared_ptr<Source> src = source_loader_(qualifier);

  // The importer, when looking for classes/functions doesn't know if 'foo'
  // contains definitions or if it is a prefix of 'foo.bar', we only figure it
  // out by testing if `foo.py` exists in the source loader. If it doesn't
  // then there is nothing to load here
  if (!src) {
    return;
  }
  Parser p(src);
  parsePossibleVersionNumber(p.lexer());

  auto& L = p.lexer();

  while (L.cur().kind != TK_EOF) {
    parseImports(L);
    auto tk = L.cur();
    auto kind = tk.kind;
    switch (kind) {
      case TK_CLASS_DEF: {
        auto parsed_treeref = ClassDef(p.parseClass());
        to_be_defined_[QualifiedName(qualifier, parsed_treeref.name().name())] =
            parsed_treeref;
      } break;
      case TK_DEF: {
        auto parsed_treeref = Def(p.parseFunction(/*is_method=*/false));
        to_be_defined_[QualifiedName(qualifier, parsed_treeref.name().name())] =
            parsed_treeref;
      } break;
      default:
        throw ErrorReport(L.cur().range)
            << "Unexpected token in code import: " << kindToString(kind);
    }
  }
}

void SourceImporterImpl::LEGACY_import_methods(
    const Module& mod,
    const std::shared_ptr<Source>& src) {
  auto self = SimpleSelf(mod.type());
  c10::QualifiedName prefix = *mod.type()->name();
  Parser p(src);

  parsePossibleVersionNumber(p.lexer());

  parseImports(p.lexer());

  std::vector<Def> definitions;
  std::vector<ResolverPtr> resolvers;
  while (p.lexer().cur().kind != TK_EOF) {
    auto def = Def(p.parseFunction(/*is_method=*/true));
    definitions.emplace_back(def);
    resolvers.emplace_back(shared_from_this());
  }
  cu_->define(
      prefix,
      /*properties=*/{},
      /*propResolvers=*/{},
      definitions,
      resolvers,
      &self);
}

std::shared_ptr<SugaredValue> SourceImporterImpl::resolveValue(
    const std::string& name,
    GraphFunction& m,
    const SourceRange& loc) {
  auto it = env_.find(name);
  if (it != env_.end()) {
    return it->second;
  }
  auto graph = m.graph();
  if (name == "inf") {
    return std::make_shared<SimpleValue>(
        graph->insertConstant(std::numeric_limits<double>::infinity(), loc));
  }
  if (name == "nan") {
    return std::make_shared<SimpleValue>(
        graph->insertConstant(std::numeric_limits<double>::quiet_NaN(), loc));
  }
  if (name == "infj") {
    return std::make_shared<SimpleValue>(graph->insertConstant(
        c10::complex<double>(0, std::numeric_limits<double>::infinity()), loc));
  }
  if (name == "nanj") {
    return std::make_shared<SimpleValue>(graph->insertConstant(
        c10::complex<double>(0, std::numeric_limits<double>::quiet_NaN()),
        loc));
  }
  if (name == "__torch__") {
    return std::make_shared<ClassNamespaceValue>(
        c10::QualifiedName(name), shared_from_this());
  }
  return nullptr;
}

TypePtr SourceImporterImpl::resolveType(
    const std::string& name,
    const SourceRange& loc) {
  return findNamedType(QualifiedName(name));
}

void SourceImporterImpl::importFunction(
    const std::string& qualifier,
    const Def& def) {
  std::vector<Def> definitions{def};
  std::vector<ResolverPtr> resolvers{shared_from_this()};
  cu_->define(
      qualifier,
      /*properties=*/{},
      /*propResolvers=*/{},
      definitions,
      resolvers,
      nullptr);
}

void SourceImporterImpl::importNamedType(
    const std::string& qualifier,
    const ClassDef& class_def) {
  const auto qualified_name =
      QualifiedName(QualifiedName(qualifier), class_def.name().name());
  if (!class_def.superclass().present()) {
    return importClass(qualified_name, class_def, /*is_module=*/false);
  }
  const auto& superclass_name = Var(class_def.superclass().get()).name().name();
  if (superclass_name == "Module") {
    importClass(qualified_name, class_def, /*is_module=*/true);
  } else if (superclass_name == "NamedTuple") {
    // NamedTuples have special rules (since they are TupleTypes and not
    // ClassTypes)
    return importNamedTuple(qualified_name, class_def);
  } else if (superclass_name == "Interface") {
    cu_->define_interface(
        qualified_name, class_def, shared_from_this(), /*is_module=*/false);
  } else if (superclass_name == "ModuleInterface") {
    cu_->define_interface(
        qualified_name, class_def, shared_from_this(), /*is_module=*/true);
  } else if (superclass_name == "Enum") {
    importEnum(qualified_name, class_def);
  } else {
    throw ErrorReport(class_def.range())
        << "Torchscript does not support class inheritance.";
  }
}

std::optional<Assign> SourceImporterImpl::
    attributeAssignmentSpecialHandlingHack(
        const QualifiedName& qualified_classname,
        const Assign& assign) {
  struct AttrTypeReplacementDescr {
    std::string attr_name;
    std::string expected_type;
    std::string replacement_type;
  };

  // module demangled qualname -> ReplacementDescr
  static std::unordered_map<std::string, AttrTypeReplacementDescr> replacements{
      {"__torch__.torch.ao.nn.quantized.modules.linear.LinearPackedParams",
       {"_packed_params",
        "Tensor",
        "__torch__.torch.classes.quantized.LinearPackedParamsBase"}},
      {"__torch__.torch.ao.nn.quantized.modules.linear.Linear",
       {"_packed_params",
        "Tensor",
        "__torch__.torch.classes.quantized.LinearPackedParamsBase"}},
      {"__torch__.torch.ao.nn.quantized.dynamic.modules.linear.Linear",
       {"_packed_params",
        "Tensor",
        "__torch__.torch.classes.quantized.LinearPackedParamsBase"}},
      {"__torch__.torch.ao.nn.quantized.modules.conv.Conv2d",
       {"_packed_params",
        "Tensor",
        "__torch__.torch.classes.quantized.Conv2dPackedParamsBase"}},
      {"__torch__.torch.nn.intrinsic.quantized.modules.conv_relu.ConvReLU2d",
       {"_packed_params",
        "Tensor",
        "__torch__.torch.classes.quantized.Conv2dPackedParamsBase"}},
      {"__torch__.torch.ao.nn.quantized.modules.conv.Conv3d",
       {"_packed_params",
        "Tensor",
        "__torch__.torch.classes.quantized.Conv3dPackedParamsBase"}},
      {"__torch__.torch.nn.intrinsic.quantized.modules.conv_relu.ConvReLU3d",
       {"_packed_params",
        "Tensor",
        "__torch__.torch.classes.quantized.Conv3dPackedParamsBase"}},
      // BC Stuff
      {"__torch__.torch.nn.quantized.modules.linear.LinearPackedParams",
       {"_packed_params",
        "Tensor",
        "__torch__.torch.classes.quantized.LinearPackedParamsBase"}},
      {"__torch__.torch.nn.quantized.modules.linear.Linear",
       {"_packed_params",
        "Tensor",
        "__torch__.torch.classes.quantized.LinearPackedParamsBase"}},
      {"__torch__.torch.nn.quantized.modules.conv.Conv2d",
       {"_packed_params",
        "Tensor",
        "__torch__.torch.classes.quantized.Conv2dPackedParamsBase"}},
      {"__torch__.torch.nn.quantized.modules.conv.Conv3d",
       {"_packed_params",
        "Tensor",
        "__torch__.torch.classes.quantized.Conv3dPackedParamsBase"}},
      {"__torch__.torch.nn.quantized.dynamic.modules.linear.Linear",
       {"_packed_params",
        "Tensor",
        "__torch__.torch.classes.quantized.LinearPackedParamsBase"}}};
  // @lint-ignore-every CLANGTIDY facebook-hte-StdRegexIsAwful
  static std::regex mangle_re("\\.___torch_mangle_\\d+");
  auto demangled_classname =
      std::regex_replace(qualified_classname.qualifiedName(), mangle_re, "");
  if (replacements.count(demangled_classname)) {
    auto lhs = Var(assign.lhs());
    if (!assign.type().present() || assign.type().get().kind() != TK_VAR) {
      return c10::nullopt;
    }
    auto type = Var(assign.type().get());

    auto& attr_name = replacements.at(demangled_classname).attr_name;
    auto& expected_type = replacements.at(demangled_classname).expected_type;
    auto& replacement_type =
        replacements.at(demangled_classname).replacement_type;
    if (lhs.name().name() == attr_name && type.name().name() == expected_type) {
      Parser p(std::make_shared<Source>(replacement_type));
      auto typename_expr = p.parseExp();
      auto maybe_typename =
          Maybe<Expr>::create(typename_expr.range(), typename_expr);
      return Assign::create(
          assign.range(), assign.lhs_list(), assign.rhs(), maybe_typename);
    }
  }
  return c10::nullopt;
}

void SourceImporterImpl::importClass(
    const QualifiedName& qualified_classname,
    const ClassDef& class_def,
    bool is_module) {
  // BC for TorchBind classes
  //
  // Previously we would serialize TorchBind classes as actual
  // classes with methods that delegate to things in the
  // torch.ops.* namespace. We've switched away from this and
  // now just rely on those classes being present in the binary
  // and emit code for them based on the ClassType in memory.
  //
  // TODO: remove this once we no longer have old TorchBind code
  // in production models
  {
    static QualifiedName torch_classes_qualname("__torch__.torch.classes");
    if (torch_classes_qualname.isPrefixOf(qualified_classname)) {
      return;
    }
  }
  auto class_type = ClassType::create(
      c10::QualifiedName(qualified_classname), cu_, is_module);

  std::vector<Def> methods;
  std::vector<ResolverPtr> method_resolvers;
  std::map<std::string, Def> pre_hook_def_map;
  std::map<std::string, Def> hook_def_map;
  std::map<std::string, ResolverPtr> pre_hook_resolver_map;
  std::map<std::string, ResolverPtr> hook_resolver_map;
  std::vector<Assign> attributes;
  std::vector<Assign> constants;

  // Module-specific: which attrs are parameters?
  std::unordered_set<std::string> parameter_names;
  std::unordered_set<std::string> buffer_names;
  std::unordered_set<std::string> pre_hook_names;
  std::unordered_set<std::string> hook_names;
  // used to keep track of original ordering of hooks and prehooks
  // in case any are called more than once
  std::vector<std::string> pre_hooks_order;
  std::vector<std::string> hooks_order;
  // Process statements, splitting things into attribute and method
  // definitions.
  for (const auto& statement : class_def.body()) {
    switch (statement.kind()) {
      case TK_ASSIGN: {
        const auto assign = Assign(statement);
        auto check_assign_values = [&assign](const std::string& name) {
          TORCH_CHECK(
              assign.rhs().present(),
              "Malformed assignment statement: missing values to assign in ",
              name);
        };
        switch (assign.lhs().kind()) {
          case TK_VAR: {
            const auto name = Var(assign.lhs()).name().name();
            if (name == "__parameters__") {
              // Populate the module parameter list. This is a field that
              // looks like:
              //   __parameters__ = ["foo", "bar", "baz"]
              // which tells us which attributes are module parameters.
              TORCH_INTERNAL_ASSERT(
                  is_module,
                  "Assignments in class body only "
                  "supported on modules right now");
              check_assign_values(name);
              const auto param_list = ListLiteral(assign.rhs().get()).inputs();
              for (const auto& param : param_list) {
                parameter_names.insert(StringLiteral(param).text());
              }
            } else if (name == "__annotations__") {
              // This is to initialize the annotations dict, just ignore.
              continue;
            } else if (name == "__buffers__") {
              TORCH_INTERNAL_ASSERT(
                  is_module, "Buffers only exist on modules at the moment");
              check_assign_values(name);
              const auto buffer_list = ListLiteral(assign.rhs().get()).inputs();
              for (const auto& buffer : buffer_list) {
                buffer_names.insert(StringLiteral(buffer).text());
              }
            } else if (name == "__forward_pre_hooks__") {
              TORCH_INTERNAL_ASSERT(
                  is_module,
                  "Forward pre hooks only exist on modules at the moment");
              check_assign_values(name);
              const auto pre_hook_list =
                  ListLiteral(assign.rhs().get()).inputs();
              for (const auto& pre_hook : pre_hook_list) {
                std::string pre_hook_name = StringLiteral(pre_hook).text();
                pre_hook_names.insert(pre_hook_name);
                pre_hooks_order.emplace_back(pre_hook_name);
              }
            } else if (name == "__forward_hooks__") {
              TORCH_INTERNAL_ASSERT(
                  is_module,
                  "Forward hooks only exist on modules at the moment");
              check_assign_values(name);
              const auto hook_list = ListLiteral(assign.rhs().get()).inputs();
              for (const auto& hook : hook_list) {
                std::string hook_name = StringLiteral(hook).text();
                hook_names.insert(hook_name);
                hooks_order.emplace_back(hook_name);
              }
            } else {
              if (auto fixed_up = attributeAssignmentSpecialHandlingHack(
                      qualified_classname, assign)) {
                attributes.push_back(std::move(*fixed_up));
              } else if (assign.rhs().present()) {
                // This is a constant assignment, of the form:
                // foo : Final[int] = 3
                constants.push_back(assign);
              } else {
                // This is a regular attribute assignment, of the form:
                // foo : Tensor
                attributes.push_back(assign);
              }
            }
          } break;
          case TK_SUBSCRIPT: {
            // This is a special attribute assignment where the attribute
            // is not a valid python, identifier. Looks like:
            //    __annotations__["0"] = Tensor
            const auto lhs = Subscript(assign.lhs());
            TORCH_INTERNAL_ASSERT(
                Var(lhs.value()).name().name() == "__annotations__");
            TORCH_INTERNAL_ASSERT(lhs.subscript_exprs().size() == 1);
            attributes.push_back(assign);
          } break;
          default: {
            TORCH_INTERNAL_ASSERT(
                false,
                "Unexpected statement kind in module metadata: ",
                kindToString(statement.kind()));
          }
        }
      } break;
      case TK_DEF: {
        Def def = Def(statement);
        const auto def_name = def.name().name();
        if (pre_hook_names.find(def_name) != pre_hook_names.end()) {
          pre_hook_def_map.emplace(def_name, def);
          pre_hook_resolver_map.emplace(def_name, shared_from_this());
        } else if (hook_names.find(def_name) != hook_names.end()) {
          hook_def_map.emplace(def_name, def);
          hook_resolver_map.emplace(def_name, shared_from_this());
        } else {
          methods.emplace_back(def);
          method_resolvers.push_back(shared_from_this());
        }
      } break;
      default: {
        TORCH_INTERNAL_ASSERT(
            false,
            "Unexpected statement kind in class body: ",
            kindToString(statement.kind()));
      }
    }
  }

  // Populate class attributes
  ScriptTypeParser type_parser(shared_from_this());
  for (const auto& assign : attributes) {
    switch (assign.lhs().kind()) {
      case TK_VAR: {
        const auto name = Var(assign.lhs()).name().name();
        TORCH_INTERNAL_ASSERT(name != "__parameters__");
        const auto type = assign.type().present()
            ? type_parser.parseTypeFromExpr(assign.type().get())
            : type_parser.parseTypeFromExpr(assign.rhs().get());
        const bool is_parameter = parameter_names.count(name);
        const bool is_buffer = buffer_names.count(name);
        class_type->addAttribute(name, type, is_parameter, is_buffer);
      } break;
      case TK_SUBSCRIPT: {
        const auto name =
            StringLiteral(Subscript(assign.lhs()).subscript_exprs()[0]).text();
        const auto type = assign.type().present()
            ? type_parser.parseTypeFromExpr(assign.type().get())
            : type_parser.parseTypeFromExpr(assign.rhs().get());
        const bool is_parameter = parameter_names.count(name);
        const bool is_buffer = buffer_names.count(name);
        class_type->addAttribute(name, type, is_parameter, is_buffer);
      }
    }
  }

  // Populate class constants
  for (const auto& assign : constants) {
    auto const_val = type_parser.parseClassConstant(assign);
    const auto name = Var(assign.lhs()).name().name();
    class_type->addConstant(name, const_val);
  }

  // build pre hook and hook def/resolver pairs
  // pairs are dedupped in ir_emitter.cpp's CompilationUnit::define_hooks()
  // ordering here is call order for hooks
  std::vector<Def> hooks;
  std::vector<ResolverPtr> hook_resolvers;
  for (const std::string& hook_name : hooks_order) {
    hooks.emplace_back(hook_def_map.find(hook_name)->second);
    hook_resolvers.push_back(hook_resolver_map.find(hook_name)->second);
  }
  std::vector<Def> pre_hooks;
  std::vector<ResolverPtr> pre_hook_resolvers;
  for (const std::string& pre_hook_name : pre_hooks_order) {
    pre_hooks.emplace_back(pre_hook_def_map.find(pre_hook_name)->second);
    pre_hook_resolvers.push_back(
        pre_hook_resolver_map.find(pre_hook_name)->second);
  }

  cu_->register_type(class_type);
  const auto self = SimpleSelf(class_type);
  // TODO (this will include the version number later)
  cu_->define(
      qualified_classname,
      /*properties=*/{},
      /*propResolvers=*/{},
      methods,
      method_resolvers,
      &self,
      /*shouldMangle=*/false,
      /*operator_set_version=*/version_);
  cu_->define_hooks(
      qualified_classname,
      hooks,
      hook_resolvers,
      pre_hooks,
      pre_hook_resolvers,
      &self);
}

void SourceImporterImpl::importEnum(
    const QualifiedName& qualified_name,
    const ClassDef& enum_def) {
  std::vector<at::EnumNameValue> names_values;

  TypePtr value_type = nullptr;
  auto set_or_check_type = [&value_type](
                               const TypePtr& t, const SourceRange& loc) {
    if (!value_type) {
      value_type = t;
    } else if (value_type != t) {
      throw ErrorReport(loc)
          << "Enum class with varying value types are not supported.";
    }
  };

  for (const auto& statement : enum_def.body()) {
    if (statement.kind() != TK_ASSIGN) {
      throw ErrorReport(statement.range())
          << "Unexpected statement in Enum class body: "
             "only enum attribute definitions are currently supported.";
    }

    const auto assign = Assign(statement);
    const auto name = Var(assign.lhs()).name().name();

    IValue ivalue;
    auto rhs = assign.rhs().get();
    switch (rhs.kind()) {
      case TK_STRINGLITERAL:
        ivalue = IValue(StringLiteral(rhs).text());
        set_or_check_type(StringType::get(), statement.range());
        break;
      case TK_CONST: {
        auto numeric_const = Const(rhs);
        if (numeric_const.isFloatingPoint()) {
          ivalue = IValue(numeric_const.asFloatingPoint());
          set_or_check_type(FloatType::get(), statement.range());
        } else if (numeric_const.isIntegral()) {
          ivalue = IValue(numeric_const.asIntegral());
          set_or_check_type(IntType::get(), statement.range());
        }
        break;
      }
      default:
        throw ErrorReport(rhs.range())
            << "Unsupported enum value type: " << rhs.kind()
            << ". Only Integers, Floats and Strings are supported.";
    }

    names_values.emplace_back(name, ivalue);
  }

  if (!value_type) {
    throw ErrorReport(enum_def.range())
        << "No enum values defined for " << qualified_name.qualifiedName();
  }

  auto enum_type = EnumType::create(
      qualified_name, std::move(value_type), std::move(names_values), cu_);
  cu_->register_type(enum_type);
}

void SourceImporterImpl::importNamedTuple(
    const QualifiedName& qualified_name,
    const ClassDef& named_tuple_def) {
  ScriptTypeParser type_parser(shared_from_this());
  std::vector<std::string> field_names;
  std::vector<TypePtr> field_types;
  std::vector<IValue> field_defaults;
  for (const auto& statement : named_tuple_def.body()) {
    if (statement.kind() != TK_ASSIGN) {
      throw ErrorReport(statement.range())
          << "Unexpected statement in NamedTuple body: "
             "only attribute annotations are currently supported.";
    }
    const auto assign = Assign(statement);

    auto name = Var(Assign(statement).lhs()).name().name();
    std::optional<IValue> default_val;
    if (assign.rhs().present()) {
      std::vector<IValue> parsed = type_parser.evaluateDefaults(
          assign.rhs().range(), {assign.rhs().get()}, {assign.type().get()});
      TORCH_INTERNAL_ASSERT(parsed.size() == 1);
      default_val = parsed[0];
    }

    auto type = type_parser.parseTypeFromExpr(assign.type().get());

    field_names.emplace_back(std::move(name));
    field_types.emplace_back(std::move(type));
    if (default_val) {
      field_defaults.emplace_back(std::move(*default_val));
    }
  }

  auto tt = TupleType::createNamed(
      qualified_name, field_names, field_types, field_defaults);
  cu_->register_type(tt);
}

void SourceImporterImpl::parsePossibleVersionNumber(Lexer& L) {
  // Older versions of serialization produced an op_version_set string
  // per-file We now just use a single version which is handled by
  // PyTorchStreamReader. We used to check if op_version_set was _newer_ for
  // forward compatibility reasons but now that it doesn't exist there can't
  // be a newer one, so we just discard this.
  if (L.cur().kind == TK_IDENT && L.cur().text() == "op_version_set") {
    auto range = L.cur().range;
    L.next();
    L.expect('=');
    std::string version_text = L.expect(TK_NUMBER).text();
    L.expect(TK_NEWLINE);
  }
}

// older versions of serialization required import statements,
// and defined classes file-at-a-time in import order.
// The problem is that in Python
// it is possible to construct cyclic dependencies between files even
// when there are none between individual classes. New versions of loading
// just compile class-at-a-time, so we no longer need to follow the import
// order. Future serialization may stop producing the import code.
void SourceImporterImpl::parseImports(Lexer& L) {
  while (L.nextIf(TK_IMPORT)) {
    std::ostringstream s;
    while (L.cur().kind != TK_NEWLINE) {
      s << L.cur().text();
      L.next();
    }
    L.expect(TK_NEWLINE);
  }
}

std::shared_ptr<SugaredValue> ClassNamespaceValue::attr(
    const SourceRange& loc,
    GraphFunction& m,
    const std::string& name) {
  auto fullName = c10::QualifiedName(basename_, name);
  // Could be a ClassType or NamedTuple constructor
  if (auto serializable_type = si_->findNamedType(fullName)) {
    if (auto classType = serializable_type->cast<ClassType>()) {
      return std::make_shared<ClassValue>(classType);
    } else if (auto tupleType = serializable_type->cast<TupleType>()) {
      return std::make_shared<NamedTupleConstructor>(tupleType);
    } else if (auto enumType = serializable_type->cast<EnumType>()) {
      return std::make_shared<SugaredEnumClass>(enumType);
    }
  }

  // Or it could be a free function
  if (auto fn = si_->findFunction(fullName)) {
    return std::make_shared<FunctionValue>(fn);
  }

  // If it's none of those things, assume it's another namespace
  return std::make_shared<ClassNamespaceValue>(std::move(fullName), si_);
}

SourceImporter::SourceImporter(
    // The compilation unit that will own the imported source
    std::shared_ptr<CompilationUnit> cu,
    const std::vector<IValue>* constant_table,
    SourceLoader loader,
    size_t version)
    : pImpl(std::make_shared<SourceImporterImpl>(
          std::move(cu),
          constant_table,
          std::move(loader),
          version)) {}

TypePtr SourceImporter::loadType(const QualifiedName& name) const {
  ScriptTypeParser type_parser(pImpl);
  TypePtr t = type_parser.parseType(name.qualifiedName());
  return t;
}

void SourceImporter::LEGACY_import_methods(
    const Module& mod,
    const std::shared_ptr<Source>& src) {
  pImpl->LEGACY_import_methods(mod, src);
}
SourceImporter::~SourceImporter() = default;

} // namespace torch::jit
