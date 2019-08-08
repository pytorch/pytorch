#include "import_source.h"

#include <ATen/core/qualified_name.h>
#include <torch/csrc/jit/export.h>
#include <torch/csrc/jit/script/parser.h>
#include <torch/csrc/jit/script/resolver.h>
#include <torch/csrc/jit/script/script_type_parser.h>

namespace torch {
namespace jit {
namespace script {

struct OpsValue : public SugaredValue {
  OpsValue(size_t version) : version_(version) {}
  std::string kind() const override {
    return "ops";
  }
  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      Function& m,
      const std::string& field) override {
    return std::make_shared<BuiltinModule>(field, version_);
  }
  size_t version_;
};

struct ConstantValue : public SugaredValue {
  ConstantValue(IValue value) : value_(std::move(value)) {}
  IValue value_;
  std::string kind() const override {
    return "constant";
  }
  Value* asValue(const SourceRange& loc, Function& m) override {
    return m.graph()->insertConstant(value_);
  }
};

// Represents nested class namespaces, like `foo.bar.Baz`.
// Right now these namespaces can only contain other namespaces or a class type.
struct TORCH_API ClassNamespaceValue : public SugaredValue {
  /**
   * @param  name  The fully qualified path, which can resolve either to a
   *               namespace or a class value.
   * @param  cu    The compilation unit to search for classes in
   */
  explicit ClassNamespaceValue(
      c10::QualifiedName name,
      const CompilationUnit& cu)
      : basename_(std::move(name)), cu_(cu) {}

  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      Function& m,
      const std::string& name) override {
    auto fullName = c10::QualifiedName(basename_, name);
    if (auto serializable_type = cu_.get_type(fullName)) {
      if (auto classType = serializable_type->cast<ClassType>()) {
        return std::make_shared<ClassValue>(classType);
      } else if (auto tupleType = serializable_type->cast<TupleType>()) {
        return std::make_shared<NamedTupleConstructor>(tupleType);
      }
    }

    if (auto fn = cu_.find_function(fullName)) {
      return std::make_shared<FunctionValue>(fn);
    }

    return std::make_shared<ClassNamespaceValue>(std::move(fullName), cu_);
  }
  std::string kind() const override {
    return "Class Namespace";
  }

 private:
  c10::QualifiedName basename_;
  const CompilationUnit& cu_;
};

// This value maps attributes CONSTANTS.c0 CONSTANTS.c1 to entries
// in the 'constants' vector. This table is will be stored in a container format
// and given to the import_method when restoring the code.
struct ConstantTableValue : public SugaredValue {
  ConstantTableValue(ArrayRef<at::Tensor> constants) : constants_(constants) {}
  std::string kind() const override {
    return "CONSTANTS";
  }
  // select an attribute on it, e.g. `this.field`
  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      Function& m,
      const std::string& field) override {
    const char* field_s = field.c_str();
    char* end;
    int64_t offset = std::strtoll(field_s + 1, &end, 10);
    if (field.size() < 2 || *end != 0)
      throw ErrorReport(loc) << "invalid constant specifier: " << field;
    if (offset < 0 || size_t(offset) >= constants_.size()) {
      throw ErrorReport(loc) << "constant index " << offset
                             << " is out of bounds (constant table has "
                             << constants_.size() << " entries)";
    }
    Value* value = m.graph()->insertConstant(constants_[offset], nullptr, loc);

    // specializing tensor type on compilation messes up typing relations
    value->setType(unshapedType(value->type()));

    return std::make_shared<SimpleValue>(value);
  }

 private:
  ArrayRef<at::Tensor> constants_;
};

// A resolver that doesn't rely on Python, and understands references to model
// constants.
struct SourceResolver : public Resolver {
  explicit SourceResolver(
      std::shared_ptr<CompilationUnit> cu,
      size_t version,
      const std::vector<at::Tensor>& tensor_table)
      : cu_(std::move(cu)) {
    env_ = {
        {"torch", std::make_shared<BuiltinModule>("aten", version)},
        {"ops", std::make_shared<OpsValue>(version)},
        // Constants present in the model. Used to resolve "CONSTANTS.n" to the
        // actual value
        {"CONSTANTS", std::make_shared<ConstantTableValue>(tensor_table)},
        {"fork", std::make_shared<ForkValue>()},
        {"annotate", std::make_shared<AnnotateValue>()},
        {"uninitialized", std::make_shared<UninitializedValue>()},
        {"inf",
         std::make_shared<ConstantValue>(
             std::numeric_limits<double>::infinity())},
        {"nan",
         std::make_shared<ConstantValue>(
             std::numeric_limits<double>::quiet_NaN())},
    };
  }

  std::shared_ptr<SugaredValue> resolveValue(
      const std::string& name,
      Function& m,
      const SourceRange& loc) const override {
    auto it = env_.find(name);
    if (it != env_.end()) {
      return it->second;
    }

    if (name == "__torch__") {
      return std::make_shared<ClassNamespaceValue>(
          c10::QualifiedName(name), *cu_);
    }
    return nullptr;
  }

  TypePtr resolveType(const std::string& name, const SourceRange& loc) const override {
    return cu_->get_type(c10::QualifiedName(name));
  }

 private:
  // Compilation unit to look classes up in
  std::shared_ptr<CompilationUnit> cu_;
  std::unordered_map<std::string, std::shared_ptr<SugaredValue>> env_;
};

struct SourceImporter {
  SourceImporter(
      const std::shared_ptr<CompilationUnit> cu,
      const std::shared_ptr<Source>& src,
      const std::vector<at::Tensor>& tensor_table,
      const std::function<void(const std::string&)>& import_callback)
      : p_(src),
        cu_(cu),
        import_callback_(import_callback),
        tensor_table_(tensor_table) {
    version_ = parseVersionNumber();
    resolver_ = std::make_shared<SourceResolver>(cu_, version_, tensor_table_);
  }

  void import(const std::string& qualifier) {
    checkVersionNumber();
    auto& L = p_.lexer();

    while (L.cur().kind != TK_EOF) {
      parseImportsAndDoCallback();

      auto tk = L.cur();
      auto kind = tk.kind;
      switch (kind) {
        case TK_CLASS_DEF: {
          auto parsed_treeref = p_.parseClassLike();
          importClass(qualifier, ClassDef(parsed_treeref));
        } break;
        case TK_NAMED_TUPLE_DEF: {
          auto parsed_treeref = p_.parseClassLike();
          importNamedTuple(qualifier, NamedTupleDef(parsed_treeref));
        } break;
        case TK_DEF: {
          auto parsed_treeref = p_.parseFunction(/*is_method=*/false);
          importFunction(qualifier, Def(parsed_treeref));
        } break;
        default:
          throw ErrorReport(L.cur().range)
              << "Unexpected token in code import: " << kindToString(kind);
      }
    }
  }

  void LEGACY_importFunctions(
      const c10::optional<c10::QualifiedName>& prefix,
      const Self* self) {
    checkVersionNumber();
    parseImportsAndDoCallback();

    std::vector<Def> definitions;
    std::vector<ResolverPtr> resolvers;
    while (p_.lexer().cur().kind != TK_EOF) {
      auto def = Def(p_.parseFunction(/*is_method=*/bool(self)));
      definitions.emplace_back(def);
      resolvers.emplace_back(resolver_);
    }
    cu_->define(prefix, definitions, resolvers, self);
  }

 private:
  void importFunction(const std::string& qualifier, const Def& def) {
    std::vector<Def> definitions{def};
    std::vector<ResolverPtr> resolvers{resolver_};
    cu_->define(qualifier, definitions, resolvers, nullptr);
  }

  void importClass(const std::string& qualifier, const ClassDef& class_def) {
    bool is_module = class_def.superclass().present();
    if (is_module &&
        Var(class_def.superclass().get()).name().name() != "Module") {
      throw ErrorReport(class_def.range())
          << "Torchscript does not support class inheritance.";
    }
    const auto qualified_classname =
        QualifiedName(QualifiedName(qualifier), class_def.name().name());
    auto class_type = ClassType::create(
        c10::QualifiedName(qualified_classname), cu_, is_module);

    std::vector<Def> methods;
    std::vector<ResolverPtr> resolvers;
    std::vector<Assign> attributes;

    // Module-specific: which attrs are parameters?
    std::unordered_set<std::string> parameter_names;
    // Process statements, splitting things into attribute and method
    // definitions.
    for (const auto& statement : class_def.body()) {
      switch (statement.kind()) {
        case TK_ASSIGN: {
          const auto assign = Assign(statement);
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
                const auto param_list =
                    ListLiteral(assign.rhs().get()).inputs();
                for (const auto& param : param_list) {
                  parameter_names.insert(StringLiteral(param).text());
                }
              } else if (name == "__annotations__") {
                // This is to initialize the annotations dict, just ignore.
                continue;
              } else {
                // This is a regular attribute assignment, of the form:
                //   foo : Tensor
                attributes.push_back(assign);
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
          methods.emplace_back(Def(statement));
          resolvers.push_back(resolver_);
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
    ScriptTypeParser type_parser(resolver_);
    for (const auto& assign : attributes) {
      switch (assign.lhs().kind()) {
        case TK_VAR: {
          const auto name = Var(assign.lhs()).name().name();
          TORCH_INTERNAL_ASSERT(name != "__parameters__");
          const auto type = type_parser.parseTypeFromExpr(assign.type().get());
          const bool is_parameter = parameter_names.count(name);
          class_type->addAttribute(name, type, is_parameter);
        } break;
        case TK_SUBSCRIPT: {
          const auto name =
              StringLiteral(Subscript(assign.lhs()).subscript_exprs()[0])
                  .text();
          const auto type = type_parser.parseTypeFromExpr(assign.rhs().get());
          const bool is_parameter = parameter_names.count(name);
          class_type->addAttribute(name, type, is_parameter);
        }
      }
    }

    cu_->register_type(class_type);
    const auto self = SimpleSelf(class_type);
    cu_->define(qualified_classname, methods, resolvers, &self);
  }

  void importNamedTuple(
      const std::string& qualifier,
      const NamedTupleDef& named_tuple_def) {
    auto qualified_name =
        c10::QualifiedName(qualifier + "." + named_tuple_def.name().name());

    std::vector<std::string> field_names;
    std::vector<TypePtr> field_types;

    for (const auto& name_ident : named_tuple_def.fields()) {
      field_names.push_back(name_ident.name());
    }

    ScriptTypeParser type_parser(resolver_);
    for (const auto& maybe_type_expr : named_tuple_def.type_exprs()) {
      if (maybe_type_expr.present()) {
        field_types.push_back(
            type_parser.parseTypeFromExpr(maybe_type_expr.get()));
      } else {
        field_types.push_back(TensorType::get());
      }
    }

    auto tt = TupleType::create(
        field_types,
        qualified_name,
        TupleType::namedTupleSchemaFromNamesAndTypes(
            qualified_name, field_names, field_types));
    cu_->register_type(tt);
  }

  void checkVersionNumber() {
    // note: this cannot be called in the constructor because it may throw
    if (version_ > CURRENT_OP_VERSION_SET) {
      throw ErrorReport(p_.lexer().cur().range)
          << "Attempting to load a script generated from a newer version of "
          << "PyTorch. Maximum supported TorchScript version is "
          << CURRENT_OP_VERSION_SET
          << " but the script being loaded is version " << version_;
    }
  }

  size_t parseVersionNumber() {
    auto& L = p_.lexer();
    auto range = L.cur().range;
    auto name = L.expect(TK_IDENT).text();
    L.expect('=');
    std::string version_text = L.expect(TK_NUMBER).text();
    L.expect(TK_NEWLINE);
    auto version = Const::create(L.cur().range, version_text);
    if (name != "op_version_set")
      throw ErrorReport(range) << "expected an assignment to op_version_set";
    if (!version.isIntegral())
      throw ErrorReport(range)
          << "expected an integral version but found " << version.text();
    return size_t(version.asIntegral());
  }

  void parseImportsAndDoCallback() {
    // Gather all imports
    auto& L = p_.lexer();
    std::vector<std::string> imports;
    while (L.nextIf(TK_IMPORT)) {
      std::ostringstream s;
      while (L.cur().kind != TK_NEWLINE) {
        s << L.cur().text();
        L.next();
      }
      L.expect(TK_NEWLINE);
      const auto str = s.str();
      AT_ASSERT(!str.empty());
      imports.push_back(str);
    }

    // Call theregister_typectually compile them
    for (const auto& import : imports) {
      if (import_callback_) {
        import_callback_(import);
      }
    }
  }

  Parser p_;
  size_t version_;
  std::shared_ptr<CompilationUnit> cu_;
  const std::function<void(const std::string&)>& import_callback_;
  const std::vector<at::Tensor>& tensor_table_;
  std::shared_ptr<SourceResolver> resolver_;
};

void LEGACY_import_methods(
    const Module& mod,
    const std::shared_ptr<Source>& src,
    const std::vector<at::Tensor>& constant_table,
    const std::function<void(const std::string&)>& import_callback) {
  SourceImporter importer(
      mod.class_compilation_unit(), src, constant_table, import_callback);
  auto self = SimpleSelf(mod.type());
  importer.LEGACY_importFunctions(mod.name(), &self);
}

void import_libs(
    std::shared_ptr<CompilationUnit> cu,
    const std::string& qualifier,
    const std::shared_ptr<Source>& src,
    const std::vector<at::Tensor>& tensor_table,
    const std::function<void(const std::string&)>& import_callback) {
  SourceImporter importer(std::move(cu), src, tensor_table, import_callback);
  importer.import(qualifier);
}
} // namespace script
} // namespace jit
} // namespace torch
