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
      const std::vector<at::Tensor>& constant_table)
      : cu_(cu) {
    env_ = {
        {"torch", std::make_shared<BuiltinModule>("aten", version)},
        {"ops", std::make_shared<OpsValue>(version)},
        // Constants present in the model. Used to resolve "CONSTANTS.n" to the
        // actual value
        {"CONSTANTS", std::make_shared<ConstantTableValue>(constant_table)},
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
      const std::vector<at::Tensor>& constant_table,
      const std::function<void(const std::string&)>& import_callback)
      : p_(src),
        cu_(cu),
        import_callback_(import_callback),
        constant_table_(constant_table) {
    version_ = parseVersionNumber();
    resolver_ =
        std::make_shared<SourceResolver>(cu_, version_, constant_table_);
  }

  void checkVersionNumber() {
    // note: this cannot be called in the constructor because it may throw
    if (version_ > CURRENT_OP_VERSION_SET) {
      throw ErrorReport(p_.lexer().cur().range)
          << "Attempting to load a script generated from a newer version of PyTorch. Maximum supported TorchScript version is "
          << CURRENT_OP_VERSION_SET
          << " but the script being loaded is version " << version_;
    }
  }

  void importLibs(std::shared_ptr<CompilationUnit> owner, const std::string& class_qualifier) {
    checkVersionNumber();
    auto& L = p_.lexer();

    while (L.cur().kind != TK_EOF) {
      parseImportsAndDoCallback();

      auto parsed_treeref = p_.parseClassLike();
      if (parsed_treeref->kind() == TK_CLASS_DEF) {
        auto class_def = ClassDef(parsed_treeref);
        const auto qualified_classname = QualifiedName(
            QualifiedName(class_qualifier), class_def.name().name());

        std::vector<Def> definitions;
        std::vector<ResolverPtr> resolvers;
        for (const auto& method_def : class_def.defs()) {
          definitions.emplace_back(method_def);
          resolvers.emplace_back(resolver_);
        }

        auto class_type =
            ClassType::create(c10::QualifiedName(qualified_classname), owner);
        owner->register_class(class_type);
        const auto self = SimpleSelf(class_type);
        owner->define(qualified_classname, definitions, resolvers, &self);
      } else if (parsed_treeref->kind() == TK_NAMED_TUPLE_DEF) {
        auto named_tuple_def = NamedTupleDef(parsed_treeref);

        auto qualified_name = c10::QualifiedName(
            class_qualifier + "." + named_tuple_def.name().name());

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
            TupleType::namedTupleSchemaFromNamesAndTypes(qualified_name, field_names, field_types));
        owner->register_class(tt);
      } else {
        TORCH_INTERNAL_ASSERT(
            false,
            "Got an unrecognized type from "
            "parseClassLike");
      }
    }
  }

  void importFunctions(
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

    // Call the callback to actually compile them
    for (const auto& import : imports) {
      if (import_callback_) {
        import_callback_(import);
      }
    }
  }

 private:
  Parser p_;
  size_t version_;
  std::shared_ptr<CompilationUnit> cu_;
  const std::function<void(const std::string&)>& import_callback_;
  const std::vector<at::Tensor>& constant_table_;
  std::shared_ptr<SourceResolver> resolver_;
};

void import_functions(
    const c10::optional<c10::QualifiedName>& prefix,
    std::shared_ptr<CompilationUnit> cu,
    const std::shared_ptr<Source>& src,
    const std::vector<at::Tensor>& constant_table,
    const Self* self,
    const std::function<void(const std::string&)>& import_callback) {
  SourceImporter importer(cu, src, constant_table, import_callback);
  importer.importFunctions(prefix, self);
}

void import_methods(
    const Module& mod,
    const std::shared_ptr<Source>& src,
    const std::vector<at::Tensor>& constant_table,
    const std::function<void(const std::string&)>& import_callback) {
  auto self = SimpleSelf(mod.type());
  import_functions(
      mod.name(),
      mod.class_compilation_unit(),
      src,
      constant_table,
      &self,
      import_callback);
}

void import_libs(
    std::shared_ptr<CompilationUnit> cu,
    const std::string& class_qualifier,
    const std::shared_ptr<Source>& src,
    const std::vector<at::Tensor>& constant_table,
    const std::function<void(const std::string&)>& import_callback) {
  SourceImporter importer(cu, src, constant_table, import_callback);
  importer.importLibs(cu, class_qualifier);
}

} // namespace script
} // namespace jit
} // namespace torch
