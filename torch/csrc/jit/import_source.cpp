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
      Function& m,
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
  ConstantTableValue(const std::vector<at::Tensor>* constants)
      : constants_(constants) {}
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
    int64_t offset = strtoll(field_s + 1, &end, 10);
    if (field.size() < 2 || *end != 0)
      throw ErrorReport(loc) << "invalid constant specifier: " << field;
    if (offset < 0 || size_t(offset) >= constants_->size()) {
      throw ErrorReport(loc) << "constant index " << offset
                             << " is out of bounds (constant table has "
                             << constants_->size() << " entries)";
    }
    Value* value = m.graph()->insertConstant(constants_->at(offset), loc);

    // specializing tensor type on compilation messes up typing relations
    value->setType(unshapedType(value->type()));

    return std::make_shared<SimpleValue>(value);
  }

 private:
  const std::vector<at::Tensor>* constants_;
};

struct SourceImporterImpl : public Resolver,
                            std::enable_shared_from_this<SourceImporterImpl> {
  SourceImporterImpl(
      const std::shared_ptr<CompilationUnit> cu,
      const std::vector<at::Tensor>* tensor_table,
      SourceLoader source_loader,
      size_t version)
      : cu_(cu), source_loader_(std::move(source_loader)) {
    env_ = {
        {"torch", std::make_shared<BuiltinModule>("aten", version)},
        {"ops", std::make_shared<OpsValue>(version)},
        // Constants present in the model. Used to resolve "CONSTANTS.n" to the
        // actual value
        {"CONSTANTS", std::make_shared<ConstantTableValue>(tensor_table)},
        {"fork", SpecialFormValue::create(prim::fork)},
        {"annotate", SpecialFormValue::create(prim::annotate)},
        {"unchecked_cast", SpecialFormValue::create(prim::unchecked_cast)},
        {"uninitialized", SpecialFormValue::create(prim::Uninitialized)},
        {"inf",
         std::make_shared<ConstantValue>(
             std::numeric_limits<double>::infinity())},
        {"nan",
         std::make_shared<ConstantValue>(
             std::numeric_limits<double>::quiet_NaN())},
    };
  }

  TypePtr findNamedType(const QualifiedName& name) {
    parseSourceIfNeeded(name.prefix());
    auto it = to_be_defined_.find(name);
    if (it != to_be_defined_.end() && it->second->kind() == TK_CLASS_DEF) {
      ClassDef cd(it->second);
      to_be_defined_.erase(it);
      importNamedType(name.prefix(), cd);
    }
    return cu_->get_type(name);
  }

  Function* findFunction(const QualifiedName& name) {
    parseSourceIfNeeded(name.prefix());
    auto it = to_be_defined_.find(name);
    if (it != to_be_defined_.end() && it->second->kind() == TK_DEF) {
      Def d(it->second);
      to_be_defined_.erase(it);
      importFunction(name.prefix(), d);
    }
    return cu_->find_function(name);
  }

  void parseSourceIfNeeded(const std::string& qualifier) {
    // qualifier may be blank, for instance checking if __torch__ is a class.
    if (qualifier == "" || loaded_sources_.count(qualifier)) {
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
          to_be_defined_[QualifiedName(
              qualifier, parsed_treeref.name().name())] = parsed_treeref;
        } break;
        case TK_DEF: {
          auto parsed_treeref = Def(p.parseFunction(/*is_method=*/false));
          to_be_defined_[QualifiedName(
              qualifier, parsed_treeref.name().name())] = parsed_treeref;
        } break;
        default:
          throw ErrorReport(L.cur().range)
              << "Unexpected token in code import: " << kindToString(kind);
      }
    }
  }

  void LEGACY_import_methods(
      const script::Module& mod,
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
    cu_->define(prefix, definitions, resolvers, &self);
  }

  std::shared_ptr<SugaredValue> resolveValue(
      const std::string& name,
      Function& m,
      const SourceRange& loc) override {
    auto it = env_.find(name);
    if (it != env_.end()) {
      return it->second;
    }

    if (name == "__torch__") {
      return std::make_shared<ClassNamespaceValue>(
          c10::QualifiedName(name), shared_from_this());
    }
    return nullptr;
  }

  TypePtr resolveType(const std::string& name, const SourceRange& loc)
      override {
    return findNamedType(QualifiedName(name));
  }

 private:
  void importFunction(const std::string& qualifier, const Def& def) {
    std::vector<Def> definitions{def};
    std::vector<ResolverPtr> resolvers{shared_from_this()};
    cu_->define(qualifier, definitions, resolvers, nullptr);
  }

  void importNamedType(
      const std::string& qualifier,
      const ClassDef& class_def) {
    const auto qualified_name =
        QualifiedName(QualifiedName(qualifier), class_def.name().name());
    if (!class_def.superclass().present()) {
      return importClass(qualified_name, class_def, /*is_module=*/false);
    }
    const auto& superclass_name =
        Var(class_def.superclass().get()).name().name();
    if (superclass_name == "Module") {
      importClass(qualified_name, class_def, /*is_module=*/true);
    } else if (superclass_name == "NamedTuple") {
      // NamedTuples have special rules (since they are TupleTypes and not
      // ClassTypes)
      return importNamedTuple(qualified_name, class_def);
    } else if (superclass_name == "Interface") {
      cu_->define_interface(qualified_name, class_def, shared_from_this(), /*is_module=*/false);
    } else if (superclass_name == "ModuleInterface") {
      cu_->define_interface(qualified_name, class_def, shared_from_this(), /*is_module=*/true);
    } else {
      throw ErrorReport(class_def.range())
          << "Torchscript does not support class inheritance.";
    }
  }

  void importClass(
      const QualifiedName& qualified_classname,
      const ClassDef& class_def,
      bool is_module) {
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
                if (assign.rhs().present()) {
                  throw ErrorReport(assign.rhs())
                      << "Unexpected right-hand found in assignment in class body. "
                         "This is not yet supported.";
                }
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
          resolvers.push_back(shared_from_this());
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
      const QualifiedName& qualified_name,
      const ClassDef& named_tuple_def) {
    ScriptTypeParser type_parser(shared_from_this());
    std::vector<std::string> field_names;
    std::vector<TypePtr> field_types;
    for (const auto& statement : named_tuple_def.body()) {
      if (statement.kind() != TK_ASSIGN) {
        throw ErrorReport(statement.range())
            << "Unexpected statement in NamedTuple body: "
               "only attribute annotations are currently supported.";
      }

      const auto assign = Assign(statement);
      auto name = Var(assign.lhs()).name().name();
      field_names.emplace_back(std::move(name));
      auto type = type_parser.parseTypeFromExpr(assign.type().get());
      field_types.emplace_back(std::move(type));
    }

    auto tt = TupleType::createNamed(qualified_name, field_names, field_types);
    cu_->register_type(tt);
  }

  void parsePossibleVersionNumber(Lexer& L) {
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
  void parseImports(Lexer& L) {
    while (L.nextIf(TK_IMPORT)) {
      std::ostringstream s;
      while (L.cur().kind != TK_NEWLINE) {
        s << L.cur().text();
        L.next();
      }
      L.expect(TK_NEWLINE);
    }
  }

  std::shared_ptr<CompilationUnit> cu_;
  std::unordered_map<std::string, std::shared_ptr<SugaredValue>> env_;
  SourceLoader source_loader_;
  std::unordered_set<std::string> loaded_sources_;
  // named types and functions loaded from a file but not yet defined because
  // their type has not been requested yet.
  std::unordered_map<QualifiedName, TreeRef> to_be_defined_;
};

std::shared_ptr<SugaredValue> ClassNamespaceValue::attr(
    const SourceRange& loc,
    Function& m,
    const std::string& name) {
  auto fullName = c10::QualifiedName(basename_, name);
  // Could be a ClassType or NamedTuple constructor
  if (auto serializable_type = si_->findNamedType(fullName)) {
    if (auto classType = serializable_type->cast<ClassType>()) {
      return std::make_shared<ClassValue>(classType);
    } else if (auto tupleType = serializable_type->cast<TupleType>()) {
      return std::make_shared<NamedTupleConstructor>(tupleType);
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
    const std::vector<at::Tensor>* tensor_table,
    SourceLoader loader,
    size_t version)
    : pImpl(std::make_shared<SourceImporterImpl>(
          std::move(cu),
          tensor_table,
          std::move(loader),
          version)) {}

TypePtr SourceImporter::loadNamedType(const QualifiedName& name) const {
  TypePtr t = pImpl->findNamedType(name);
  TORCH_INTERNAL_ASSERT(t != nullptr);
  return t;
}

void SourceImporter::LEGACY_import_methods(
    const script::Module& mod,
    const std::shared_ptr<Source>& src) {
  pImpl->LEGACY_import_methods(mod, src);
}
SourceImporter::~SourceImporter() = default;

} // namespace script
} // namespace jit
} // namespace torch
