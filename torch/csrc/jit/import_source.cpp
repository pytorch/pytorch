#include "import_source.h"

#include <torch/csrc/jit/script/parser.h>
#include <torch/csrc/jit/script/resolver.h>

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
                             << constants_.size() << " entries).";
    }
    Value* value = m.graph()->insertConstant(constants_[offset], nullptr, loc);
    return std::make_shared<SimpleValue>(value);
  }

 private:
  ArrayRef<at::Tensor> constants_;
};

// A resolver that doesn't rely on Python, and understands references to model
// constants.
struct SourceResolver : public Resolver {
  explicit SourceResolver(
      size_t version,
      const std::vector<at::Tensor>& constant_table) {
    env_ = {
        {"torch", std::make_shared<BuiltinModule>("aten", version)},
        {"ops", std::make_shared<OpsValue>(version)},
        // Constants present in the model. Used to resolve "CONSTANTS.n" to the
        // actual value
        {"CONSTANTS", std::make_shared<ConstantTableValue>(constant_table)},
        {"fork", std::make_shared<ForkValue>()},
        {"annotate", std::make_shared<AnnotateValue>()},
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
    if (it == env_.end()) {
      return nullptr;
    }
    return it->second;
  }

  TypePtr resolveType(const std::string& name) const override {
    return ClassType::get(name);
  }

 private:
  std::unordered_map<std::string, std::shared_ptr<SugaredValue>> env_;
};

// Helper that contains the state for a parsing a TorchScript source string.
struct SourceImporter {
  SourceImporter(
      const std::string& src,
      const std::vector<at::Tensor>& constant_table)
      : parser_(src) {
    const auto version = parseVersionNumber();
    resolver_ = std::make_shared<SourceResolver>(version, constant_table);
  }

  Parser parser_;
  ResolverPtr resolver_;

  size_t parseVersionNumber() {
    auto& L = parser_.lexer();
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
};

void import_functions(
    CompilationUnit& cu,
    const std::string& src,
    const std::vector<at::Tensor>& constant_table,
    const Self& self) {
  SourceImporter importer(src, constant_table);
  auto& p = importer.parser_;

  std::vector<Def> definitions;
  std::vector<ResolverPtr> resolvers;
  while (p.lexer().cur().kind != TK_EOF) {
    auto def = Def(p.parseFunction(/*is_method=*/bool(self)));
    definitions.emplace_back(def);
    resolvers.emplace_back(importer.resolver_);
  }
  cu.define(definitions, resolvers, self);
}

void import_methods(
    const std::shared_ptr<Module>& mod,
    const std::string& src,
    const std::vector<at::Tensor>& constant_table) {
  auto self = [&](Value* v) {
    v->setType(mod->module_object()->type());
    return std::make_shared<SimpleValue>(v);
  };
  import_functions(
      mod->module_object()->type()->compilation_unit(),
      src,
      constant_table,
      self);
}

void import_libs(
    const std::string& src,
    const std::vector<at::Tensor>& constant_table) {
  SourceImporter importer(src, constant_table);
  auto& p = importer.parser_;

  while (p.lexer().cur().kind != TK_EOF) {
    std::vector<Def> definitions;
    std::vector<ResolverPtr> resolvers;
    auto class_def = ClassDef(p.parseClass());
    for (const auto& method_def : class_def.defs()) {
      definitions.emplace_back(method_def);
      resolvers.emplace_back(importer.resolver_);
    }

    auto cu = std::make_shared<CompilationUnit>();
    auto class_type = ClassType::create(class_def.name().name(), cu);
    auto self = [&](Value* v) {
      v->setType(class_type);
      return std::make_shared<SimpleValue>(v);
    };
    cu->define(definitions, resolvers, self);
  }
}

} // namespace script
} // namespace jit
} // namespace torch
