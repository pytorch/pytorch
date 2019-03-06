#include <torch/csrc/jit/import_method.h>
#include <torch/csrc/jit/script/parser.h>

namespace torch {
namespace jit {

// this is a much simpler accessor that only handles modules, parameters, and
// and methods. It does not depend on python to work.
struct ModuleAccessorValue : public script::SugaredValue {
  ModuleAccessorValue(std::shared_ptr<script::Module> module)
      : module(std::move(module)) {}
  std::string kind() const override {
    return "module";
  }
  // select an attribute on it, e.g. `this.field`
  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      script::Method& m,
      const std::string& field) override {
    if (script::NamedModule* v = module->find_module(field)) {
      return std::make_shared<ModuleAccessorValue>(v->module);
    } else if (script::NamedIValue* v = module->find_parameter(field)) {
      return std::make_shared<script::SimpleValue>(
          m.get_or_add_parameter(v->slot()));
    } else if (script::NamedIValue* v = module->find_buffer(field)) {
      return std::make_shared<script::SimpleValue>(
          m.get_or_add_parameter(v->slot()));
    } else if (script::Method* m = module->find_method(field)) {
      return std::make_shared<script::MethodValue>(shared_from_this(), *m);
    } else {
      throw script::ErrorReport(loc) << "unknown attr: " << field;
    }
  }

 private:
  std::shared_ptr<script::Module> module;
};

struct OpsValue : public script::SugaredValue {
  OpsValue(size_t version) : version_(version) {}
  std::string kind() const override {
    return "ops";
  }
  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      script::Method& m,
      const std::string& field) override {
    return std::make_shared<script::BuiltinModule>(field, version_);
  }
  size_t version_;
};

struct ConstantValue : public script::SugaredValue {
  ConstantValue(IValue value) : value_(std::move(value)) {}
  IValue value_;
  std::string kind() const override {
    return "constant";
  }
  Value* asValue(const SourceRange& loc, script::Method& m) override {
    return m.graph()->insertConstant(value_);
  }
};

// This value maps attributes CONSTANTS.c0 CONSTANTS.c1 to entries
// in the 'constants' vector. This table is will be stored in a container format
// and given to the import_method when restoring the code.
struct ConstantTableValue : public script::SugaredValue {
  ConstantTableValue(ArrayRef<at::Tensor> constants) : constants_(constants) {}
  std::string kind() const override {
    return "CONSTANTS";
  }
  // select an attribute on it, e.g. `this.field`
  std::shared_ptr<SugaredValue> attr(
      const SourceRange& loc,
      script::Method& m,
      const std::string& field) override {
    const char* field_s = field.c_str();
    char* end;
    int64_t offset = std::strtoll(field_s + 1, &end, 10);
    if (field.size() < 2 || *end != 0)
      throw script::ErrorReport(loc) << "invalid constant specifier: " << field;
    if (offset < 0 || size_t(offset) >= constants_.size()) {
      throw script::ErrorReport(loc) << "constant index " << offset
                                     << " is out of bounds (constant table has "
                                     << constants_.size() << " entries).";
    }
    Value* value = m.graph()->insertConstant(constants_[offset], nullptr, loc);
    return std::make_shared<script::SimpleValue>(value);
  }

 private:
  ArrayRef<at::Tensor> constants_;
};

static size_t parseVersionNumber(script::Lexer& L) {
  auto range = L.cur().range;
  auto name = L.expect(script::TK_IDENT).text();
  L.expect('=');
  std::string version_text = L.expect(script::TK_NUMBER).text();
  L.expect(script::TK_NEWLINE);
  auto version = script::Const::create(L.cur().range, version_text);
  if (name != "op_version_set")
    throw script::ErrorReport(range)
        << "expected an assignment to op_version_set";
  if (!version.isIntegral())
    throw script::ErrorReport(range)
        << "expected an integral version but found " << version.text();
  return size_t(version.asIntegral());
}

void import_methods(
    const std::shared_ptr<script::Module>& mod,
    const std::string& src,
    const std::vector<at::Tensor>& constant_table) {
  script::Parser p(src);

  size_t version = parseVersionNumber(p.lexer());

  std::unordered_map<std::string, std::shared_ptr<script::SugaredValue>> env = {
      {"torch", std::make_shared<script::BuiltinModule>("aten", version)},
      {"ops", std::make_shared<OpsValue>(version)},
      {"CONSTANTS", std::make_shared<ConstantTableValue>(constant_table)},
      {"fork", std::make_shared<script::ForkValue>()},
      {"annotate", std::make_shared<script::AnnotateValue>()},
      {"inf",
       std::make_shared<ConstantValue>(
           std::numeric_limits<double>::infinity())},
      {"nan",
       std::make_shared<ConstantValue>(
           std::numeric_limits<double>::quiet_NaN())},
  };

  auto resolver =
      [&](const std::string& name,
          script::Method& m,
          const SourceRange& loc) -> std::shared_ptr<script::SugaredValue> {
    auto it = env.find(name);
    if (it == env.end())
      return nullptr;
    return it->second;
  };

  std::vector<script::Def> definitions;
  std::vector<script::Resolver> resolvers;

  while (p.lexer().cur().kind != script::TK_EOF) {
    auto def = script::Def(p.parseFunction(/*is_method=*/true));
    definitions.emplace_back(def);
    resolvers.emplace_back(resolver);
  }
  auto self = std::make_shared<ModuleAccessorValue>(mod);
  script::defineMethodsInModule(mod, definitions, resolvers, self);
}

} // namespace jit
} // namespace torch
