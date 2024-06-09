#pragma once

#include <ATen/core/ivalue_inl.h>
#include <ATen/core/qualified_name.h>
#include <c10/util/Optional.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/frontend/parser.h>
#include <torch/csrc/jit/frontend/resolver.h>
#include <torch/csrc/jit/frontend/script_type_parser.h>
#include <torch/csrc/jit/frontend/source_range.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/serialization/export.h>
#include <torch/custom_class.h>
#include <functional>
#include <memory>
#include <regex>
#include <string>
#include <vector>

namespace torch {
namespace jit {

using SourceLoader = std::function<std::shared_ptr<Source>(const std::string&)>;

struct SourceImporterImpl : public Resolver,
                            std::enable_shared_from_this<SourceImporterImpl> {
  SourceImporterImpl(
      std::shared_ptr<CompilationUnit> cu,
      const std::vector<at::IValue>* constant_table,
      SourceLoader source_loader,
      size_t version);
  TypePtr findNamedType(const QualifiedName& name);
  Function* findFunction(const QualifiedName& name);
  void parseSourceIfNeeded(const std::string& qualifier);
  void LEGACY_import_methods(
      const Module& mod,
      const std::shared_ptr<Source>& src);

  std::shared_ptr<SugaredValue> resolveValue(
      const std::string& name,
      GraphFunction& m,
      const SourceRange& loc) override;
  TypePtr resolveType(const std::string& name, const SourceRange& loc) override;

 private:
  void importFunction(const std::string& qualifier, const Def& def);
  void importNamedType(const std::string& qualifier, const ClassDef& class_def);
  c10::optional<Assign> attributeAssignmentSpecialHandlingHack(
      const QualifiedName& qualified_classname,
      const Assign& assign);
  void importClass(
      const QualifiedName& qualified_classname,
      const ClassDef& class_def,
      bool is_module);
  void importEnum(
      const QualifiedName& qualified_name,
      const ClassDef& enum_def);
  void importNamedTuple(
      const QualifiedName& qualified_name,
      const ClassDef& named_tuple_def);

  void parsePossibleVersionNumber(Lexer& L);

  void parseImports(Lexer& L);

  std::shared_ptr<CompilationUnit> cu_;
  std::unordered_map<std::string, std::shared_ptr<SugaredValue>> env_;
  SourceLoader source_loader_;
  c10::optional<size_t> version_ = c10::nullopt;
  std::unordered_set<std::string> loaded_sources_;
  // named types and functions loaded from a file but not yet defined because
  // their type has not been requested yet.
  std::unordered_map<QualifiedName, TreeRef> to_be_defined_;
};

// Given a directory of serialized TorchScript sources,
// This class allows the loading of individual named types in source.
// Resolves the dependencies between source files and parses
// the source files as necessary.

struct TORCH_API SourceImporter {
  SourceImporter(
      // The compilation unit that will own the imported source
      std::shared_ptr<CompilationUnit> cu,
      const std::vector<at::IValue>* constant_table,
      SourceLoader loader,
      size_t version);

  TypePtr loadType(const QualifiedName& name) const;

  // Add the methods defined in `src` to the module `mod`, using SourceImporter
  // to resolve any classes via loadType
  void LEGACY_import_methods(
      const Module& mod,
      const std::shared_ptr<Source>& src);
  ~SourceImporter();

 private:
  std::shared_ptr<SourceImporterImpl> pImpl;
};

} // namespace jit
} // namespace torch
