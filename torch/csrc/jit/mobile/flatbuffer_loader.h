#pragma once

#include <ATen/core/ivalue.h>
#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/mobile/function.h>
#include <torch/csrc/jit/mobile/interpreter.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/runtime/instruction.h>
#include <torch/csrc/jit/serialization/mobile_bytecode_generated.h> // NOLINT
#include <torch/custom_class.h>

#include <string>
#include <vector>

namespace torch {
namespace jit {

using ExtraFilesMap = std::unordered_map<std::string, std::string>;

// On high level, to produce a Module from a file on disk, we need to go
// through the follow steps:
// 1. Read: Read the file from disk -> memory
// 2. Deserialize: Parse the bytes to produce some in memory manipulable
//    structure
// 3. Module initialization: Produce mobile::Module out of the structure
//    produced in 2.
// Under this context, the structure described in 2. is
// mobile::serialization::Module

// Parse a mobile::Module from flatbuffer's in-memory Module representation.
// The caller is assumed to manage the lifetimes of Module.
// This function does step 3 described above.
TORCH_API mobile::Module initialize_mobile_module(
    mobile::serialization::Module* flatbuffer_module,
    c10::optional<at::Device> device = c10::nullopt);

// Parse a mobile::Module from raw bytes.
// ownership of data is shared to the returned Module.
// (Feel free to pass in a unique_ptr too!)
// This function does steps 2+3 described above
TORCH_API mobile::Module parse_and_initialize_mobile_module(
    std::shared_ptr<char> data,
    size_t size,
    c10::optional<at::Device> device = c10::nullopt);

// Load a mobile::Module from a filepath.
// This function does steps 1+2+3 described above.
// We need to have this as a convienience because Python
// API will need to wrap this. C++ clients should use one
// versions above.
TORCH_API mobile::Module load_mobile_module_from_file(
    const std::string& filename,
    c10::optional<at::Device> device = c10::nullopt);

TORCH_API void parseExtraFiles(
    mobile::serialization::Module* module,
    ExtraFilesMap& extra_files);

TORCH_API std::tuple<std::shared_ptr<char>, size_t> get_file_content(
    const char* filename);

TORCH_API std::tuple<std::shared_ptr<char>, size_t> get_stream_content(
    std::istream& in);

TORCH_API uint64_t get_bytecode_version(std::istream& in);
TORCH_API uint64_t get_bytecode_version(const std::string& filename);

class TORCH_API FlatbufferLoader {
 public:
  FlatbufferLoader();

  typedef IValue (
      *IValueParser)(FlatbufferLoader&, const mobile::serialization::IValue&);
  void registerIValueParser(
      mobile::serialization::IValueUnion ivalue_type,
      IValueParser parser);
  mobile::Module parseModule(mobile::serialization::Module* module);

  void extractJitSourceAndConstants(
      ExtraFilesMap* jit_sources,
      std::vector<IValue>* constants);

  typedef TypePtr (*TypeResolver)(
      const std::string& type_str,
      std::shared_ptr<CompilationUnit> cu);

  void internal_registerTypeResolver(TypeResolver type_resolver);

  IValue& getIValue(uint32_t pos) {
    TORCH_CHECK(pos < all_ivalues_.size());
    return all_ivalues_[pos];
  }

  mobile::Function* getFunction(uint32_t pos) {
    return all_functions_[pos];
  }

  ClassTypePtr getType(uint32_t pos) {
    TORCH_CHECK(pos < all_ivalues_.size());
    return all_types_[pos];
  }

  c10::Storage getStorage(uint32_t index);
  TypePtr getOrCreateTypeAnnotations(const flatbuffers::String* offset);
  ClassTypePtr getOrCreateClassTypeForObject(
      const mobile::serialization::Object* object);

  const mobile::serialization::Module* getCurrentFlatbufferInput() {
    return module_;
  }

  std::shared_ptr<mobile::CompilationUnit> mcu_;
  std::shared_ptr<CompilationUnit> cu_;

 private:
  IValue parseIValue(const mobile::serialization::IValue* ivalue);
  std::unique_ptr<mobile::Function> parseFunction(
      const mobile::serialization::Function* method);

  std::unordered_map<uint32_t, mobile::Function*> all_functions_;
  std::vector<ClassTypePtr> all_types_;
  std::unordered_set<uint32_t> initialized_types_;
  std::unordered_map<const flatbuffers::String*, TypePtr> type_annotations_;
  std::vector<bool> storage_loaded_;
  std::vector<c10::Storage> storages_;
  std::vector<IValue> all_ivalues_;
  std::array<
      IValueParser,
      static_cast<uint8_t>(mobile::serialization::IValueUnion::MAX) + 1>
      ivalue_parsers_;
  TypeResolver type_resolver_ = nullptr;
  mobile::serialization::Module* module_ = nullptr;
  bool module_parsed_ = false;
};

} // namespace jit
} // namespace torch
