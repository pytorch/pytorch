#include <torch/csrc/jit/serialization/flatbuffer_serializer_jit.h>

#include <torch/csrc/jit/mobile/flatbuffer_loader.h>
#include <torch/csrc/jit/serialization/export.h>
#include <torch/csrc/jit/serialization/export_bytecode.h>
#include <torch/csrc/jit/serialization/flatbuffer_serializer.h>
#include <torch/csrc/jit/serialization/import.h>

namespace torch {
namespace jit {

Module parse_and_initialize_jit_module(
    std::shared_ptr<char> data,
    size_t size,
    ExtraFilesMap& extra_files,
    c10::optional<at::Device> device) {
  auto* flatbuffer_module = mobile::serialization::GetMutableModule(data.get());
  FlatbufferLoader loader;
  mobile::Module mobilem = loader.parseModule(flatbuffer_module);
  parseExtraFiles(flatbuffer_module, extra_files);
  ExtraFilesMap files;
  std::vector<IValue> constants;
  loader.extractJitSourceAndConstants(&files, &constants);
  Module m = jitModuleFromSourceAndConstants(
      mobilem._ivalue(),
      files,
      constants,
      flatbuffer_module->bytecode_version());
  m.set_delete_memory(data);
  return m;
}

Module load_jit_module_from_file(
    const std::string& filename,
    ExtraFilesMap& extra_files,
    c10::optional<at::Device> device) {
  auto data = get_file_content(filename.c_str());
  return parse_and_initialize_jit_module(
      std::move(std::get<0>(data)), std::get<1>(data), extra_files, device);
}

Module load_jit_module_from_stream(
    std::istream& in,
    ExtraFilesMap& extra_files,
    c10::optional<at::Device> device) {
  auto data = get_stream_content(in);
  return parse_and_initialize_jit_module(
      std::move(std::get<0>(data)), std::get<1>(data), extra_files, device);
}

void save_jit_module(
    const Module& module,
    const std::string& filename,
    const ExtraFilesMap& extra_files) {
  auto buffer = save_jit_module_to_bytes(module, extra_files);
  std::fstream ofile(filename, std::ios::binary | std::ios::out);
  ofile.write(reinterpret_cast<char*>(buffer.data()), buffer.size()); // NOLINT
  ofile.close();
}

flatbuffers::DetachedBuffer save_jit_module_to_bytes(
    const Module& module,
    const ExtraFilesMap& extra_files) {
  ExtraFilesMap jitfiles;
  std::vector<IValue> constants;
  jitModuleToPythonCodeAndConstants(module, &jitfiles, &constants);
  CompilationOptions options;
  mobile::Module mobilem = jitModuleToMobile(module, options);
  return save_mobile_module_to_bytes(mobilem, extra_files, jitfiles, constants);
}

} // namespace jit
} // namespace torch
