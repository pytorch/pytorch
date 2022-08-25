#include <torch/csrc/jit/serialization/flatbuffer_serializer_jit.h>

#ifdef FLATBUFFERS_VERSION_MAJOR
#error "flatbuffer_serializer_jit.h must not include any flatbuffers headers"
#endif // FLATBUFFERS_VERSION_MAJOR

#include <torch/csrc/jit/mobile/file_format.h>
#include <torch/csrc/jit/mobile/flatbuffer_loader.h>
#include <torch/csrc/jit/operator_upgraders/upgraders_entry.h>
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
  populate_upgraders_graph_map();
  ExtraFilesMap jit_files;
  std::vector<IValue> jit_constants;
  mobile::Module mobilem = parse_and_initialize_mobile_module_for_jit(
      data.get(), size, jit_files, jit_constants, device, &extra_files);

  Module m = jitModuleFromSourceAndConstants(
      mobilem._ivalue(),
      jit_files,
      jit_constants,
      static_cast<int32_t>(mobilem.bytecode_version()));
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
  ofile.write(
      reinterpret_cast<char*>(buffer->data()), buffer->size()); // NOLINT
  ofile.close();
}

DetachedBuffer::UniqueDetachedBuffer save_jit_module_to_bytes(
    const Module& module,
    const ExtraFilesMap& extra_files) {
  ExtraFilesMap jitfiles;
  std::vector<IValue> constants;
  jitModuleToPythonCodeAndConstants(module, &jitfiles, &constants);
  CompilationOptions options = getOptionsFromGlobal();
  mobile::Module mobilem = jitModuleToMobile(module, options);
  return save_mobile_module_to_bytes(mobilem, extra_files, jitfiles, constants);
}

static void save_jit_module_to_write_func(
    const Module& module,
    const ExtraFilesMap& extra_files,
    bool save_mobile_debug_info,
    const std::function<size_t(const void*, size_t)>& writer_func) {
  (void)save_mobile_debug_info;
  auto buffer = save_jit_module_to_bytes(module, extra_files);
  writer_func(reinterpret_cast<void*>(buffer->data()), buffer->size());
}

bool register_flatbuffer_all() {
  (void)register_flatbuffer_loader();
  (void)register_flatbuffer_serializer();
  _save_jit_module_to = save_jit_module_to_write_func;
  _load_jit_module_from_flatbuffer_bytes = parse_and_initialize_jit_module;
  return true;
}

#if !defined(__APPLE__)
const bool kFlatbufferSerializerJitInitialized = register_flatbuffer_all();
#endif

} // namespace jit
} // namespace torch
