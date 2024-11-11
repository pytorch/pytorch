#pragma once

#include <istream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <ATen/core/ivalue.h>
#include <c10/core/Device.h>
#include <c10/macros/Macros.h>
#include <torch/csrc/jit/mobile/module.h>
#include <optional>

/**
 * Defines the public API for loading flatbuffer-serialized mobile modules.
 * Note that this header must not include or depend on flatbuffer-defined
 * types, to avoid leaking those details to PyTorch clients.
 */

namespace torch::jit {

/// All non-copied data pointers provided to `parse_and_initialize_*` functions
/// must be aligned to this boundary. Since the Module will point directly into
/// the data, this alignment is necessary to ensure that certain types/structs
/// are properly aligned.
constexpr size_t kFlatbufferDataAlignmentBytes = 16;

/// Maps file names to file contents.
using ExtraFilesMap = std::unordered_map<std::string, std::string>;

// On high level, to produce a Module from a file on disk, we need to go
// through the follow steps:
// 1. Read: Read the file from disk -> memory
// 2. Deserialize: Parse the bytes to produce some in memory manipulable
//    structure
// 3. Module initialization: Produce mobile::Module out of the structure
//    produced in 2.
// Under this context, the structure described in 2. is the flatbuffer-defined
// type mobile::serialization::Module. However, this step/type is not visible in
// the public API.

// Parse a mobile::Module from raw bytes.
//
// This function does steps 2+3 described above.
//
// Does not take ownership of `data`; if you want it to take ownership, see the
// shared_ptr overload of this function.
//
// If should_copy_tensor_memory is true, then the returned module will NOT have
// refences to `data`, so `data` can be freed immediately.
//
// If should_copy_tensor_memory is false, then returned module will have tensors
// that points inside of `data`; the caller will need to make sure that `data`
// outlives the returned Module. Also, `data` must be aligned to
// kFlatbufferDataAlignmentBytes.
TORCH_API mobile::Module parse_and_initialize_mobile_module(
    void* data,
    size_t size, // of `data`, in bytes.
    std::optional<at::Device> device = std::nullopt,
    ExtraFilesMap* extra_files = nullptr,
    bool should_copy_tensor_memory = false);

// Parse a mobile::Module from raw bytes.
//
// This function does steps 2+3 described above.
//
// The returned Module holds a reference to `data`, which must be aligned to
// kFlatbufferDataAlignmentBytes.
//
// If you do not want the Module to hold a reference to `data`, see the raw
// pointer overload of this function.
TORCH_API mobile::Module parse_and_initialize_mobile_module(
    std::shared_ptr<char> data,
    size_t size, // of `data`, in bytes.
    std::optional<at::Device> device = std::nullopt,
    ExtraFilesMap* extra_files = nullptr);

// Parse a mobile::Module from raw bytes, also returning JIT-related metadata.
//
// This is the same as parse_and_initialize_mobile_module() except that it also
// extracts JIT source files and constants. Can be used to construct a
// jit::Module.
TORCH_API mobile::Module parse_and_initialize_mobile_module_for_jit(
    void* data,
    size_t size, // of `data`, in bytes.
    ExtraFilesMap& jit_sources,
    std::vector<IValue>& jit_constants,
    std::optional<at::Device> device = std::nullopt,
    ExtraFilesMap* extra_files = nullptr);

// Load a mobile::Module from a filepath.
//
// This function does steps 1+2+3 described above.
//
// We need to have this as a convienience because Python API will need to wrap
// this. C++ clients should use one of the versions of
// parse_and_initialize_mobile_module() so they can manage the raw data more
// directly.
TORCH_API mobile::Module load_mobile_module_from_file(
    const std::string& filename,
    std::optional<at::Device> device = std::nullopt,
    ExtraFilesMap* extra_files = nullptr);

TORCH_API uint64_t get_bytecode_version(std::istream& in);
TORCH_API uint64_t get_bytecode_version(const std::string& filename);
TORCH_API uint64_t get_bytecode_version_from_bytes(char* flatbuffer_content);

TORCH_API mobile::ModuleInfo get_module_info_from_flatbuffer(
    char* flatbuffer_content);

// The methods below are less efficient because it need to read the stream in
// its entirity to a buffer
TORCH_API mobile::Module load_mobile_module_from_stream_with_copy(
    std::istream& in,
    std::optional<at::Device> device = std::nullopt,
    ExtraFilesMap* extra_files = nullptr);

TORCH_API mobile::Module parse_flatbuffer_no_object(
    std::shared_ptr<char> data,
    size_t size,
    std::optional<at::Device> device);

TORCH_API mobile::Module parse_and_initialize_mobile_module(
    void* data,
    size_t,
    std::optional<at::Device>,
    ExtraFilesMap* extra_files,
    bool should_copy_tensor_memory);

// no op, TODO(qihan) delete
TORCH_API bool register_flatbuffer_loader();

} // namespace torch::jit
