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

} // namespace jit
} // namespace torch
