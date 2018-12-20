#include <google/protobuf/util/json_util.h>
#include <google/protobuf/util/type_resolver_util.h>

#include <torch/csrc/jit/import.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/assertions.h>

#include <caffe2/core/types.h>
#include <caffe2/proto/caffe2_pb.h>
#include <caffe2/proto/torch_pb.h>
#include <caffe2/serialize/inline_container.h>
#include <caffe2/serialize/script_module_deserializer.h>

#include <ATen/ATen.h>

#include <vector>
#include <string>
#include <fstream>

namespace torch { namespace jit {

void import_ir_module(
    ModuleLookup module_lookup,
    std::istream& in,
    c10::optional<at::Device> device) {
  auto reader = caffe2::make_unique<PyTorchStreamReader>(&in);
  ScriptModuleDeserializer<PyTorchStreamReader> deserializer(std::move(reader));
  deserializer.deserialize(module_lookup, device);
}

void import_ir_module(
    ModuleLookup module_lookup,
    const std::string& filename,
    c10::optional<at::Device> device) {
  auto reader = caffe2::make_unique<PyTorchStreamReader>(filename);
  ScriptModuleDeserializer<PyTorchStreamReader> deserializer(std::move(reader));
  deserializer.deserialize(module_lookup, device);
}

std::shared_ptr<script::Module> load(std::istream& in,
    c10::optional<at::Device> device) {
  auto module = std::make_shared<script::Module>();

  auto module_lookup = [&](const std::vector<std::string>& qualified_name) {
    std::shared_ptr<script::Module> curr = module;
    for (const auto& name : qualified_name) {
      if (curr->find_module(name) == nullptr) {
        curr->register_module(name, std::make_shared<script::Module>());
      }
      curr = curr->get_module(name);
    }
    return curr;
  };

  auto reader = caffe2::make_unique<PyTorchStreamReader>(&in);
  ScriptModuleDeserializer<PyTorchStreamReader> deserializer(std::move(reader));
  deserializer.deserialize(module_lookup, device);

  return module;
}

std::shared_ptr<script::Module> load(const std::string& filename,
    c10::optional<at::Device> device) {
  std::ifstream in(filename, std::ios_base::binary);

  AT_CHECK(! in.fail(), "load: could not open file ", filename);

  auto module = load(in, device);

  return module;
}

}}
