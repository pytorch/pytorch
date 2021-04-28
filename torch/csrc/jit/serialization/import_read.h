#pragma once

#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/serialization/unpickler.h>

namespace torch {
namespace jit {

TORCH_API IValue readArchiveAndTensors(
    const std::string& archive_name,
    c10::optional<TypeResolver> type_resolver,
    c10::optional<ObjLoader> obj_loader,
    c10::optional<at::Device> device,
    caffe2::serialize::PyTorchStreamReader& stream_reader);

} // namespace jit
} // namespace torch
