#pragma once

#include <torch/nn/pimpl.h>
#include <torch/optim/optimizer.h>
#include <torch/serialize/base.h>
#include <torch/serialize/default.h>

#ifdef TORCH_USE_CEREAL
#include <torch/serialize/cereal.h>
#endif

#include <utility>

namespace torch {
namespace serialize {

template <typename ModuleType>
void save(
    const nn::ModuleHolder<ModuleType>& module,
    serialize::Writer& writer) {
  module->save(writer);
}

void save(const Tensor& tensor, serialize::Writer& writer);
void save(const optim::Optimizer& optimizer, serialize::Writer& writer);

template <typename ModuleType>
void load(nn::ModuleHolder<ModuleType>& module, serialize::Reader& reader) {
  module->load(reader);
}

void load(Tensor& tensor, serialize::Reader& reader);
void load(optim::Optimizer& optimizer, serialize::Reader& reader);
} // namespace serialize

template <
#ifdef TORCH_USE_CEREAL
    typename WriterType = serialize::CerealWriter,
#else
    typename WriterType = serialize::DefaultWriter,
#endif
    typename T,
    typename... Args>
void save(const T& value, Args&&... args) {
  WriterType writer(std::forward<Args>(args)...);
  serialize::save(value, writer);
  writer.finish();
}

template <
#ifdef TORCH_USE_CEREAL
    typename ReaderType = serialize::CerealReader,
#else
    typename ReaderType = serialize::DefaultReader,
#endif
    typename T,
    typename... Args>
void load(T& value, Args&&... args) {
  ReaderType reader(std::forward<Args>(args)...);
  serialize::load(value, reader);
  reader.finish();
}
} // namespace torch
