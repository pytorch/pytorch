#pragma once

#include <torch/nn/module.h>
#include <torch/nn/pimpl.h>
#include <torch/serialize/archive.h>
#include <torch/tensor.h>

#include <string>
#include <utility>

namespace torch {
namespace optim {
class Optimizer;
} // namespace optim
} // namespace torch

namespace torch {
namespace serialize {
template <typename ModuleType>
void save(const nn::ModuleHolder<ModuleType>& module, OutputArchive& archive) {
  module->save(archive);
}

template <typename ModuleType>
void load(nn::ModuleHolder<ModuleType>& module, InputArchive& archive) {
  module->load(archive);
}

void save(const Tensor& tensor, OutputArchive& archive);

void save(const optim::Optimizer& optimizer, OutputArchive& archive);
void load(optim::Optimizer& optimizer, InputArchive& archive);
} // namespace serialize

template <typename T>
void save(const T& value, const std::string& filename) {
  serialize::OutputArchive archive;
  serialize::save(value, archive);
  serialize::save_to_file(archive, filename);
}

template <typename T>
void load(T& value, const std::string& filename) {
  serialize::InputArchive archive = serialize::load_from_file(filename);
  serialize::load(value, archive);
}

Tensor load(const std::string& filename);
} // namespace torch
