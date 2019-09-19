#include <torch/csrc/jit/pickle.h>
#include <torch/csrc/jit/pickler.h>
#include <torch/serialize.h>

#include <vector>

namespace torch {

std::vector<char> pickle_save(const at::IValue& ivalue) {
  return jit::pickle_save(ivalue);
}

IValue pickle_load(const std::function<bool(char*, size_t)>& reader) {
  return jit::pickle_load(reader);
}

} // namespace torch
