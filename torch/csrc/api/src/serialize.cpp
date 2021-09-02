#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/csrc/jit/serialization/pickler.h>
#include <torch/serialize.h>

#include <vector>
#include <iostream>


namespace torch {

std::vector<char> pickle_save(const at::IValue& ivalue) {
  return save(ivalue);
}

torch::IValue pickle_load(const std::vector<char>& data) {
  return load(data);
}

std::vector<char> save(const torch::IValue& ivalue) {
  return jit::pickle_save(ivalue);
}

torch::IValue load(const std::vector<char>& data) {
  return jit::pickle_load(data);
}

void save(const torch::IValue& ivalue, const std::string& filename) {
  auto bytes = jit::pickle_save(ivalue);
  std::ofstream outfile(filename, std::ios::out | std::ios::binary);
  outfile.write(&bytes[0], bytes.size());
}

torch::IValue load(const std::string& filename) {
  std::ifstream input_stream(filename);
  std::vector<char> input;
  input.insert(
      input.begin(),
      std::istream_iterator<char>(input_stream),
      std::istream_iterator<char>());

  for (const auto& x : input) {
    std::cout << "c " << std::hex << int(x) << "\n";
  }
  return jit::pickle_load(input);
}

} // namespace torch
