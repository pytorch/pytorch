#include <torch/cuda.h>
#include <torch/script.h>

#include <string>

#include "custom_backend.h"

// Load a module lowered for the custom backend from \p path and test that
// it can be executed and produces correct results.
void load_serialized_lowered_module_and_execute(const std::string& path) {
  torch::jit::Module module = torch::jit::load(path);
  // The custom backend is hardcoded to compute f(a, b) = (a + b, a - b).
  auto tensor = torch::ones(5);
  std::vector<torch::jit::IValue> inputs{tensor, tensor};
  auto output = module.forward(inputs);
  AT_ASSERT(output.isTuple());
  auto output_elements = output.toTuple()->elements();
  for (auto& e : output_elements) {
    AT_ASSERT(e.isTensor());
  }
  AT_ASSERT(output_elements.size(), 2);
  AT_ASSERT(output_elements[0].toTensor().allclose(tensor + tensor));
  AT_ASSERT(output_elements[1].toTensor().allclose(tensor - tensor));
}

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr
        << "usage: test_custom_backend <path-to-exported-script-module>\n";
    return -1;
  }
  const std::string path_to_exported_script_module = argv[1];

  std::cout << "Testing " << torch::custom_backend::getBackendName() << "\n";
  load_serialized_lowered_module_and_execute(path_to_exported_script_module);

  std::cout << "OK\n";
  return 0;
}
