// This is a simple predictor binary that loads a mobile CV model and runs
// a forward pass with fixed input `torch::ones({1, 3, 224, 224})` using lite interpreter.
// It's used for end-to-end integration test for custom mobile build.

#include <iostream>
#include <string>
#include <c10/util/irange.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/module.h>

using namespace std;

namespace {

torch::jit::mobile::Module loadModel(const std::string& path) {
  auto module = torch::jit::_load_for_mobile(path);
  module.eval();
  return module;
}

} // namespace

int main(int argc, const char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model_path>\n";
    return 1;
  }
  auto module = loadModel(argv[1]);
  auto input = torch::ones({1, 3, 224, 224});
  auto output = [&]() {
    return module.forward({input}).toTensor();
  }();

  std::cout << std::setprecision(3) << std::fixed;
  for (const auto i : c10::irange(5)) {
    std::cout << output.data_ptr<float>()[i] << std::endl;
  }
  return 0;
}
