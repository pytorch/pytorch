// This is a simple predictor binary that loads a TorchScript CV model and runs
// a forward pass with fixed input `torch::ones({1, 3, 224, 224})`.
// It's used for end-to-end integration test for custom mobile build.

#include <iostream>
#include <string>
#include <torch/script.h>

using namespace std;

namespace {

struct MobileCallGuard {
  // AutoGrad is disabled for mobile by default.
  torch::autograd::AutoGradMode no_autograd_guard{false};
  // VariableType dispatch is not included in default mobile build. We need set
  // this guard globally to avoid dispatch error (only for dynamic dispatch).
  // Thanks to the unification of Variable class and Tensor class it's no longer
  // required to toggle the NonVariableTypeMode per op - so it doesn't hurt to
  // always set NonVariableTypeMode for inference only use case.
  torch::AutoNonVariableTypeMode non_var_guard{true};
  // Disable graph optimizer to ensure list of unused ops are not changed for
  // custom mobile build.
  torch::jit::GraphOptimizerEnabledGuard no_optimizer_guard{false};
};

void init() {
  at::globalContext().setQEngine(at::QEngine::QNNPACK);
}

torch::jit::Module loadModel(const std::string& path) {
  MobileCallGuard guard;
  auto module = torch::jit::load(path);
  module.eval();
  return module;
}

} // namespace

int main(int argc, const char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model_path>\n";
    return 1;
  }
  init();
  auto module = loadModel(argv[1]);
  auto input = torch::ones({1, 3, 224, 224});
  auto output = [&]() {
    MobileCallGuard guard;
    return module.forward({input}).toTensor();
  }();

  std::cout << std::setprecision(3) << std::fixed;
  for (int i = 0; i < 5; i++) {
    std::cout << output.data_ptr<float>()[i] << std::endl;
  }
  return 0;
}
