#include <test/cpp/jit/test_base.h>
#include <torch/csrc/jit/script/module.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/import.h>

// Tests go in torch::jit
namespace torch {
namespace jit {

void testLiteInterpreterAdd() {
  script::Module m("m");
  m.register_parameter("foo", torch::ones({}), false);
  // TODO: support default param val, which was pushed in
  // function schema's checkAndNormalizeInputs()
//  m.define(R"(
//    def add_it(self, x, b : int = 4):
//      return self.foo + x + b
//  )");
  m.define(R"(
    def add_it(self, x):
      b = 4
      return self.foo + x + b
  )");

  std::vector<IValue> inputs;
  auto minput = 5 * torch::ones({});
  inputs.emplace_back(minput);
  auto ref = m.run_method("add_it", minput);

  std::stringstream ss;
  m._save_for_mobile(ss);
  mobile::Module bc = _load_for_mobile(ss);
  IValue res;
  for (int i = 0; i < 3; ++i) {
    auto bcinputs = inputs;
    res = bc.run_method("add_it", bcinputs);
  }

  auto resd = res.toTensor().item<float>();
  auto refd = ref.toTensor().item<float>();
  AT_ASSERT(resd == refd);
}

void testLiteInterpreterConv() {
  std::vector<torch::jit::IValue> inputs;

  script::Module m("m");
  m.register_parameter("weight", torch::ones({20, 1, 5, 5}), false);
  m.register_parameter("bias", torch::ones({20}), false);
  m.define(R"(
    def forward(self, input):
      return torch._convolution(input, self.weight, self.bias, [1, 1], [0, 0], [1, 1], False, [0, 0], 1, False, False, True)
  )");

  inputs.push_back(torch::ones({1, 1, 28, 28}));

  auto outputref = m.forward(inputs).toTensor();

  std::stringstream ss;
  m._save_for_mobile(ss);
  mobile::Module bc = _load_for_mobile(ss);
  IValue res;
  for (int i = 0; i < 3; ++i) {
    auto bcinputs = inputs;
    res = bc.run_method("forward", bcinputs);
  }
  auto output = res.toTensor();
  AT_ASSERT(outputref.dim() == output.dim());
  AT_ASSERT(outputref[0][0][0][0].item<int>() == output[0][0][0][0].item<int>());
}
} // namespace torch
} // namespace jit
