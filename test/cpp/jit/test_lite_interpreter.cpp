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
  auto s = std::getenv("PYTORCH_TEST_WITH_TSAN");
  if (s && strcmp(s, "1") == 0)
    return;

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

void testLiteInterpreterModel() {
  std::vector<torch::jit::IValue> inputs;
  auto L = c10::List<int64_t>({1, 1, 1});
  auto length = L.size();
  auto LL = c10::List<c10::List<int64_t>>({L});
  inputs.emplace_back(torch::jit::IValue(LL));
  auto bite_lens = c10::List<int64_t>({3});
  inputs.emplace_back(torch::jit::IValue(bite_lens));

  auto m = load("/Users/myuan/data/pytext/BI/bi-model.pt1");
  auto ref = m.forward(inputs);
  std::cout << ref << std::endl;

  std::stringstream ss;
  m._save_for_mobile(ss);
  m._save_for_mobile("/Users/myuan/data/pytext/BI/bi-model.bc");
  mobile::Module bc = _load_for_mobile(ss);
  IValue res;
  for (int i = 0; i < 3; ++i) {
    auto bcinputs = inputs;
    res = bc.run_method("forward", bcinputs);
  }
  std::cout << res << std::endl;
}
} // namespace torch
} // namespace jit
