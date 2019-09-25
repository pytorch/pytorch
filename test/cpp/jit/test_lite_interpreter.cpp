#include <test/cpp/jit/test_base.h>
#include <torch/csrc/jit/script/module.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/import.h>
#include <torch/types.h>
#include <torch/csrc/autograd/grad_mode.h>

// Tests go in torch::jit
namespace torch {
namespace jit {

void printSlots(const c10::intrusive_ptr<c10::ivalue::Object>& obj) {
  auto slots = obj->slots();
  for (auto value : slots) {
    if (value.isObject()) {
      auto subobj = value.toObject();
      std::cout << "sub obj : " << subobj->name() << ", "
                << subobj->slots().size() << " slots."
                << std::endl;
      printSlots(value.toObject());
    } else if (value.isTensor()) {
      auto tensor = value.toTensor();
      std::cout << "tensor with dim " << tensor.dim() << std::endl;
    }
  }
}

void testLiteInterpreter() {
  torch::autograd::AutoGradMode guard(false);
  at::AutoNonVariableTypeMode non_var_type_mode(true);
  std::vector<torch::jit::IValue> inputs;

//  auto m = load("/Users/myuan/data/fbnet/fbnet.pt");
//  inputs.push_back(torch::ones({1, 3, 224, 224}));

//  auto m = load("/Users/myuan/data/lenet/Lenet_trace.pt");
//  inputs.push_back(torch::ones({1, 1, 30, 30}));

  // pytext
  auto m = load("/Users/myuan/data/pytext/pytext_pickle.pt");
  auto options = torch::TensorOptions().dtype(torch::kI64)
                     .requires_grad(false).device(torch::kCPU);
  int length = 5;
  inputs.push_back(torch::ones({1, length}, options));
  auto stensor = length * torch::ones({1}, options);
  inputs.push_back(stensor);

  at::Tensor outputref = m.forward(inputs).toTensor();
  std::cout << outputref.slice(/*dim=*/1, /*start=*/0, /*end=*/5);

  std::stringstream ss;
  m._save_for_mobile(ss);
  at::Tensor res;
  auto bc = _load_for_mobile(ss);

  std::cout << "ref slots: \n";
  printSlots(m.module_object());

  std::cout << "bytecode slots: \n";
  printSlots(bc.module_object());

  for (int i = 0; i < 1; ++i) {
    auto bcinputs = inputs;
    res = bc.run_method("forward", bcinputs).toTensor();
  }
  std::cout << res.slice(/*dim=*/1, /*start=*/0, /*end=*/5);
//  script::Module m("m");
//  m.register_parameter("foo", torch::ones({}), false);
//  // TODO: support default param val, which was pushed in
//  // function schema's checkAndNormalizeInputs()
////  m.define(R"(
////    def add_it(self, x, b : int = 4):
////      return self.foo + x + b
////  )");
//  m.define(R"(
//    def add_it(self, x):
//      b = 4
//      return self.foo + x + b
//  )");

//  std::vector<IValue> inputs;
//  auto minput = 5 * torch::ones({});
//  inputs.emplace_back(minput);
//  auto ref = m.run_method("add_it", minput);

//  std::stringstream ss;
//  m.save_for_mobile(ss);
//  auto bc = load_bytecode(ss);
//  IValue res;
//  for (int i = 0; i < 3; ++i) {
//    auto bcinputs = inputs;
//    res = bc.run_method("add_it", bcinputs);
//  }

//  auto resd = res.toTensor().item<float>();
//  auto refd = ref.toTensor().item<float>();
//  AT_ASSERT(resd == refd);
}

} // namespace torch
} // namespace jit
