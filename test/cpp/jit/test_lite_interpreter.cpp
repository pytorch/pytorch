#include <test/cpp/jit/test_base.h>
#include <torch/csrc/jit/script/module.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/jit/lite_interpreter/import_bytecode.h>
#include <torch/csrc/jit/lite_interpreter/bytecode.h>
#include <torch/csrc/jit/import.h>

// Tests go in torch::jit
namespace torch {
namespace jit {

void testLiteInterpreter() {
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

//  std::stringstream s_script;
//  m.save(s_script);
//  auto m_load = load(s_script);

//  script::Module m = load("/Users/myuan/data/lenet/lenet_traced.pt");
  std::stringstream ss;
  m.save_for_mobile(ss);
  auto bc = load_bytecode(ss);
  IValue res;
  for (int i = 0; i < 3; ++i) {
    auto bcinputs = inputs;
    res = bc.run_method("add_it", bcinputs);
    std::cout << "output: " << std::endl;
    std::cout << res.toTensor().item<float>();
    std::cout << std::endl;
  }

  auto resd = res.toTensor().item<float>();
  auto refd = ref.toTensor().item<float>();

  AT_ASSERT(resd == refd);
}

void testLiteWithFile() {
//  // Deserialize the ScriptModule from a file using torch::jit::load().
//  torch::jit::script::Module module = torch::jit::load(argv[1]);

//  // Save in new format with code and pkl files.
//  std::string nfile(argv[1]);
//  nfile += "1.pt";
//  module.save(nfile);

//  // Save in bytecode format.
//  module.save(argv[2], torch::jit::script::ExtraFilesMap(), true /*bytecode_format*/);
}
}
}
