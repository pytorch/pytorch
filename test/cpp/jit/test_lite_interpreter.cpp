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
  m.define(R"(
    def add_it(self, x, b : int = 4):
      return self.foo + x + b
  )");

  std::stringstream s_script;
  m.save(s_script);
  auto m_load = load(s_script);

//  script::Module m = load("/Users/myuan/data/lenet/lenet_traced.pt");
  std::stringstream ss;
  m.save_for_mobile(ss);
  auto bc = load_bytecode(ss);

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
