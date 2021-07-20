#include <gtest/gtest.h>
#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/frontend/resolver.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/mobile/parse_bytecode.h>
#include <torch/csrc/jit/serialization/import_export_functions.h>

#include <unordered_set>

namespace torch {
namespace jit {
namespace mobile {

TEST(RunTimeTest, LoadAndForward) {
  // Load check in model: sequence.ptl
  std::string filePath(__FILE__);
  auto testModelFile = filePath.substr(0, filePath.find_last_of("/\\") + 1);
  testModelFile.append("sequence.ptl");

  //  sequence.ptl source code:
  //  class A(torch.nn.Module):
  //    def __init__(self):
  //      super(A, self).__init__()
  //
  //    def forward(self, x):
  //      return x + 1
  //
  //  class B(torch.nn.Module):
  //    def __init__(self):
  //      super(B, self).__init__()
  //
  //    def forward(self, x):
  //      return x + 2
  //
  //  class C(torch.nn.Module):
  //    def __init__(self):
  //      super(C, self).__init__()
  //      self.A0 = A()
  //      self.B0 = B()
  //
  //    def forward(self, x):
  //      return self.A0.forward(self.B0.forward(x))

  Module bc = _load_for_mobile(testModelFile);
  auto forward_method = bc.find_method("forward");
  std::vector<c10::IValue> input{c10::IValue(at::tensor(1))};
  const auto result = bc.forward(input);
  const auto expected_result = c10::IValue(at::tensor(4));
  ASSERT_EQ(result, expected_result);
}

TEST(RunTimeTest, Delegate) {
  std::string filePath(__FILE__);
  auto testModelFile = filePath.substr(0, filePath.find_last_of("/\\") + 1);
  // "delegate_test.ptl" is generated from test/cpp/jit/test_backend.cpp,
  // BackendTest.TestCompiler. This test is on target runtime. It has
  // model running capability, but no compilation and serialization.
  // The mobile model delegated to the "backend_with_compiler_demo" backend
  // The model is from the jit code:
  //  Module m("m");
  //  m.define(R"(
  //    def forward(self, x, h):
  //        return x + h
  //  )");
  testModelFile.append("delegate_test.ptl");
  auto mlm = _load_for_mobile(testModelFile);
  std::vector<IValue> inputs;
  inputs.emplace_back(2.0 * at::ones({}));
  inputs.emplace_back(1.0 * at::ones({}));

  auto mres = mlm.forward(inputs);
  AT_ASSERT(mres.toTensor().equal(3 * at::ones({})));
}

TEST(RunTimeTest, DelegateException) {
  std::string filePath(__FILE__);
  auto testModelFile = filePath.substr(0, filePath.find_last_of("/\\") + 1);
  /*
   * Model: delegated_submodule_with_debug_info.ptl
   * Model structure:
   * def AA(..):
   *   def forward(self, x, y):
   *     return x + y
   *
   * def A(..):
   *   def __init__(..):
   *     self.AA0 = AA()
   *   def forward(self, x, y):
   *     return self.AA0.forward(x, y) + 3
   *
   * def B(..):
   *   def forward(self, x):
   *     return x + 2
   *
   * def C(..):
   *   def __init__(..):
   *     self.A0 = A()
   *     self.B0 = B()
   *   def forward(self, x, y):
   *     return self.A0.forward(x, y) + self.B0.forward(x)
   *
   * std::vector<IValue> inputs;
   * inputs.emplace_back(torch::rand({2, 4}));
   * inputs.emplace_back(torch::rand({13, 9}));
   * Run with inputs and expect exception
   * Erro stack trace will look like this:
   * Module hierarchy:top(C).A0(backend_with_compiler_demoLoweredModule).AA0(AA)
   * Traceback of TorchScript (most recent call last):
   *  File "<string>", line 3, in FunctionName_UNKNOWN
   *
   *    def forward(self, x, y):
   *      return self.A0.forward(x, y) + self.B0.forward(x)
   *             ~~~~~~~~~~~~~~~ <--- HERE
   *
   *  File "<string>", line 5, in FunctionName_UNKNOWN
   *                typed_inputs: List[Any] = [x, y, ]
   *                if self.__backend.is_available() :
   *                  _0, = self.__backend.execute(self.__handles["forward"],
   * typed_inputs)
   *                        ~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
   *                  assert isinstance(_0, Tensor)
   *                  return _0
   *  File "<string>", line 3, in FunctionName_UNKNOWN
   *
   *    def forward(self, x, y):
   *      return self.AA0.forward(x, y) + 3
   *             ~~~~~~~~~~~~~~~~ <--- HERE
   *
   *  File "<string>", line 3, in FunctionName_UNKNOWN
   *
   *    def forward(self, x, y):
   *      return x + y
   *             ~~~~~ <--- HERE
   *
   *
   */
  testModelFile.append("delegated_submodule_with_debug_info.ptl");
  auto mlm = _load_for_mobile(testModelFile);
  std::vector<IValue> inputs;
  inputs.emplace_back(torch::rand({2, 4}));
  inputs.emplace_back(torch::rand({13, 9}));

  std::string error_pattern = R"(
  Module hierarchy:top(C).A0(backend_with_compiler_demoLoweredModule).AA0(AA).aten::add
Traceback of TorchScript (most recent call last):
  File "<string>", line 3, in FunctionName_UNKNOWN

    def forward(self, x, y):
      return self.A0.forward(x, y) + self.B0.forward(x)
             ~~~~~~~~~~~~~~~ <--- HERE

  File "<string>", line 5, in forward
                typed_inputs: List[Any] = [x, y, ]
                if self.__backend.is_available() :
                  _0, = self.__backend.execute(self.__handles["forward"], typed_inputs)
                        ~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
                  assert isinstance(_0, Tensor)
                  return _0
  File "<string>", line 3, in FunctionName_UNKNOWN

    def forward(self, x, y):
      return self.AA0.forward(x, y) + 3
             ~~~~~~~~~~~~~~~~ <--- HERE

  File "<string>", line 3, in forward

    def forward(self, x, y):
      return x + y
             ~~~~~ <--- HERE
  )";
  ASSERT_THROWS_WITH_MESSAGE(mlm.forward(inputs), error_pattern);
}

TEST(RunTimeTest, ParseBytecode) {
  // A simple example to show a simple bytecode that can be used independent of
  // PyTorch TorchScript serialization (unpickler, etc) and operator library.
  // It has basic control flow (if, else) and basic data orchestration (list
  // construction). The original PyTorch program:

  //  class Module(torch.nn.Module):
  //
  //    def __init__(self):
  //      super().__init__()
  //
  //    def forward(self, x: int, h: int, xfirst: bool):
  //      if xfirst:
  //        return [x, h]
  //      else:
  //        return [h, x]

  // 1. Prepare for the bytecode. In reality it can be from a customized
  // deserializer.
  std::vector<IValue> instructions{
      Tup({"STOREN", 1, 4}),
      Tup({"DROPR", 1, 0}),
      Tup({"MOVE", 4, 0}),
      Tup({"JF", 5, 0}),
      Tup({"LOAD", 2, 0}),
      Tup({"LOAD", 3, 0}),
      Tup({"LIST_CONSTRUCT", 0, 2}),
      Tup({"JMP", 4, 0}),
      Tup({"LOAD", 3, 0}),
      Tup({"LOAD", 2, 0}),
      Tup({"LIST_CONSTRUCT", 1, 2}),
      Tup({"STORE", 5, 0}),
      Tup({"DROPR", 3, 0}),
      Tup({"DROPR", 2, 0}),
      Tup({"MOVE", 5, 0}),
      Tup({"RET", 0, 0}),
  };
  std::vector<IValue> operators; // empty for this example
  std::vector<IValue> constants; // empty for this example

  std::vector<IValue> types{"List[int]", "List[int]"};
  auto codeTable = Table(
      {{"instructions", Tup(instructions)},
       {"operators", Tup(operators)},
       {"constants", Tup(constants)},
       {"types", Tup(types)},
       {"register_size", 5}});

  // 2. Parse the function
  std::string function_name("test_functoin");
  auto function = std::unique_ptr<mobile::Function>(
      new mobile::Function(c10::QualifiedName(function_name)));
  parseInstructions(function_name, codeTable, IValue(), function.get());
  parseConstants(codeTable, function.get());
  parseTypes(codeTable, function.get());
  parseRegisterSize(codeTable, function.get());

  // 3. Prepare for inputs and run the function
  // Note that the first input is reserved for Module object.
  // Since this is a function test and Module object is not required,
  // a dummy IValue (0) is added here.
  std::vector<IValue> inputs{0, 1, 2, true};
  function->run(inputs);
  auto output = inputs[0].toList();
  ASSERT_EQ(output[0], 1);
  ASSERT_EQ(output[1], 2);

  std::vector<IValue> inputs1{0, 1, 2, false};
  function->run(inputs1);
  auto output1 = inputs1[0].toList();
  ASSERT_EQ(output1[0], 2);
  ASSERT_EQ(output1[1], 1);
}

} // namespace mobile
} // namespace jit
} // namespace torch
