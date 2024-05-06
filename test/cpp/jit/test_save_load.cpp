#include <gtest/gtest.h>

#include <test/cpp/jit/test_utils.h>
#include <cstdlib>
#include <iostream>
#include <sstream>

#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/runtime/calculate_necessary_args.h>
#include <torch/csrc/jit/serialization/export.h>
#include <torch/csrc/jit/serialization/export_bytecode.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/csrc/jit/serialization/import_source.h>
#include <torch/script.h>
#include <torch/torch.h>

#include "caffe2/serialize/istream_adapter.h"

namespace torch {
namespace jit {

namespace {

Module roundtripThroughMobile(const Module& m) {
  ExtraFilesMap files;
  std::vector<IValue> constants;
  jitModuleToPythonCodeAndConstants(m, &files, &constants);
  CompilationOptions options;
  mobile::Module mobilem = jitModuleToMobile(m, options);
  return jitModuleFromSourceAndConstants(
      mobilem._ivalue(), files, constants, 8);
}

template <class Functor>
inline void expectThrowsEq(Functor&& functor, const char* expectedMessage) {
  try {
    std::forward<Functor>(functor)();
  } catch (const Error& e) {
    EXPECT_STREQ(e.what_without_backtrace(), expectedMessage);
    return;
  }
  ADD_FAILURE() << "Expected to throw exception with message \""
                << expectedMessage << "\" but didn't throw";
}

} // namespace

TEST(SerializationTest, ExtraFilesHookPreference) {
  // Tests that an extra file written explicitly has precedence over
  //   extra files written by a hook
  // TODO: test for the warning, too
  const auto script = R"JIT(
    def forward(self):
        x = torch.rand(5, 5)
        x = x.mm(x)
        return x
  )JIT";

  auto module =
      std::make_shared<Module>("Module", std::make_shared<CompilationUnit>());
  module->define(script);
  std::ostringstream oss;
  std::unordered_map<std::string, std::string> extra_files;
  extra_files["metadata.json"] = "abc";
  SetExportModuleExtraFilesHook([](const Module&) -> ExtraFilesMap {
    return {{"metadata.json", "def"}};
  });
  module->save(oss, extra_files);
  SetExportModuleExtraFilesHook(nullptr);

  std::istringstream iss(oss.str());
  caffe2::serialize::IStreamAdapter adapter{&iss};
  std::unordered_map<std::string, std::string> loaded_extra_files;
  loaded_extra_files["metadata.json"] = "";
  auto loaded_module = torch::jit::load(iss, torch::kCPU, loaded_extra_files);
  ASSERT_EQ(loaded_extra_files["metadata.json"], "abc");
}

TEST(SerializationTest, ExtraFileHooksNoSecret) {
  // no secrets
  std::stringstream ss;
  {
    Module m("__torch__.m");
    ExtraFilesMap extra;
    extra["metadata.json"] = "abc";
    m.save(ss, extra);
  }
  ss.seekg(0);
  {
    ExtraFilesMap extra;
    extra["metadata.json"] = "";
    extra["secret.json"] = "";
    jit::load(ss, c10::nullopt, extra);
    ASSERT_EQ(extra["metadata.json"], "abc");
    ASSERT_EQ(extra["secret.json"], "");
  }
}

TEST(SerializationTest, ExtraFileHooksWithSecret) {
  std::stringstream ss;
  {
    SetExportModuleExtraFilesHook([](const Module&) -> ExtraFilesMap {
      return {{"secret.json", "topsecret"}};
    });
    Module m("__torch__.m");
    ExtraFilesMap extra;
    extra["metadata.json"] = "abc";
    m.save(ss, extra);
    SetExportModuleExtraFilesHook(nullptr);
  }
  ss.seekg(0);
  {
    ExtraFilesMap extra;
    extra["metadata.json"] = "";
    extra["secret.json"] = "";
    jit::load(ss, c10::nullopt, extra);
    ASSERT_EQ(extra["metadata.json"], "abc");
    ASSERT_EQ(extra["secret.json"], "topsecret");
  }
}

TEST(SerializationTest, TypeTags) {
  auto list = c10::List<c10::List<int64_t>>();
  list.push_back(c10::List<int64_t>({1, 2, 3}));
  list.push_back(c10::List<int64_t>({4, 5, 6}));
  auto dict = c10::Dict<std::string, at::Tensor>();
  dict.insert("Hello", torch::ones({2, 2}));
  auto dict_list = c10::List<c10::Dict<std::string, at::Tensor>>();
  for (size_t i = 0; i < 5; i++) {
    auto another_dict = c10::Dict<std::string, at::Tensor>();
    another_dict.insert("Hello" + std::to_string(i), torch::ones({2, 2}));
    dict_list.push_back(another_dict);
  }
  auto tuple = std::tuple<int, std::string>(2, "hi");
  struct TestItem {
    IValue value;
    TypePtr expected_type;
  };
  std::vector<TestItem> items = {
      {list, ListType::create(ListType::create(IntType::get()))},
      {2, IntType::get()},
      {dict, DictType::create(StringType::get(), TensorType::get())},
      {dict_list,
       ListType::create(
           DictType::create(StringType::get(), TensorType::get()))},
      {tuple, TupleType::create({IntType::get(), StringType::get()})}};
  // NOLINTNEXTLINE(performance-for-range-copy)
  for (auto item : items) {
    auto bytes = torch::pickle_save(item.value);
    auto loaded = torch::pickle_load(bytes);
    ASSERT_TRUE(loaded.type()->isSubtypeOf(*item.expected_type));
    ASSERT_TRUE(item.expected_type->isSubtypeOf(*loaded.type()));
  }
}

TEST(SerializationTest, TestJitStream_CUDA) {
  torch::jit::Module model;
  std::vector<torch::jit::IValue> inputs;
  // Deserialize the ScriptModule from a file using torch::jit::load().
  // Load the scripted model. This should have been generated by tests_setup.py
  // Refer: TorchSaveJitStream_CUDA in test/cpp/jit/tests_setup.py
  model = torch::jit::load("saved_stream_model.pt");

  auto output = model.forward(inputs);
  const auto& list_of_elements = output.toTupleRef().elements();
  auto is_stream_s = list_of_elements[0].toBool();

  // a,b: These are the two input tensors
  // c: This is output tensor generated by the operation torch.cat(a,b)
  auto a = list_of_elements[1].toTensor();
  auto b = list_of_elements[2].toTensor();
  auto c = list_of_elements[3].toTensor();
  // op: this is used to verify if the cat operation produced the same results
  // as that on the GPU with torch.cat
  auto op = at::cat({a, b}, 0);

  // Check if the stream is set
  ASSERT_TRUE(is_stream_s);
  // Check if the sizes of the outputs (op and c) is same on the GPU and CPU
  ASSERT_EQ(op.sizes(), c.sizes());
  // Check if both the output tensors are equal
  ASSERT_TRUE(op.equal(c));
}

TEST(TestSourceRoundTrip, UpsampleNearest2d) {
  Module m("m");
  m.define(R"(
    def forward(self, input: Tensor, scale:float):
      return torch.upsample_nearest2d(input, [1, 1], float(scale), float(scale))
  )");

  std::vector<IValue> inputs;
  inputs.emplace_back(torch::rand({1, 3, 128, 128}));
  inputs.emplace_back(at::Scalar(2.0));
  auto ref = m.forward(inputs);

  Module m2 = roundtripThroughMobile(m);
  auto res = m2.forward(inputs);

  auto resd = res.toTensor();
  auto refd = ref.toTensor();
  ASSERT_TRUE(resd.equal(refd));
}

TEST(TestSourceRoundTrip, CheckAttrAccess) {
  Module m("m");
  m.register_attribute("mobile_optimized", BoolType::get(), true);
  Module m2 = roundtripThroughMobile(m);
  bool mobile_optimized = m2.attr("mobile_optimized", false).toBool();
  AT_ASSERT(mobile_optimized);
}

TEST(TestSourceRoundTrip,
     MethodInvocation) { // NOLINT (use =delete in gtest)
  const std::vector<std::string> test_programs{
      // test invoking a method with default parameter
      R"(
      def test_func(self, x, b : int = 4):
        return self.foo + x + b
      )",
      // inner method call with default parameter (gets inlined)
      R"(
      def add_with_default_arg(self, x, b : int = 4):
        return self.foo + x + b
      def test_func(self, x):
        return self.add_with_default_arg(x)  # invoke method w/ default arg
      )",
      // simple method call
      R"(
      def test_func(self, x):
        b = 4
        return self.foo + x + b
      )",
  };
  for (const auto& test_program : test_programs) {
    Module m("m");
    m.register_parameter("foo", torch::ones({}), false);
    m.define(test_program);

    const int fortyTwo = 42; // (keep linter happy)
    auto minput = fortyTwo * torch::ones({});
    auto ref = m.run_method("test_func", minput);

    Module m2 = roundtripThroughMobile(m);
    const auto& test_func = m2.get_method("test_func");
    IValue res;
    for (int i = 0; i < 3; ++i) {
      res = test_func({minput});
    }

    auto resd = res.toTensor().item<float>();
    auto refd = ref.toTensor().item<float>();
    AT_ASSERT(resd == refd);
  }
}

TEST(SerializationTest, ParentDirNotExist) {
  expectThrowsEq(
      []() {
        auto t = torch::nn::Linear(5, 5);
        torch::save(t, "./doesnotexist/file.pt");
      },
      "Parent directory ./doesnotexist does not exist.");
}

#ifdef WIN32
TEST(SerializationTest, WindowsDrivePathTest) {
  // "ZZZ" is typically not a valid drive letter.
  // We expect to see "ZZZ:\\" or "ZZZ:/" in the error message.
  // Note: slash should be included for the drive letter parent in Windows.
  expectThrowsEq(
      []() {
        auto t = torch::nn::Linear(5, 5);
        torch::save(t, "ZZZ:\\file.pt");
      },
      "Parent directory ZZZ:\\ does not exist.");
  expectThrowsEq(
      []() {
        auto t = torch::nn::Linear(5, 5);
        torch::save(t, "ZZZ:/file.pt");
      },
      "Parent directory ZZZ:/ does not exist.");
}

TEST(SerializationTest, WindowsTempPathTest) {
  // Test for verifying file saving and loading in the temporary folder
  std::string temp_dir = std::getenv("TEMP");
  std::string file_path = temp_dir + "/file.pt";
  auto t1 = torch::tensor(1.0);
  torch::save(t1, file_path);
  torch::Tensor t2;
  torch::load(t2, file_path);
  ASSERT_TRUE(t1.allclose(t2, 0.0, 0.0));
}
#endif

TEST(SerializationTest, CalculateNecessaryArgsTest) {
  auto schema = torch::schema(
      "sync_stream(int stream_id = -1) -> ()",
      c10::AliasAnalysisKind::CONSERVATIVE);

  auto graph = std::make_shared<Graph>();
  auto one_val = graph->insertConstant(-1);
  auto necessary = CalculateNecessaryArgs(schema.arguments(), {one_val}, true);
  EXPECT_EQ(0, necessary.first);
  EXPECT_EQ(0, necessary.second);
}

TEST(TestSaveLoad, LoadWithoutDebugInfo) { // NOLINT (use =delete in gtest)
  Module m("m");
  m.register_parameter("foo", torch::ones({}), false);
  m.define(
      R"(
    def test_func(self, x):
      b = 4
      return self.foo + x + b
    )");
  m.define(
      R"(
    def exception(self):
      assert False, "message"
    )");
  std::stringstream ss;
  m.save(ss);
  ss.seekg(0);
  caffe2::serialize::PyTorchStreamReader reader(&ss);
  reader.setShouldLoadDebugSymbol(true);
  EXPECT_TRUE(reader.hasRecord("code/__torch__.py.debug_pkl"));
  reader.setShouldLoadDebugSymbol(false);
  EXPECT_FALSE(reader.hasRecord("code/__torch__.py.debug_pkl"));
  ss.seekg(0);
  Module m2 = torch::jit::load(ss);
  std::string error_msg = R"(
    def exception(self):
      assert False, "message"
      ~~~~~~~~~~~~~~~~~~~~~~~ <--- HERE)";
  ASSERT_THROWS_WITH_MESSAGE(m2.run_method("exception"), error_msg);

  ss.seekg(0);
  // NO DEBUG trace so error message points to torchscript generated
  // source instead of original python source.
  std::string error2 = R"(
    def exception(self: __torch__.m) -> NoneType:
      _0 = uninitialized(NoneType)
      ops.prim.RaiseException("AssertionError: message")
      ~~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
      return _0
  )";
  Module m3 = torch::jit::load(ss, c10::nullopt, false);
  ASSERT_THROWS_WITH_MESSAGE(m3.run_method("exception"), error2);
}

TEST(SerializationTest, TestPickleAppend) {
  auto data = std::vector<char>({'\x80', char(2), ']', 'K', char(2), 'a', '.'});

  torch::IValue actual = torch::jit::unpickle(data.data(), data.size());

  torch::IValue expected = c10::impl::GenericList(at::AnyType::get());
  expected.toList().push_back(2);
  ASSERT_EQ(expected, actual);
}

} // namespace jit
} // namespace torch
