#include <test/cpp/jit/test_base.h>
#include <test/cpp/jit/test_utils.h>
#include <sstream>

#include <torch/csrc/jit/serialization/export.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/csrc/jit/serialization/import_source.h>
#include <torch/torch.h>

#include "caffe2/serialize/istream_adapter.h"

namespace torch {
namespace jit {

// Tests that an extra file written explicitly has precedence over
//   extra files written by a hook
// TODO: test for the warning, too
void testExtraFilesHookPreference() {
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

void testSaveExtraFilesHook() {
  // no secrets
  {
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
  // some secret
  {
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
}

void testTypeTags() {
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
  for (auto item : items) {
    auto bytes = torch::pickle_save(item.value);
    auto loaded = torch::pickle_load(bytes);
    ASSERT_TRUE(loaded.type()->isSubtypeOf(item.expected_type));
    ASSERT_TRUE(item.expected_type->isSubtypeOf(loaded.type()));
  }
}

} // namespace jit
} // namespace torch
