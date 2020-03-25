#include <test/cpp/jit/test_base.h>
#include <test/cpp/jit/test_utils.h>

#include <sstream>

#include <torch/csrc/jit/serialization/export.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/csrc/jit/serialization/import_source.h>
#include <torch/torch.h>

namespace torch {
namespace jit {

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


// TODO: Re-enable when add_type_tags is true
void testTypeTags() {
//   auto list = c10::List<c10::List<int64_t>>();
//   list.push_back(c10::List<int64_t>({1, 2, 3}));
//   list.push_back(c10::List<int64_t>({4, 5, 6}));
//
//   auto dict = c10::Dict<std::string, at::Tensor>();
//   dict.insert("Hello", torch::ones({2, 2}));
//
//   auto dict_list = c10::List<c10::Dict<std::string, at::Tensor>>();
//   for (size_t i = 0; i < 5; i++) {
//     auto another_dict = c10::Dict<std::string, at::Tensor>();
//     another_dict.insert("Hello" + std::to_string(i), torch::ones({2, 2}));
//     dict_list.push_back(another_dict);
//   }
//
//   auto tuple = std::tuple<int, std::string>(2, "hi");
//
//   struct TestItem {
//     IValue value;
//     TypePtr expected_type;
//   };
//   std::vector<TestItem> items = {
//     {list, ListType::create(ListType::create(IntType::get()))},
//     {2, IntType::get()},
//     {dict, DictType::create(StringType::get(), TensorType::get())},
//     {dict_list, ListType::create(DictType::create(StringType::get(), TensorType::get()))},
//     {tuple, TupleType::create({IntType::get(), StringType::get()})}
//   };
//
//   for (auto item : items) {
//     auto bytes = torch::pickle_save(item.value);
//     auto loaded = torch::pickle_load(bytes);
//     ASSERT_TRUE(loaded.type()->isSubtypeOf(item.expected_type));
//     ASSERT_TRUE(item.expected_type->isSubtypeOf(loaded.type()));
//   }
}

} // namespace jit
} // namespace torch
