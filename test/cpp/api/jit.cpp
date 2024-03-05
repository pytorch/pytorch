#include <gtest/gtest.h>

#include <torch/jit.h>
#include <torch/script.h>
#include <torch/types.h>

#include <string>

TEST(TorchScriptTest, CanCompileMultipleFunctions) {
  auto module = torch::jit::compile(R"JIT(
      def test_mul(a, b):
        return a * b
      def test_relu(a, b):
        return torch.relu(a + b)
      def test_while(a, i):
        while bool(i < 10):
          a += a
          i += 1
        return a
      def test_len(a : List[int]):
        return len(a)
    )JIT");
  auto a = torch::ones(1);
  auto b = torch::ones(1);

  ASSERT_EQ(1, module->run_method("test_mul", a, b).toTensor().item<int64_t>());

  ASSERT_EQ(
      2, module->run_method("test_relu", a, b).toTensor().item<int64_t>());

  ASSERT_TRUE(
      0x200 ==
      module->run_method("test_while", a, b).toTensor().item<int64_t>());

  at::IValue list = c10::List<int64_t>({3, 4});
  ASSERT_EQ(2, module->run_method("test_len", list).toInt());
}

TEST(TorchScriptTest, TestNestedIValueModuleArgMatching) {
  auto module = torch::jit::compile(R"JIT(
      def nested_loop(a: List[List[Tensor]], b: int):
        return torch.tensor(1.0) + b
    )JIT");

  auto b = 3;

  torch::List<torch::Tensor> list({torch::rand({4, 4})});

  torch::List<torch::List<torch::Tensor>> list_of_lists;
  list_of_lists.push_back(list);
  module->run_method("nested_loop", list_of_lists, b);

  auto generic_list = c10::impl::GenericList(at::TensorType::get());
  auto empty_generic_list =
      c10::impl::GenericList(at::ListType::create(at::TensorType::get()));
  empty_generic_list.push_back(generic_list);
  module->run_method("nested_loop", empty_generic_list, b);

  auto too_many_lists = c10::impl::GenericList(
      at::ListType::create(at::ListType::create(at::TensorType::get())));
  too_many_lists.push_back(empty_generic_list);
  try {
    module->run_method("nested_loop", too_many_lists, b);
    AT_ASSERT(false);
  } catch (const c10::Error& error) {
    AT_ASSERT(
        std::string(error.what_without_backtrace())
            .find("nested_loop() Expected a value of type 'List[List[Tensor]]'"
                  " for argument 'a' but instead found type "
                  "'List[List[List[Tensor]]]'") == 0);
  };
}

TEST(TorchScriptTest, TestDictArgMatching) {
  auto module = torch::jit::compile(R"JIT(
      def dict_op(a: Dict[str, Tensor], b: str):
        return a[b]
    )JIT");
  c10::Dict<std::string, at::Tensor> dict;
  dict.insert("hello", torch::ones({2}));
  auto output = module->run_method("dict_op", dict, std::string("hello"));
  ASSERT_EQ(1, output.toTensor()[0].item<int64_t>());
}

TEST(TorchScriptTest, TestTupleArgMatching) {
  auto module = torch::jit::compile(R"JIT(
      def tuple_op(a: Tuple[List[int]]):
        return a
    )JIT");

  c10::List<int64_t> int_list({1});
  auto tuple_generic_list = c10::ivalue::Tuple::create({int_list});

  // doesn't fail on arg matching
  module->run_method("tuple_op", tuple_generic_list);
}

TEST(TorchScriptTest, TestOptionalArgMatching) {
  auto module = torch::jit::compile(R"JIT(
      def optional_tuple_op(a: Optional[Tuple[int, str]]):
        if a is None:
          return 0
        else:
          return a[0]
    )JIT");

  auto optional_tuple = c10::ivalue::Tuple::create({2, std::string("hi")});

  ASSERT_EQ(2, module->run_method("optional_tuple_op", optional_tuple).toInt());
  ASSERT_EQ(
      0, module->run_method("optional_tuple_op", torch::jit::IValue()).toInt());
}

TEST(TorchScriptTest, TestPickle) {
  torch::IValue float_value(2.3);

  // TODO: when tensors are stored in the pickle, delete this
  std::vector<at::Tensor> tensor_table;
  auto data = torch::jit::pickle(float_value, &tensor_table);

  torch::IValue ivalue = torch::jit::unpickle(data.data(), data.size());

  double diff = ivalue.toDouble() - float_value.toDouble();
  double eps = 0.0001;
  ASSERT_TRUE(diff < eps && diff > -eps);
}
