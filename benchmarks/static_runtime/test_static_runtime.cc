#include <gtest/gtest.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/runtime/static/fusion.h>
#include <torch/csrc/jit/runtime/static/impl.h>
#include <torch/csrc/jit/runtime/static/passes.h>
#include "deep_wide_pt.h"
#include "test_scripts.h"

using namespace caffe2;
using namespace torch;
using namespace torch::jit;
using c10::IValue;

C10_DECLARE_bool(
    static_runtime_enable_fast_math);

namespace {
static at::Tensor getTensor(const at::IValue& ival) {
  if (ival.isTensor()) {
    return ival.toTensor();
  } else if (ival.isTensorList()) {
    auto tensor_vec = ival.toTensorVector();
    TORCH_CHECK(tensor_vec.size() == 1);
    return tensor_vec[0];
  } else if (ival.isTuple()) {
    auto tuple = ival.toTuple();
    auto ivalue_vec = tuple->elements();
    TORCH_CHECK(ivalue_vec.size() == 1);
    return ivalue_vec[0].toTensor();
  } else {
    CAFFE_THROW("Unknown input IValue");
  }
}

void compareTensorLists(
    const std::vector<IValue>& l, /* expects */
    const std::vector<IValue>& r /* values */) {
  EXPECT_TRUE(l.size() == r.size());
  for (int i = 0; i < l.size(); ++i) {
    ASSERT_TRUE(l[i].isTensor());
    ASSERT_TRUE(r[i].isTensor());
    VLOG(2) << "expect " << i << ": \n" << l[i] << std::endl;
    VLOG(2) << "output " << i << ": \n" << r[i] << std::endl;
    if (!l[i].toTensor().defined()) {
      EXPECT_TRUE(!r[i].toTensor().defined());
    } else {
      EXPECT_TRUE(l[i].toTensor().equal(r[i].toTensor()));
    }
  }
}

void compareTensorLists(
    const std::vector<at::Tensor>& l, /* expects */
    const std::vector<at::Tensor>& r /* values */) {
  EXPECT_TRUE(l.size() == r.size());
  for (int i = 0; i < l.size(); ++i) {
    VLOG(2) << "expect " << i << ": \n" << l[i] << std::endl;
    VLOG(2) << "output " << i << ": \n" << r[i] << std::endl;
    if (!l[i].defined()) {
      EXPECT_TRUE(!r[i].defined());
    } else {
      EXPECT_TRUE(l[i].equal(r[i]));
    }
  }
}

void compareResults(const IValue& expect, const IValue& actual) {
  if (expect.isTensor()) {
    VLOG(2) << "expect " << expect.toTensor() << std::endl;
    VLOG(2) << "output " << actual.toTensor() << std::endl;
    EXPECT_TRUE(actual.isTensor());
    EXPECT_TRUE(expect.toTensor().equal(actual.toTensor()));
    return;
  } else if (expect.isTuple()) {
    EXPECT_TRUE(actual.isTuple());
    auto lhs = expect.toTuple()->elements();
    auto rhs = actual.toTuple()->elements();
    EXPECT_TRUE(lhs.size() == rhs.size());
    for (size_t i = 0; i < lhs.size(); i++) {
      compareResults(lhs[i], rhs[i]);
    }
  } else if (expect.isList()) {
    EXPECT_TRUE(actual.isList());
    auto lhs = expect.toList();
    auto rhs = actual.toList();
    EXPECT_TRUE(lhs.size() == rhs.size());
    for (size_t i = 0; i < lhs.size(); i++) {
      compareResults(lhs[i], rhs[i]);
    }
  } else if (expect.isGenericDict()) {
    EXPECT_TRUE(actual.isGenericDict());
    auto lhs = expect.toGenericDict();
    auto rhs = actual.toGenericDict();
    EXPECT_TRUE(lhs.size() == rhs.size());
    for (auto& lh : lhs) {
      auto f = rhs.find(lh.key());
      EXPECT_FALSE(f == rhs.end());
      compareResults(lh.value(), f->value());
    }
  } else {
    // fall back to the default comparison impl in IValue
    EXPECT_TRUE(expect == actual);
  }
}

// Given a model/function in jit script, run the model/function
// with the jit interpreter and static runtime, and compare the results
void testStaticRuntime(
    const std::string& jit_script,
    const std::vector<IValue>& args,
    const std::vector<IValue>& args2 = {}) {
  script::Module module("module");
  module.define(jit_script);

  std::vector<IValue> args_tensors, args_copy;
  for (const auto& ival : args) {
    if (ival.isTensor()) {
      args_tensors.emplace_back(ival);
      const at::Tensor& t = ival.toTensor();
      args_copy.emplace_back(t.clone());
    }
  }

  auto expect = module.forward(args);

  for (bool enable_out_variant : {true, false}) {
    torch::jit::StaticModule smodule(
        module, {true, enable_out_variant, enable_out_variant});
    auto actual = smodule(args, {});
    smodule.runtime().check_for_memory_leak();
    // first run
    compareResults(expect, actual);

    // args2 is used to check for dynamic shapes
    // it also exercises the memory planner
    if (!args2.empty()) {
      expect = module.forward(args2);
      actual = smodule(args2, {});
      smodule.runtime().check_for_memory_leak();
      // second run
      compareResults(expect, actual);

      expect = module.forward(args);
      actual = smodule(args, {});
      smodule.runtime().check_for_memory_leak();
      // third run
      compareResults(expect, actual);
    } else {
      // run static runtime again to exercise the memory planner
      actual = smodule(args, {});
      smodule.runtime().check_for_memory_leak();
      // second run
      compareResults(expect, actual);
    }
  }

  // make sure inputs were not modified
  compareTensorLists(args_tensors, args_copy);
}

bool testHasInplaceOp(const std::string& jit_script) {
  script::Module module("module");
  module.define(jit_script);

  Method method = module.get_method("forward");
  auto graph = module.get_method("forward").graph();

  torch::jit::AliasDb alias_db(graph);
  return torch::jit::HasInplaceOp(graph, alias_db);
}

static Node* getNodeWithKind(const torch::jit::StaticModule& smodule, const string& kind) {
  for (auto& pnode : smodule.nodes()) {
    if (std::string(pnode.node()->kind().toQualString()) == kind) {
      return pnode.node();
    }
  }
  return nullptr;
}

bool testCanEnableStaticRuntime(const std::string& jit_script) {
  script::Module module("module");
  module.define(jit_script);

  Method method = module.get_method("forward");
  auto graph = module.get_method("forward").graph();

  // here we do not freeze graph
  return torch::jit::canEnableStaticRuntime(graph);
}
} // namespace

TEST(StaticRuntime, InPlace) {
  EXPECT_TRUE(testHasInplaceOp(reshape_inplace_script));
  EXPECT_TRUE(testHasInplaceOp(sigmoid_inplace_script));
  EXPECT_FALSE(testHasInplaceOp(sigmoid_out_script));
}

TEST(StaticRuntime, CanEnableStaticRuntime) {
  EXPECT_TRUE(testCanEnableStaticRuntime(reshape_inplace_script));
  EXPECT_FALSE(testCanEnableStaticRuntime(if_script));
}

TEST(StaticRuntime, NestedOutput) {
  auto run_test = [](std::vector<int64_t> shapes) {
    auto a = at::randn(shapes);
    auto b = at::randn(shapes);

    std::vector<IValue> args{a, b};
    testStaticRuntime(nested_output_script_0, args);
    testStaticRuntime(nested_output_script_1, args);
    testStaticRuntime(nested_output_script_2, args);
    testStaticRuntime(nested_output_script_3, args);

    if (shapes.size() > 0 && shapes[0] != 0) {
      shapes[0] *= 2;
      testStaticRuntime(
          nested_output_script_0, args, {at::randn(shapes), at::randn(shapes)});
      testStaticRuntime(
          nested_output_script_1, args, {at::randn(shapes), at::randn(shapes)});
    }
  };
  run_test({2, 3, 1, 4});
  run_test({2, 3});
}

TEST(StaticRuntime, UnaryOps) {
  auto a = at::randn({2, 3});
  auto b = at::randn({4, 3, 2});

  std::vector<IValue> args{a}, args2{b};

  // sum
  testStaticRuntime(aten_sum, args);
  testStaticRuntime(aten_sum_0, args);
  testStaticRuntime(aten_sum_1, args);
  testStaticRuntime(aten_sum_0_true, args);
  testStaticRuntime(aten_sum_1_true, args);

  testStaticRuntime(aten_sum, args, args2);
  testStaticRuntime(aten_sum_0, args, args2);
  testStaticRuntime(aten_sum_1, args, args2);
  testStaticRuntime(aten_sum_0_true, args, args2);
  testStaticRuntime(aten_sum_1_true, args, args2);
}

TEST(StaticRuntime, Sigmoid) {
  auto a = at::randn({2, 3});
  auto b = at::randn({4, 3, 2});

  std::vector<IValue> args{a}, args2{b};

  testStaticRuntime(sigmoid_script, args);
  testStaticRuntime(sigmoid_script, args, {args2});

  FLAGS_static_runtime_enable_fast_math = false;
  testStaticRuntime(sigmoid_script, args);
  testStaticRuntime(sigmoid_script, args, {args2});
  FLAGS_static_runtime_enable_fast_math = true;
}

TEST(StaticRuntime, Clone) {
  auto a = at::randn({2, 3});
  auto b = at::empty_strided({3, 2}, {1, 3});
  auto c = at::randn({1, 2, 3, 4});
  auto d = at::randn({1, 0, 3, 4});
  std::vector<IValue> args_0{b, c10::MemoryFormat::Contiguous};
  std::vector<IValue> args_1{b, c10::MemoryFormat::Preserve};
  std::vector<IValue> args_2{c, c10::MemoryFormat::ChannelsLast};
  std::vector<IValue> args_3{d, c10::MemoryFormat::ChannelsLast};

  testStaticRuntime(clone_script_0, {a});
  testStaticRuntime(clone_script_0, {a}, {b});

  testStaticRuntime(clone_script_1, args_0);
  testStaticRuntime(clone_script_1, args_1);
  testStaticRuntime(clone_script_1, args_2);
  testStaticRuntime(clone_script_1, args_3);
  testStaticRuntime(clone_script_1, args_0, args_1);
  testStaticRuntime(clone_script_1, args_3, args_2);
}

TEST(StaticRuntime, Clamp) {
  auto a = at::randn({2, 3});
  auto max_t = at::full_like(a, 1);
  auto min_t = at::full_like(a, -1);

  auto b = at::randn({4, 3, 2});
  auto max_t1 = at::full_like(b, 1);
  auto min_t1 = at::full_like(b, -1);

  testStaticRuntime(clamp_script_1, {a, -1, 1});
  testStaticRuntime(clamp_script_2, {a, min_t, max_t});

  testStaticRuntime(clamp_script_1, {a, -1, 1}, {b, -1, 1});
  testStaticRuntime(clamp_script_2, {a, min_t, max_t}, {b, max_t1, min_t1});
}

TEST(StaticRuntime, Logit) {
  auto a = at::ones({2, 3});
  double b = 1e-6;
  std::vector<IValue> args_1{a};
  std::vector<IValue> args_2({a, b});

  auto c = at::ones({4, 3, 2});

  // logit
  testStaticRuntime(logit_script_1, args_1);
  testStaticRuntime(logit_script_2, args_1);
  testStaticRuntime(logit_script_3, args_2);

  testStaticRuntime(logit_script_1, args_1, {c});
  testStaticRuntime(logit_script_2, args_1, {c});
  testStaticRuntime(logit_script_3, args_2, {c, b});
}

// TODO: check for dynamic shapes
TEST(StaticRuntime, EmbeddingBag) {
  at::Tensor weight = torch::randn({3, 11}, at::ScalarType::Float);
  at::Tensor input = torch::tensor({0, 1, 0, 2});
  at::Tensor offset = torch::tensor({0, 2, 4});

  std::vector<IValue> args{weight, input, offset};

  testStaticRuntime(embedding_bag_default, args);
  testStaticRuntime(embedding_bag_mean, args);
  testStaticRuntime(embedding_bag_max, args);
  testStaticRuntime(embedding_bag_sum_last_offset, args);
  testStaticRuntime(embedding_bag_mean_last_offset, args);
  testStaticRuntime(embedding_bag_max_last_offset, args);
}

TEST(StaticRuntime, LayerNorm) {
  const auto a = torch::rand({1, 2, 2, 2});
  const auto b = torch::rand({3, 2, 2, 2});
  for (int normalized_size : {2, 3}) {
    std::vector<int64_t> normalized_shape(normalized_size, 2);
    const auto weight = torch::rand(normalized_shape);
    const auto bias = torch::rand(normalized_shape);

    std::vector<IValue> args{a, normalized_shape, weight, bias};
    std::vector<IValue> args1{b, normalized_shape, weight, bias};
    testStaticRuntime(layer_norm_with_weights, args);
    testStaticRuntime(layer_norm_with_weights, args, args1);

    args = {a, normalized_shape};
    testStaticRuntime(layer_norm_without_weights, args);
    testStaticRuntime(layer_norm_without_weights, args, {b, normalized_shape});
  }
}

TEST(StaticRuntime, Bmm) {
  auto a = at::randn({10, 4, 5});
  auto b = at::randn({10, 5, 6});

  auto c = at::randn({12, 5, 6});
  auto d = at::randn({12, 6, 7});

  std::vector<IValue> args{a, b};
  std::vector<IValue> args1{c, d};
  testStaticRuntime(bmm_script, args);
  testStaticRuntime(bmm_script, args1);
  testStaticRuntime(bmm_script, args, args1);
}

TEST(StaticRuntime, Addmm) {
  auto inp1 = at::randn({5});
  auto mat1 = at::randn({3, 4});
  auto mat2 = at::randn({4, 5});

  auto inp2 = at::randn({3, 7});
  auto mat3 = at::randn({3, 6});
  auto mat4 = at::randn({6, 7});

  std::vector<IValue> args{inp1, mat1, mat2, 1.0, 2.0};
  std::vector<IValue> args1{inp2, mat3, mat4, 2.0, 1.0};
  testStaticRuntime(addmm_script, args);
  testStaticRuntime(addmm_script, args1);
  testStaticRuntime(addmm_script, args, args1);
}

TEST(StaticRuntime, IndividualOps_Binary) {
  auto a = at::randn({2, 3});
  auto b = at::ones({2, 3});

  auto c = at::randn({4, 2, 3});
  auto d = at::ones({4, 2, 3});

  std::vector<IValue> args{a, b};

  testStaticRuntime(add_script, args);
  testStaticRuntime(add_script, args, {c, d});
  testStaticRuntime(list_construct_script, args);
  testStaticRuntime(list_construct_script_2, args);
  testStaticRuntime(list_construct_script_3, args);
  testStaticRuntime(list_unpack_script, args);
  testStaticRuntime(list_unpack_script_2, args);
  testStaticRuntime(tuple_construct_script, args);
  testStaticRuntime(tuple_construct_script_2, args);
}

TEST(StaticRuntime, IndividualOps_Binary_MatMul) {
  // 1-D, 1-D
  std::vector<IValue> args{at::randn({3}), at::randn({3})};
  testStaticRuntime(aten_matmul, args);
  // 2-D, 2-D
  std::vector<IValue> args1 = {at::randn({3, 2}), at::randn({2, 3})};
  testStaticRuntime(aten_matmul, args1);
  // 1-D, 2-D
  std::vector<IValue> args2 = {at::randn({3}), at::randn({3, 5})};
  testStaticRuntime(aten_matmul, args2);
  // 2-D, 1-D
  std::vector<IValue> args3 = {at::randn({3, 5}), at::randn({5})};
  testStaticRuntime(aten_matmul, args3);
  // > 2-D , > 2-D
  std::vector<IValue> args4 = {at::randn({3, 1, 4, 5}), at::randn({2, 5, 6})};
  testStaticRuntime(aten_matmul, args4);

  testStaticRuntime(aten_matmul, args3, args4);
}

TEST(StaticRuntime, IndividualOps_Sign) {
  auto a = at::randn({2, 3});
  auto b = at::randn({4, 3, 2});

  std::vector<IValue> args{a};
  testStaticRuntime(sign_tensor, args);
  testStaticRuntime(sign_tensor, args, {b});
}

TEST(StaticRuntime, IndividualOps_Div) {
  auto a = at::randn({2, 3});
  auto b = at::randn({2, 3});
  auto c = at::randn({4, 3, 2});
  auto d = at::randn({4, 3, 2});

  std::vector<IValue> args0{a, b};
  testStaticRuntime(div_tensor, args0);
  testStaticRuntime(div_tensor, args0, {c, d});

  std::vector<IValue> args1{a, 3};
  testStaticRuntime(div_scalar, args1);
  testStaticRuntime(div_scalar, args1, {c, 4});

  std::vector<IValue> args2{a, b, "floor"};
  testStaticRuntime(div_tensor_mode, args2);
  testStaticRuntime(div_tensor_mode, args2, {c, d, "floor"});

  std::vector<IValue> args3{a, 2.3, "trunc"};
  testStaticRuntime(div_scalar_mode, args3);
  testStaticRuntime(div_scalar_mode, args3, {a, 1.5, "trunc"});
}

TEST(StaticRuntime, IndividualOps_Log) {
  // Ensure that the input values are valid.
  auto a = at::abs(at::randn({2, 3}));
  auto b = at::abs(at::randn({4, 3, 2}));

  std::vector<IValue> args{a};
  testStaticRuntime(log_tensor, args);
  testStaticRuntime(log_tensor, args, {b});
}

TEST(StaticRuntime, IndividualOps_Sub) {
  auto a = at::randn({2, 3});
  auto b = at::randn({2, 3});
  auto c = at::randn({4, 3, 2});
  auto d = at::randn({4, 3, 2});

  std::vector<IValue> args0{a, b};
  testStaticRuntime(sub_tensor, args0);
  testStaticRuntime(sub_tensor, args0, {c, d});

  std::vector<IValue> args1{a, 3};
  testStaticRuntime(sub_scalar, args1);
  testStaticRuntime(sub_scalar, args1, {c, 4});

  std::vector<IValue> args2{a, b, 2.3};
  testStaticRuntime(sub_tensor_alpha, args2);
  testStaticRuntime(sub_tensor_alpha, {c, d, 3.1});

  std::vector<IValue> args3{a, 2.3, 4};
  testStaticRuntime(sub_scalar_alpha, args3);
  testStaticRuntime(sub_scalar_alpha, {c, 1.3, 2});
}

TEST(StaticRuntime, IndividualOps_Norm) {
  auto a = at::randn({2, 3});
  auto b = at::randn({4, 3, 2});
  auto dim = std::vector<int64_t>({1});
  auto dtype = at::ScalarType::Float;

  std::vector<IValue> args2{a, 2};
  testStaticRuntime(norm_2arg, args2);
  testStaticRuntime(norm_2arg, args2, {b, 2});

  std::vector<IValue> args3{a, 2, dtype};
  testStaticRuntime(norm_3arg, args3);
  testStaticRuntime(norm_3arg, args3, {b, 2, dtype});

  std::vector<IValue> args4{a, 3, dim, false};
  testStaticRuntime(norm_4arg, args4);
  testStaticRuntime(norm_4arg, args4, {b, 3, dim, false});

  std::vector<IValue> args5{a, 4, dim, true, dtype};
  testStaticRuntime(norm_5arg, args5);
  testStaticRuntime(norm_5arg, args5, {b, 4, dim, true, dtype});
}

TEST(StaticRuntime, IndividualOps_Reshape) {
  auto a = at::randn({2, 3});
  auto b = std::vector<int64_t>({3, 2});
  std::vector<IValue> args{a, b};

  auto c = at::randn({4, 2});
  auto d = std::vector<int64_t>({2, 4});
  std::vector<IValue> args1{c, d};

  testStaticRuntime(reshape_script_1, args);
  testStaticRuntime(reshape_script_2, args);
  testStaticRuntime(reshape_script_3, args);
  testStaticRuntime(reshape_script_4, args);
  testStaticRuntime(reshape_script_5, args);
  testStaticRuntime(reshape_inplace_script, args);
  testStaticRuntime(reshape_incontiguous_script, args);

  testStaticRuntime(reshape_script_1, args, args1);
  testStaticRuntime(reshape_script_2, args, args1);
  testStaticRuntime(reshape_script_3, args, args1);
  testStaticRuntime(reshape_script_4, args, args1);
  testStaticRuntime(reshape_script_5, args, args1);
  testStaticRuntime(reshape_inplace_script, args, args1);
  testStaticRuntime(reshape_incontiguous_script, args, args1);
}

TEST(StaticRuntime, IndividualOps_Repeat) {
  auto a = at::randn({2, 3});
  auto b = at::randn({4, 3});
  auto c = std::vector<int64_t>({1, 2});
  auto d = std::vector<int64_t>({2, 3});
  std::vector<IValue> args1{a, c};
  std::vector<IValue> args2{b, d};

  testStaticRuntime(repeat, args1);
  testStaticRuntime(repeat, args2);
  testStaticRuntime(repeat, args1, args2);
}

TEST(StaticRuntime, IndividualOps_flatten) {
  auto test_flatten =
      [](std::vector<int64_t> shape, int64_t start_dim, int64_t end_dim) {
        std::vector<int64_t> shape1(shape);
        if (shape1.size() > 0) {
          shape1[0] *= 2;
        }
        auto a = at::randn(shape);
        auto b = at::randn(shape1);
        std::vector<IValue> args{a, start_dim, end_dim};
        testStaticRuntime(flatten_script_1, args);
        testStaticRuntime(flatten_script_1, args, {b, start_dim, end_dim});
        if (shape.size() > 2) {
          testStaticRuntime(flatten_script_2, args);
          testStaticRuntime(flatten_script_2, args, {b, start_dim, end_dim});
        }
      };

  test_flatten({2, 3}, 0, 1);
  test_flatten({2, 1, 3}, 1, 2);
  test_flatten({0, 1, 3, 0}, 1, 2);
  test_flatten({2, 3}, 1, 1);
  test_flatten({}, 0, 0);
}

TEST(StaticRuntime, IndividualOps_pow) {
  auto a = at::randn({2, 3});
  auto b = at::randn({2, 3});
  auto c = at::randn({4, 3, 2});
  auto d = at::randn({4, 3, 2});

  std::vector<IValue> args0{a, 4};
  testStaticRuntime(pow_script_ten_sca, args0);
  testStaticRuntime(pow_script_ten_sca, args0, {c, 4});

  std::vector<IValue> args1{at::abs(a), b};
  testStaticRuntime(pow_script_ten_ten, args1);
  testStaticRuntime(pow_script_ten_ten, args1, {at::abs(c), d});

  std::vector<IValue> args2{5, b};
  testStaticRuntime(pow_script_sca_ten, args2);
  testStaticRuntime(pow_script_sca_ten, args2, {3, d});
}

TEST(StaticRuntime, IndividualOps_to) {
  auto test_to = [](at::ScalarType b, bool c, bool d, c10::MemoryFormat e) {
    auto a = at::randn({4, 3, 1, 2});
    auto other = at::randn({4, 3, 1, 2}, b);
    auto a2 = at::randn({3, 2, 2, 4});
    auto a2_other = at::randn({3, 2, 2, 4}, b);

    std::vector<IValue> args0{a, b, c, d, e};
    std::vector<IValue> args1{a, b, c, d};
    std::vector<IValue> args2{a, other, c, d, e};

    testStaticRuntime(to_script_0, args0); // to.dtype
    testStaticRuntime(to_script_1, args0); // to.dtype, strided
    testStaticRuntime(to_script_2, args1); // to.prim_dtype
    testStaticRuntime(to_script_3, args2); // to.other
    testStaticRuntime(to_script_4, {a}); // alias

    // dynamic shapes
    testStaticRuntime(to_script_0, args0, {a2, b, c, d, e}); // to.dtype
    testStaticRuntime(to_script_1, args0, {a2, b, c, d, e}); // to.dtype
    testStaticRuntime(to_script_2, args1, {a2, b, c, d}); // to.prim_dtype
    testStaticRuntime(to_script_3, args2, {a2, a2_other, c, d, e}); // to.other
    testStaticRuntime(to_script_4, {a}, {a2});
  };
  // float->float, NCHW->NHWC
  test_to(at::ScalarType::Float, true, true, c10::MemoryFormat::ChannelsLast);
  // float->half
  test_to(at::ScalarType::Half, true, false, c10::MemoryFormat::Preserve);
  // float->float
  test_to(at::ScalarType::Float, false, false, c10::MemoryFormat::Contiguous);
  // TODO: check if fbgemm is enabled properly in this case
  // half->float, NCHW->NHWC
  test_to(at::ScalarType::Half, false, true, c10::MemoryFormat::ChannelsLast);
}

TEST(StaticRuntime, IndividualOps_FullLike) {
  auto a = at::randn({2, 3});
  auto b = at::randn({3, 2, 2});
  auto dtype = at::ScalarType::Int;
  auto cpu = at::Device(DeviceType::CPU);
  std::vector<IValue> args{
      a, 4, dtype, at::kStrided, cpu, false, c10::MemoryFormat::Contiguous};
  std::vector<IValue> args2{
      b, 4, dtype, at::kStrided, cpu, false, c10::MemoryFormat::Contiguous};
  testStaticRuntime(full_like_script, args);
  testStaticRuntime(full_like_script, args, args2);
}

TEST(StaticRuntime, LongModel) {
  torch::jit::Module mod = getLongScriptModel();
  auto a = torch::randn({2, 2});
  auto b = torch::randn({2, 2});
  auto c = torch::randn({2, 2});

  // run jit graph executor
  std::vector<at::IValue> input_ivalues({a, b, c});
  at::Tensor output_1 = mod.forward(input_ivalues).toTensor();

  // run static runtime
  std::vector<at::Tensor> input_tensors({a, b, c});
  torch::jit::StaticModule smod(mod);
  at::Tensor output_2 = smod(input_tensors)[0];
  smod.runtime().check_for_memory_leak();
  EXPECT_TRUE(torch::allclose(output_1, output_2, 1e-6));
}

TEST(StaticRuntime, TrivialModel) {
  torch::jit::Module mod = getTrivialScriptModel();
  auto a = torch::randn({2, 2});
  auto b = torch::randn({2, 2});
  auto c = torch::randn({2, 2});

  // run jit graph executor
  std::vector<at::IValue> input_ivalues({a, b, c});
  at::Tensor output_1 = mod.forward(input_ivalues).toTensor();

  // run static runtime
  std::vector<at::Tensor> input_tensors({a, b, c});
  torch::jit::StaticModule smod(mod);
  at::Tensor output_2 = smod(input_tensors)[0];
  smod.runtime().check_for_memory_leak();
  EXPECT_TRUE(torch::allclose(output_1, output_2, 1e-6));
}

TEST(StaticRuntime, LeakyReLU) {
  torch::jit::Module mod = getLeakyReLUConstScriptModel();
  auto inputs = torch::randn({2, 2});

  // run jit graph executor
  std::vector<at::IValue> input_ivalues({inputs});
  at::Tensor output_1 = mod.forward(input_ivalues).toTensor();

  // run static runtime
  std::vector<at::Tensor> input_tensors({inputs});
  torch::jit::StaticModule smod(mod);
  at::Tensor output_2 = smod(input_tensors)[0];
  smod.runtime().check_for_memory_leak();
  EXPECT_TRUE(torch::allclose(output_1, output_2, 1e-6));
}

TEST(StaticRuntime, DeepWide) {
  const int embedding_size = 32;
  const int num_features = 50;
  torch::jit::Module mod = getDeepAndWideSciptModel();
  torch::jit::StaticModule smod(mod);

  for (int batch_size : {1, 8, 32}) {
    for (int i = 0; i < 2; ++i) {
      auto ad_emb_packed = torch::randn({batch_size, 1, embedding_size});
      auto user_emb = torch::randn({batch_size, 1, embedding_size});
      auto wide = torch::randn({batch_size, num_features});

      // run jit graph executor
      std::vector<at::IValue> inputs({ad_emb_packed, user_emb, wide});
      auto output_1 = getTensor(mod.forward(inputs));

      // run static runtime
      std::vector<at::Tensor> input_tensors({ad_emb_packed, user_emb, wide});
      at::Tensor output_2 = smod(input_tensors)[0];
      smod.runtime().check_for_memory_leak();
      EXPECT_TRUE(torch::allclose(output_1, output_2, 1e-6));
    }
  }
}

TEST(StaticRuntime, KWargsAPI_1) {
  const int embedding_size = 32;
  const int num_features = 50;
  auto module = getDeepAndWideSciptModel();
  torch::jit::StaticModule smod(module);

  for (int batch_size : {1, 8, 32}) {
    for (int i = 0; i < 2; ++i) {
      auto ad_emb_packed = torch::randn({batch_size, 1, embedding_size});
      auto user_emb = torch::randn({batch_size, 1, embedding_size});
      auto wide = torch::randn({batch_size, num_features});
      {
        std::vector<at::IValue> inputs({ad_emb_packed, user_emb, wide});

        // run jit graph executor
        at::Tensor output_1 = getTensor(module.forward(inputs));

        // run static runtime
        c10::IValue output_ivalue = smod(inputs, {});
        smod.runtime().check_for_memory_leak();

        at::Tensor output_2 = getTensor(output_ivalue);
        EXPECT_TRUE(torch::allclose(output_1, output_2, 1e-6));

        // check for output aliasing
        EXPECT_EQ(output_ivalue.use_count(), 1);
        output_ivalue = IValue();

        EXPECT_EQ(output_2.getIntrusivePtr().use_count(), 1);
      }

      // check for input aliasing (deep & wide does not have ops
      // that create aliases of input tensors)
      EXPECT_EQ(ad_emb_packed.getIntrusivePtr().use_count(), 1);
      EXPECT_EQ(user_emb.getIntrusivePtr().use_count(), 1);
      EXPECT_EQ(wide.getIntrusivePtr().use_count(), 1);
    }
  }
}

TEST(StaticRuntime, KWargsAPI_2) {
  const int embedding_size = 32;
  const int num_features = 50;
  auto module = getDeepAndWideSciptModel();
  torch::jit::StaticModule smod(module);

  for (int batch_size : {1, 8, 32}) {
    for (int i = 0; i < 2; ++i) {
      auto ad_emb_packed = torch::randn({batch_size, 1, embedding_size});
      auto user_emb = torch::randn({batch_size, 1, embedding_size});
      auto wide = torch::randn({batch_size, num_features});
      {
        // run jit graph executor
        std::vector<at::IValue> args({ad_emb_packed, user_emb, wide});
        at::Tensor output_1 = getTensor(module.forward(args));

        std::unordered_map<std::string, c10::IValue> kwargs(
            {{"ad_emb_packed", ad_emb_packed},
             {"user_emb", user_emb},
             {"wide", wide}});

        // run static runtime
        c10::IValue output_ivalue = smod({}, kwargs);
        smod.runtime().check_for_memory_leak();

        at::Tensor output_2 = getTensor(output_ivalue);
        EXPECT_TRUE(torch::allclose(output_1, output_2, 1e-6));

        // check for output aliasing
        EXPECT_EQ(output_ivalue.use_count(), 1);
        output_ivalue = IValue();

        EXPECT_EQ(output_2.getIntrusivePtr().use_count(), 1);
      }

      EXPECT_EQ(ad_emb_packed.getIntrusivePtr().use_count(), 1);
      EXPECT_EQ(user_emb.getIntrusivePtr().use_count(), 1);
      EXPECT_EQ(wide.getIntrusivePtr().use_count(), 1);
    }
  }
}

TEST(StaticRuntime, CleanUpMemory) {
  const int embedding_size = 32;
  const int num_features = 50;
  torch::jit::Module mod = getDeepAndWideSciptModel();

  for (auto cleanup_activations : {true, false}) {
    for (auto enable_out_variant : {true, false}) {
      for (auto optimize_memory : {true, false}) {
        for (auto optimize_graph_output_memory : {true, false}) {
          if (optimize_graph_output_memory && !optimize_memory) {
            // when optimize_graph_output_memory is enabled, optimize_memory
            // must be enabled too
            continue;
          }
          if (optimize_memory && !enable_out_variant) {
            // when optimize_memory is enabled, enable_out_variant must be
            // enabled too
            continue;
          }
          VLOG(1) << "cleanup_activations: " << cleanup_activations
                  << ", enable_out_variant: " << enable_out_variant
                  << ", optimize_memory: " << optimize_memory
                  << ", optimize_graph_output_memory: "
                  << optimize_graph_output_memory;
          torch::jit::StaticModuleOptions opts{
              cleanup_activations,
              enable_out_variant,
              optimize_memory,
              optimize_graph_output_memory};
          torch::jit::StaticModule smod(mod, opts);

          for (int batch_size : {1, 8, 32}) {
            for (int i = 0; i < 2; ++i) {
              auto ad_emb_packed =
                  torch::randn({batch_size, 1, embedding_size});
              auto user_emb = torch::randn({batch_size, 1, embedding_size});
              auto wide = torch::randn({batch_size, num_features});

              // run jit graph executor
              std::vector<at::IValue> inputs({ad_emb_packed, user_emb, wide});
              auto output_1 = getTensor(mod.forward(inputs));

              // run static runtime
              std::vector<at::Tensor> input_tensors(
                  {ad_emb_packed, user_emb, wide});
              at::Tensor output_2 = smod(input_tensors)[0];
              smod.runtime().check_for_memory_leak();
              EXPECT_TRUE(torch::allclose(output_1, output_2, 1e-6));
            }
          }
        }
      }
    }
  }
}

TEST(StaticRuntime, FusionPass) {
  const int embedding_size = 32;
  const int num_features = 50;
  for (int batch_size : {1, 8, 32}) {
    for (int i = 0; i < 2; ++i) {
      torch::jit::Module module = getDeepAndWideSciptModel();
      auto ad_emb_packed = torch::randn({batch_size, 1, embedding_size});
      auto user_emb = torch::randn({batch_size, 1, embedding_size});
      auto wide = torch::randn({batch_size, num_features});

      // run jit graph executor
      std::vector<at::IValue> inputs({ad_emb_packed, user_emb, wide});
      auto output_1 = getTensor(module.forward(inputs));

      Method method = module.get_method("forward");
      auto graph = method.graph();
      fuseStaticSubgraphs(graph, 2);
      bool hit = false;
      for (const auto& n : module.get_method("forward").graph()->nodes()) {
        if (n->kind() == torch::jit::prim::StaticSubgraph) {
          hit = true;
        }
      }
      EXPECT_TRUE(hit);
      auto output_2 = getTensor(module.forward(inputs));
      EXPECT_TRUE(torch::allclose(output_1, output_2, 1e-6));
    }
  }
}

TEST(ProcessedNode, VerifyOutputsNotOverlappingWithImmutableInputsWithImmutableArguments) {
  script::Module module("module");
  // Not using out= variant.
  module.define(sigmoid_script);
  torch::jit::StaticModule smodule(module);
  Node* sigmoid_node = getNodeWithKind(smodule, "aten::sigmoid");
  const at::IValue a = torch::randn({2, 3});
  at::IValue b = torch::randn({3, 1});
  std::vector<const IValue*> ivalue_inputs{&a};
  ProcessedNode pnode(sigmoid_node, std::move(ivalue_inputs), true);

  pnode.Output(0) = b;
  EXPECT_TRUE(pnode.verify_outputs_not_overlapping_with_immutable_inputs());

  pnode.Output(0) = a;
  EXPECT_FALSE(pnode.verify_outputs_not_overlapping_with_immutable_inputs());
}

TEST(ProcessedNode, VerifyOutputsNotOverlappingWithImmutableInputsWithMutableArguments) {
  script::Module module("module");
  // Using out= variant.
  module.define(sigmoid_inplace_script);
  torch::jit::StaticModule smodule(module);
  Node* sigmoid_node = getNodeWithKind(smodule, "aten::sigmoid");
  const at::IValue a = torch::randn({2, 3});
  at::IValue b = torch::randn({3, 1});
  std::vector<const IValue*> ivalue_inputs{&a};
  ProcessedNode pnode(sigmoid_node, std::move(ivalue_inputs), true);

  pnode.Output(0) = b;
  EXPECT_TRUE(pnode.verify_outputs_not_overlapping_with_immutable_inputs());

  pnode.Output(0) = a;
  EXPECT_TRUE(pnode.verify_outputs_not_overlapping_with_immutable_inputs());
}
