#include <c10/core/ScalarType.h>
#include <gtest/gtest.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/runtime/static/fusion.h>
#include <torch/csrc/jit/runtime/static/impl.h>
#include <torch/csrc/jit/runtime/static/passes.h>
#include "deep_wide_pt.h"
#include "test_scripts.h"
#include "test_utils.h"

using namespace caffe2;
using namespace torch;
using namespace torch::jit;
using namespace torch::jit::test;
using c10::IValue;

C10_DECLARE_bool(static_runtime_enable_fast_math);

namespace {

at::Tensor getTensor(const at::IValue& ival) {
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

bool testCanEnableStaticRuntime(const std::string& jit_script) {
  script::Module module("module");
  module.define(jit_script);

  Method method = module.get_method("forward");
  auto graph = module.get_method("forward").graph();

  // here we do not freeze graph
  return canEnableStaticRuntime(graph);
}

bool testHasInplaceOp(const std::string& jit_script) {
  script::Module module("module");
  module.define(jit_script);

  Method method = module.get_method("forward");
  auto graph = module.get_method("forward").graph();

  AliasDb alias_db(graph);
  return HasInplaceOp(graph, alias_db);
}

bool testModuleHasOp(const std::string& jit_script, const char* op_name) {
  script::Module module("module");
  module.define(jit_script);

  return forwardHasOp(module, op_name);
}

Node* getNodeWithKind(const StaticModule& smodule, const std::string& kind) {
  for (auto& pnode : smodule.nodes()) {
    if (std::string(pnode.node()->kind().toQualString()) == kind) {
      return pnode.node();
    }
  }
  return nullptr;
}

bool hasNodeWithKind(const StaticModule& smodule, const std::string& kind) {
  return getNodeWithKind(smodule, kind) != nullptr;
}

} // namespace

TEST(StaticRuntime, InPlace) {
  EXPECT_TRUE(testHasInplaceOp(reshape_inplace_script));
  EXPECT_TRUE(testHasInplaceOp(reshape_inplace_script_1));
  EXPECT_TRUE(testHasInplaceOp(sigmoid_inplace_script));
  EXPECT_FALSE(testHasInplaceOp(sigmoid_out_script));
}

TEST(StaticRuntime, ModuleHasOp) {
  EXPECT_TRUE(testModuleHasOp(reshape_inplace_script, "aten::sigmoid_"));
  EXPECT_TRUE(testModuleHasOp(reshape_inplace_script_1, "aten::reshape"));
  EXPECT_TRUE(testModuleHasOp(sigmoid_inplace_script, "aten::clone"));
  EXPECT_FALSE(testModuleHasOp(reshape_inplace_script_1, "aten::add_"));
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

  testStaticRuntime(sigmoid_script, args, /*args2=*/{}, /*use_allclose=*/true);
  testStaticRuntime(sigmoid_script, args, {args2}, /*use_allclose=*/true);

  FLAGS_static_runtime_enable_fast_math = false;
  testStaticRuntime(sigmoid_script, args, /*args2=*/{}, /*use_allclose=*/true);
  testStaticRuntime(sigmoid_script, args, {args2}, /*use_allclose=*/true);
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
#ifdef FBCODE_CAFFE2
  script::Module module("module");
  module.define(layer_norm_with_weights);
  torch::jit::StaticModule smodule(module);
  ASSERT_EQ(getNodeWithKind(smodule, "aten::layer_norm"), nullptr);
  ASSERT_NE(getNodeWithKind(smodule, "static_runtime::layer_norm"), nullptr);
#endif
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

TEST(StaticRuntime, IndividualOps_Abs) {
  auto a = at::randn({2, 3});
  auto b = at::randn({4, 2, 3});
  std::vector<IValue> args{a};
  std::vector<IValue> args2{b};
  testStaticRuntime(abs_script, args);
  testStaticRuntime(abs_script, args, args2);
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

TEST(StaticRuntime, IndividualOps_Mul) {
  auto a = at::randn({3, 3});
  auto b = at::randn({3, 3});
  auto c = at::randn({3, 3, 3});
  auto d = at::randn({3, 3, 3});

  std::vector<IValue> tensor_args1{a, b};
  std::vector<IValue> tensor_args2{c, d};

  testStaticRuntime(mul_tensor, tensor_args1);
  testStaticRuntime(mul_tensor, tensor_args1, tensor_args2);

  std::vector<IValue> scalar_args1{a, 42};
  std::vector<IValue> scalar_args2{c, 42};

  testStaticRuntime(mul_scalar, scalar_args1);
  testStaticRuntime(mul_scalar, scalar_args1, scalar_args2);
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

TEST(StaticRuntime, IndividualOps_NanToNum) {
  const auto inf = std::numeric_limits<double>::infinity();
  const auto nan = std::numeric_limits<double>::quiet_NaN();

  auto a = torch::tensor({{1.0, nan}, {-inf, inf}});
  auto b = torch::tensor({{1.0, nan, -inf}, {-inf, inf, inf}, {nan, 1.0, 1.0}});

  std::vector<IValue> args1{a, 1.0, 2.0, -2.0};
  std::vector<IValue> args2{b, 1.0, 2.0, -2.0};

  testStaticRuntime(
      nan_to_num_script,
      args1,
      /*args2*/ {},
      /*use_allclose*/ true,
      /*use_equalnan*/ true);
  testStaticRuntime(
      nan_to_num_script,
      args1,
      args2,
      /*use_allclose*/ true,
      /*use_equalnan*/ true);
}

TEST(StaticRuntime, IndividualOps_Stack) {
  auto a = torch::tensor({{1.0, 2.0}, {3.0, 4.0}});
  auto b = torch::tensor({{1.0, 2.0}, {3.0, 4.0}});
  auto c = torch::tensor({{1.0, 2.0}, {3.0, 4.0}});

  auto d = torch::tensor({{1.0, 2.0, 3.0}, {4.0, 4.0, 4.0}});
  auto e = torch::tensor({{1.0, 2.0, 3.0}, {4.0, 4.0, 4.0}});
  auto f = torch::tensor({{1.0, 2.0, 3.0}, {4.0, 4.0, 4.0}});

  std::vector<IValue> args1_dim{a, b, 0};
  std::vector<IValue> args2_dim{d, e, 1};

  std::vector<IValue> args1_three_tensors{a, b, c};
  std::vector<IValue> args2_three_tensors{d, e, f};

  testStaticRuntime(stack_dim, args1_dim);
  testStaticRuntime(stack_dim, args1_dim, args2_dim);

  testStaticRuntime(stack_three, args1_three_tensors);
  testStaticRuntime(stack_three, args1_three_tensors, args2_three_tensors);
}

TEST(StaicRuntime, IndividualOps_ReLU) {
  auto a = torch::tensor({{1, -1}, {2, 0}});
  auto b = torch::tensor({{1, -1, -1}, {2, 0, -1}});

  std::vector<IValue> args1{a};
  std::vector<IValue> args2{b};

  testStaticRuntime(relu_script, args1);
  testStaticRuntime(relu_script, args1, args2);
}

TEST(StaicRuntime, IndividualOps_Tanh) {
  auto a = at::randn({2, 2});
  auto b = at::randn({3, 3, 3});

  std::vector<IValue> args1{a};
  std::vector<IValue> args2{b};

  testStaticRuntime(tanh_script, args1, /*args2*/ {}, /*use_allclose*/ true);
  testStaticRuntime(tanh_script, args1, args2, /*use_allclose*/ true);
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

TEST(StaticRuntime, IndividualOps_Detach) {
  auto a = at::randn({4, 3, 1, 2});
  auto b = at::randn({3, 2, 2});
  std::vector<IValue> args{a};
  std::vector<IValue> args2{b};
  testStaticRuntime(detach_script_0, args);
  testStaticRuntime(detach_script_0, args, args2);
  testStaticRuntime(detach_script_1, args);
  testStaticRuntime(detach_script_1, args, args2);
}

TEST(StaticRuntime, IndividualOps_ExpandAs) {
  auto a = at::randn({3, 1});
  auto b = at::randn({3, 2});
  auto c = at::randn({4, 1});
  auto d = at::randn({4, 2});
  std::vector<IValue> args{a, b};
  std::vector<IValue> args2{c, d};
  testStaticRuntime(expand_as_script, args);
  testStaticRuntime(expand_as_script, args, args2);
}

TEST(StaticRuntime, IndividualOps_Full) {
  auto dtype = at::ScalarType::Int;
  auto cpu = at::Device(DeviceType::CPU);
  c10::List<int64_t> size0{4, 5};
  std::vector<IValue> args{size0, 4, dtype, at::kStrided, cpu, false};
  c10::List<int64_t> size1{5, 6};
  std::vector<IValue> args2{size1, 5, dtype, at::kStrided, cpu, false};
  testStaticRuntime(full_script, args);
  testStaticRuntime(full_script, args, args2);
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

TEST(StaticRuntime, Linear) {
  auto input = at::randn({1, 2});
  auto weights = at::randn({1, 2});
  auto bias = at::randn({1, 1});

  std::vector<IValue> args{input, weights, bias};
  std::vector<IValue> args_no_bias{input, weights, c10::nullopt};

  auto input2 = at::randn({2, 3});
  auto weights2 = at::randn({2, 3});
  auto bias2 = at::randn({2, 2});

  std::vector<IValue> args2{input2, weights2, bias2};
  std::vector<IValue> args2_no_bias{input2, weights2, c10::nullopt};

  testStaticRuntime(linear_script, args);
  testStaticRuntime(linear_script, args_no_bias);

  testStaticRuntime(linear_script, args, args2);
  testStaticRuntime(linear_script, args, args2_no_bias);
}

TEST(StaticRuntime, IndividualOps_VarCat) {
  // 2D tensors - cat dim = 0
  std::vector<IValue> args1 = {at::randn({4, 6}), at::randn({5, 6}), 0};
  testStaticRuntime(var_cat_script, args1);

  // 3D tensors - cat dim = 1
  std::vector<IValue> args2 = {at::randn({4, 5, 6}), at::randn({4, 8, 6}), 1};
  testStaticRuntime(var_cat_script, args2);

  // 3D tensors - cat dim = 2
  std::vector<IValue> args3 = {at::randn({4, 5, 6}), at::randn({4, 5, 7}), 2};
  testStaticRuntime(var_cat_script, args3);

  testStaticRuntime(var_cat_script, args1, args2);
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
          torch::jit::StaticModule smod(mod, false, opts);

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

TEST(
    ProcessedNode,
    VerifyNoMemoryOverlapWithImmutableInputsWithImmutableArguments) {
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
  EXPECT_TRUE(pnode.verify_no_memory_overlap());

  pnode.Output(0) = a;
  EXPECT_FALSE(pnode.verify_no_memory_overlap());
}

TEST(
    ProcessedNode,
    VerifyNoMemoryOverlapWithImmutableInputsWithMutableArguments) {
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
  EXPECT_TRUE(pnode.verify_no_memory_overlap());

  pnode.Output(0) = a;
  EXPECT_TRUE(pnode.verify_no_memory_overlap());
}

TEST(ProcessedNode, VerifyNoMemoryOverlapWithOverlappingOutputs) {
  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(
      R"IR(
    graph(%0):
      %1 : Tensor, %2 : Tensor = prim::ListUnpack(%0)
      return (%1, %2))IR",
      g.get());
  torch::jit::StaticModule smodule(g);
  Node* list_unpack_node = getNodeWithKind(smodule, "prim::ListUnpack");
  {
    auto a = at::randn({2, 3});
    IValue ivalue(a);
    std::vector<const IValue*> inputs{&ivalue};
    ProcessedNode list_unpack_pnode(list_unpack_node, std::move(inputs), /*enable_out_variant=*/true);
    ASSERT_EQ(list_unpack_pnode.outputs().size(), 2);
    EXPECT_TRUE(list_unpack_pnode.verify_no_memory_overlap());
  }
  {
    auto a = at::randn({2, 3});
    IValue ivalue(a);
    std::vector<const IValue*> inputs{&ivalue};
    ProcessedNode list_unpack_pnode(list_unpack_node, std::move(inputs), /*enable_out_variant=*/true);
    auto b = at::randn({2, 3});
    list_unpack_pnode.Output(0) = b;
    list_unpack_pnode.Output(1) = b;
    EXPECT_FALSE(list_unpack_pnode.verify_no_memory_overlap());
  }
}

TEST(StaticRuntime, IndividualOps_isinstance) {
  auto a = at::randn({2, 2});
  auto b = at::randn({2, 2, 2});

  std::vector<at::IValue> args{a};
  std::vector<at::IValue> args2{b};

  testStaticRuntime(isinstance_int_script, args);
  testStaticRuntime(isinstance_int_script, args, args2);

  testStaticRuntime(isinstance_tensor_script, args);
  testStaticRuntime(isinstance_tensor_script, args, args2);

  testStaticRuntime(isinstance_many_types_script, args);
  testStaticRuntime(isinstance_many_types_script, args, args2);
}

TEST(StaticRuntime, IndividualOps_TypeCheck) {
  auto a = at::zeros({2, 2}, at::kFloat);
  a.to(at::kCPU);
  auto b = at::ones({3, 3}, at::kFloat);
  auto c = at::ones({2, 2, 2}, at::kFloat);

  std::vector<IValue> args_correct = {a, b};
  std::vector<IValue> args_incorrect = {a, c};

  testStaticRuntime(typecheck_ir, args_correct);
  testStaticRuntime(typecheck_ir, args_correct, args_incorrect);
}

TEST(StaticRuntime, IndividualOps_Index) {
  // Index with boolean mask
  auto a = at::rand({2, 2});
  auto idx_a = torch::tensor({{0, 1}, {0, 0}}, at::kBool);
  std::vector<IValue> args_a{a, idx_a};

  // Index with tensor
  auto b = at::rand({3, 3, 3});
  auto idx_b = torch::tensor({0, 1, 2}, at::kLong);
  std::vector<IValue> args_b{b, idx_b};

  testStaticRuntime(index_without_none_script, args_a);
  testStaticRuntime(index_without_none_script, args_a, args_b);

  // Index with None
  // When indexing with none, the shape of `a` becomes [2, 1, 2],
  // so the mask must be reshaped appropriately.
  auto idx_a_reshape = torch::tensor({{{0, 1}}, {{0, 0}}}, at::kBool);
  std::vector<IValue> args_a_with_none{a, idx_a_reshape};

  testStaticRuntime(index_with_none_script, args_a_with_none);
  testStaticRuntime(index_with_none_script, args_a_with_none, args_b);

  // Index with multiple tensors
  auto c = at::randn({2, 2});
  auto idx_c1 = torch::tensor({0, 0}, at::kLong);
  auto idx_c2 = torch::tensor({0}, at::kLong);
  std::vector<IValue> args_c{c, idx_c1, idx_c2};

  auto d = at::randn({3, 3, 3});
  auto idx_d1 = torch::tensor({{0, 0}, {0, 1}}, at::kLong);
  auto idx_d2 = torch::tensor({{1, 1}, {1, 0}}, at::kLong);
  std::vector<IValue> args_d{d, idx_d1, idx_d2};

  testStaticRuntime(index_with_two_tensors_script, args_c);
  testStaticRuntime(index_with_two_tensors_script, args_c, args_d);
}

TEST(StaticRuntime, IndividualOps_ClampMin) {
  auto a = at::randn({2, 2});
  auto b = at::randn({3, 3, 3});
  int scalar_int = 1;
  float scalar_float = 3.14;

  std::vector<IValue> args_a_int{a, scalar_int};
  std::vector<IValue> args_b_int{b, scalar_int};

  testStaticRuntime(clamp_min_int_script, args_a_int);
  testStaticRuntime(clamp_min_int_script, args_a_int, args_b_int);

  std::vector<IValue> args_a_float{a, scalar_float};
  std::vector<IValue> args_b_float{b, scalar_float};

  testStaticRuntime(clamp_min_float_script, args_a_float);
  testStaticRuntime(clamp_min_float_script, args_a_float, args_b_float);
}

TEST(StaticRuntime, IndividualOps_Argmin) {
  auto a = at::randn({2, 2});
  auto b = at::randn({3, 3, 3});

  testStaticRuntime(argmin_script, {a});
  testStaticRuntime(argmin_script, {a}, {b});

  int dim_a = 0;
  int dim_b = 1;

  std::vector<IValue> args_a{a, dim_a};
  std::vector<IValue> args_b{b, dim_b};

  testStaticRuntime(argmin_with_dim_script, args_a);
  testStaticRuntime(argmin_with_dim_script, args_a, args_b);

  testStaticRuntime(argmin_with_keep_dim_script, args_a);
  testStaticRuntime(argmin_with_keep_dim_script, args_a, args_b);
}

TEST(StaticRuntime, IndividualOps_Softmax) {
  auto a = at::randn({2, 3});
  auto b = at::randn({3, 3, 3});

  testStaticRuntime(softmax_script, {a, 0});
  testStaticRuntime(softmax_script, {a, 1});

  testStaticRuntime(softmax_script, {b, 0});
  testStaticRuntime(softmax_script, {b, 1});
  testStaticRuntime(softmax_script, {b, 2});

  testStaticRuntime(softmax_script_with_dtype, {a, 1, at::ScalarType::Float});
  testStaticRuntime(softmax_script_with_dtype, {b, 1, at::ScalarType::Float});
}

TEST(StaticRuntime, IndividualOps_GetItem_Dict) {
  int int_key = 0;
  std::string str_key = "str";

  // No need to test these multiple times, args are not tensors
  testStaticRuntime(getitem_dict_int_script, {int_key});
  testStaticRuntime(getitem_dict_str_script, {str_key});

  auto a = torch::tensor({1});
  auto b = torch::tensor({1, 1});

  testStaticRuntime(getitem_dict_tensor_script, {a});
  testStaticRuntime(getitem_dict_tensor_script, {a}, {b});
}

TEST(StaticRuntime, IndividualOps_GetItem_List) {
  testStaticRuntime(getitem_list_int_script, {1});
  testStaticRuntime(getitem_list_int_script, {-1});

  auto a = torch::tensor({1});
  auto b = torch::tensor({1, 1});

  testStaticRuntime(getitem_list_tensor_script, {a, 1});
  testStaticRuntime(getitem_list_tensor_script, {a, 1}, {b, -1});
}

TEST(StaticRuntime, IndividualOps_Transpose) {
  auto a = at::randn({2, 2});
  int dim1_a = 0;
  int dim2_a = 1;
  std::vector<IValue> args_a{a, dim1_a, dim2_a};

  auto b = at::randn({3, 3, 3});
  int dim1_b = 0;
  int dim2_b = 2;
  std::vector<IValue> args_b{b, dim1_b, dim2_b};

  testStaticRuntime(transpose_script, args_a);
  testStaticRuntime(transpose_script, args_a, args_b);
}

TEST(StaticRuntime, IndividualOps_Permute) {
  auto a = at::randn({2, 2});
  c10::List<int64_t> dims_a{1, 0};
  std::vector<IValue> args_a{a, dims_a};

  auto b = at::randn({3, 3, 3});
  c10::List<int64_t> dims_b{0, 2, 1};
  std::vector<IValue> args_b{b, dims_b};

  testStaticRuntime(permute_script, args_a);
  testStaticRuntime(permute_script, args_a, args_b);
}

TEST(StaticRuntime, IndividualOps_Slice) {
  auto a = at::randn({2, 2});
  int dim_a = 1;
  int start_a = 0;
  int end_a = 1;
  int step_a = 1;
  std::vector<IValue> args_a{a, dim_a, start_a, end_a, step_a};

  auto b = at::randn({3, 3, 3});
  int dim_b = 2;
  int start_b = 0;
  int end_b = 1;
  int step_b = 2;
  std::vector<IValue> args_b{b, dim_b, start_b, end_b, step_b};

  testStaticRuntime(slice_script, args_a);
  testStaticRuntime(slice_script, args_a, args_b);
}

TEST(StaticRuntime, IndividualOps_Narrow) {
  auto a = at::randn({5, 5});
  int dim_a = 0;
  int start_a_int = 3;
  int len_a = 2;
  std::vector<IValue> args_a{a, dim_a, start_a_int, len_a};

  auto b = at::randn({5, 5, 5});
  int dim_b = 1;
  int start_b_int = 2;
  int len_b = 3;
  std::vector<IValue> args_b{b, dim_b, start_b_int, len_b};

  testStaticRuntime(narrow_with_int_script, args_a);
  testStaticRuntime(narrow_with_int_script, args_a, args_b);
}

TEST(StaticRuntime, InvidualOps_TupleUnpack) {
  auto two_tup = c10::ivalue::Tuple::create({at::randn({1}), at::randn({1})});
  auto two_tup_large =
      c10::ivalue::Tuple::create({at::randn({2, 2}), at::randn({2, 2})});

  auto three_tup = c10::ivalue::Tuple::create(
      {at::randn({1}), at::randn({1}), at::randn({1})});
  auto three_tup_large = c10::ivalue::Tuple::create(
      {at::randn({2, 2}), at::randn({2, 2}), at::randn({2, 2})});

  testStaticRuntime(two_tuple_unpack_script, {two_tup});
  testStaticRuntime(two_tuple_unpack_script, {two_tup}, {two_tup_large});

  testStaticRuntime(three_tuple_unpack_script, {three_tup});
  testStaticRuntime(three_tuple_unpack_script, {three_tup}, {three_tup_large});
}

TEST(StaticRuntime, IndividualOps_Append) {
  std::vector<IValue> args_int{1};

  testStaticRuntime(append_int_script, args_int);

  std::vector<IValue> args_tensor{at::randn({1})};
  std::vector<IValue> args_tensor_large{at::randn({2, 2})};

  testStaticRuntime(append_tensor_script, args_tensor);
  testStaticRuntime(append_tensor_script, args_tensor, args_tensor_large);
}

TEST(StaticRuntime, QuantizedLinear) {
  at::Tensor weight =
      at::quantize_per_tensor(torch::randn({3, 2}), 2, 3, torch::kQInt8);
  at::Tensor input =
      at::quantize_per_tensor(torch::randn({3, 2}), 2, 3, torch::kQUInt8);

  at::Tensor weight_2 =
      at::quantize_per_tensor(torch::randn({4, 3}), 2, 3, torch::kQInt8);
  at::Tensor input_2 =
      at::quantize_per_tensor(torch::randn({4, 3}), 2, 3, torch::kQUInt8);

  testStaticRuntime(quantize_script, {input, weight}, {input_2, weight_2});
}

TEST(StaticRuntime, IndividualOps_VarStack) {
  // 2D tensors - stack dim = 0
  std::vector<IValue> args1 = {at::randn({6, 6}), at::randn({6, 6}), 0};
  testStaticRuntime(var_stack_script, args1);

  // 3D tensors - stack dim = 1
  std::vector<IValue> args2 = {at::randn({4, 5, 6}), at::randn({4, 5, 6}), 1};
  testStaticRuntime(var_stack_script, args2);

  // 3D tensors - stack dim = 2
  std::vector<IValue> args3 = {at::randn({4, 5, 6}), at::randn({4, 5, 6}), 2};
  testStaticRuntime(var_stack_script, args3);

  testStaticRuntime(var_stack_script, args1, args2);
}

TEST(StaticRuntime, IndividualOps_FmodTensor) {
  // fmod tensor version
  auto a = at::randn({2, 3});
  auto b = at::randn({2, 3});
  std::vector<IValue> args0{a, b};
  testStaticRuntime(fmod_tensor, args0);

  // check for dynamic shapes
  auto c = at::randn({4, 3, 2});
  auto d = at::randn({4, 3, 2});
  std::vector<IValue> args1{c, d};
  testStaticRuntime(fmod_tensor, args0, args1);
}

TEST(StaticRuntime, IndividualOps_FmodScalar) {
  auto a = at::randn({2, 3});

  // fmod scalar version
  std::vector<IValue> args2{a, 3};
  testStaticRuntime(fmod_scalar, args2);

  // check for dynamic shapes
  auto c = at::randn({4, 3, 2});
  std::vector<IValue> args3{c, 4};
  testStaticRuntime(fmod_scalar, args2, args3);
}

TEST(StaticRuntime, QEmbeddingBagByteUnpack) {
  auto a = torch::randn({8, 16}, at::ScalarType::Float);
  auto b = torch::randn({8 * 2, 16 * 2}, at::ScalarType::Float);

  testStaticRuntime(embedding_bag_byte_prepack_script, {a});
  testStaticRuntime(embedding_bag_byte_prepack_script, {a}, {b});
}

TEST(StaticRuntime, IndividualOps_LinalgNorm_ScalarOrd) {
  auto a = at::randn({2, 3});
  auto dim = std::vector<int64_t>({1});
  auto dtype = at::ScalarType::Float;

  std::vector<IValue> args0{a, 4, dim, true, dtype};
  testStaticRuntime(linalg_norm_ord_scalar, args0);

  auto b = at::randn({4, 5});
  std::vector<IValue> args1{b, 4, dim, true, dtype};
  testStaticRuntime(linalg_norm_ord_scalar, args0, args1);
}

TEST(StaticRuntime, IndividualOps_LinalgNorm_StringOrd) {
  auto a = at::randn({2, 3});
  auto dim = std::vector<int64_t>({0, 1});
  auto dtype = at::ScalarType::Float;

  std::vector<IValue> args0{a, "fro", dim, true, dtype};
  testStaticRuntime(linalg_norm_ord_str, args0);

  auto b = at::randn({4, 5});
  std::vector<IValue> args1{b, "fro", dim, true, dtype};
  testStaticRuntime(linalg_norm_ord_str, args0, args1);
}

TEST(StaticRuntime, IndividualOps_Cat) {
  auto graph = std::make_shared<Graph>();
  std::unordered_map<std::string, Value*> vmap;
  parseIR(cat_script, graph.get(), vmap);
  torch::jit::StaticModule smodule(graph);
  ASSERT_TRUE(getNodeWithKind(smodule, "aten::cat"));

  auto a = at::randn({2, 4});
  auto b = at::randn({3, 4});
  std::vector<IValue> args0{a, b, 0};

  testStaticRuntime(cat_script, args0);

  auto c = at::randn({3, 4});
  auto d = at::randn({3, 5});
  std::vector<IValue> args1{c, d, 1};
  testStaticRuntime(cat_script, args0, args1);
}

TEST(StaticRuntime, IndividualOps_Cumsum) {
  auto a = at::randn({2, 3});
  std::vector<IValue> args0{a, 0};
  testStaticRuntime(cumsum_script, args0);

  auto b = at::randn({4, 3});
  std::vector<IValue> args1{b, 1};
  testStaticRuntime(cumsum_script, args0, args1);
}

TEST(StaticRuntime, IndividualOps_CumsumDtype) {
  auto a = at::randn({1, 2});
  auto dtype = at::ScalarType::Float;
  std::vector<IValue> args0{a, 0, dtype};
  testStaticRuntime(cumsum_script_dtype, args0);

  auto b = at::randn({3, 4});
  std::vector<IValue> args1{b, 1, dtype};
  testStaticRuntime(cumsum_script_dtype, args0, args1);
}

TEST(StaticRuntime, IndividualOps_Nonzero) {
  auto a = at::randint(0, 2, {2, 3});
  testStaticRuntime(nonzero_tensor, {a});

  auto b = at::randint(0, 2, {4, 3, 2});
  testStaticRuntime(nonzero_tensor, {a}, {b});
}

TEST(StaticRuntime, SignedLog1p) {
  std::vector<IValue> args1 = {at::randn({2, 2})};
  testStaticRuntime(signed_log1p_script, args1, {}, true);

  std::vector<IValue> args2 = {at::randn({3, 3, 3})};
  testStaticRuntime(signed_log1p_script, args1, args2, true);
}

TEST(StaticRuntime, RemoveImmutableInputDictLookupsWithImmutableInputDict) {
  script::Module module("module");
  module.define(getitem_immutable_input_dict_script);
  torch::jit::StaticModule smodule(module);
  EXPECT_FALSE(hasNodeWithKind(smodule, "aten::__getitem__"));
  EXPECT_TRUE(hasNodeWithKind(smodule, "static_runtime::dict_unpack"));

  auto a = at::randn({2, 4});
  auto b = at::randn({2, 4});
  c10::Dict<c10::IValue, c10::IValue> dict(
      c10::IntType::get(), c10::TensorType::get());
  dict.insert(0, a);
  dict.insert(1, b);
  testStaticRuntime(getitem_immutable_input_dict_script, {dict});

  c10::Dict<c10::IValue, c10::IValue> dict0(
      c10::IntType::get(), c10::TensorType::get());
  auto a0 = at::randn({3, 4});
  auto b0 = at::randn({3, 4});
  dict0.insert(0, a0);
  dict0.insert(1, b0);
  testStaticRuntime(getitem_immutable_input_dict_script, {dict0});
}

TEST(StaticRuntime, RemoveImmutableInputDictLookupsWithMutableInputDict) {
  script::Module module("module");
  module.define(getitem_mutable_input_dict_script);
  torch::jit::StaticModule smodule(module);
  EXPECT_TRUE(hasNodeWithKind(smodule, "aten::__getitem__"));
  EXPECT_FALSE(hasNodeWithKind(smodule, "static_runtime::dict_unpack"));
}

TEST(StaticRuntime, VarTupleUnpack) {
  script::Module module("module");
  module.define(var_tuple_unpack_script);
  torch::jit::StaticModule smodule(module);
  EXPECT_FALSE(hasNodeWithKind(smodule, "prim::TupleUnpack"));
  EXPECT_TRUE(hasNodeWithKind(smodule, "static_runtime::VarTupleUnpack"));

  auto a = at::randn({2, 2});
  auto b = at::randn({3, 3, 3});
  std::vector<IValue> args1{c10::ivalue::Tuple::create(a, a), c10::ivalue::Tuple::create(1, 2)};
  std::vector<IValue> args2{c10::ivalue::Tuple::create(b, b), c10::ivalue::Tuple::create(1, 2)};

  testStaticRuntime(var_tuple_unpack_script, args1);
  testStaticRuntime(var_tuple_unpack_script, args1, args2);
}

TEST(StaticRuntime, VarTupleUnpack_NotApplied) {
  script::Module module("module");
  // In this script, the optimization is not applied since there is a computation between
  // the TupleUnpack nodes.
  module.define(var_tuple_unpack_not_applied_script);
  torch::jit::StaticModule smodule(module);
  EXPECT_FALSE(hasNodeWithKind(smodule, "static_runtime::VarTupleUnpack"));
  EXPECT_TRUE(hasNodeWithKind(smodule, "prim::TupleUnpack"));
}

TEST(StaticRuntime, IndividualOps_RemainderTensor) {
  const auto remainder_tensor = R"JIT(
    def forward(self, x, y):
        return torch.remainder(x, y).clone()
  )JIT";

  std::vector<IValue> args1 = {
      at::randint(0, 10, {2, 2}), at::randint(0, 10, {2, 2})};
  std::vector<IValue> args2 = {
      at::randint(0, 10, {3, 3}), at::randint(0, 10, {3, 3})};

  // Use allclose and equalnan since outputs may be NaN.
  testStaticRuntime(
      remainder_tensor,
      args1,
      /*args2*/ {},
      /*use_alloclose*/ true,
      /*use_equalnan*/ true);
  testStaticRuntime(
      remainder_tensor,
      args1,
      args2,
      /*use_allclose*/ true,
      /*use_equalnan*/ true);
}

TEST(StaticRuntime, IndividualOps_RemainderScalar) {
  const auto remainder_scalar = R"JIT(
    def forward(self, x, y: int):
        return torch.remainder(x, y).clone()
  )JIT";

  std::vector<IValue> args1 = {at::randint(0, 10, {2, 2}), 4};
  std::vector<IValue> args2 = {at::randint(0, 10, {3, 3}), 4};

  // Use allclose and equalnan since outputs may be NaN.
  testStaticRuntime(
      remainder_scalar,
      args1,
      /*args2*/ {},
      /*use_alloclose*/ true,
      /*use_equalnan*/ true);
  testStaticRuntime(
      remainder_scalar,
      args1,
      args2,
      /*use_allclose*/ true,
      /*use_equalnan*/ true);
}

TEST(StaticRuntime, ControlFlow_JumpIf) {
  auto test_jump = [](const std::string& ir, bool jump_arg) {
    auto graph = getGraphFromIR(ir);
    StaticModuleOptions opt = {
      .cleanup_activations = true,
      .enable_out_variant = true,
      .optimize_memory = true
    };
    torch::jit::StaticModule smodule(graph, opt);

    // Both adds are executed
    auto result_no_jmp = smodule({!jump_arg}, {});
    ASSERT_TRUE(result_no_jmp.isTuple());
    const auto& elems_no_jmp = result_no_jmp.toTuple()->elements();
    ASSERT_EQ(elems_no_jmp.size(), 2);
    for (const auto& e : elems_no_jmp) {
      ASSERT_TRUE(e.isInt());
      ASSERT_EQ(e.toInt(), 2);
    }

    // One add is skipped
    auto results_jmp = smodule({jump_arg}, {});
    ASSERT_TRUE(results_jmp.isTuple());
    const auto& elems_jmp = results_jmp.toTuple()->elements();
    ASSERT_EQ(elems_jmp.size(), 2);
    ASSERT_TRUE(elems_jmp[0].isNone());
    ASSERT_TRUE(elems_jmp[1].isInt());
    ASSERT_EQ(elems_jmp[1].toInt(), 2);
  };

  const std::string jump_if = R"IR(
    graph(%0: bool):
        %target : int = prim::Constant[value=2]()
        %a : int = prim::Constant[value=1]()
        static_runtime::JumpIf(%0, %target)
        %2 : int = aten::add(%a, %a)
        %3 : int = aten::add(%a, %a)
        return (%2, %3)
  )IR";

  const std::string jump_if_not = R"IR(
    graph(%0: bool):
        %target : int = prim::Constant[value=2]()
        %a : int = prim::Constant[value=1]()
        static_runtime::JumpIfNot(%0, %target)
        %2 : int = aten::add(%a, %a)
        %3 : int = aten::add(%a, %a)
        return (%2, %3)
  )IR";

  test_jump(jump_if, true);
  test_jump(jump_if_not, false);
}

TEST(StaticRuntime, ControlFlow_Jump) {
  const std::string ir = R"IR(
    graph():
        %target : int = prim::Constant[value=2]()
        %a : int = prim::Constant[value=1]()
        static_runtime::Jump(%target)
        %2 : int = aten::add(%a, %a)
        %3 : int = aten::add(%a, %a)
        return (%2, %3)
  )IR";

  auto graph = getGraphFromIR(ir);
  StaticModuleOptions opt = {
    .cleanup_activations = true,
    .enable_out_variant = true,
    .optimize_memory = true
  };
  torch::jit::StaticModule smodule(graph, opt);

  // One add is skipped
  auto results_jmp = smodule({}, {});
  ASSERT_TRUE(results_jmp.isTuple());
  const auto& elems_jmp = results_jmp.toTuple()->elements();
  ASSERT_EQ(elems_jmp.size(), 2);
  ASSERT_TRUE(elems_jmp[0].isNone());
  ASSERT_TRUE(elems_jmp[1].isInt());
  ASSERT_EQ(elems_jmp[1].toInt(), 2);
}
