#include <ATen/core/dispatch/OperatorOptions.h>
#include <c10/core/ScalarType.h>
#include <gtest/gtest.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/runtime/static/ProcessedNodeInputs.h>
#include <torch/csrc/jit/runtime/static/impl.h>
#include <torch/csrc/jit/runtime/static/passes.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <stdexcept>

#include "deep_wide_pt.h"
#include "test_utils.h"

using namespace caffe2;
using namespace torch;
using namespace torch::jit;
using namespace torch::jit::test;
using c10::IValue;

/*
 When adding a test for an operator implemented in static runtime, there are
 several things that you need to pay attention to:

 1) if the op is an out variant, in the test script of the op,
 instead of:
    def forward(self, input):
      return myop(input)

  do:
    def forward(self, input):
      return myop(input).clone()

 This makes sure that the output of myop is managed by the memory planner and
 exercise the code path in the op impl that otherwise doesn't get exercised. The
 output of the model is not managed by the memory planner, because it needs to
 be returned to the client.

 2) The memory planner rounds up the size of each Tensor's storage to multiples
 of 64 bytes (alignment requirement on AVX512). Make sure the sizes of the input
 tensors in args2 are big enough to trigger resizing.

 3) for view ops such as aten::reshape or aten::to, if you want it to be
 replaced by the copy version with the ReplaceWithCopy pass in passes.h, you
 also want to make sure its output is not returned as the model output. The
 reason is that ReplaceWithCopy only replaces the op whose output is not an
 alias of the model output.
*/

C10_DECLARE_bool(static_runtime_enable_fast_math);

TEST(StaticRuntime, UnaryOps) {
  const auto aten_sum = R"JIT(
    def forward(self, input):
        return torch.sum(input).clone()
  )JIT";

  const auto aten_sum_0 = R"JIT(
    def forward(self, input):
        return torch.sum(input, 0).clone()
  )JIT";

  const auto aten_sum_1 = R"JIT(
    def forward(self, input):
        return torch.sum(input, 1).clone()
  )JIT";

  const auto aten_sum_0_true = R"JIT(
    def forward(self, input):
        return torch.sum(input, 0, True).clone()
  )JIT";

  const auto aten_sum_1_true = R"JIT(
    def forward(self, input):
        return torch.sum(input, 1, True).clone()
  )JIT";

  auto a = at::randn({2, 3});
  auto b = at::randn({3, 3, 6});

  std::vector<IValue> args{a}, args2{b};

  // sum
  testStaticRuntime(aten_sum, args);
  testStaticRuntime(aten_sum_0, args);
  testStaticRuntime(aten_sum_1, args);
  testStaticRuntime(aten_sum_0_true, args);
  testStaticRuntime(aten_sum_1_true, args);

  testStaticRuntime(aten_sum, args, args2, false, false, false);
  testStaticRuntime(aten_sum_0, args, args2);
  testStaticRuntime(aten_sum_1, args, args2);
  testStaticRuntime(aten_sum_0_true, args, args2);
  testStaticRuntime(aten_sum_1_true, args, args2);
}

TEST(StaticRuntime, Max) {
  auto src_max_reduce = R"JIT(
    def forward(self, input):
        return torch.max(input).clone()
  )JIT";

  auto src_max_dim = R"JIT(
    def forward(self, input, dim: int):
        values, indices = torch.max(input, dim)
        return values.clone(), indices.clone()
  )JIT";

  auto src_max_dim_keepdim = R"JIT(
    def forward(self, input, dim: int):
        values, indices = torch.max(input, dim, keepdim=True)
        return values.clone(), indices.clone()
  )JIT";

  auto src_max_pointwise = R"JIT(
    def forward(self, input, other):
        return torch.max(input, other).clone()
  )JIT";

  auto input = at::randn({2, 3, 2});
  auto input_other = at::randn({2, 3, 2});
  auto large_input = at::randn({8, 9, 10});
  auto large_input_other = at::randn({8, 9, 10});

  testStaticRuntime(src_max_reduce, {input});
  testStaticRuntime(src_max_dim, {input, 1});
  testStaticRuntime(src_max_dim, {input, 1}, {large_input, 0});
  testStaticRuntime(src_max_dim_keepdim, {input, 0});
  testStaticRuntime(src_max_dim_keepdim, {input, 0}, {large_input, 2});
  testStaticRuntime(src_max_pointwise, {input, input_other});
  testStaticRuntime(src_max_pointwise, {input, input_other}, {large_input, large_input_other});
}

TEST(StaticRuntime, Mean) {
  const auto src_default = R"JIT(
    def forward(self, input):
        return torch.mean(input).clone()
  )JIT";
  const auto src_dtype = R"JIT(
    def forward(self, input, dtype: int):
        return torch.mean(input, dtype=dtype).clone()
  )JIT";
  const auto src_dim = R"JIT(
    def forward(self, input, dim: List[int]):
        return torch.mean(input, dim).clone()
  )JIT";
  const auto src_dim_keepdim = R"JIT(
    def forward(self, input, dim: List[int]):
        return torch.mean(input, dim, keepdim=True).clone()
  )JIT";
  const auto src_dim_dtype = R"JIT(
    def forward(self, input, dim: List[int], dtype: int):
        return torch.mean(input, dim, dtype=dtype).clone()
  )JIT";

  auto input = at::randn({2, 3, 2});
  auto large_input = at::randn({8, 7, 6, 8});

  std::vector<IValue> args_default = {input};
  std::vector<IValue> args_dtype = {input, torch::kFloat};
  std::vector<IValue> args_dim = {input, c10::List<int64_t>{0, 2}};
  std::vector<IValue> args_dim_keepdim = {input, c10::List<int64_t>{1, 2}};
  std::vector<IValue> args_dim_dtype = {input, c10::List<int64_t>{0, 1}, torch::kBFloat16};

  testStaticRuntime(src_default, args_default);
  testStaticRuntime(src_dtype, args_dtype);
  testStaticRuntime(src_dim, args_dim);
  testStaticRuntime(src_dim_keepdim, args_dim_keepdim);
  testStaticRuntime(src_dim_dtype, args_dim_dtype);

  std::vector<IValue> large_args_dim = {large_input, c10::List<int64_t>{0, 3}};
  std::vector<IValue> large_args_dim_keepdim = {large_input, c10::List<int64_t>{1, 2}};
  std::vector<IValue> large_args_dim_dtype = {large_input, c10::List<int64_t>{1, 3}, torch::kBFloat16};

  testStaticRuntime(src_dim, args_dim, large_args_dim);
  testStaticRuntime(src_dim_keepdim, args_dim_keepdim, large_args_dim_keepdim);
  testStaticRuntime(src_dim_dtype, args_dim_dtype, large_args_dim_dtype);
}

TEST(StaticRuntime, Sigmoid) {
  const auto sigmoid_script = R"JIT(
    def forward(self, inp: Tensor):
        b = torch.sigmoid(inp).clone()
        return (b)
  )JIT";
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
  /*
  Clone called two times to trigger memory planner for output of first clone.
  The output of last op(second clone) is not managed by memory planner since it
  needs to be returned to the client and cannot be reused by planner.
  */
  const auto clone_script_0 = R"JIT(
    def forward(self, input):
        a = torch.clone(input).clone()
        return (a * a)
  )JIT";

  // Case: clone with different set of memory_formats
  const auto clone_script_1 = R"JIT(
    def forward(self, input: Tensor, memory_format: int):
        a = torch.clone(input, memory_format=memory_format).clone()
        return (a * a)
  )JIT";

  /*
  Case: input stride set to 0 (due to expand op)
  calls native clone instead of out variant
  */
  const auto clone_script_2 = R"JIT(
    def forward(self, input: Tensor, other:Tensor):
        a = input.expand_as(other)
        return a.clone().clone()
  )JIT";

  /*
  Case: testing the case of sliced tensor for
  testing non-contiguous tensor storage
  */
  const auto clone_script_3 = R"JIT(
    def forward(self, input: Tensor):
        a = input[:, 0:10:2]
        return a.clone().clone()
  )JIT";

  auto a = at::randn({2, 3});
  auto b = at::randn({3, 2}).as_strided({3, 2}, {1, 3});
  auto b_larger = at::randn({30, 20}).as_strided({30, 20}, {1, 3});
  auto c = at::randn({1, 20, 13, 8});
  auto d = at::randn({1, 0, 3, 4});
  auto e = at::randn({2, 1});
  auto f = at::randn({2, 10});
  auto g = at::randn({3, 20});
  std::vector<IValue> args_0{b, c10::MemoryFormat::Contiguous};
  std::vector<IValue> args_1{b_larger, c10::MemoryFormat::Preserve};
  std::vector<IValue> args_2{c, c10::MemoryFormat::ChannelsLast};
  std::vector<IValue> args_3{d, c10::MemoryFormat::ChannelsLast};
  std::vector<IValue> args_4{e,a};
  std::vector<IValue> args_5{e,f};

  testStaticRuntime(clone_script_0, {a});
  testStaticRuntime(clone_script_0, {a}, {b_larger});

  testStaticRuntime(clone_script_1, args_0);
  testStaticRuntime(clone_script_1, args_1);
  testStaticRuntime(clone_script_1, args_2);
  testStaticRuntime(clone_script_1, args_3);
  testStaticRuntime(clone_script_1, args_0, args_1);
  testStaticRuntime(clone_script_1, args_3, args_2);

  testStaticRuntime(clone_script_2, args_4);
  testStaticRuntime(clone_script_2, args_4, args_5);

  testStaticRuntime(clone_script_3, {f});
  testStaticRuntime(clone_script_3, {f}, {g});
}

TEST(StaticRuntime, Clamp) {
  const auto clamp_script_1 = R"JIT(
    def forward(self, inp: Tensor, min: int, max: int):
        a = torch.clamp(inp, min, max).clone()
        return (a)
  )JIT";

  const auto clamp_script_2 = R"JIT(
    def forward(self, inp: Tensor, min: Tensor, max: Tensor):
        a = torch.clamp(inp, min, max).clone()
        return (a)
  )JIT";
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

TEST(StaticRuntime, ClampMinOnly) {
  const auto src = R"JIT(
    def forward(self, inp: Tensor, min: float):
        a = torch.clamp(inp, min, None).clone()
        return (a)
  )JIT";
  auto a = at::randn({2, 3});
  auto b = at::randn({4, 3, 2});
  testStaticRuntime(src, {a, 0.5});
  testStaticRuntime(src, {a, 0.5}, {b, 0.25});
}

TEST(StaticRuntime, ClampMaxOnly) {
  const auto src = R"JIT(
    def forward(self, inp: Tensor, max: float):
        a = torch.clamp(inp, None, max).clone()
        return (a)
  )JIT";
  auto a = at::randn({2, 3});
  auto b = at::randn({4, 3, 2});
  testStaticRuntime(src, {a, 0.5});
  testStaticRuntime(src, {a, 0.5}, {b, 0.25});
}

TEST(StaticRuntime, ClampIntTensor) {
  const auto src = R"JIT(
    def forward(self, inp: Tensor, min: float, max: float):
        a = torch.clamp(inp, min, max).clone()
        return (a)
  )JIT";
  auto a = at::randint(0, 20, {2, 3}, at::kFloat);
  auto b = at::randint(0, 20, {4, 3, 2}, at::kFloat);
  auto min = 5.0f;
  auto max = 5.0f;
  testStaticRuntime(src, {a, min, max});
  testStaticRuntime(src, {a, min, max}, {b, min, max});
}

TEST(StaticRuntime, LenWithTuple) {
  const auto src = R"IR(
    graph(%input : int[]):
        %res : int = aten::len(%input)
        return (%res)
  )IR";

  testStaticRuntime(src, {c10::List<int64_t>(4)});
}

TEST(StaticRuntime, LenWithTensor) {
  const auto src = R"IR(
    graph(%input : Tensor):
        %res : int = aten::len(%input)
        return (%res)
  )IR";

  testStaticRuntime(src, {at::randn({2, 2, 2})});
}

TEST(StaticRuntime, LenWithStr) {
  const auto src = R"IR(
    graph(%input : str):
        %res : int = aten::len(%input)
        return (%res)
  )IR";

  testStaticRuntime(src, {"static_runtime"});
}

TEST(StaticRuntime, LenWithDict_str) {
  const auto script = R"JIT(
    def forward(self, input: Dict[str, str]):
        return len(input)
  )JIT";

  c10::Dict<std::string, std::string> dict;
  dict.insert("abc", "123");
  dict.insert("def", "456");
  testStaticRuntime(script, {dict});
}

TEST(StaticRuntime, LenWithDict_int) {
  const auto script = R"JIT(
    def forward(self, input: Dict[int, int]):
        return len(input)
  )JIT";

  c10::Dict<int64_t, int64_t> dict;
  dict.insert(0, 1);
  dict.insert(2, 3);
  testStaticRuntime(script, {dict});
}

TEST(StaticRuntime, LenWithDict_bool) {
  const auto script = R"JIT(
    def forward(self, input: Dict[bool, bool]):
        return len(input)
  )JIT";

  c10::Dict<bool, bool> dict;
  dict.insert(true, false);
  dict.insert(false, true);
  testStaticRuntime(script, {dict});
}

TEST(StaticRuntime, LenWithDict_float) {
  const auto script = R"JIT(
    def forward(self, input: Dict[float, float]):
        return len(input)
  )JIT";

  c10::Dict<double, double> dict;
  dict.insert(0.1, 0.9);
  dict.insert(0.8, 0.18);
  testStaticRuntime(script, {dict});
}

TEST(StaticRuntime, LenWithDict_complex) {
  const auto script = R"JIT(
    def forward(self, input: Dict[complex, complex]):
        return len(input)
  )JIT";

  c10::Dict<c10::complex<double>, c10::complex<double>> dict;
  dict.insert(0.1, 0.4);
  dict.insert(0.9, 0.45);
  testStaticRuntime(script, {dict});
}

TEST(StaticRuntime, LenWithDict_Tensor) {
  const auto script = R"JIT(
    def forward(self, input: Dict[Tensor, Tensor]):
        return len(input)
  )JIT";

  c10::Dict<at::Tensor, at::Tensor> dict;
  dict.insert(at::randn({1, 2}), at::randn({1, 2}));
  dict.insert(at::randn({1, 2}), at::randn({1, 2}));
  testStaticRuntime(script, {dict});
}

TEST(StaticRuntime, Logit) {
  // no nnc
  const auto logit_script_1 = R"JIT(
    def forward(self, inp: Tensor):
        a = torch.logit(inp).clone()
        return (a)
  )JIT";

  // with nnc
  const auto logit_script_2 = R"JIT(
    def forward(self, inp: Tensor):
        a = torch.logit(inp, 1e-6).clone()
        return (a)
  )JIT";

  // no nnc
  const auto logit_script_3 = R"JIT(
    def forward(self, inp: Tensor, eps: float):
        a = torch.logit(inp, eps).clone()
        return (a)
  )JIT";
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

TEST(StaticRuntime, EmbeddingBag) {
  const std::string embedding_bag_default = R"JIT(
    def forward(self, a: Tensor, b: Tensor, c: Tensor):
        x, y, z, _ = torch.embedding_bag(a, b, c)
        return (x.clone(), y.clone(), z.clone(), _.clone())
  )JIT";

  const std::string embedding_bag_mean = R"JIT(
    def forward(self, a: Tensor, b: Tensor, c: Tensor):
        x, y, z, _ = torch.embedding_bag(a, b, c, False, 1)
        return (x.clone(), y.clone(), z.clone(), _.clone())
  )JIT";

  const std::string embedding_bag_max = R"JIT(
    def forward(self, a: Tensor, b: Tensor, c: Tensor):
        x, y, z, _ = torch.embedding_bag(a, b, c, False, 2)
        return (x.clone(), y.clone(), z.clone(), _.clone())
  )JIT";

  const std::string embedding_bag_sum_last_offset = R"JIT(
    def forward(self, a: Tensor, b: Tensor, c: Tensor):
        x, y, z, _ = torch.embedding_bag(a, b, c, False, 0, False, None, True)
        return (x.clone(), y.clone(), z.clone(), _.clone())
  )JIT";

  const std::string embedding_bag_mean_last_offset = R"JIT(
    def forward(self, a: Tensor, b: Tensor, c: Tensor):
        x, y, z, _ = torch.embedding_bag(a, b, c, False, 1, False, None, True)
        return (x.clone(), y.clone(), z.clone(), _.clone())
  )JIT";

  const std::string embedding_bag_max_last_offset = R"JIT(
    def forward(self, a: Tensor, b: Tensor, c: Tensor):
        x, y, z, _ = torch.embedding_bag(a, b, c, False, 2, False, None, True)
        return (x.clone(), y.clone(), z.clone(), _.clone())
  )JIT";

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

  at::Tensor weight2 = torch::randn({10, 11}, at::ScalarType::Float);
  at::Tensor input2 = torch::tensor({0, 1, 0, 2, 1});
  at::Tensor offset2 = torch::tensor({0, 1, 2, 3, 4, 5});
  std::vector<IValue> args2{weight2, input2, offset2};
  testStaticRuntime(embedding_bag_default, args, args2);
  testStaticRuntime(embedding_bag_mean, args, args2);
  testStaticRuntime(embedding_bag_max, args, args2);
  testStaticRuntime(embedding_bag_sum_last_offset, args, args2);
  testStaticRuntime(embedding_bag_mean_last_offset, args, args2);
  testStaticRuntime(embedding_bag_max_last_offset, args, args2);
}

TEST(StaticRuntime, EmbeddingBagWithManagedOutput) {
  const std::string embedding_bag_managed_output = R"JIT(
    def forward(self, a: Tensor, b: Tensor, c: Tensor):
        # The outputs of embedding_bag become an intermediate tensors
        # since they are not directly returned from the graph.
        x, y, z, _ = torch.embedding_bag(a, b, c)
        return x + x
  )JIT";

  at::Tensor weight = torch::randn({3, 8}, at::ScalarType::Float);
  at::Tensor input = torch::tensor({0, 1, 0, 2});
  at::Tensor offset = torch::tensor({0, 2});
  std::vector<IValue> args{weight, input, offset};

  at::Tensor weight2 = torch::randn({6, 8}, at::ScalarType::Float);
  at::Tensor input2 = torch::tensor({0, 1, 0, 2, 3, 4});
  at::Tensor offset2 = torch::tensor({0, 2, 4, 5});
  std::vector<IValue> args2{weight2, input2, offset2};

  testStaticRuntime(embedding_bag_managed_output, args);
  testStaticRuntime(embedding_bag_managed_output, args, args2);
}

TEST(StaticRuntime, EmbeddingBagWithExtraneousOutput) {
  const std::string embedding_bag_default_ir = R"IR(
    graph(%weight, %indices, %offsets):
        %scale_grad_by_freq : bool = prim::Constant[value=0]()
        %mode : int = prim::Constant[value=0]()
        %sparse : bool = prim::Constant[value=0]()
        %per_sample_weights : NoneType = prim::Constant()
        %include_last_offset : bool = prim::Constant[value=0]()
        %y0 : Tensor, %y1 : Tensor, %y2 : Tensor, %y3 : Tensor = aten::embedding_bag(%weight, %indices, %offsets, %scale_grad_by_freq, %mode, %sparse, %per_sample_weights, %include_last_offset)
        %none : NoneType = prim::Constant()
        %res : Tensor = aten::clone(%y0, %none)
        return (%res)
  )IR";
  auto graph = getGraphFromIR(embedding_bag_default_ir);
  RemoveUnnecessaryOutputs(graph);
  torch::jit::testing::FileCheck()
      .check("static_runtime::embedding_bag")
      ->run(*graph);

  const std::string embedding_bag_mean_ir = R"IR(
    graph(%weight, %indices, %offsets):
        %scale_grad_by_freq : bool = prim::Constant[value=0]()
        %mode : int = prim::Constant[value=1]()
        %sparse : bool = prim::Constant[value=0]()
        %per_sample_weights : NoneType = prim::Constant()
        %include_last_offset : bool = prim::Constant[value=0]()
        %y0 : Tensor, %y1 : Tensor, %y2 : Tensor, %y3 : Tensor = aten::embedding_bag(%weight, %indices, %offsets, %scale_grad_by_freq, %mode, %sparse, %per_sample_weights, %include_last_offset)
        %none : NoneType = prim::Constant()
        %res : Tensor = aten::clone(%y0, %none)
        return (%res)
  )IR";
  graph = getGraphFromIR(embedding_bag_mean_ir);
  RemoveUnnecessaryOutputs(graph);
  torch::jit::testing::FileCheck()
      .check("static_runtime::embedding_bag")
      ->run(*graph);

  const std::string embedding_bag_max_last_offset_ir = R"IR(
    graph(%weight, %indices, %offsets):
        %scale_grad_by_freq : bool = prim::Constant[value=0]()
        %mode : int = prim::Constant[value=2]()
        %sparse : bool = prim::Constant[value=0]()
        %per_sample_weights : NoneType = prim::Constant()
        %include_last_offset : bool = prim::Constant[value=1]()
        %y0 : Tensor, %y1 : Tensor, %y2 : Tensor, %y3 : Tensor = aten::embedding_bag(%weight, %indices, %offsets, %scale_grad_by_freq, %mode, %sparse, %per_sample_weights, %include_last_offset)
        %none : NoneType = prim::Constant()
        %res : Tensor = aten::clone(%y0, %none)
        return (%res)
  )IR";
  graph = getGraphFromIR(embedding_bag_max_last_offset_ir);
  RemoveUnnecessaryOutputs(graph);
  torch::jit::testing::FileCheck()
      .check("static_runtime::embedding_bag")
      ->run(*graph);

  const std::string embedding_bag_normal_ir = R"IR(
    graph(%weight, %indices, %offsets):
        %scale_grad_by_freq : bool = prim::Constant[value=0]()
        %mode : int = prim::Constant[value=0]()
        %sparse : bool = prim::Constant[value=0]()
        %per_sample_weights : NoneType = prim::Constant()
        %include_last_offset : bool = prim::Constant[value=0]()
        %y0 : Tensor, %y1 : Tensor, %y2 : Tensor, %y3 : Tensor = aten::embedding_bag(%weight, %indices, %offsets, %scale_grad_by_freq, %mode, %sparse, %per_sample_weights, %include_last_offset)
        %none : NoneType = prim::Constant()
        %res0 : Tensor = aten::clone(%y0, %none)
        %res1 : Tensor = aten::clone(%y1, %none)
        %res2 : Tensor = aten::clone(%y2, %none)
        %res3 : Tensor = aten::clone(%y3, %none)
        return (%res0, %res1, %res2, %res3)
  )IR";
  graph = getGraphFromIR(embedding_bag_normal_ir);
  RemoveUnnecessaryOutputs(graph);
  torch::jit::testing::FileCheck()
      .check_not("static_runtime::embedding_bag")
      ->run(*graph);

  at::Tensor weight = torch::randn({3, 11}, at::ScalarType::Float);
  at::Tensor input = torch::tensor({0, 1, 0, 2});
  at::Tensor offset = torch::tensor({0, 2, 4});
  std::vector<IValue> args{weight, input, offset};
  testStaticRuntime(embedding_bag_default_ir, args);
  testStaticRuntime(embedding_bag_mean_ir, args);
  testStaticRuntime(embedding_bag_max_last_offset_ir, args);

  at::Tensor weight2 = torch::randn({10, 11}, at::ScalarType::Float);
  at::Tensor input2 = torch::tensor({0, 1, 0, 2, 1});
  at::Tensor offset2 = torch::tensor({0, 1, 2, 3, 4, 5});
  std::vector<IValue> args2{weight2, input2, offset2};
  testStaticRuntime(embedding_bag_default_ir, args, args2);
  testStaticRuntime(embedding_bag_mean_ir, args, args2);
  testStaticRuntime(embedding_bag_max_last_offset_ir, args, args2);
}

TEST(StaticRuntime, LayerNorm) {
  const std::string layer_norm_with_weights = R"JIT(
    def forward(self, input: Tensor, normalized_shape: List[int], weight: Tensor, bias: Tensor):
        return torch.layer_norm(input, normalized_shape, weight, bias, 1e-05, False).clone()
  )JIT";

  const std::string layer_norm_without_weights = R"JIT(
    def forward(self, input: Tensor, normalized_shape: List[int]):
        return torch.layer_norm(input, normalized_shape, None, None, 1e-05, False).clone()
  )JIT";

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
  const auto bmm_script = R"JIT(
    def forward(self, inp: Tensor, mat2: Tensor):
      return torch.bmm(inp, mat2).clone()
  )JIT";

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
  const auto addmm_script = R"JIT(
    def forward(self, inp: Tensor, mat1: Tensor, mat2: Tensor, beta: float, alpha: float):
      return torch.addmm(inp, mat1, mat2, alpha=alpha, beta=beta).clone()
  )JIT";
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

TEST(StaticRuntime, Abs) {
  const auto abs_script = R"JIT(
    def forward(self, a):
      return a.abs().clone()
  )JIT";
  auto a = at::randn({2, 3});
  auto b = at::randn({4, 2, 3});
  std::vector<IValue> args{a};
  std::vector<IValue> args2{b};
  testStaticRuntime(abs_script, args);
  testStaticRuntime(abs_script, args, args2);
}

TEST(StaticRuntime, Binary) {
  const auto add_script = R"JIT(
    def forward(self, a, b):
        c = a + b
        return (c.clone())
  )JIT";

  const auto add_script_ints = R"JIT(
    def forward(self, a: int, b: int):
        c = a + b
        d = c + 1
        return d
  )JIT";

  const auto add_list_script = R"JIT(
    def forward(self, a: List[int], b: List[int]):
        c = a + b
        return c[::]
  )JIT";

  const auto list_construct_script = R"JIT(
    def forward(self, a, b):
      return [a, b]
  )JIT";

  const auto list_construct_script_2 = R"JIT(
    def forward(self, a, b):
      c = a + a
      return [c, c]
  )JIT";

  const auto list_construct_script_3 = R"JIT(
    def forward(self, a, b):
      c = a + a
      return [c, c.flatten()]
  )JIT";

  const auto list_unpack_script = R"JIT(
    def forward(self, a, b):
      c = [a, b]
      x, y = c
      z = x + y
      return z.clone()
  )JIT";

  const auto list_unpack_script_2 = R"JIT(
    def forward(self, a, b):
      c = [a, b]
      x, y = c
      z = (x, y)
      return z
  )JIT";

  const auto tuple_construct_script = R"JIT(
    def forward(self, a, b):
      return (a, b)
  )JIT";

  const auto tuple_construct_script_2 = R"JIT(
    def forward(self, a, b):
      return (a.flatten(), b)
  )JIT";

  auto a = at::randn({2, 3});
  auto b = at::ones({2, 3});

  auto c = at::randn({4, 2, 3});
  auto d = at::ones({4, 2, 3});

  std::vector<IValue> args{a, b};

  testStaticRuntime(add_script, args);
  testStaticRuntime(add_script_ints, {1, 2});
  testStaticRuntime(add_script, args, {c, d});
  testStaticRuntime(list_construct_script, args);
  testStaticRuntime(list_construct_script_2, args);
  testStaticRuntime(list_construct_script_3, args);
  testStaticRuntime(list_unpack_script, args);
  testStaticRuntime(list_unpack_script_2, args);
  testStaticRuntime(tuple_construct_script, args);
  testStaticRuntime(tuple_construct_script_2, args);

  std::vector<IValue> list_args{
      c10::List<int64_t>{1, 2, 3}, c10::List<int64_t>{4, 5, 6}};
  testStaticRuntime(add_list_script, list_args);
}

TEST(StaticRuntime, MatMul) {
  const auto aten_matmul = R"JIT(
    def forward(self, a: Tensor, b: Tensor):
        return torch.matmul(a, b).clone()
  )JIT";

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

TEST(StaticRuntime, Sign) {
  const auto sign_tensor = R"JIT(
    def forward(self, input: Tensor):
        return torch.sign(input).clone()
  )JIT";

  auto a = at::randn({2, 3});
  auto b = at::randn({4, 3, 2});

  std::vector<IValue> args{a};
  testStaticRuntime(sign_tensor, args);
  testStaticRuntime(sign_tensor, args, {b});
}

TEST(StaticRuntime, Div) {
  const auto div_tensor = R"JIT(
    def forward(self, a: Tensor, b: Tensor):
        return torch.div(a, b).clone()
  )JIT";

  const auto div_scalar = R"JIT(
    def forward(self, a: Tensor, b: int):
        return torch.div(a, b).clone()
  )JIT";

  const auto div_tensor_mode = R"JIT(
    def forward(self, a: Tensor, b: Tensor, c: str):
        return torch.div(a, b, rounding_mode=c).clone()
  )JIT";

  const auto div_scalar_mode = R"JIT(
    def forward(self, a: Tensor, b: float, c: str):
        return torch.div(a, b, rounding_mode=c).clone()
  )JIT";

  const auto div_strided = R"JIT(
    def forward(self, a: Tensor, b: Tensor):
        a_strided = torch.transpose(a, 0, 1)
        b_strided = torch.transpose(b, 0, 1)
        return torch.div(a_strided, b_strided).clone()
  )JIT";

  auto a = at::randn({2, 3});
  auto b = at::randn({2, 3});
  auto bs = at::randn({3, 2}).transpose(0, 1);
  auto c = at::randn({4, 3, 2});
  auto d = at::randn({4, 3, 2});
  auto ds = at::randn({3, 4, 2}).transpose(0, 1);

  std::vector<IValue> args0{a, b};
  testStaticRuntime(div_tensor, args0);
  testStaticRuntime(div_tensor, args0, {c, d});

  testStaticRuntime(div_strided, args0);
  testStaticRuntime(div_strided, args0, {c, d});

  testStaticRuntime(div_tensor, {a, bs});
  testStaticRuntime(div_tensor, {a, bs}, {c, ds});

  std::vector<IValue> args1{a, 3};
  testStaticRuntime(div_scalar, args1);
  testStaticRuntime(div_scalar, args1, {c, 4});

  std::vector<IValue> args2{a, b, "floor"};
  testStaticRuntime(div_tensor_mode, args2);
  testStaticRuntime(div_tensor_mode, args2, {c, d, "floor"});

  std::vector<IValue> args3{a, 2.3, "trunc"};
  testStaticRuntime(div_scalar_mode, args3);
  testStaticRuntime(div_scalar_mode, args3, {c, 1.5, "trunc"});
}

TEST(StaticRuntime, Mul) {
  const auto mul_tensor = R"JIT(
    def forward(self, a: Tensor, b: Tensor):
        return torch.mul(a, b).clone()
  )JIT";

  const auto mul_scalar = R"JIT(
    def forward(self, a: Tensor, b: int):
        return torch.mul(a, b).clone()
  )JIT";

  const auto mul_list = R"JIT(
    def forward(self, a: List[int], n: int):
        b = a * n
        return b[::]
  )JIT";

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

  std::vector<IValue> list_args{c10::List<int64_t>{1, 2}, 3};
  testStaticRuntime(mul_list, list_args);
}

TEST(StaticRuntime, Log) {
  const auto log_tensor = R"JIT(
    def forward(self, inp: Tensor):
        a = torch.log(inp).clone()
        return (a)
  )JIT";

  // Ensure that the input values are valid.
  auto a = at::abs(at::randn({2, 3}));
  auto b = at::abs(at::randn({4, 3, 2}));

  std::vector<IValue> args{a};
  testStaticRuntime(log_tensor, args);
  testStaticRuntime(log_tensor, args, {b});
}

TEST(StaticRuntime, Sub) {
  const auto sub_tensor = R"JIT(
    def forward(self, a: Tensor, b: Tensor):
        return torch.sub(a, b).clone()
  )JIT";

  const auto sub_scalar = R"JIT(
    def forward(self, a: Tensor, b: int):
        return torch.sub(a, b).clone()
  )JIT";

  const auto sub_tensor_alpha = R"JIT(
    def forward(self, a: Tensor, b: Tensor, c: float):
        return torch.sub(a, b, alpha=c).clone()
  )JIT";

  const auto sub_scalar_alpha = R"JIT(
    def forward(self, a: Tensor, b: float, c: int):
        return torch.sub(a, b, alpha=c).clone()
  )JIT";

  const auto sub_two_scalars = R"JIT(
    def forward(self, a: int, b: int):
        return (a - b - b)
  )JIT";

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

  std::vector<IValue> args4{1, 2};
  testStaticRuntime(sub_two_scalars, args4);
}

TEST(StaticRuntime, NanToNum) {
  const auto nan_to_num_script = R"JIT(
    def forward(self, a: Tensor, nan: float, posinf: float, neginf: float):
        return torch.nan_to_num(a, nan, posinf, neginf).clone()
  )JIT";

  const auto inf = std::numeric_limits<double>::infinity();
  const auto nan = std::numeric_limits<double>::quiet_NaN();

  auto a = torch::tensor({{1.0, nan}, {-inf, inf}});
  auto b = at::randn({3, 6});
  float* b_data = b.data_ptr<float>();
  b_data[0] = nan;
  b_data[4] = -inf;
  b_data[11] = inf;
  b_data[13] = nan;

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

TEST(StaticRuntime, Stack) {
  const auto stack_dim = R"JIT(
    def forward(self, a: Tensor, b: Tensor, dim: int):
        inputs = [a]
        inputs.append(b) # mutation to avoid using VarStack
        return torch.stack(inputs, dim = dim).clone()
  )JIT";

  const auto stack_three = R"JIT(
    def forward(self, a: Tensor, b: Tensor, c: Tensor):
        inputs = [a, b]
        inputs.append(c) # mutation to avoid using VarStack
        return torch.stack(inputs).clone()
  )JIT";

  auto a = at::randn({2, 2});
  auto b = at::randn({2, 2});
  auto c = at::randn({2, 2});

  auto d = at::randn({3, 3, 3});
  auto e = at::randn({3, 3, 3});
  auto f = at::randn({3, 3, 3});

  std::vector<IValue> args1_dim{a, b, 0};
  std::vector<IValue> args2_dim{d, e, 1};
  std::vector<IValue> args_dim_negative{d, e, -1};

  std::vector<IValue> args1_three_tensors{a, b, c};
  std::vector<IValue> args2_three_tensors{d, e, f};

  testStaticRuntime(stack_dim, args1_dim);
  testStaticRuntime(stack_dim, args1_dim, args2_dim);

  testStaticRuntime(stack_dim, args_dim_negative);

  testStaticRuntime(stack_three, args1_three_tensors);
  testStaticRuntime(stack_three, args1_three_tensors, args2_three_tensors);
}

TEST(StaticRuntime, ReLU) {
  const auto relu_script = R"JIT(
    def forward(self, a: Tensor):
        return torch.relu(a).clone()
  )JIT";
  auto a = at::randint(-10, 10, {2, 4});
  auto b = at::randint(-10, 10, {3, 6});

  std::vector<IValue> args1{a};
  std::vector<IValue> args2{b};

  testStaticRuntime(relu_script, args1);
  testStaticRuntime(relu_script, args1, args2);
}

TEST(StaticRuntime, Tanh) {
  const auto tanh_script = R"JIT(
    def forward(self, a):
        return torch.tanh(a).clone()
  )JIT";
  auto a = at::randn({2, 2});
  auto b = at::randn({3, 3, 3});

  std::vector<IValue> args1{a};
  std::vector<IValue> args2{b};

  testStaticRuntime(tanh_script, args1, /*args2*/ {}, /*use_allclose*/ true);
  testStaticRuntime(tanh_script, args1, args2, /*use_allclose*/ true);
}

TEST(StaticRuntime, Norm) {
  const auto norm_2arg = R"JIT(
    def forward(self, a: Tensor, p: int):
        return torch.norm(a, p).clone()
  )JIT";

  const auto norm_3arg = R"JIT(
    def forward(self, a: Tensor, p: int, dtype: int):
        return torch.norm(a, p, dtype=dtype).clone()
  )JIT";

  const auto norm_4arg = R"JIT(
    def forward(self, a: Tensor, p: int, dim: List[int], keepdim: bool):
        return torch.norm(a, p, dim, keepdim).clone()
  )JIT";

  const auto norm_5arg = R"JIT(
    def forward(self, a: Tensor, p: int, dim: List[int], keepdim: bool, dtype: int):
        return torch.norm(a, p, dim, keepdim, dtype=dtype).clone()
  )JIT";

  auto a = at::randn({2, 3});
  auto b = at::randn({4, 3, 5});
  auto dim = std::vector<int64_t>({1});
  auto dtype = at::ScalarType::Float;

  std::vector<IValue> args2{a, 2};
  testStaticRuntime(norm_2arg, args2);
  testStaticRuntime(norm_2arg, args2, {b, 2}, false, false, false);

  std::vector<IValue> args3{a, 2, dtype};
  testStaticRuntime(norm_3arg, args3);
  testStaticRuntime(norm_3arg, args3, {b, 2, dtype}, false, false, false);

  std::vector<IValue> args4{a, 3, dim, false};
  testStaticRuntime(norm_4arg, args4);
  testStaticRuntime(norm_4arg, args4, {b, 3, dim, false});

  std::vector<IValue> args5{a, 4, dim, true, dtype};
  testStaticRuntime(norm_5arg, args5);
  testStaticRuntime(norm_5arg, args5, {b, 4, dim, true, dtype});
}

TEST(StaticRuntime, Reshape) {
  const auto reshape_script_1 = R"JIT(
    def forward(self, a: Tensor, shape: List[int]):
        b = a.reshape(shape)
        return b + b
  )JIT";

  const auto reshape_script_2 = R"JIT(
    def forward(self, a: Tensor, shape: List[int]):
        b = a.transpose(0, 1)
        return b.reshape(shape)
  )JIT";

  const auto reshape_script_3 = R"JIT(
    def forward(self, inp: Tensor, shape: List[int]):
        a = inp + inp
        b = a.reshape(shape)
        c = a.reshape(shape)
        d = c + c
        e = d + d
        f = e * e
        g = f * f
        return b.reshape(shape), g
  )JIT";

  // exercise reshape_copy and flatten_copy
  const auto reshape_script_4 = R"JIT(
    def forward(self, inp: Tensor, shape: List[int]):
        k = inp + inp
        a = k + k
        b = a.reshape(shape)
        c = a.flatten().reshape(shape)
        return b + c
  )JIT";

  // exercise reshape_copy
  const auto reshape_script_5 = R"JIT(
    def forward(self, inp: Tensor, shape: List[int]):
        a = inp + inp
        b = a.reshape(shape)
        c = a.reshape(shape).relu()
        d = c + c
        e = d + d
        f = e * e
        g = f * f
        return g
  )JIT";

  const auto reshape_inplace_script = R"JIT(
    def forward(self, inp: Tensor, shape: List[int]):
        a = inp + inp
        b = a.reshape(shape)
        c = b.sigmoid_()
        d = c + c
        e = a + a
        f = b + b
        return (d, e, f)
  )JIT";

  // b is in_contiguous
  const auto reshape_incontiguous_script = R"JIT(
    def forward(self, a: Tensor, shape: List[int]):
        b = a.transpose(0, 1)
        c = b.reshape(shape)
        c = c.relu()
        return (c)
  )JIT";

  auto a = at::randn({2, 3});
  auto b = std::vector<int64_t>({3, 2});
  std::vector<IValue> args{a, b};

  auto c = at::randn({4, 5});
  auto d = std::vector<int64_t>({5, 1, 2, 2});
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

TEST(StaticRuntime, Repeat) {
  const std::string repeat = R"JIT(
    def forward(self, a: Tensor, repeats: List[int]):
        return torch.repeat(a, repeats).clone()
  )JIT";

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

TEST(StaticRuntime, Flatten) {
  // exercise flatten_copy
  const auto flatten_script_1 = R"JIT(
    def forward(self, a: Tensor, start_dim: int, end_dim: int):
        b = a * a
        c = torch.flatten(b, start_dim, end_dim)
        d = torch.relu(c)
        return d
  )JIT";

  const auto flatten_script_2 = R"JIT(
    def forward(self, a: Tensor, start_dim: int, end_dim: int):
        b = a.transpose(0, 1)
        return torch.flatten(b, start_dim, end_dim).clone()
  )JIT";

  auto test_flatten =
      [&](std::vector<int64_t> shape, int64_t start_dim, int64_t end_dim) {
        std::vector<int64_t> shape1(shape);
        if (shape1.size() > 0) {
          shape1[0] *= 6;
        }
        auto a = at::randn(shape);
        auto b = at::randn(shape1);
        std::vector<IValue> args{a, start_dim, end_dim};
        bool check_resize = shape1.size() > 0;
        testStaticRuntime(flatten_script_1, args);
        testStaticRuntime(
            flatten_script_1,
            args,
            {b, start_dim, end_dim},
            false, /* use_allclose */
            false, /* use_equalnan */
            check_resize);
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

TEST(StaticRuntime, pow) {
  const auto pow_script_ten_sca = R"JIT(
    def forward(self, input : Tensor, exponent : int):
        return torch.pow(input, exponent).clone()
  )JIT";

  const auto pow_script_ten_ten = R"JIT(
    def forward(self, input : Tensor, exponent : Tensor):
        return torch.pow(input, exponent).clone()
  )JIT";

  const auto pow_script_sca_ten = R"JIT(
    def forward(self, input : int, exponent : Tensor):
        return torch.pow(input, exponent).clone()
  )JIT";

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

TEST(StaticRuntime, to) {
  const auto to_script_dtype = R"JIT(
    def forward(self, input: Tensor, dtype: int, non_blocking: bool, copy: bool, memory_format: int):
        a = input + input
        return torch.to(a, dtype, non_blocking, copy, memory_format).clone()
  )JIT";

  const auto to_script_dtype_strided = R"JIT(
    def forward(self, input: Tensor, dtype: int, non_blocking: bool, copy: bool, memory_format: int):
        b = input.permute(0, 2, 3, 1)
        return torch.to(b, dtype, non_blocking, copy, memory_format).clone()
  )JIT";

  const auto to_script_prim_dtype = R"JIT(
    def forward(self, input:Tensor, dtype: Optional[int], non_blocking: bool, copy: bool):
        a = input + input
        return torch.to(a, dtype, non_blocking, copy).clone()
  )JIT";

  const auto to_script_other = R"JIT(
    def forward(self, input:Tensor, other: Tensor, non_blocking: bool, copy: bool, memory_format: int):
        a = input + input
        return torch.to(a, other, non_blocking, copy, memory_format).clone()
  )JIT";

  // if input is float tensor, b could be alias of a
  const auto to_script_alias = R"JIT(
    def forward(self, input:Tensor):
        a = input + input
        b = a.float()
        c = b * b
        return (c)
  )JIT";

  const auto to_script_fails_managed_output_check = R"JIT(
    def forward(self, a, b):
        d = a.half() * b.half()
        e = d.float()
        return e
  )JIT";

  const auto to_script_select_tensor_output_into_tuple = R"JIT(
    def forward(self, a, b):
        d = a.half() * b.half()
        e = d.float()
        return (d, e)
  )JIT";

  const auto to_script_memory_planning_fail = R"JIT(
    def forward(self, a, b):
        d = a.half() * b.half()
        e = d.float().relu()
        return e
  )JIT";

  auto test_to = [&](at::ScalarType b, bool c, bool d, c10::MemoryFormat e) {
    auto a = at::randn({4, 3, 1, 2});
    auto other = at::randn({4, 3, 1, 2}).to(b);
    auto a2 = at::randn({3, 2, 2, 4});
    auto a2_other = at::randn({3, 2, 2, 4}).to(b);

    std::vector<IValue> args0{a, b, c, d, e};
    std::vector<IValue> args1{a, b, c, d};
    std::vector<IValue> args2{a, other, c, d, e};
    std::vector<IValue> args2WithDifferentOtherType{
        a, at::randn({4, 3, 1, 2}, ScalarType::Double), c, d, e};
    std::vector<IValue> args3{a, c10::nullopt, c, d};

    std::vector<IValue> args0WithInt{a, ScalarType::Int, c, d, e};
    testStaticRuntime(
        to_script_dtype,
        args0,
        args0WithInt,
        /* default for use_allclose */ false,
        /* default for use_equalnan */ false,
        /* check_resize */ false);
    testStaticRuntime(to_script_dtype_strided, args0);
    testStaticRuntime(to_script_prim_dtype, args1);
    if (!d) {
      testStaticRuntime(to_script_prim_dtype, args3);
    }
    // Second set of args tests case where the `other` tensor's dtype
    // changes between iterations.
    testStaticRuntime(
        to_script_other,
        args2,
        args2WithDifferentOtherType,
        /* default for use_allclose */ false,
        /* default for use_equalnan */ false,
        /* check_resize */ false);
    testStaticRuntime(to_script_alias, {a});

    testStaticRuntime(to_script_memory_planning_fail, {a, a});
    testStaticRuntime(to_script_fails_managed_output_check, {a, a});
    testStaticRuntime(to_script_select_tensor_output_into_tuple, {a, a});

    // dynamic shapes
    testStaticRuntime(to_script_dtype, args0, {a2, b, c, d, e});
    testStaticRuntime(to_script_dtype_strided, args0, {a2, b, c, d, e});
    testStaticRuntime(to_script_prim_dtype, args1, {a2, b, c, d});
    if (!d) {
      testStaticRuntime(to_script_prim_dtype, args3, {a2, c10::nullopt, c, d});
    }
    testStaticRuntime(to_script_other, args2, {a2, a2_other, c, d, e});
    testStaticRuntime(to_script_alias, {a}, {a2});
  };
  for (const bool non_blocking : {false, true}) {
    for (const bool copy : {false, true}) {
      // float->float, NCHW->NHWC
      test_to(
          at::ScalarType::Float,
          non_blocking,
          copy,
          c10::MemoryFormat::ChannelsLast);
      // float->half
      test_to(
          at::ScalarType::Half,
          non_blocking,
          copy,
          c10::MemoryFormat::Preserve);
      // float->float
      test_to(
          at::ScalarType::Float,
          non_blocking,
          copy,
          c10::MemoryFormat::Contiguous);
      test_to(
          at::ScalarType::Bool,
          non_blocking,
          copy,
          c10::MemoryFormat::Contiguous);
      // TODO: check if fbgemm is enabled properly in this case
      // half->float, NCHW->NHWC
      test_to(
          at::ScalarType::Half,
          non_blocking,
          copy,
          c10::MemoryFormat::ChannelsLast);
    }
  }
}

TEST(StaticRuntime, ExpandAs) {
  const auto expand_as_script = R"JIT(
    def forward(self, input: Tensor, other:Tensor):
        a = input.expand_as(other)
        return a.clone()
  )JIT";

  auto a = at::randn({3, 1});
  auto b = at::randn({3, 2});
  auto c = at::randn({4, 1});
  auto d = at::randn({4, 2});
  std::vector<IValue> args{a, b};
  std::vector<IValue> args2{c, d};
  testStaticRuntime(expand_as_script, args);
  testStaticRuntime(expand_as_script, args, args2);
}

TEST(StaticRuntime, Full) {
  const auto full_script = R"JIT(
    def forward(self,
                size: List[int],
                fill_value: int,
                dtype: Optional[int],
                layout: Optional[int],
                device: Optional[Device],
                pin_memory: Optional[bool]):
        a = torch.full(size,
                      fill_value,
                      dtype=dtype,
                      layout=layout,
                      device=device,
                      pin_memory=pin_memory)
        return (a.clone())
  )JIT";

  auto cpu = at::Device(DeviceType::CPU);
  c10::List<int64_t> size0{2, 5};
  std::vector<IValue> args{
      size0, 4, at::ScalarType::Int, at::kStrided, cpu, false};
  std::vector<IValue> args1{
      size0, 4, at::ScalarType::Float, at::kStrided, cpu, false};
  c10::List<int64_t> size1{5, 6};
  std::vector<IValue> args2{
      size1, 5, at::ScalarType::Float, at::kStrided, cpu, false};
  testStaticRuntime(full_script, args);
  testStaticRuntime(
      full_script,
      args,
      args1,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/false);
  testStaticRuntime(full_script, args, args2);
}

TEST(StaticRuntime, FullLike) {
  const auto full_like_script = R"JIT(
    def forward(self,
                a: Tensor,
                fill_value: int,
                dtype: Optional[int],
                layout: Optional[int],
                device: Optional[Device],
                pin_memory: Optional[bool],
                memory_format: Optional[int]):
        b = torch.full_like(a,
                            fill_value,
                            dtype=dtype,
                            layout=layout,
                            device=device,
                            pin_memory=pin_memory,
                            memory_format=memory_format)
        return (b.clone())
  )JIT";

  auto a = at::randn({2, 3});
  auto b = at::randn({3, 4, 2});
  auto cpu = at::Device(DeviceType::CPU);
  std::vector<IValue> args{
      a,
      4,
      at::ScalarType::Int,
      at::kStrided,
      cpu,
      false,
      c10::MemoryFormat::Contiguous};
  std::vector<IValue> args1{
      a,
      4,
      at::ScalarType::Float,
      at::kStrided,
      cpu,
      false,
      c10::MemoryFormat::Contiguous};
  std::vector<IValue> args2{
      b,
      4,
      at::ScalarType::Float,
      at::kStrided,
      cpu,
      false,
      c10::MemoryFormat::Contiguous};
  testStaticRuntime(full_like_script, args);
  testStaticRuntime(
      full_like_script,
      args,
      args1,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/false);
  testStaticRuntime(full_like_script, args, args2);
}

TEST(StaticRuntime, Ones) {
  const auto script = R"JIT(
    def forward(self,
                size: List[int],
                dtype: Optional[int],
                layout: Optional[int],
                device: Optional[Device],
                pin_memory: Optional[bool]):
        a = torch.ones(size,
                       dtype=dtype,
                       layout=layout,
                       device=device,
                       pin_memory=pin_memory)
        return (a.clone())
  )JIT";

  auto dtype = at::ScalarType::Int;
  auto cpu = at::Device(DeviceType::CPU);
  c10::List<int64_t> size0{2, 5};
  std::vector<IValue> args{size0, dtype, at::kStrided, cpu, false};
  c10::List<int64_t> size1{5, 6};
  std::vector<IValue> args2{size1, dtype, at::kStrided, cpu, false};
  testStaticRuntime(script, args);
  testStaticRuntime(script, args, args2);
}

TEST(StaticRuntime, OnesLike) {
  const auto script = R"JIT(
    def forward(self,
                input: Tensor,
                dtype: Optional[int],
                layout: Optional[int],
                device: Optional[Device],
                pin_memory: Optional[bool],
                memory_format: Optional[int]):
        a = torch.ones_like(input,
                            dtype=dtype,
                            layout=layout,
                            device=device,
                            pin_memory=pin_memory,
                            memory_format=memory_format)
        return (a.clone())
  )JIT";

  auto cpu = at::Device(DeviceType::CPU);
  auto input0 = at::randn({2, 5});
  std::vector<IValue> args{
      input0,
      at::ScalarType::Int,
      at::kStrided,
      cpu,
      false,
      c10::MemoryFormat::Contiguous};
  std::vector<IValue> args1{
      input0,
      at::ScalarType::Float,
      at::kStrided,
      cpu,
      false,
      c10::MemoryFormat::Contiguous};
  auto input1 = at::randn({5, 6});
  std::vector<IValue> args2{
      input1,
      at::ScalarType::Float,
      at::kStrided,
      cpu,
      false,
      c10::MemoryFormat::Contiguous};
  testStaticRuntime(script, args);
  testStaticRuntime(
      script,
      args,
      args1,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/false);
  testStaticRuntime(script, args, args2);
}

TEST(StaticRuntime, Zeros) {
  const auto script = R"JIT(
    def forward(self,
                size: List[int],
                dtype: Optional[int],
                layout: Optional[int],
                device: Optional[Device],
                pin_memory: Optional[bool]):
        a = torch.zeros(size,
                       dtype=dtype,
                       layout=layout,
                       device=device,
                       pin_memory=pin_memory)
        return (a.clone())
  )JIT";

  auto cpu = at::Device(DeviceType::CPU);
  c10::List<int64_t> size0{2, 5};
  std::vector<IValue> args{
      size0, at::ScalarType::Int, at::kStrided, cpu, false};
  std::vector<IValue> args1{
      size0, at::ScalarType::Float, at::kStrided, cpu, false};
  c10::List<int64_t> size1{5, 6};
  std::vector<IValue> args2{
      size1, at::ScalarType::Float, at::kStrided, cpu, false};
  testStaticRuntime(script, args);
  testStaticRuntime(
      script,
      args,
      args1,
      /*use_allclose=*/false,
      /*use_equalnan=*/false,
      /*check_resize=*/false);
  testStaticRuntime(script, args, args2);
}

TEST(StaticRuntime, Linear) {
  const auto linear_script = R"JIT(
    def forward(self, inp: Tensor, weights: Tensor, bias: Optional[Tensor]) -> Tensor:
        return torch.linear(inp, weights, bias).clone()
  )JIT";

  auto input = at::randn({1, 2});
  auto weights = at::randn({1, 2});
  auto bias = at::randn({1, 1});

  std::vector<IValue> args{input, weights, bias};
  std::vector<IValue> args_no_bias{input, weights, c10::nullopt};

  auto input2 = at::randn({6, 3});
  auto weights2 = at::randn({6, 3});
  auto bias2 = at::randn({6, 6});

  std::vector<IValue> args2{input2, weights2, bias2};
  std::vector<IValue> args2_no_bias{input2, weights2, c10::nullopt};

  testStaticRuntime(linear_script, args);
  testStaticRuntime(linear_script, args_no_bias);

  testStaticRuntime(linear_script, args, args2);
  testStaticRuntime(linear_script, args, args2_no_bias);
}

TEST(StaticRuntime, VarCat) {
  const auto var_cat_script = R"JIT(
    def forward(self, inp1: Tensor, inp2: Tensor, dim: int):
      return torch.cat([inp1, inp2], dim).clone()
  )JIT";

  // 2D tensors - cat dim = 0
  std::vector<IValue> args1 = {at::randn({4, 6}), at::randn({5, 6}), 0};
  testStaticRuntime(var_cat_script, args1);

  // 3D tensors - cat dim = 1
  std::vector<IValue> args2 = {at::randn({4, 5, 6}), at::randn({4, 8, 6}), 1};
  testStaticRuntime(var_cat_script, args2);

  // 3D tensors - cat dim = 2
  std::vector<IValue> args3 = {at::randn({4, 5, 6}), at::randn({4, 5, 7}), 2};
  testStaticRuntime(var_cat_script, args3);

  // Negative dim
  std::vector<IValue> args4 = {at::randn({4, 5, 6}), at::randn({4, 5, 7}), -1};
  testStaticRuntime(var_cat_script, args4);

  testStaticRuntime(var_cat_script, args1, args2);
}

TEST(StaticRuntime, LeakyReLU) {
  torch::jit::Module mod = getLeakyReLUConstScriptModel();
  auto inputs = torch::randn({2, 2});

  // run jit graph executor
  std::vector<at::IValue> input_ivalues({inputs});
  at::Tensor output_1 = mod.forward(input_ivalues).toTensor();

  // run static runtime
  std::vector<c10::IValue> input_tensors({inputs});
  torch::jit::StaticModule smod(mod);
  at::Tensor output_2 = smod(input_tensors, {}).toTensor();
  smod.runtime().check_for_memory_leak();
  EXPECT_TRUE(torch::allclose(output_1, output_2, 1e-6));
}

static ProcessedNodeInputs createProcessedNodeInputs(
    c10::ArrayRef<uint16_t> inputs) {
  ProcessedNodeInputs result(inputs.size());
  for (const auto idx : c10::irange(inputs.size())) {
    result[idx] = inputs[idx];
  }
  return result;
}

static void checkProcessedNodeInputs(
    const ProcessedNodeInputs& io,
    c10::ArrayRef<uint16_t> inputs) {
  ASSERT_EQ(inputs.size(), io.size());
  for (const auto idx : c10::irange(inputs.size())) {
    EXPECT_EQ(inputs[idx], io[idx]);
  }
}

static void testProcessedNodeInputsRoundTrip(c10::ArrayRef<uint16_t> inputs) {
  auto io = createProcessedNodeInputs(inputs);
  checkProcessedNodeInputs(io, inputs);

  ProcessedNodeInputs copied(io);
  checkProcessedNodeInputs(copied, inputs);
  ProcessedNodeInputs moved(std::move(io));
  checkProcessedNodeInputs(moved, inputs);
}

TEST(ProcessedNodeInputs, Basic) {
  std::vector<std::vector<uint16_t>> testCases = {
      {}, // empty
      {0xABCD, 0x5a5a}, // inline
      {0x11, 0x22, 0x33, 0x44, 0x55}, // max inline size
      {0x11, 0x22, 0x33, 0x44, 0x55, 0x66}, // minimum outline size
      std::vector<uint16_t>(100, 0x5a), // large outline size
  };

  for (const auto& values : testCases) {
    testProcessedNodeInputsRoundTrip(values);
    for (const auto& values2 : testCases) {
      auto from = createProcessedNodeInputs(values);
      auto to = createProcessedNodeInputs(values2);

      to = from;
      checkProcessedNodeInputs(to, values);

      auto toMoveInto = createProcessedNodeInputs(values2);
      toMoveInto = std::move(from);
      checkProcessedNodeInputs(toMoveInto, values);
    }
  }
}

TEST(StaticRuntime, isinstance) {
  const auto isinstance_int_script = R"JIT(
    def forward(self, a: Any):
        return isinstance(a, int)
  )JIT";

  const auto isinstance_tensor_script = R"JIT(
    def forward(self, a: Any):
        return isinstance(a, torch.Tensor)
  )JIT";

  const auto isinstance_many_types_script = R"JIT(
    def forward(self, a: Any):
        return isinstance(a, (bool, int))
  )JIT";

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

TEST(StaticRuntime, TypeCheck) {
  const auto typecheck_ir = R"IR(
  graph(%a.1 : Tensor,
        %b.1 : Tensor):
    %t0 : Float(2, 2, strides=[2, 1], device=cpu), %t1 : Float(3, 3, strides=[3, 1]), %type_matched : bool = prim::TypeCheck[types=[Float(2, 2, strides=[2, 1], device=cpu), Float(3, 3, strides=[3, 1])]](%a.1, %b.1)
    return (%t0, %t1, %type_matched)
  )IR";

  auto a = at::zeros({2, 2}, at::kFloat);
  a.to(at::kCPU);
  auto b = at::ones({3, 3}, at::kFloat);
  auto c = at::ones({2, 2, 2}, at::kFloat);

  std::vector<IValue> args_correct = {a, b};
  std::vector<IValue> args_incorrect = {a, c};

  testStaticRuntime(typecheck_ir, args_correct);
  testStaticRuntime(typecheck_ir, args_correct, args_incorrect);
}

TEST(StaticRuntime, Index) {
  const auto index_without_none_script = R"JIT(
    def forward(self, a: Tensor, idx: Tensor):
        return a[idx].clone()
  )JIT";

  // Index with boolean mask
  auto a = at::arange(4, at::kFloat).view({2, 2});
  auto idx_a = torch::tensor({{0, 1}, {0, 0}}, at::kBool);
  std::vector<IValue> args_a{a, idx_a};

  // Index with tensor
  auto b = at::arange(27, at::kFloat).view({3, 3, 3});
  auto idx_b = torch::tensor({0, 1, 2}, at::kLong);
  std::vector<IValue> args_b{b, idx_b};

  testStaticRuntime(index_without_none_script, args_a);
  testStaticRuntime(index_without_none_script, args_a, args_b);

  const auto index_with_none_script = R"JIT(
    def forward(self, a: Tensor, idx: Tensor, none: Optional[Tensor]):
        return a[idx, none].clone()
  )JIT";

  // Index with None
  // When indexing with none, the shape of `f` becomes [2, 1, 2],
  // so the mask must be reshaped appropriately.
  auto f = at::arange(4, at::kFloat).view({2, 1, 2});
  auto idx_f_reshape = torch::tensor({{{0, 1}}, {{0, 0}}}, at::kBool);
  std::vector<IValue> args_f_with_none{f, idx_f_reshape};
  args_f_with_none.emplace_back();

  testStaticRuntime(index_with_none_script, args_f_with_none);
  testStaticRuntime(
      index_with_none_script,
      args_f_with_none,
      {IValue(b), IValue(idx_b), IValue()});

  const auto index_with_two_tensors_script = R"JIT(
    def forward(self, a: Tensor, idx_a: Tensor, idx_b: Tensor):
        return a[idx_a, idx_b].clone()
  )JIT";

  // Index with multiple tensors
  const auto& c = a; // 2x2 tensor
  auto idx_c1 = torch::tensor({0, 0}, at::kLong);
  auto idx_c2 = torch::tensor({0}, at::kLong);
  std::vector<IValue> args_c{c, idx_c1, idx_c2};

  const auto& d = b; // 3x3x3 tensor
  auto idx_d1 = torch::tensor({{0, 0, 2}, {0, 1, 1}}, at::kLong);
  auto idx_d2 = torch::tensor({{1, 1, 0}, {1, 0, 2}}, at::kLong);
  std::vector<IValue> args_d{d, idx_d1, idx_d2};

  testStaticRuntime(index_with_two_tensors_script, args_c, args_d);
}

TEST(StaticRuntime, IndexSelect) {
  const std::string script = R"IR(
    graph(%self: Tensor, %dim: int, %index: Tensor):
        %bias: None = prim::Constant()
        %ret = aten::index_select(%self, %dim, %index)
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  auto self0 = at::rand({6});
  auto dim0 = 0;
  auto index0 = at::randint(0, 5, {6}, torch::kInt32);
  std::vector<IValue> args{self0, dim0, index0};
  testStaticRuntime(script, args);

  auto self1 = at::rand({128});
  auto dim1 = 0;
  auto index1 = at::randint(0, 127, {127}, torch::kInt32);
  std::vector<IValue> args2{self1, dim1, index1};
  testStaticRuntime(script, args, args2);
}

TEST(StaticRuntime, ClampMin) {
  const auto clamp_min_int_script = R"JIT(
    def forward(self, a: Tensor, b: int):
        return torch.clamp_min(a, b).clone()
  )JIT";

  const auto clamp_min_float_script = R"JIT(
    def forward(self, a: Tensor, b: float):
        return torch.clamp_min(a, b).clone()
  )JIT";

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

TEST(StaticRuntime, Argmin) {
  const auto argmin_script = R"JIT(
    def forward(self, a: Tensor):
        return torch.argmin(a).clone()
  )JIT";

  const auto argmin_with_dim_script = R"JIT(
    def forward(self, a: Tensor, dim: int):
        return torch.argmin(a, dim).clone()
  )JIT";

  const auto argmin_with_keep_dim_script = R"JIT(
    def forward(self, a: Tensor, dim: int):
        return torch.argmin(a, dim, True).clone()
  )JIT";

  auto a = at::randn({2, 2});
  auto b = at::randn({17, 2, 1});

  testStaticRuntime(argmin_script, {a});
  testStaticRuntime(
      argmin_script,
      {a},
      {b},
      /* use_allclose */ false,
      /* use_equalnan */ false,
      /* check_resize */ false);

  int dim_a = 0;
  int dim_b = 1;

  std::vector<IValue> args_a{a, dim_a};
  std::vector<IValue> args_b{b, dim_b};

  testStaticRuntime(argmin_with_dim_script, args_a);
  testStaticRuntime(argmin_with_dim_script, args_a, args_b);

  testStaticRuntime(argmin_with_keep_dim_script, args_a);
  testStaticRuntime(argmin_with_keep_dim_script, args_a, args_b);
}

TEST(StaticRuntime, Softmax) {
  const auto softmax_script = R"JIT(
    def forward(self, a: Tensor, dim: int):
        return torch.softmax(a, dim).clone()
  )JIT";

  const auto softmax_script_with_dtype = R"JIT(
    def forward(self, a: Tensor, dim: int, dtype: int):
        return torch.softmax(a, dim, dtype=dtype).clone()
  )JIT";

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

TEST(StaticRuntime, GetItem_Dict) {
  const auto getitem_dict_tensor_script = R"JIT(
    def forward(self, key: Tensor):
        d = {key: 1}
        return d[key]
  )JIT";

  const auto getitem_dict_int_script = R"JIT(
    def forward(self, key: int):
        d = {key: 1}
        return d[key]
  )JIT";

  const auto getitem_dict_str_script = R"JIT(
    def forward(self, key: str):
        d = {key: 1}
        return d[key]
  )JIT";

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

TEST(StaticRuntime, GetItem_List) {
  const auto getitem_list_int_script = R"JIT(
    def forward(self, idx: int):
        lst = [1, 2, 3]
        return lst[idx]
  )JIT";

  const auto getitem_list_tensor_script = R"JIT(
    def forward(self, tensor: Tensor, idx: int):
        lst = [tensor, tensor]
        return lst[idx]
  )JIT";

  testStaticRuntime(getitem_list_int_script, {1});
  testStaticRuntime(getitem_list_int_script, {-1});

  auto a = torch::tensor({1});
  auto b = torch::tensor({1, 1});

  testStaticRuntime(getitem_list_tensor_script, {a, 1});
  testStaticRuntime(getitem_list_tensor_script, {a, 1}, {b, -1});
}

TEST(StaticRuntime, Transpose) {
  const auto transpose_script = R"JIT(
    def forward(self, a: Tensor, dim1: int, dim2: int):
        return torch.transpose(a, dim1, dim2).clone()
  )JIT";

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

TEST(StaticRuntime, Permute) {
  auto permute_script = R"JIT(
    def forward(self, a: Tensor, dims: List[int]):
        return torch.permute(a, dims).clone()
  )JIT";

  auto a = at::randn({2, 2});
  c10::List<int64_t> dims_a{1, 0};
  std::vector<IValue> args_a{a, dims_a};

  auto b = at::randn({3, 3, 3});
  c10::List<int64_t> dims_b{0, 2, 1};
  std::vector<IValue> args_b{b, dims_b};

  auto c = at::randn({3, 3, 3});
  c10::List<int64_t> dims_c{0, -1, 1};
  std::vector<IValue> args_c{c, dims_c};

  testStaticRuntime(permute_script, args_a);
  testStaticRuntime(permute_script, args_c);
  testStaticRuntime(permute_script, args_a, args_b);

  permute_script = R"JIT(
    def forward(self, a: Tensor, dims: List[int], shape: List[int]):
        return torch.permute(a, dims).reshape(shape).clone()
  )JIT";

  a = at::randn({8, 16, 4});
  dims_a = {0, 2, 1};
  dims_b = {-1, 16};
  testStaticRuntime(permute_script, {a, dims_a, dims_b});
}

TEST(StaticRuntime, Slice) {
  const auto slice_script = R"JIT(
    def forward(self, a: Tensor, dim: int, start: int, end: int, step: int):
      return a.slice(dim, start, end, step).clone()
  )JIT";

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

  const auto slice_script2 = R"JIT(
    def forward(self, a: Tensor, dim: int, step: int):
      return a.slice(dim, None, None, step).clone()
  )JIT";
  std::vector<IValue> args_c{b, dim_b, step_b};
  testStaticRuntime(slice_script2, args_c);
}

TEST(StaticRuntime, Narrow) {
  const auto narrow_with_int_script = R"JIT(
    def forward(self, a: Tensor, dim: int, start: int, length: int):
        return a.narrow(dim, start, length).clone()
  )JIT";

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

TEST(StaticRuntime, TupleUnpack) {
  const auto two_tuple_unpack_script = R"JIT(
    def forward(self, tup: Tuple[Tensor, Tensor]):
        a, b = tup
        return (a, b)
  )JIT";

  const auto three_tuple_unpack_script = R"JIT(
    def forward(self, tup: Tuple[Tensor, Tensor, Tensor]):
        a, b, c = tup
        return (a, b, c)
  )JIT";

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

TEST(StaticRuntime, Append) {
  const auto append_int_script = R"JIT(
    def forward(self, a: int):
        lst = [1, 2, 3]
        lst.append(a)
        return lst
  )JIT";

  const auto append_tensor_script = R"JIT(
    def forward(self, a: Tensor):
        lst = []
        lst.append(a)
        return lst
  )JIT";

  std::vector<IValue> args_int{1};

  testStaticRuntime(append_int_script, args_int);

  std::vector<IValue> args_tensor{at::randn({1})};
  std::vector<IValue> args_tensor_large{at::randn({2, 2})};

  testStaticRuntime(append_tensor_script, args_tensor);
  testStaticRuntime(append_tensor_script, args_tensor, args_tensor_large);
}

TEST(StaticRuntime, QuantizedLinear) {
  const std::string quantize_script = R"IR(
    graph(%input: Tensor, %weights: Tensor):
        %scale: float = prim::Constant[value=1.]()
        %zero_point: int = prim::Constant[value=1]()
        %bias: None = prim::Constant()
        %packed_params = quantized::linear_prepack(%weights, %bias)
        %1254 = quantized::linear(%input, %packed_params, %scale, %zero_point)
        %1249: Tensor = aten::dequantize(%1254)
        return (%1249)
  )IR";
  at::Tensor weight =
      at::quantize_per_tensor(torch::randn({3, 2}), 2, 3, torch::kQInt8);
  at::Tensor input =
      at::quantize_per_tensor(torch::randn({3, 2}), 2, 3, torch::kQUInt8);

  at::Tensor weight_2 =
      at::quantize_per_tensor(torch::randn({8, 3}), 2, 3, torch::kQInt8);
  at::Tensor input_2 =
      at::quantize_per_tensor(torch::randn({9, 3}), 2, 3, torch::kQUInt8);

  testStaticRuntime(quantize_script, {input, weight}, {input_2, weight_2});
}

TEST(StaticRuntime, QuantizedLinearDynamicFp16) {
  const std::string quantized_linear_dynamic_fp16_script = R"IR(
    graph(%input: Tensor, %weights: Tensor):
        %bias: None = prim::Constant()
        %packed_params = quantized::linear_prepack_fp16(%weights, %bias)
        %output = quantized::linear_dynamic_fp16(%input, %packed_params)
        %ret = aten::clone(%output, %bias)
        return (%ret)
  )IR";
  at::Tensor weight = torch::randn({3, 2}, torch::kFloat);
  at::Tensor input = torch::randn({3, 2}, torch::kFloat);

  at::Tensor weight_2 = torch::randn({4, 3}, torch::kFloat);
  at::Tensor input_2 = torch::randn({5, 3}, torch::kFloat);

  testStaticRuntime(
      quantized_linear_dynamic_fp16_script,
      {input, weight},
      {input_2, weight_2});
}

TEST(StaticRuntime, QuantizedLinearReluDynamicFp16) {
  const std::string quantized_linear_relu_dynamic_fp16_script = R"IR(
    graph(%input: Tensor, %weights: Tensor):
        %bias: None = prim::Constant()
        %packed_params = quantized::linear_prepack_fp16(%weights, %bias)
        %output = quantized::linear_relu_dynamic_fp16(%input, %packed_params)
        %ret = aten::clone(%output, %bias)
        return (%ret)
  )IR";
  at::Tensor weight = torch::randn({3, 2}, torch::kFloat);
  at::Tensor input = torch::randn({3, 2}, torch::kFloat);

  at::Tensor weight_2 = torch::randn({4, 3}, torch::kFloat);
  at::Tensor input_2 = torch::randn({5, 3}, torch::kFloat);

  testStaticRuntime(
      quantized_linear_relu_dynamic_fp16_script,
      {input, weight},
      {input_2, weight_2});
}

TEST(StaticRuntime, VarStack) {
  const auto var_stack_script = R"JIT(
    def forward(self, inp1: Tensor, inp2: Tensor, dim: int):
        return torch.stack([inp1, inp2], dim).clone()
  )JIT";

  // 2D tensors - stack dim = 0
  std::vector<IValue> args1 = {at::randn({6, 6}), at::randn({6, 6}), 0};
  testStaticRuntime(var_stack_script, args1);

  // 3D tensors - stack dim = 1
  std::vector<IValue> args2 = {at::randn({4, 5, 6}), at::randn({4, 5, 6}), 1};
  testStaticRuntime(var_stack_script, args2);

  // 3D tensors - stack dim = 2
  std::vector<IValue> args3 = {at::randn({4, 5, 6}), at::randn({4, 5, 6}), 2};
  testStaticRuntime(var_stack_script, args3);

  // Negative dim
  std::vector<IValue> args4 = {at::randn({4, 5, 6}), at::randn({4, 5, 6}), -1};
  testStaticRuntime(var_stack_script, args4);

  // Non-serial path
  std::vector<IValue> args5 = {at::randn({1, 2, 3}), at::randn({1, 2, 3}), 3};
  testStaticRuntime(var_stack_script, args5);

  // Fast path
  std::vector<IValue> args6 = {at::randn({1}), at::randn({1}), 0};
  testStaticRuntime(var_stack_script, args6);

  testStaticRuntime(var_stack_script, args1, args2);
}

TEST(StaticRuntime, FmodTensor) {
  const auto fmod_tensor = R"JIT(
    def forward(self, a: Tensor, b: Tensor):
        return torch.fmod(a, b).clone()
  )JIT";

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

TEST(StaticRuntime, FmodScalar) {
  const auto fmod_scalar = R"JIT(
    def forward(self, a: Tensor, b: int):
        return torch.fmod(a, b).clone()
  )JIT";

  auto a = at::randn({2, 3});

  // fmod scalar version
  std::vector<IValue> args2{a, 3};
  testStaticRuntime(fmod_scalar, args2);

  // check for dynamic shapes
  auto c = at::randn({4, 3, 2});
  std::vector<IValue> args3{c, 4};
  testStaticRuntime(fmod_scalar, args2, args3);

  // test int32 version
  a = at::randint(-100, 100, {2, 3}, at::kInt);
  c = at::randint(-100, 100, {4, 3, 2}, at::kInt);
  testStaticRuntime(fmod_scalar, {a, 3});
  testStaticRuntime(fmod_scalar, {a, 3}, {c, 4});
}

TEST(StaticRuntime, QEmbeddingBagBytePrepack) {
  const std::string embedding_bag_byte_prepack_script = R"IR(
    graph(%input: Tensor):
        %none : None = prim::Constant()
        %output: Tensor = quantized::embedding_bag_byte_prepack(%input)
        %res: Tensor = aten::clone(%output, %none)
        return (%res)
  )IR";

  auto a = torch::randn({8, 16}, at::ScalarType::Float);
  auto b = torch::randn({8 * 2, 16 * 2}, at::ScalarType::Float);

  testStaticRuntime(embedding_bag_byte_prepack_script, {a});
  testStaticRuntime(embedding_bag_byte_prepack_script, {a}, {b});
}

TEST(StaticRuntime, QEmbeddingBagByteUnpack) {
  const auto src = R"IR(
    graph(%input: Tensor):
        %none : None = prim::Constant()
        %weight: Tensor = quantized::embedding_bag_byte_prepack(%input)
        %output: Tensor = quantized::embedding_bag_byte_unpack(%weight)
        %res: Tensor = aten::clone(%output, %none)
        return (%res)
  )IR";

  auto a = torch::randn({8, 16}, at::ScalarType::Float);
  auto b = torch::randn({8 * 2, 16 * 2}, at::ScalarType::Float);

  testStaticRuntime(src, {a});
  testStaticRuntime(src, {a}, {b});
}

TEST(StaticRuntime, LinalgNorm_ScalarOrd) {
  const auto linalg_norm_ord_scalar = R"JIT(
    def forward(self, a: Tensor, ord: int, dim: List[int], keepdim: bool, dtype: int):
        return torch.linalg_norm(a, ord, dim, keepdim, dtype=dtype).clone()
  )JIT";

  auto a = at::randn({2, 3});
  auto dim = std::vector<int64_t>({1});
  auto dtype = at::ScalarType::Float;

  std::vector<IValue> args0{a, 4, dim, true, dtype};
  testStaticRuntime(linalg_norm_ord_scalar, args0);

  auto b = at::randn({3, 2, 6});
  std::vector<IValue> args1{b, 4, dim, true, dtype};
  testStaticRuntime(linalg_norm_ord_scalar, args0, args1);
}

TEST(StaticRuntime, LinalgNorm_StringOrd) {
  const auto linalg_norm_ord_str = R"JIT(
    def forward(self, a: Tensor, ord: str, dim: List[int], keepdim: bool, dtype: int):
        return torch.linalg_norm(a, ord, dim, keepdim, dtype=dtype).clone()
  )JIT";

  auto a = at::randn({2, 3});
  auto dim = std::vector<int64_t>({0, 1});
  auto dtype = at::ScalarType::Float;

  std::vector<IValue> args0{a, "fro", dim, true, dtype};
  testStaticRuntime(linalg_norm_ord_str, args0);

  auto b = at::randn({3, 2, 17});
  std::vector<IValue> args1{b, "fro", dim, true, dtype};
  testStaticRuntime(linalg_norm_ord_str, args0, args1);
}

TEST(StaticRuntime, Index_Put) {
  const auto index_put_str = R"JIT(
    def forward(self, a: Tensor, indices: Tuple[Optional[Tensor]], values: Tensor, accumulate: bool):
        return torch.index_put(a, indices, values, accumulate).clone()
  )JIT";

  auto a = at::randn({2});
  auto indices_a = std::make_tuple(torch::tensor({0}, at::kLong));
  auto values_a = at::randn({1});

  std::vector<IValue> args0{a, indices_a, values_a, false};
  testStaticRuntime(index_put_str, args0);

  const auto index_put_non_optional_str = R"JIT(
    def forward(self, a: Tensor, indices: List[Tensor], values: Tensor, accumulate: bool):
        return torch.index_put(a, indices, values, accumulate).clone()
  )JIT";

  auto indices_b = c10::List<at::Tensor>{torch::tensor({0}, at::kLong)};
  std::vector<IValue> args1{a, indices_b, values_a, false};
  testStaticRuntime(index_put_non_optional_str, args1);

  const auto index_put_list_construct = R"JIT(
    def forward(self, a: Tensor, indices: Tensor, values: Tensor, accumulate: bool):
        indices: List[Optional[Tensor]] = [indices]
        return torch.index_put(a, indices, values, accumulate).clone()
  )JIT";

  std::vector<IValue> args2{a, torch::tensor({0}, at::kLong), values_a, false};
  testStaticRuntime(index_put_list_construct, args2);
}

TEST(StaticRuntime, Item) {
  const auto item_str = R"JIT(
    def forward(self, a: Tensor):
        return torch.item(a)
  )JIT";

  auto a = at::randn({1});

  std::vector<IValue> args0{a};
  testStaticRuntime(item_str, args0);
}

TEST(StaticRuntime, Tensor_Split) {
  const auto tensor_split_str1 = R"JIT(
    def forward(self, a: Tensor, sections: int, dim: int):
        return torch.tensor_split(a, sections, dim)
  )JIT";
  std::vector<IValue> args1{at::randn({8}), 3, 0};

  const auto tensor_split_str2 = R"JIT(
    def forward(self, a: Tensor, sections: Tensor, dim: int):
        return torch.tensor_split(a, sections, dim)
  )JIT";
  std::vector<IValue> args2{at::randn({8}), torch::tensor(3), 0};

  const auto tensor_split_str3 = R"JIT(
    def forward(self, a: Tensor, indicies: List[int], dim: int):
        return torch.tensor_split(a, indicies, dim)
  )JIT";
  std::vector<IValue> args3{at::randn({8}), c10::List<int64_t>({1, 6}), 0};

  testStaticRuntime(tensor_split_str1, args1);
  testStaticRuntime(tensor_split_str2, args2);
  testStaticRuntime(tensor_split_str3, args3);
}

TEST(StaticRuntime, JIT_Aten_Cpu) {
  const std::string script = R"IR(
    graph(%a: Tensor):
        %1 : int = prim::Constant[value=0]()
        %aa: Tensor = aten::add(%a, %a, %1)
        %ret: Tensor = aten::cpu(%aa)
        return (%ret)
  )IR";

  auto graph = std::make_shared<Graph>();
  std::unordered_map<std::string, Value*> vmap;
  vmap.reserve(0);
  parseIR(script, graph.get(), vmap);
  torch::jit::StaticModule smodule(graph);

  auto a = at::randn({2, 4});
  std::vector<IValue> args0{a};

  testStaticRuntime(script, args0);
}

TEST(StaticRuntime, JIT_Aten_Numel) {
  const std::string script = R"IR(
    graph(%a: Tensor):
        %1 : int = prim::Constant[value=0]()
        %aa: Tensor = aten::add(%a, %a, %1)
        %ret: int = aten::numel(%aa)
        return (%ret)
  )IR";

  auto graph = std::make_shared<Graph>();
  std::unordered_map<std::string, Value*> vmap;
  vmap.reserve(0);
  parseIR(script, graph.get(), vmap);
  torch::jit::StaticModule smodule(graph);

  auto a = at::randn({2, 4});
  std::vector<IValue> args0{a};

  testStaticRuntime(script, args0);
}

TEST(StaticRuntime, JIT_Aten_List) {
  const auto script_str = R"IR(
    graph(%a: str):
        %ret: str[] = aten::list(%a)
        return (%ret)
  )IR";
  std::string a = "abcd";
  std::vector<IValue> args0{a};
  testStaticRuntime(script_str, args0);

  // Update the result of aten::list to ensure that a deep copy
  // took place
  const auto script_list = R"IR(
    graph(%a : int[]):
        %idx : int = prim::Constant[value=0]()
        %value : int = prim::Constant[value=42]()
        %res : int[] = aten::list(%a)
        %updated : int[] = aten::_set_item(%res, %idx, %value)
        return (%res, %a)
  )IR";

  std::vector<IValue> args1{c10::List<int64_t>{1, 2, 3}};
  testStaticRuntime(script_list, args1);
}

TEST(StaticRuntime, JIT_Aten_Range_Length) {
  const std::string script = R"IR(
    graph(%lo: int, %hi: int, %step: int):
        %1 : int = prim::Constant[value=0]()
        %ret: int = aten::__range_length(%lo, %hi, %step)
        return (%ret)
  )IR";

  auto graph = std::make_shared<Graph>();
  std::unordered_map<std::string, Value*> vmap;
  vmap.reserve(0);
  parseIR(script, graph.get(), vmap);
  torch::jit::StaticModule smodule(graph);

  std::vector<IValue> args0{0, 10, 2};

  testStaticRuntime(script, args0);
}

TEST(StaticRuntime, Cat) {
  const std::string cat_script = R"IR(
    graph(%a: Tensor, %b: Tensor, %dim: int):
        %ten_list: Tensor[] = prim::ListConstruct(%a, %b)
        %1 : int = prim::Constant[value=0]()
        %2 : int = prim::Constant[value=1]()
        %3 : int = prim::Constant[value=1]()
        %ten_list2 : Tensor[] = aten::slice(%ten_list, %1, %2, %3)
        %ret: Tensor = aten::cat(%ten_list2, %dim)
        return (%ret)
  )IR";

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

  std::vector<IValue> args_dim_negative{c, d, -1};
  testStaticRuntime(cat_script, args_dim_negative);
}

TEST(StaticRuntime, Cumsum) {
  const auto cumsum_script = R"JIT(
    def forward(self, a: Tensor, dim: int):
        return torch.cumsum(a, dim).clone()
  )JIT";

  auto a = at::randn({2, 3});
  std::vector<IValue> args0{a, 0};
  testStaticRuntime(cumsum_script, args0);

  auto b = at::randn({3, 6});
  std::vector<IValue> args1{b, 1};
  testStaticRuntime(cumsum_script, args0, args1);
}

TEST(StaticRuntime, CumsumDtype) {
  const auto cumsum_script_dtype = R"JIT(
    def forward(self, a: Tensor, dim: int, dtype: int):
        return torch.cumsum(a, dim, dtype=dtype).clone()
  )JIT";

  auto a = at::randn({1, 2});
  auto dtype = at::ScalarType::Float;
  std::vector<IValue> args0{a, 0, dtype};
  testStaticRuntime(cumsum_script_dtype, args0);

  auto b = at::randn({3, 6});
  std::vector<IValue> args1{b, 1, dtype};
  testStaticRuntime(cumsum_script_dtype, args0, args1);
}

TEST(StaticRuntime, Nonzero) {
  const auto nonzero_tensor = R"JIT(
    def forward(self, input: Tensor):
        a = torch.nonzero(input).clone()
        return (a)
  )JIT";

  auto a = at::randint(0, 2, {2, 3});
  testStaticRuntime(nonzero_tensor, {a});

  auto b = at::randint(0, 2, {4, 3, 2});
  testStaticRuntime(nonzero_tensor, {a}, {b});
}

TEST(StaticRuntime, SignedLog1p) {
  const std::string signed_log1p_script = R"IR(
    graph(%input):
        %0 : Tensor = aten::sign(%input)
        %1 : Tensor = aten::abs(%input)
        %2 : Tensor = aten::log1p(%1)
        %3 : Tensor = aten::mul(%0, %2)
        %none : NoneType = prim::Constant()
        %res : Tensor = aten::clone(%3, %none)
        return (%res)
  )IR";

  std::vector<IValue> args1 = {at::randn({2, 2})};
  testStaticRuntime(signed_log1p_script, args1, {}, true);

  std::vector<IValue> args2 = {at::randn({3, 3, 3})};
  testStaticRuntime(signed_log1p_script, args1, args2, true);
}

TEST(StaticRuntime, RemoveImmutableInputDictLookupsWithImmutableInputDict) {
  const auto getitem_immutable_input_dict_script = R"JIT(
    def forward(self, input: Dict[int, Tensor]):
        a = input[0]
        b = input[1]
        c = a + b
        return c.clone()
  )JIT";

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
  const auto getitem_mutable_input_dict_script = R"JIT(
    def forward(self, input: Dict[int, Tensor]):
        a = input[0]
        input[1] = a
        b = input[1]
        c = a + b
        return c.clone()
  )JIT";

  script::Module module("module");
  module.define(getitem_mutable_input_dict_script);
  torch::jit::StaticModule smodule(module);
  EXPECT_TRUE(hasNodeWithKind(smodule, "aten::__getitem__"));
  EXPECT_FALSE(hasNodeWithKind(smodule, "static_runtime::dict_unpack"));
}

TEST(StaticRuntime, VarTupleUnpack) {
  const auto var_tuple_unpack_script = R"JIT(
    def forward(self, input_0: Tuple[Tensor, Tensor], input_1: Tuple[int, int]):
        a, b = input_0
        c, d = input_1
        res = a * c + b * d
        return res.clone()
  )JIT";

  script::Module module("module");
  module.define(var_tuple_unpack_script);
  torch::jit::StaticModule smodule(module);
  EXPECT_FALSE(hasNodeWithKind(smodule, "prim::TupleUnpack"));
  EXPECT_TRUE(hasNodeWithKind(smodule, "static_runtime::VarTupleUnpack"));

  auto a = at::randn({2, 2});
  auto b = at::randn({3, 3, 3});
  std::vector<IValue> args1{
      c10::ivalue::Tuple::create(a, a), c10::ivalue::Tuple::create(1, 2)};
  std::vector<IValue> args2{
      c10::ivalue::Tuple::create(b, b), c10::ivalue::Tuple::create(1, 2)};

  testStaticRuntime(var_tuple_unpack_script, args1);
  testStaticRuntime(var_tuple_unpack_script, args1, args2);
}

TEST(StaticRuntime, VarTupleUnpack_NotApplied) {
  const auto var_tuple_unpack_not_applied_script = R"JIT(
    def forward(self, input_0: Tuple[Tensor, Tensor], input_1: Tuple[int, int]):
        a, b = input_0
        x = a + b
        c, d = input_1
        res = a * c + b * d + x
        return res.clone()
  )JIT";

  script::Module module("module");
  // In this script, the optimization is not applied since there is a
  // computation between the TupleUnpack nodes.
  module.define(var_tuple_unpack_not_applied_script);
  torch::jit::StaticModule smodule(module);
  EXPECT_FALSE(hasNodeWithKind(smodule, "static_runtime::VarTupleUnpack"));
  EXPECT_TRUE(hasNodeWithKind(smodule, "prim::TupleUnpack"));
}

TEST(StaticRuntime, RemainderTensor) {
  const auto remainder_tensor = R"JIT(
    def forward(self, x, y):
        return torch.remainder(x, y).clone()
  )JIT";

  std::vector<IValue> args1 = {
      at::randint(0, 10, {2, 2}), at::randint(1, 10, {2, 2})};
  std::vector<IValue> args2 = {
      at::randint(0, 10, {3, 6}), at::randint(1, 10, {3, 6})};

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

TEST(StaticRuntime, RemainderScalar) {
  const auto remainder_scalar = R"JIT(
    def forward(self, x, y: int):
        return torch.remainder(x, y).clone()
  )JIT";

  std::vector<IValue> args1 = {at::randint(0, 10, {2, 2}), 4};
  std::vector<IValue> args2 = {at::randint(0, 10, {3, 6}), 4};

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

TEST(StaticRuntime, Where) {
  const auto where_script = R"JIT(
    def forward(self, x, y):
        return torch.where(x > 0, x, y).clone()
  )JIT";

  std::vector<IValue> args1 = {at::randn({2, 2}), at::randn({2, 2})};
  std::vector<IValue> args2 = {at::randn({8, 10}), at::randn({8, 10})};

  testStaticRuntime(where_script, args1);
  testStaticRuntime(where_script, args1, args2);
}

TEST(StaticRuntime, WhereBroadcast) {
  const auto where_script = R"JIT(
    def forward(self, cond_1d, x, y):
        shape = [-1] + [1] * (x.dim() - 1)
        cond = cond_1d.view(shape)
        return torch.where(cond, x, y).clone()
  )JIT";

  std::vector<IValue> args1 = {
      at::tensor({0, 1}).to(at::kBool), at::randn({2, 2}), at::randn({2, 2})};
  std::vector<IValue> args2 = {
      at::tensor({1, 0, 0}).to(at::kBool),
      at::randn({3, 6}),
      at::randn({3, 6})};

  testStaticRuntime(where_script, args1);
  testStaticRuntime(where_script, args1, args2);
}

TEST(StaticRuntime, View) {
  // Note that clone is not technically necessary here since this is not
  // an out variant, but it suppresses warnings about only have one op
  // in testStaticRuntime
  const auto src = R"IR(
    graph(%input : Tensor, %shape : int[]):
        %none : NoneType = prim::Constant()
        %view : Tensor = aten::view(%input, %shape)
        %res : Tensor = aten::clone(%view, %none)
        return (%res)
  )IR";

  std::vector<IValue> args1{at::randn({2, 2}), c10::List<int64_t>(4)};
  std::vector<IValue> args2{at::randn({2, 2, 2}), c10::List<int64_t>({4, 2})};

  testStaticRuntime(src, args1);
  testStaticRuntime(src, args1, args2);
}

TEST(StaticRuntime, Size) {
  const auto src_with_dim = R"JIT(
      def forward(self, x, dim: int):
          return x.size(dim)
  )JIT";

  const auto src_no_dim = R"JIT(
      def forward(self, x):
          return x.size()
  )JIT";

  std::vector<IValue> args1{at::randn({1}), 0};
  std::vector<IValue> args2{at::randn({1}), -1};
  std::vector<IValue> args3{at::randn({2, 4}), 1};
  std::vector<IValue> args_no_dim{at::randn({2, 4})};

  testStaticRuntime(src_with_dim, args1);
  testStaticRuntime(src_with_dim, args2);
  testStaticRuntime(src_with_dim, args1, args3);
  testStaticRuntime(src_no_dim, args_no_dim);
}

TEST(StaticRuntime, Squeeze) {
  // Note: this is a native op, not an out variant, but clone anyways
  // to silence warnings in testStaticRuntime
  const auto src = R"JIT(
    def forward(self, inp, dim: int):
        return inp.squeeze(dim).clone()
  )JIT";

  const auto a = at::randn({2, 2});
  const auto b = at::randn({3, 2, 3});

  testStaticRuntime(src, {a, 0});
  testStaticRuntime(src, {a, 1});
  testStaticRuntime(src, {a, -1}, {b, 2});
}

TEST(StaticRuntime, NumToTensorScalar) {
  const auto num_to_tensor_ir = R"IR(
    graph(%1 : int):
      %2 : NoneType = prim::Constant()
      %3 : Tensor = prim::NumToTensor(%1)
      %4 : Tensor = aten::clone(%3, %2)
      return (%4)
  )IR";

  IValue arg{5};
  std::vector<IValue> args = {arg};
  testStaticRuntime(num_to_tensor_ir, args);
}

TEST(StaticRuntime, NumToTensorFalse) {
  const auto num_to_tensor_ir = R"IR(
    graph(%1 : bool):
      %2 : NoneType = prim::Constant()
      %3 : Tensor = prim::NumToTensor(%1)
      %4 : Tensor = aten::clone(%3, %2)
      return (%4)
  )IR";

  IValue arg{false};
  std::vector<IValue> args = {arg};
  testStaticRuntime(num_to_tensor_ir, args);
}

TEST(StaticRuntime, NumToTensorTrue) {
  const auto num_to_tensor_ir = R"IR(
    graph(%1 : bool):
      %2 : NoneType = prim::Constant()
      %3 : Tensor = prim::NumToTensor(%1)
      %4 : Tensor = aten::clone(%3, %2)
      return (%4)
  )IR";

  IValue arg{true};
  std::vector<IValue> args = {arg};
  testStaticRuntime(num_to_tensor_ir, args);
}

TEST(StaticRuntime, Split) {
  const auto src = R"JIT(
    def forward(self, inp, split_size: int, dim: int):
        return inp.split(split_size, dim)
  )JIT";

  const auto a = at::randn({2, 2});
  const auto b = at::randn({2, 2, 2});

  testStaticRuntime(src, {a, 1, 0});
  testStaticRuntime(src, {a, 1, 1});
  testStaticRuntime(src, {a, 2, -1}, {b, 2, 2});
}

TEST(StaticRuntime, SplitWithSizes) {
  const auto src = R"JIT(
    def forward(self, inp, split_sizes: List[int], dim: int):
        return inp.split(split_sizes, dim)
  )JIT";

  const auto a = at::randn({2, 2});
  const auto b = at::randn({2, 2, 2});
  const auto split_sizes = c10::List<int64_t>{1, 1};

  testStaticRuntime(src, {a, split_sizes, 0});
  testStaticRuntime(src, {a, split_sizes, 1});
  testStaticRuntime(src, {a, split_sizes, -1}, {b, split_sizes, 2});
}

namespace {

void maybe_throw(bool should_throw) {
  if (should_throw) {
    throw std::runtime_error("test exception");
  }
}

TORCH_LIBRARY(static_runtime_tests, m) {
  // Conservative so this op doesn't get deleted by dead
  // code elimination
  m.def(torch::schema(
      "static_runtime_tests::maybe_throw(bool throw) -> ()",
      at::AliasAnalysisKind::CONSERVATIVE));
  m.impl("maybe_throw", maybe_throw);
}

} // namespace

TEST(StaticRuntime, ModelCrashOnFirstRun) {
  const auto src = R"JIT(
    graph(%0: Tensor, %throw: bool):
        %1: Tensor = aten::mul(%0, %0)
        static_runtime_tests::maybe_throw(%throw)
        %2: Tensor = aten::mul(%1, %1)
        %3: Tensor = aten::mul(%2, %2)
        return (%3)
  )JIT";

  auto graph = getGraphFromIR(src);
  auto static_module = StaticModule(graph);
  auto& runtime = static_module.runtime();

  std::vector<IValue> args_crash{at::randn({1}), true};
  std::vector<IValue> args_no_crash{at::randn({1}), false};
  EXPECT_THROW(runtime(args_crash, {}), std::runtime_error);

  // The run didn't finish, we didn't allocate the memory planner
  EXPECT_EQ(runtime.get_memory_planner(), nullptr);
  runtime.check_for_memory_leak();

  // We guarantee that the runtime is still usable after the crash.
  // Run again to verify this.
  compareResultsWithJIT(runtime, graph, args_no_crash);
  EXPECT_NE(runtime.get_memory_planner(), nullptr);
}

TEST(StaticRuntime, ModelCrashOnSecondRun) {
  const auto src = R"JIT(
    graph(%0: Tensor, %throw: bool):
        %1: Tensor = aten::mul(%0, %0)
        static_runtime_tests::maybe_throw(%throw)
        %2: Tensor = aten::mul(%1, %1)
        %3: Tensor = aten::mul(%2, %2)
        return (%3)
  )JIT";

  auto graph = getGraphFromIR(src);
  auto static_module = StaticModule(graph);
  auto& runtime = static_module.runtime();

  std::vector<IValue> args_crash{at::randn({1}), true};
  std::vector<IValue> args_no_crash{at::randn({1}), false};
  runtime(args_no_crash, {});
  EXPECT_NE(runtime.get_memory_planner(), nullptr);
  runtime.check_for_memory_leak();

  EXPECT_THROW(runtime(args_crash, {}), std::runtime_error);
  runtime.check_for_memory_leak();

  // We guarantee that the runtime is still usable after the crash.
  // Run again to verify this.
  compareResultsWithJIT(runtime, graph, args_no_crash);
}

TEST(StaticRuntime, ModelCrashOnFirstRunWithBorrows) {
  const auto src = R"JIT(
    graph(%0: Tensor):
        %1: Tensor = aten::mul(%0, %0)
        %2: Tensor = aten::mul(%1, %1)
        %3: bool = prim::Constant[value=1]()
        %4: Tensor = static_runtime::select_tensor(%1, %2, %3)
        static_runtime_tests::maybe_throw(%3)
        return (%4)
  )JIT";
  auto graph = getGraphFromIR(src);
  auto static_module = StaticModule(graph);
  auto& runtime = static_module.runtime();

  std::vector<IValue> args{at::randn({1})};
  EXPECT_THROW(runtime(args), std::runtime_error);
}

TEST(StaticRuntime, ModelCrashOnFirstRunWithBorrowedInputs) {
  const auto src = R"JIT(
    graph(%0: Tensor, %1: Tensor):
        %2: bool = prim::Constant[value=1]()
        %3: Tensor = static_runtime::select_tensor(%0, %1, %2)
        static_runtime_tests::maybe_throw(%2)
        return (%3)
  )JIT";
  auto graph = getGraphFromIR(src);
  auto static_module = StaticModule(graph);
  auto& runtime = static_module.runtime();

  std::vector<IValue> args{at::randn({1}), at::randn({1})};
  EXPECT_THROW(runtime(std::move(args)), std::runtime_error);
}

TEST(StaticRuntime, ReplaceWithMaybeCopy) {
  const std::string to = R"IR(
    graph(%0 : Tensor):
      %1: int = prim::Constant[value=4]()
      %2: bool = prim::Constant[value=0]()
      %3: None = prim::Constant()
      %res : Tensor = aten::to(%0, %1, %2, %2, %3)
      return (%res)
  )IR";

  at::Tensor a = at::tensor({1.1, 2.2, 3.3, 4.0}, at::ScalarType::Float);
  std::vector<IValue> args{a};
  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(to, g.get());

  // Jit Interpreter.
  Stack stack(args);
  torch::jit::GraphExecutor graph_exec(g, "");
  graph_exec.run(stack);
  ASSERT_EQ(stack.size(), 1);
  auto expected = stack[0].toTensor();

  // Static Runtime.
  torch::jit::StaticModule smodule(g);
  auto actual = smodule(args, {}).toTensor();
  smodule.runtime().check_for_memory_leak();

  EXPECT_TRUE(expected.equal(actual));

  // Make a fresh graph to ensure the pass works in isolation
  auto new_graph = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(to, new_graph.get());
  ReplaceWithMaybeCopy(new_graph);
  EXPECT_FALSE(hasNodeWithKind(new_graph, "aten::to"));
  EXPECT_TRUE(
      hasNodeWithKind(new_graph, "static_runtime::to_maybe_copy_out"));
}

TEST(StaticRuntime, Int) {
  const auto src = R"JIT(
    def forward(self, x):
        return int(x) + int(x)
  )JIT";
  std::vector<IValue> args{at::tensor({3.14})};
  testStaticRuntime(src, args);
}

TEST(StaticRuntime, ReturnConstant) {
  const auto src = R"JIT(
    def forward(self):
        return 1
  )JIT";

  testStaticRuntime(src, {});
}

TEST(StaticRuntime, SimpleIf) {
  const auto src = R"JIT(
    def forward(self, cond: bool, x):
        if cond:
            return torch.mul(x, 42).clone()
        else:
            return x.clone()
  )JIT";

  std::vector<IValue> args_false{false, at::randn({1})};
  std::vector<IValue> args_true{true, at::randn({1})};
  std::vector<IValue> args_big_tensor{true, at::randn({3, 3, 3})};

  testStaticRuntime(src, args_false);
  testStaticRuntime(src, args_true);
  testStaticRuntime(src, args_true, args_big_tensor);
}

TEST(StaticRuntime, NestedIf) {
  const auto src = R"JIT(
    def forward(self, cond1: bool, cond2: bool, x):
        y = x * 42
        if cond1:
            y = y * y
            if cond2:
                y += x
        else:
            if cond2:
                return x.clone()

        return y.clone()
  )JIT";

  for (auto cond1 : {true, false}) {
    for (auto cond2 : {true, false}) {
      std::vector<IValue> args1{cond1, cond2, at::randn({1})};
      std::vector<IValue> args2{cond1, cond2, at::randn({3, 3, 3})};
      testStaticRuntime(src, args1, args2);
    }
  }
}

TEST(StaticRuntime, DeeplyNestedIf) {
  const auto src = R"JIT(
    def forward(self, cond1: bool, cond2: bool, cond3: bool, x):
        y = x * 42
        if cond1:
            y = y * y
            if cond2:
                y += x

            if cond2 and cond3:
                y += 1

            if cond2:
                if cond3:
                    y += 2
                else:
                    y = y * y
                    y += 4
        else:
            if cond2:
                return x.clone()
            if cond3 or cond2:
                y += 42

        return y.clone()
  )JIT";

  for (auto cond1 : {true, false}) {
    for (auto cond2 : {true, false}) {
      for (auto cond3 : {true, false}) {
        std::vector<IValue> args1{cond1, cond2, cond3, at::randn({1})};
        std::vector<IValue> args2{cond1, cond2, cond3, at::randn({3, 3, 3})};
        testStaticRuntime(src, args1, args2);
      }
    }
  }
}

TEST(StaticRuntime, BasicForLoop) {
  const auto src = R"JIT(
    def forward(self, x, loop_max: int):
        y = x.clone()
        for i in range(loop_max):
            y += 1
        return y
  )JIT";

  std::vector<IValue> args1{at::randn({1}), 10};
  std::vector<IValue> args2{at::randn({3, 3, 3}), 10};

  testStaticRuntime(src, args1, args2);
}

TEST(StaticRuntime, BasicWhileLoop) {
  const auto src = R"JIT(
    def forward(self, x, loop_max: int):
        y = x.clone()
        loop_count = 0
        while loop_count < loop_max:
            y += 1
            loop_count += 1
        return y
  )JIT";

  std::vector<IValue> args1{at::randn({1}), 10};
  std::vector<IValue> args2{at::randn({3, 3, 3}), 10};

  testStaticRuntime(src, args1, args2);
}

TEST(StaticRuntime, NestedLoops) {
  const auto src = R"JIT(
    def forward(self, x, loop_max: int):
        y = x.clone()
        even: List[int] = []
        odd: List[int] = []

        for i in range(loop_max):
            if i % 2:
                odd.append(i)
            else:
                even.append(i)

            for j in range(i):
                y += 1

        return y, even, odd
  )JIT";

  std::vector<IValue> args1{at::randn({1}), 10};
  std::vector<IValue> args2{at::randn({3, 3, 3}), 10};

  testStaticRuntime(src, args1, args2);
}

TEST(StaticRuntime, TupleIndex) {
  const auto src = R"JIT(
    def forward(self, idx: int, tup: Tuple[int, int]):
        a = tup[idx]
        return a * a
  )JIT";
  const auto tuple = c10::ivalue::Tuple::create({1, 2});
  testStaticRuntime(src, {1, tuple}, {-1, tuple});

  torch::jit::Module mod("module");
  mod.define(src);
  StaticModule smod(mod);
  EXPECT_THROW(smod({100, tuple}), std::out_of_range);
}

TEST(StaticRuntime, RaiseException) {
  const auto src = R"IR(
    graph(%str: str):
        %none: NoneType = prim::Constant()
        prim::RaiseException(%str, %none)
        return (%none)
  )IR";
  auto graph = getGraphFromIR(src);
  StaticModule smod(graph);
  const auto msg = "exception message";
  EXPECT_THROW(
      {
        try {
          smod({msg});
        } catch (const std::runtime_error& e) {
          EXPECT_STREQ(msg, e.what());
          throw;
        }
      },
      std::runtime_error);
}

TEST(StaticRuntime, Uninitialized) {
  const auto src = R"IR(
    graph():
      %0: int = prim::Uninitialized()
      return (%0)
  )IR";
  auto graph = getGraphFromIR(src);
  StaticModule smod(graph);
  const auto ret = smod({});
  // If a and b are both uninitialized, then a != b. So just check that the type
  // is Any
  EXPECT_EQ(ret.type()->kind(), c10::TypeKind::AnyType);
}

TEST(StaticRuntime, Format) {
  const auto src = R"JIT(
    def forward(self, arg1: int, arg2: Tensor, arg3: str):
        a = "arg1: {}, arg2: {}, arg3: {}".format(arg1, arg2, arg3)
        return a[::]
  )JIT";
  testStaticRuntime(src, {1, at::randn({3}), "str"});
}

TEST(StaticRuntime, Device) {
  const auto src = R"JIT(
    def forward(self, x):
        return x.device, x.device
  )JIT";
  testStaticRuntime(src, {at::tensor({1})});
}

TEST(StaticRuntime, Dtype) {
  const auto src = R"JIT(
    def forward(self, x, y):
        return x.dtype, y.dtype
  )JIT";
  testStaticRuntime(
      src, {at::tensor({1}, at::kLong), at::tensor({1}, at::kFloat)});
}

TEST(StaticRuntime, Dim) {
  const auto src = R"JIT(
    def forward(self, x, y):
        return x.dim(), y.dim()
  )JIT";
  testStaticRuntime(src, {at::randn({2, 2}), at::randn({1})});
}

TEST(StaticRuntime, Not) {
  const auto src = R"JIT(
    def forward(self, x: bool, y: bool):
        return not x, not y
  )JIT";
  testStaticRuntime(src, {true, false});
}

TEST(StaticRuntime, Bool) {
  const auto src = R"JIT(
      def forward(self, x: Tensor, y: int, z: float):
          return bool(x), bool(y), bool(z)
  )JIT";
  testStaticRuntime(src, {at::randn({1}), 0, 1.151}, {at::zeros({1}), 1, 0.0});
}

TEST(StaticRuntime, IsCuda) {
  const auto src = R"JIT(
      def forward(self, x: Tensor, y: Tensor):
          return x.is_cuda, y.is_cuda
  )JIT";
  testStaticRuntime(src, {at::randn({1}), at::randn({1})});
}

TEST(StaticRuntime, ToList) {
  const auto src = R"JIT(
      graph(%x: Tensor):
          %type: int = prim::Constant[value=1]()
          %dim: int = aten::dim(%x)
          %ret: float[] = prim::tolist(%x, %dim, %type)
          return (%ret)
  )JIT";
  testStaticRuntime(src, {at::randn({2, 2})});
}

TEST(StaticRuntime, IfThenElse) {
  const auto src = R"IR(
    graph(%cond: bool, %a: Tensor, %b: Tensor):
        %none: NoneType = prim::Constant()
        %c: Tensor = prim::IfThenElse(%cond, %a, %b)
        %d: Tensor = aten::clone(%c, %none)
        return (%d)
  )IR";

  std::vector<IValue> args1{true, at::randn({1}), at::randn({1})};
  std::vector<IValue> args2{false, at::randn({1}), at::randn({1})};

  testStaticRuntime(src, args1);
  testStaticRuntime(src, args2);
}

TEST(StaticRuntime, EmptyIfBlock) {
  const auto src =
      R"JIT(
      def forward(self, cond: bool, a: Tensor, b: Tensor):
          l = []
          if cond:
              l.append((a + b).clone())
          return l
  )JIT";

  testStaticRuntime(src, {true, at::rand(1), at::rand({1, 2})});
  testStaticRuntime(src, {false, at::rand(1), at::rand({1, 2})});
}

TEST(StaticRuntime, EmptyNestedIfBlock) {
  const auto src =
      R"JIT(
      def forward(self, cond: bool, a: Tensor, b: Tensor):
          l = []
          if cond:
              if cond:
                  l.append((a + b).clone())
          return l
  )JIT";

  testStaticRuntime(src, {true, at::rand(1), at::rand({1, 2})});
  testStaticRuntime(src, {false, at::rand(1), at::rand({1, 2})});
}

TEST(StaticRuntime, StackEmpty) {
  const auto src = R"JIT(
    def forward(self):
        x = torch.stack([])
        return x
  )JIT";

  torch::jit::Module mod("mod");
  mod.define(src);

  torch::jit::StaticModule smod(mod);
  EXPECT_THROW(smod({}), c10::Error);
}

TEST(StaticRuntime, ConcatEmpty) {
  const auto src = R"JIT(
    def forward(self):
        x = torch.concat([])
        return x
  )JIT";

  torch::jit::Module mod("mod");
  mod.define(src);

  torch::jit::StaticModule smod(mod);
  EXPECT_THROW(smod({}), c10::Error);
}

TEST(StaticRuntime, IntImplicit) {
  const auto src = R"IR(
    graph(%a: Tensor):
        %y: int = aten::IntImplicit(%a)
        return (%y)
  )IR";
  testStaticRuntime(src, {at::tensor({1}, at::kInt).squeeze()});
}

TEST(StaticRuntime, IntImplicit_ThrowOnBadInputs) {
  const auto src = R"IR(
    graph(%a: Tensor):
        %y: int = aten::IntImplicit(%a)
        return (%y)
  )IR";
  auto graph = getGraphFromIR(src);
  torch::jit::StaticModule smod(graph);
  // Not 0D tensor
  EXPECT_THROW(smod({at::tensor({1, 2}, at::kInt)}), std::runtime_error);
  // Wrong dtype
  EXPECT_THROW(
      smod({at::tensor({1}, at::kFloat).squeeze()}), std::runtime_error);
}

TEST(StaticRuntime, Select) {
  const auto src = R"IR(
    graph(%a: Tensor, %dim: int, %index: int):
        %none: NoneType = prim::Constant()
        %b: Tensor = aten::select(%a, %dim, %index)
        %c: Tensor = aten::clone(%b, %none)
        return (%c)
  )IR";
  testStaticRuntime(src, {at::randn({2, 2}), 0, 1});
}

TEST(StaticRuntime, ReshapeAs) {
  const auto src = R"JIT(
    def forward(self, a, b):
        return a.reshape_as(b).clone()
  )JIT";
  testStaticRuntime(src, {at::randn({2, 2}), at::randn({4})});
}

TEST(StaticRuntime, MoveCtor) {
  auto mod = getDeepAndWideSciptModel();
  std::vector<IValue> args{
      at::randn({1, 1, 32}), at::randn({1, 1, 32}), at::randn({1, 50})};

  torch::jit::StaticModule smod(mod);

  torch::jit::StaticRuntime runtime(smod);
  auto expected = runtime(args);

  torch::jit::StaticRuntime new_runtime(std::move(runtime));
  auto actual = new_runtime(args);
  compareResults(expected, actual);
}

TEST(StaticRuntime, SingleBlockIfReturnList) {
  const auto src = R"JIT(
    def forward(self, a, b, cond: bool):
        lst = []
        if cond:
            lst.append(a + b)
        return lst
  )JIT";
  std::vector<IValue> args1{at::randn({1}), at::randn({1}), true};
  std::vector<IValue> args2{at::randn({42, 42}), at::randn({42, 42}), false};
  testStaticRuntime(src, args1, args2);
}

TEST(StaticRuntime, NestedBlockIfReturnList) {
  const auto src = R"JIT(
    def forward(self, a, b, cond1: bool, cond2: bool):
        if cond1:
            lst = []
            if cond2:
                lst.append(a + b)
            lst.append(a * b)
            return lst
        return []
  )JIT";
  std::vector<IValue> args1{at::randn({1}), at::randn({1}), true, true};
  std::vector<IValue> args2{
      at::randn({42, 42}), at::randn({42, 42}), true, false};
  testStaticRuntime(src, args1, args2);
}

TEST(StaticRuntime, ClampNaNToNum) {
  const auto src1 = R"JIT(
    def forward(self, a):
        return torch.clamp(a, min=1.0, max=2.0).nan_to_num().clone()
  )JIT";

  const auto src2 = R"JIT(
    def forward(self, a, nan: float):
        return torch.clamp(a, min=-1.0, max=2.0).nan_to_num(nan=nan).clone()
  )JIT";

  const auto src3 = R"JIT(
    def forward(self, a):
        return torch.clamp(a, min=1.0, max=-1.0).nan_to_num().clone()
  )JIT";

  auto a = at::tensor({
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      0.0f,
      3.0f
    });
  auto b = a.repeat({10, 5});

  // Have to use_allclose even though all NaNs will be replaced - testStaticRuntime
  // also checks inputs at the end to make sure they're not changed
  testStaticRuntime(src1, {a}, {}, /*use_allclose=*/true, /*use_equalnan=*/true);
  testStaticRuntime(src1, {a}, {b}, /*use_allclose=*/true, /*use_equalnan=*/true);

  testStaticRuntime(src2, {a, 42.0}, {}, /*use_allclose=*/true, /*use_equalnan=*/true);
  testStaticRuntime(src2, {a, 2.0}, {b, 1.0}, /*use_allclose=*/true, /*use_equalnan=*/true);

  testStaticRuntime(src3, {a}, {}, /*use_allclose=*/true, /*use_equalnan=*/true);
  testStaticRuntime(src3, {a}, {b}, /*use_allclose=*/true, /*use_equalnan=*/true);

  // Non-NNC path
  testStaticRuntime(src1, {a.to(at::kDouble)}, {}, /*use_allclose=*/true, /*use_equalnan=*/true);
  testStaticRuntime(src1, {a.to(at::kDouble)}, {b.to(at::kDouble)}, /*use_allclose=*/true, /*use_equalnan=*/true);
}

TEST(StaticRuntime, IfReturningTuple) {
  const auto src = R"JIT(
    def forward(self, x, y, cond: bool, idx: int):
        if cond:
            tup = (x, y)
        else:
            tup = (x, x)
        return tup[idx]
  )JIT";

  std::vector<IValue> args{at::randn({3}), at::randn({3}), true, 0};
  testStaticRuntime(src, args);
}
