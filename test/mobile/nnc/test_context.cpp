#include <gtest/gtest.h>
#include <torch/csrc/jit/mobile/nnc/context.h>
#include <torch/csrc/jit/mobile/nnc/registry.h>
#include <ATen/Functions.h>

namespace torch {
namespace jit {
namespace mobile {
namespace nnc {

extern "C" {

// out = a * n (doing calculation in the `tmp` buffer)
int slow_mul_kernel(void** args) {
  const int size = 128;
  at::Tensor a = at::from_blob(args[0], {size}, at::kFloat);
  at::Tensor out = at::from_blob(args[1], {size}, at::kFloat);
  at::Tensor n = at::from_blob(args[2], {1}, at::kInt);
  at::Tensor tmp = at::from_blob(args[3], {size}, at::kFloat);

  tmp.zero_();
  for (int i = n.item().toInt(); i > 0; i--) {
    tmp.add_(a);
  }
  out.copy_(tmp);
  return 0;
}

int dummy_kernel(void** /* args */) {
  return 0;
}

} // extern "C"

REGISTER_NNC_KERNEL("slow_mul", slow_mul_kernel)
REGISTER_NNC_KERNEL("dummy", dummy_kernel)

InputSpec create_test_input_spec(const std::vector<int64_t>& sizes) {
  InputSpec input_spec;
  input_spec.sizes_ = sizes;
  input_spec.dtype_ = at::kFloat;
  return input_spec;
}

OutputSpec create_test_output_spec(const std::vector<int64_t>& sizes) {
  OutputSpec output_spec;
  output_spec.sizes_ = sizes;
  output_spec.dtype_ = at::kFloat;
  return output_spec;
}

MemoryPlan create_test_memory_plan(const std::vector<int64_t>& buffer_sizes) {
  MemoryPlan memory_plan;
  memory_plan.buffer_sizes_ = buffer_sizes;
  return memory_plan;
}

TEST(Function, ExecuteSlowMul) {
  const int a = 999;
  const int n = 100;
  const int size = 128;
  Function f;

  f.set_nnc_kernel_id("slow_mul");
  f.set_input_specs({create_test_input_spec({size})});
  f.set_output_spec({create_test_output_spec({size})});
  f.set_parameters({at::ones({1}, at::kInt).mul(n)});
  f.set_memory_plan(create_test_memory_plan({sizeof(float) * size}));

  c10::List<at::Tensor> input({
      at::ones({size}, at::kFloat).mul(a)
  });
  auto outputs = f.run(c10::impl::toList(input));
  auto output = ((const c10::IValue&) outputs[0]).toTensor();
  auto expected_output = at::ones({size}, at::kFloat).mul(a * n);
  EXPECT_TRUE(output.equal(expected_output));
}

TEST(Function, Serialization) {
  Function f;
  f.set_name("test_function");
  f.set_nnc_kernel_id("test_kernel");
  f.set_input_specs({create_test_input_spec({1, 3, 224, 224})});
  f.set_output_spec({create_test_output_spec({1000})});
  f.set_parameters({
      at::ones({1, 16, 3, 3}, at::kFloat),
      at::ones({16, 32, 1, 1}, at::kFloat),
      at::ones({32, 1, 3, 3}, at::kFloat)
  });
  f.set_memory_plan(create_test_memory_plan({
      sizeof(float) * 1024,
      sizeof(float) * 2048,
  }));

  auto serialized = f.serialize();
  Function f2(serialized);
  EXPECT_EQ(f2.name(), "test_function");
  EXPECT_EQ(f2.nnc_kernel_id(), "test_kernel");
  EXPECT_EQ(f2.input_specs().size(), 1);
  EXPECT_EQ(f2.input_specs()[0].sizes_, std::vector<int64_t>({1, 3, 224, 224}));
  EXPECT_EQ(f2.input_specs()[0].dtype_, at::kFloat);

  EXPECT_EQ(f2.output_specs().size(), 1);
  EXPECT_EQ(f2.output_specs()[0].sizes_, std::vector<int64_t>({1000}));
  EXPECT_EQ(f2.output_specs()[0].dtype_, at::kFloat);

  EXPECT_EQ(f2.parameters().size(), 3);
  EXPECT_EQ(f2.parameters()[0].sizes(), at::IntArrayRef({1, 16, 3, 3}));
  EXPECT_EQ(f2.parameters()[1].sizes(), at::IntArrayRef({16, 32, 1, 1}));
  EXPECT_EQ(f2.parameters()[2].sizes(), at::IntArrayRef({32, 1, 3, 3}));

  EXPECT_EQ(f2.memory_plan().buffer_sizes_.size(), 2);
  EXPECT_EQ(f2.memory_plan().buffer_sizes_[0], sizeof(float) * 1024);
  EXPECT_EQ(f2.memory_plan().buffer_sizes_[1], sizeof(float) * 2048);
}

TEST(Function, ValidInput) {
  const int size = 128;
  Function f;
  f.set_nnc_kernel_id("dummy");
  f.set_input_specs({create_test_input_spec({size})});

  c10::List<at::Tensor> input({
      at::ones({size}, at::kFloat)
  });
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  EXPECT_NO_THROW(
      f.run(c10::impl::toList(input)));
}

TEST(Function, InvalidInput) {
  const int size = 128;
  Function f;
  f.set_nnc_kernel_id("dummy");
  f.set_input_specs({create_test_input_spec({size})});

  c10::List<at::Tensor> input({
      at::ones({size * 2}, at::kFloat)
  });
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  EXPECT_THROW(
      f.run(c10::impl::toList(input)),
      c10::Error);
}

} // namespace nnc
} // namespace mobile
} // namespace jit
} // namespace torch
