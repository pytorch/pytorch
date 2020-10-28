#ifdef USE_VULKAN_API

#include <gtest/gtest.h>
#include <ATen/ATen.h>

// TODO: These functions should move to a common place.

namespace {

bool checkRtol(const at::Tensor& diff, const std::vector<at::Tensor>& inputs) {
  float maxValue = 0.0f;

  for (const auto& tensor : inputs) {
    maxValue = fmax(tensor.abs().max().item<float>(), maxValue);
  }

  return diff.abs().max().item<float>() < (2e-6 * maxValue);
}

bool almostEqual(const at::Tensor& a, const at::Tensor& b) {
  return checkRtol(a - b, {a, b});
}

bool exactlyEqual(const at::Tensor& a, const at::Tensor& b) {
  return (a - b).abs().max().item<float>() == 0.0f;
}

} // namespace

namespace {

TEST(VulkanAPITest, add) {
  const auto a_cpu = at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({11, 7, 139, 109}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  const auto c_cpu = at::add(a_cpu, b_cpu, 2.1f);
  const auto c_vulkan = at::add(a_vulkan, b_vulkan, 2.1f);

  ASSERT_TRUE(almostEqual(c_cpu, c_vulkan.cpu()));
}

TEST(VulkanAPITest, add_) {
  auto a_cpu = at::rand({1, 3, 2, 2}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  const auto b_cpu = at::rand({1, 3, 2, 2}, at::device(at::kCPU).dtype(at::kFloat));
  const auto b_vulkan = b_cpu.vulkan();

  a_cpu.add_(b_cpu, 2.1f);
  a_vulkan.add_(b_vulkan, 2.1f);
  auto a_cpu_test = a_vulkan.cpu();

  const auto check = almostEqual(a_cpu_test, a_cpu);
  if (!check) {
    std::cout << "expected:\n" << a_cpu << std::endl;
    std::cout << "got:\n" << a_cpu_test << std::endl;
  }
  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, add_scalar) {
  const auto a_cpu = at::rand({1, 1, 1, 1}, at::device(at::kCPU).dtype(at::kFloat));
  const auto a_vulkan = a_cpu.vulkan();

  const float b_scalar = 3.1415f;

  const auto c_cpu = at::add(a_cpu, b_scalar, 2.1f);
  const auto c_vulkan = at::add(a_vulkan, b_scalar, 2.1f);

  ASSERT_TRUE(almostEqual(c_cpu, c_vulkan.cpu()));
}

TEST(VulkanAPITest, add_scalar_) {
  auto a_cpu = at::rand({1, 3, 2, 2}, at::device(at::kCPU).dtype(at::kFloat));
  auto a_vulkan = a_cpu.vulkan();

  const float b_scalar = 3.1415f;

  a_cpu.add_(b_scalar, 2.1f);
  a_vulkan.add_(b_scalar, 2.1f);
  auto a_cpu_test = a_vulkan.cpu();

  const auto check = almostEqual(a_cpu_test, a_cpu);
  if (!check) {
    std::cout << "expected:\n" << a_cpu << std::endl;
    std::cout << "got:\n" << a_cpu_test << std::endl;
  }
  ASSERT_TRUE(check);
}

TEST(VulkanTest, addmm) {
  auto t_m1 = at::rand({2, 2}, at::device(at::kCPU).dtype(at::kFloat));
  auto t_m2 = at::rand({2, 3}, at::device(at::kCPU).dtype(at::kFloat));
  auto t_b = at::rand({2, 3}, at::device(at::kCPU).dtype(at::kFloat));

  float beta = 100;
  float alpha = 2;
  auto t_out_expected = at::addmm(t_b, t_m1, t_m2, beta, alpha);

  std::cout << "XXX " << __FILE__ << ":" << __LINE__ << std::endl;
  auto tv_m1 = t_m1.vulkan();
  std::cout << "XXX " << __FILE__ << ":" << __LINE__ << std::endl;
  auto tv_m2 = t_m2.vulkan();
  std::cout << "XXX " << __FILE__ << ":" << __LINE__ << std::endl;
  auto tv_b = t_b.vulkan();
  std::cout << "XXX " << __FILE__ << ":" << __LINE__ << std::endl;
  auto tv_out = at::addmm(tv_b, tv_m1, tv_m2, beta, alpha);
  std::cout << "XXX " << __FILE__ << ":" << __LINE__ << std::endl;
  auto t_out = tv_out.cpu();
  std::cout << "XXX " << __FILE__ << ":" << __LINE__ << std::endl;
  const auto check = almostEqual(t_out, t_out_expected);
  if (!check) {
    std::cout << "expected:\n" << t_out_expected << std::endl;
    std::cout << "got:\n" << t_out << std::endl;
  }
  ASSERT_TRUE(check);
}

TEST(VulkanTest, mm) {
  std::cout << "XXX " << __FILE__ << ":" << __LINE__ << std::endl;
  auto t_m1 = at::rand({2, 3}, at::device(at::kCPU).dtype(at::kFloat));
  auto t_m2 = at::rand({3, 2}, at::device(at::kCPU).dtype(at::kFloat));
  std::cout << "XXX " << __FILE__ << ":" << __LINE__ << std::endl;

  auto t_out_expected = t_m1.mm(t_m2);
  std::cout << "XXX " << __FILE__ << ":" << __LINE__ << std::endl;

  std::cout << "XXX " << __FILE__ << ":" << __LINE__ << std::endl;
  auto tv_m1 = t_m1.vulkan();
  auto tv_m2 = t_m2.vulkan();
  auto tv_out = tv_m1.mm(tv_m2);
  auto t_out = tv_out.cpu();
  std::cout << "XXX " << __FILE__ << ":" << __LINE__ << std::endl;
  const auto check = almostEqual(t_out, t_out_expected);
  if (!check) {
    std::cout << "expected:\n" << t_out_expected << std::endl;
    std::cout << "got:\n" << t_out << std::endl;
  }
  std::cout << "XXX " << __FILE__ << ":" << __LINE__ << std::endl;
  ASSERT_TRUE(check);
}

TEST(VulkanTest, adaptive_avg_pool2d) {
  auto t_in =
      at::rand({1, 2, 7, 7}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  auto t_out_expected = at::adaptive_avg_pool2d(t_in, {3, 3});
  auto tv_in = t_in.vulkan();

  auto tv_out = at::adaptive_avg_pool2d(tv_in, {3, 3});
  auto t_out = tv_out.cpu();

  const auto check = almostEqual(t_out, t_out_expected);
  if (!check) {
    std::cout << "expected:\n" << t_out_expected << std::endl;
    std::cout << "got:\n" << t_out << std::endl;
  }
  ASSERT_TRUE(check);
}

TEST(VulkanAPITest, copy) {
  const auto cpu = at::rand({13, 17, 37, 19}, at::device(at::kCPU).dtype(at::kFloat));
  ASSERT_TRUE(exactlyEqual(cpu, cpu.vulkan().cpu()));
}

TEST(VulkanAPITest, empty) {
  ASSERT_NO_THROW(at::empty({1, 17, 41, 53}, at::device(at::kVulkan).dtype(at::kFloat)));
}

} // namespace

#endif /* USE_VULKAN_API */
