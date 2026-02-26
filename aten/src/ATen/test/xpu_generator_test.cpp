#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>
#include <ATen/xpu/XPUGeneratorImpl.h>
#include <ATen/core/PhiloxRNGEngine.h>

#include <assert.h>
#include <thread>

TEST(XpuGeneratorTest, testGeneratorDynamicCast) {
  if (!at::xpu::is_available()) {
    return;
  }
  auto foo = at::xpu::detail::createXPUGenerator();
  auto result = foo.get<at::XPUGeneratorImpl>();
  EXPECT_EQ(typeid(at::XPUGeneratorImpl*).hash_code(), typeid(result).hash_code());
}

TEST(XpuGeneratorTest, testDefaultGenerator) {
  if (!at::xpu::is_available()) {
    return;
  }
  auto foo = at::xpu::detail::getDefaultXPUGenerator();
  auto bar = at::xpu::detail::getDefaultXPUGenerator();
  EXPECT_EQ(foo, bar);

  auto offset = foo.get_offset() << 1;
  foo.set_offset(offset);
  EXPECT_EQ(foo.get_offset(), offset);

  if (c10::xpu::device_count() >= 2) {
    foo = at::xpu::detail::getDefaultXPUGenerator(0);
    bar = at::xpu::detail::getDefaultXPUGenerator(0);
    EXPECT_EQ(foo, bar);

    foo = at::xpu::detail::getDefaultXPUGenerator(0);
    bar = at::xpu::detail::getDefaultXPUGenerator(1);
    EXPECT_NE(foo, bar);
  }
}

TEST(XpuGeneratorTest, testCloning) {
  if (!at::xpu::is_available()) {
    return;
  }
  auto gen1 = at::xpu::detail::createXPUGenerator();
  gen1.set_current_seed(123); // modify gen1 state
  auto xpu_gen1 = at::check_generator<at::XPUGeneratorImpl>(gen1);
  xpu_gen1->set_philox_offset_per_thread(4);
  auto gen2 = at::xpu::detail::createXPUGenerator();
  gen2 = gen1.clone();
  auto xpu_gen2 = at::check_generator<at::XPUGeneratorImpl>(gen2);
  EXPECT_EQ(gen1.current_seed(), gen2.current_seed());
  EXPECT_EQ(
    xpu_gen1->philox_offset_per_thread(),
    xpu_gen2->philox_offset_per_thread()
  );
}

void thread_func_get_set_current_seed(at::Generator generator) {
  std::lock_guard<std::mutex> lock(generator.mutex());
  auto current_seed = generator.current_seed();
  current_seed++;
  generator.set_current_seed(current_seed);
}

TEST(XpuGeneratorTest, testMultithreadingGetSetCurrentSeed) {
  // See Note [Acquire lock when using random generators]
  if (!at::xpu::is_available()) {
    return;
  }
  auto gen1 = at::xpu::detail::getDefaultXPUGenerator();
  auto initial_seed = gen1.current_seed();
  std::thread t0{thread_func_get_set_current_seed, gen1};
  std::thread t1{thread_func_get_set_current_seed, gen1};
  std::thread t2{thread_func_get_set_current_seed, gen1};
  t0.join();
  t1.join();
  t2.join();
  EXPECT_EQ(gen1.current_seed(), initial_seed+3);
}

TEST(XpuGeneratorTest, testRNGForking) {
  // See Note [Acquire lock when using random generators]
  if (!at::xpu::is_available()) return;
  auto default_gen = at::xpu::detail::getDefaultXPUGenerator();
  auto current_gen = at::xpu::detail::createXPUGenerator();
  {
    std::lock_guard<std::mutex> lock(default_gen.mutex());
    current_gen = default_gen.clone(); // capture the current state of default generator
  }
  auto target_value = at::randn({1000}, at::kXPU);
  // Dramatically alter the internal state of the main generator
  auto x = at::randn({100000}, at::kXPU);
  auto forked_value = at::randn({1000}, current_gen, at::kXPU);
  ASSERT_EQ(target_value.sum().item<double>(), forked_value.sum().item<double>());
}
