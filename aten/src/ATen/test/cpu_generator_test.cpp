#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/Utils.h>
#include <ATen/CPUGenerator.h>
#include <thread>

using namespace at;

TEST(CPUGenerator, TestGeneratorDynamicCast) {
  // Check dynamic cast for CPU
  std::unique_ptr<Generator> foo = at::detail::createCPUGenerator();
  auto result = dynamic_cast<CPUGenerator*>(foo.get());
  ASSERT_EQ(typeid(CPUGenerator*).hash_code(), typeid(result).hash_code());
}

TEST(CPUGenerator, TestDefaultGenerator) {
  // Check if default generator is created only once
  // address of generator should be same in all calls
  auto foo = &at::detail::getDefaultCPUGenerator();
  auto bar = &globalContext().getDefaultGenerator(kCPU);
  ASSERT_EQ(foo, bar);
}

TEST(CPUGenerator, TestCopyToDefaultGenerator) {
  auto new_gen = at::detail::createCPUGenerator();
  new_gen->random(); // advance new gen_state
  new_gen->random();
  auto default_gen = &at::detail::getDefaultCPUGenerator();
  *default_gen = *new_gen;
  ASSERT_EQ(new_gen->random(), default_gen->random());
}

void thread_func_get_engine_op(at::CPUGenerator& generator) {
  generator.random();
}

TEST(CPUGenerator, TestMultithreadingGetEngineOperator) {
  auto& gen1 = at::detail::getDefaultCPUGenerator();
  auto gen2 = at::detail::createCPUGenerator();
  *gen2 = gen1; // capture the current state of default generator
  std::thread t0{thread_func_get_engine_op, std::ref(gen1)};
  std::thread t1{thread_func_get_engine_op, std::ref(gen1)};
  std::thread t2{thread_func_get_engine_op, std::ref(gen1)};
  t0.join();
  t1.join();
  t2.join();
  gen2->random();
  gen2->random();
  gen2->random();
  ASSERT_EQ(gen1.random(), gen2->random());
}

TEST(CPUGenerator, TestGetSetCurrentSeed) {
  auto foo = &at::detail::getDefaultCPUGenerator();
  foo->setCurrentSeed(123);
  auto current_seed = foo->getCurrentSeed();
  ASSERT_EQ(current_seed, 123);
}

void thread_func_get_set_current_seed(at::CPUGenerator* generator) {
  auto current_seed = generator->getCurrentSeed();
  current_seed++;
  generator->setCurrentSeed(current_seed);
}

TEST(CPUGenerator, TestMultithreadingGetSetCurrentSeed) {
  auto gen1 = &at::detail::getDefaultCPUGenerator();
  auto initial_seed = gen1->getCurrentSeed();
  std::thread t0{thread_func_get_set_current_seed, gen1};
  std::thread t1{thread_func_get_set_current_seed, gen1};
  std::thread t2{thread_func_get_set_current_seed, gen1};
  t0.join();
  t1.join();
  t2.join();
  ASSERT_EQ(gen1->getCurrentSeed(), initial_seed+3);
}

TEST(CPUGenerator, TestRNGForking) {
  auto default_gen = &at::detail::getDefaultCPUGenerator();
  auto current_gen = at::detail::createCPUGenerator();
  *current_gen = *default_gen;

  auto target_value = at::randn({1000});
  // Dramatically alter the internal state of the main generator
  auto x = at::randn({100000});
  auto forked_value = at::randn({1000}, current_gen.get());
  ASSERT_EQ(target_value.sum().item<double>(), forked_value.sum().item<double>());
}
