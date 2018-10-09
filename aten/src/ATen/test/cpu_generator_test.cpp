#include "gtest/gtest.h"

#include "ATen/ATen.h"

TEST(CPUGenerator, TestGeneratorDynamicCast) {
  // Check dynamic cast for CPU
  auto foo = at::globalContext().createGenerator(at::kCPU);
  auto result = dynamic_cast<at::Generator*>(&foo);
  ASSERT_EQ(typeid(at::Generator*).hash_code(), typeid(result).hash_code());
}

TEST(CPUGenerator, TestDefaultGenerator) {
  // Check if default generator state is created only once
  // address of generator should be same in all calls
  auto foo = &at::globalContext().getDefaultGenerator(at::kCPU);
  auto bar = &at::globalContext().getDefaultGenerator(at::kCPU);
  ASSERT_EQ(foo, bar);

  // check setting of state for default generator
  auto new_gen = at::globalContext().createGenerator(at::kCPU);
  new_gen.random64(); // advance new gen_state
  new_gen.random64();
  auto& default_gen = at::globalContext().getDefaultGenerator(at::kCPU);
  default_gen.setState(new_gen.getState());
  ASSERT_EQ(new_gen.random64(), default_gen.random64());
  
}

TEST(CPUGenerator, TestCPUEngine) {

  auto new_gen = at::globalContext().createGenerator(at::kCPU);
  new_gen.random64(); // advance new gen_state
  new_gen.random64();
  auto& default_gen = at::globalContext().getDefaultGenerator(at::kCPU);
  ASSERT_NE(new_gen.random64(), default_gen.random64());
  default_gen.setCPUEngine(new_gen.getCPUEngine());
  ASSERT_EQ(new_gen.random64(), default_gen.random64());
  
}

TEST(CPUGenerator, TestSeeding) {
  auto& foo = at::globalContext().getDefaultGenerator(at::kCPU);
  foo.setCurrentSeed(123);
  auto current_seed = foo.getCurrentSeed();
  ASSERT_EQ(current_seed, 123);
}

TEST(CPUGenerator, TestRNGForking) {
  auto& default_gen = at::globalContext().getDefaultGenerator(at::kCPU);
  auto current_gen = at::globalContext().createGenerator(at::kCPU);
  default_gen.random64();
  auto current_default_state = default_gen.getState();
  current_gen.setState(current_default_state);
  ASSERT_NE(current_gen.getState(), current_default_state);
  
  auto target_value = at::randn({1000});
  // Dramatically alter the internal state of the main generator
  auto x = at::randn({100000});
  auto forked_value = at::randn({1000}, &current_gen);
  ASSERT_EQ(target_value.sum().item<double>(), forked_value.sum().item<double>());
}

TEST(CPUGenerator, TestCallingCUDAGeneratorMethod) {
  auto& default_gen = at::globalContext().getDefaultGenerator(at::kCPU);
  ASSERT_THROW(default_gen.incrementPhiloxOffset(1234, 123, 123, 4), c10::Error);
}