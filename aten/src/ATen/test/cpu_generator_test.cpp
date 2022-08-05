#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/Utils.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/core/PhiloxRNGEngine.h>
#include <c10/util/irange.h>
#include <thread>
#include <limits>
#include <random>

using namespace at;

TEST(CPUGeneratorImpl, TestGeneratorDynamicCast) {
  // Test Description: Check dynamic cast for CPU
  auto foo = at::detail::createCPUGenerator();
  auto result = check_generator<CPUGeneratorImpl>(foo);
  ASSERT_EQ(typeid(CPUGeneratorImpl*).hash_code(), typeid(result).hash_code());
}

TEST(CPUGeneratorImpl, TestDefaultGenerator) {
  // Test Description:
  // Check if default generator is created only once
  // address of generator should be same in all calls
  auto foo = at::detail::getDefaultCPUGenerator();
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  auto bar = at::detail::getDefaultCPUGenerator();
  ASSERT_EQ(foo, bar);
}

TEST(CPUGeneratorImpl, TestCloning) {
  // Test Description:
  // Check cloning of new generators.
  // Note that we don't allow cloning of other
  // generator states into default generators.
  auto gen1 = at::detail::createCPUGenerator();
  auto cpu_gen1 = check_generator<CPUGeneratorImpl>(gen1);
  // NOLINTNEXTLINE(clang-analyzer-security.insecureAPI.rand)
  cpu_gen1->random(); // advance gen1 state
  // NOLINTNEXTLINE(clang-analyzer-security.insecureAPI.rand)
  cpu_gen1->random();
  auto gen2 = at::detail::createCPUGenerator();
  gen2 = gen1.clone();
  auto cpu_gen2 = check_generator<CPUGeneratorImpl>(gen2);
  // NOLINTNEXTLINE(clang-analyzer-security.insecureAPI.rand)
  ASSERT_EQ(cpu_gen1->random(), cpu_gen2->random());
}

void thread_func_get_engine_op(CPUGeneratorImpl* generator) {
  std::lock_guard<std::mutex> lock(generator->mutex_);
  // NOLINTNEXTLINE(clang-analyzer-security.insecureAPI.rand)
  generator->random();
}

TEST(CPUGeneratorImpl, TestMultithreadingGetEngineOperator) {
  // Test Description:
  // Check CPUGeneratorImpl is reentrant and the engine state
  // is not corrupted when multiple threads request for
  // random samples.
  // See Note [Acquire lock when using random generators]
  auto gen1 = at::detail::createCPUGenerator();
  auto cpu_gen1 = check_generator<CPUGeneratorImpl>(gen1);
  auto gen2 = at::detail::createCPUGenerator();
  {
    std::lock_guard<std::mutex> lock(gen1.mutex());
    gen2 = gen1.clone(); // capture the current state of default generator
  }
  std::thread t0{thread_func_get_engine_op, cpu_gen1};
  std::thread t1{thread_func_get_engine_op, cpu_gen1};
  std::thread t2{thread_func_get_engine_op, cpu_gen1};
  t0.join();
  t1.join();
  t2.join();
  std::lock_guard<std::mutex> lock(gen2.mutex());
  auto cpu_gen2 = check_generator<CPUGeneratorImpl>(gen2);
  // NOLINTNEXTLINE(clang-analyzer-security.insecureAPI.rand)
  cpu_gen2->random();
  // NOLINTNEXTLINE(clang-analyzer-security.insecureAPI.rand)
  cpu_gen2->random();
  // NOLINTNEXTLINE(clang-analyzer-security.insecureAPI.rand)
  cpu_gen2->random();
  // NOLINTNEXTLINE(clang-analyzer-security.insecureAPI.rand)
  ASSERT_EQ(cpu_gen1->random(), cpu_gen2->random());
}

TEST(CPUGeneratorImpl, TestGetSetCurrentSeed) {
  // Test Description:
  // Test current seed getter and setter
  // See Note [Acquire lock when using random generators]
  auto foo = at::detail::getDefaultCPUGenerator();
  std::lock_guard<std::mutex> lock(foo.mutex());
  foo.set_current_seed(123);
  auto current_seed = foo.current_seed();
  ASSERT_EQ(current_seed, 123);
}

void thread_func_get_set_current_seed(Generator generator) {
  std::lock_guard<std::mutex> lock(generator.mutex());
  auto current_seed = generator.current_seed();
  current_seed++;
  generator.set_current_seed(current_seed);
}

TEST(CPUGeneratorImpl, TestMultithreadingGetSetCurrentSeed) {
  // Test Description:
  // Test current seed getter and setter are thread safe
  // See Note [Acquire lock when using random generators]
  auto gen1 = at::detail::getDefaultCPUGenerator();
  auto initial_seed = gen1.current_seed();
  std::thread t0{thread_func_get_set_current_seed, gen1};
  std::thread t1{thread_func_get_set_current_seed, gen1};
  std::thread t2{thread_func_get_set_current_seed, gen1};
  t0.join();
  t1.join();
  t2.join();
  ASSERT_EQ(gen1.current_seed(), initial_seed+3);
}

TEST(CPUGeneratorImpl, TestRNGForking) {
  // Test Description:
  // Test that state of a generator can be frozen and
  // restored
  // See Note [Acquire lock when using random generators]
  auto default_gen = at::detail::getDefaultCPUGenerator();
  auto current_gen = at::detail::createCPUGenerator();
  {
    std::lock_guard<std::mutex> lock(default_gen.mutex());
    current_gen = default_gen.clone(); // capture the current state of default generator
  }
  auto target_value = at::randn({1000});
  // Dramatically alter the internal state of the main generator
  auto x = at::randn({100000});
  auto forked_value = at::randn({1000}, current_gen);
  ASSERT_EQ(target_value.sum().item<double>(), forked_value.sum().item<double>());
}

/**
 * Philox CPU Engine Tests
 */

TEST(CPUGeneratorImpl, TestPhiloxEngineReproducibility) {
  // Test Description:
  //   Tests if same inputs give same results.
  //   launch on same thread index and create two engines.
  //   Given same seed, idx and offset, assert that the engines
  //   should be aligned and have the same sequence.
  at::Philox4_32 engine1(0, 0, 4);
  at::Philox4_32 engine2(0, 0, 4);
  ASSERT_EQ(engine1(), engine2());
}

TEST(CPUGeneratorImpl, TestPhiloxEngineOffset1) {
  // Test Description:
  //   Tests offsetting in same thread index.
  //   make one engine skip the first 8 values and
  //   make another engine increment to until the
  //   first 8 values. Assert that the first call
  //   of engine2 and the 9th call of engine1 are equal.
  at::Philox4_32 engine1(123, 1, 0);
  // Note: offset is a multiple of 4.
  // So if you want to skip 8 values, offset would
  // be 2, since 2*4=8.
  at::Philox4_32 engine2(123, 1, 2);
  for (const auto i : c10::irange(8)) {
    (void)i; // Suppress unused variable warning
    // Note: instead of using the engine() call 8 times
    // we could have achieved the same functionality by
    // calling the incr() function twice.
    engine1();
  }
  ASSERT_EQ(engine1(), engine2());
}

TEST(CPUGeneratorImpl, TestPhiloxEngineOffset2) {
  // Test Description:
  //   Tests edge case at the end of the 2^190th value of the generator.
  //   launch on same thread index and create two engines.
  //   make engine1 skip to the 2^64th 128 bit while being at thread 0
  //   make engine2 skip to the 2^64th 128 bit while being at 2^64th thread
  //   Assert that engine2 should be increment_val+1 steps behind engine1.
  unsigned long long increment_val = std::numeric_limits<uint64_t>::max();
  at::Philox4_32 engine1(123, 0, increment_val);
  at::Philox4_32 engine2(123, increment_val, increment_val);

  engine2.incr_n(increment_val);
  engine2.incr();
  ASSERT_EQ(engine1(), engine2());
}

TEST(CPUGeneratorImpl, TestPhiloxEngineOffset3) {
  // Test Description:
  //   Tests edge case in between thread indices.
  //   launch on same thread index and create two engines.
  //   make engine1 skip to the 2^64th 128 bit while being at thread 0
  //   start engine2 at thread 1, with offset 0
  //   Assert that engine1 is 1 step behind engine2.
  unsigned long long increment_val = std::numeric_limits<uint64_t>::max();
  at::Philox4_32 engine1(123, 0, increment_val);
  at::Philox4_32 engine2(123, 1, 0);
  engine1.incr();
  ASSERT_EQ(engine1(), engine2());
}

TEST(CPUGeneratorImpl, TestPhiloxEngineIndex) {
  // Test Description:
  //   Tests if thread indexing is working properly.
  //   create two engines with different thread index but same offset.
  //   Assert that the engines have different sequences.
  at::Philox4_32 engine1(123456, 0, 4);
  at::Philox4_32 engine2(123456, 1, 4);
  ASSERT_NE(engine1(), engine2());
}

/**
 * MT19937 CPU Engine Tests
 */

TEST(CPUGeneratorImpl, TestMT19937EngineReproducibility) {
  // Test Description:
  //   Tests if same inputs give same results when compared
  //   to std.

  // test with zero seed
  at::mt19937 engine1(0);
  std::mt19937 engine2(0);
  for (const auto i : c10::irange(10000)) {
    (void)i; // Suppress unused variable warning
    ASSERT_EQ(engine1(), engine2());
  }

  // test with large seed
  engine1 = at::mt19937(2147483647);
  engine2 = std::mt19937(2147483647);
  for (const auto i : c10::irange(10000)) {
    (void)i; // Suppress unused variable warning
    ASSERT_EQ(engine1(), engine2());
  }

  // test with random seed
  std::random_device rd;
  auto seed = rd();
  engine1 = at::mt19937(seed);
  engine2 = std::mt19937(seed);
  for (const auto i : c10::irange(10000)) {
    (void)i; // Suppress unused variable warning
    ASSERT_EQ(engine1(), engine2());
  }

}

TEST(CPUGeneratorImpl, TestPhiloxEngineReproducibilityRandN) {
  at::Philox4_32 engine1(0, 0, 4);
  at::Philox4_32 engine2(0, 0, 4);
  ASSERT_EQ(engine1.randn(1), engine2.randn(1));
}

TEST(CPUGeneratorImpl, TestPhiloxDeterministic) {
  at::Philox4_32 engine1(0, 0, 4);
  ASSERT_EQ(engine1(), 4013802324);  // Determinism!
  ASSERT_EQ(engine1(), 2979262830);  // Determinism!

  at::Philox4_32 engine2(10, 0, 1);
  ASSERT_EQ(engine2(), 2007330488);  // Determinism!
  ASSERT_EQ(engine2(), 2354548925);  // Determinism!
}
