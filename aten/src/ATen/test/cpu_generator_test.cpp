#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/Utils.h>
#include <ATen/CPUGenerator.h>
#include <ATen/core/PhiloxRNGEngine.h>
#include <thread>
#include <limits>
#include <random>

using namespace at;

TEST(CPUGenerator, TestGeneratorDynamicCast) {
  // Test Description: Check dynamic cast for CPU
  std::shared_ptr<GeneratorImpl> foo = at::detail::createCPUGenerator();
  auto result = dynamic_cast<CPUGenerator*>(foo.get());
  ASSERT_EQ(typeid(CPUGenerator*).hash_code(), typeid(result).hash_code());
}

TEST(CPUGenerator, TestDefaultGenerator) {
  // Test Description: 
  // Check if default generator is created only once
  // address of generator should be same in all calls
  auto foo = at::detail::getDefaultCPUGenerator();
  auto bar = at::detail::getDefaultCPUGenerator();
  ASSERT_EQ(foo, bar);
}

TEST(CPUGenerator, TestCloning) {
  // Test Description: 
  // Check cloning of new generators.
  // Note that we don't allow cloning of other
  // generator states into default generators.
  auto gen1 = at::detail::createCPUGenerator();
  gen1->random(); // advance gen1 state
  gen1->random();
  auto gen2 = at::detail::createCPUGenerator();
  gen2 = gen1->clone();
  ASSERT_EQ(gen1->random(), gen2->random());
}

void thread_func_get_engine_op(CPUGenerator* generator) {
  std::lock_guard<std::mutex> lock(generator->mutex_);
  generator->random();
}

TEST(CPUGenerator, TestMultithreadingGetEngineOperator) {
  // Test Description: 
  // Check CPUGenerator is reentrant and the engine state
  // is not corrupted when multiple threads request for 
  // random samples.
  // See Note [Acquire lock when using random generators]
  auto gen1 = at::detail::createCPUGenerator();
  auto gen2 = at::detail::createCPUGenerator();
  {
    std::lock_guard<std::mutex> lock(gen1->mutex_);
    gen2 = gen1->clone(); // capture the current state of default generator
  }
  std::thread t0{thread_func_get_engine_op, gen1.get()};
  std::thread t1{thread_func_get_engine_op, gen1.get()};
  std::thread t2{thread_func_get_engine_op, gen1.get()};
  t0.join();
  t1.join();
  t2.join();
  std::lock_guard<std::mutex> lock(gen2->mutex_);
  gen2->random();
  gen2->random();
  gen2->random();
  ASSERT_EQ(gen1->random(), gen2->random());
}

TEST(CPUGenerator, TestGetSetCurrentSeed) {
  // Test Description: 
  // Test current seed getter and setter
  // See Note [Acquire lock when using random generators]
  auto foo = at::detail::getDefaultCPUGenerator();
  std::lock_guard<std::mutex> lock(foo->mutex_);
  foo->set_current_seed(123);
  auto current_seed = foo->current_seed();
  ASSERT_EQ(current_seed, 123);
}

void thread_func_get_set_current_seed(CPUGenerator* generator) {
  std::lock_guard<std::mutex> lock(generator->mutex_);
  auto current_seed = generator->current_seed();
  current_seed++;
  generator->set_current_seed(current_seed);
}

TEST(CPUGenerator, TestMultithreadingGetSetCurrentSeed) {
  // Test Description: 
  // Test current seed getter and setter are thread safe
  // See Note [Acquire lock when using random generators]
  auto gen1 = at::detail::getDefaultCPUGenerator();
  auto initial_seed = gen1->current_seed();
  std::thread t0{thread_func_get_set_current_seed, gen1};
  std::thread t1{thread_func_get_set_current_seed, gen1};
  std::thread t2{thread_func_get_set_current_seed, gen1};
  t0.join();
  t1.join();
  t2.join();
  ASSERT_EQ(gen1->current_seed(), initial_seed+3);
}

TEST(CPUGenerator, TestRNGForking) {
  // Test Description: 
  // Test that state of a generator can be frozen and
  // restored
  // See Note [Acquire lock when using random generators]
  auto default_gen = at::detail::getDefaultCPUGenerator();
  auto current_gen = at::detail::createCPUGenerator();
  {
    std::lock_guard<std::mutex> lock(default_gen->mutex_);
    current_gen = default_gen->clone(); // capture the current state of default generator
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

TEST(CPUGenerator, TestPhiloxEngineReproducibility) {
  // Test Description:
  //   Tests if same inputs give same results.
  //   launch on same thread index and create two engines.
  //   Given same seed, idx and offset, assert that the engines
  //   should be aligned and have the same sequence.
  at::Philox4_32_10 engine1(0, 0, 4);
  at::Philox4_32_10 engine2(0, 0, 4);
  ASSERT_EQ(engine1(), engine2());
}

TEST(CPUGenerator, TestPhiloxEngineOffset1) {
  // Test Description:
  //   Tests offsetting in same thread index.
  //   make one engine skip the first 8 values and
  //   make another engine increment to until the
  //   first 8 values. Assert that the first call
  //   of engine2 and the 9th call of engine1 are equal.
  at::Philox4_32_10 engine1(123, 1, 0);
  // Note: offset is a multiple of 4.
  // So if you want to skip 8 values, offset would
  // be 2, since 2*4=8.
  at::Philox4_32_10 engine2(123, 1, 2);
  for(int i = 0; i < 8; i++){
    // Note: instead of using the engine() call 8 times
    // we could have achieved the same functionality by
    // calling the incr() function twice.
    engine1();
  }
  ASSERT_EQ(engine1(), engine2());
}

TEST(CPUGenerator, TestPhiloxEngineOffset2) {
  // Test Description:
  //   Tests edge case at the end of the 2^190th value of the generator.
  //   launch on same thread index and create two engines.
  //   make engine1 skip to the 2^64th 128 bit while being at thread 0
  //   make engine2 skip to the 2^64th 128 bit while being at 2^64th thread
  //   Assert that engine2 should be increment_val+1 steps behind engine1.
  unsigned long long increment_val = std::numeric_limits<uint64_t>::max();
  at::Philox4_32_10 engine1(123, 0, increment_val);
  at::Philox4_32_10 engine2(123, increment_val, increment_val);

  engine2.incr_n(increment_val);
  engine2.incr();
  ASSERT_EQ(engine1(), engine2());
}

TEST(CPUGenerator, TestPhiloxEngineOffset3) {
  // Test Description:
  //   Tests edge case in between thread indices.
  //   launch on same thread index and create two engines.
  //   make engine1 skip to the 2^64th 128 bit while being at thread 0
  //   start engine2 at thread 1, with offset 0
  //   Assert that engine1 is 1 step behind engine2.
  unsigned long long increment_val = std::numeric_limits<uint64_t>::max();
  at::Philox4_32_10 engine1(123, 0, increment_val);
  at::Philox4_32_10 engine2(123, 1, 0);
  engine1.incr();
  ASSERT_EQ(engine1(), engine2());
}

TEST(CPUGenerator, TestPhiloxEngineIndex) {
  // Test Description:
  //   Tests if thread indexing is working properly.
  //   create two engines with different thread index but same offset.
  //   Assert that the engines have different sequences.
  at::Philox4_32_10 engine1(123456, 0, 4);
  at::Philox4_32_10 engine2(123456, 1, 4);
  ASSERT_NE(engine1(), engine2());
}

/**
 * MT19937 CPU Engine Tests
 */

TEST(CPUGenerator, TestMT19937EngineReproducibility) {
  // Test Description:
  //   Tests if same inputs give same results when compared
  //   to std.
  
  // test with zero seed
  at::mt19937 engine1(0);
  std::mt19937 engine2(0);
  for(int i = 0; i < 10000; i++) {
    ASSERT_EQ(engine1(), engine2());
  }

  // test with large seed
  engine1 = at::mt19937(2147483647);
  engine2 = std::mt19937(2147483647);
  for(int i = 0; i < 10000; i++) {
    ASSERT_EQ(engine1(), engine2());
  }

  // test with random seed
  std::random_device rd;
  auto seed = rd();
  engine1 = at::mt19937(seed);
  engine2 = std::mt19937(seed);
  for(int i = 0; i < 10000; i++) {
    ASSERT_EQ(engine1(), engine2());
  }
  
}
