#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAFunctions.h>
#include <ATen/core/PhiloxRNGEngine.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <assert.h>
#include <thread>

using namespace at;

/*
* Philox Engine Tests
*/

__global__ void testEngineReproducibility(){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  at::Philox4_32_10 engine1(0, idx, 4);
  at::Philox4_32_10 engine2(0, idx, 4);
  assert(engine1() == engine2());
}

void test_engine_reproducibility(){
  testEngineReproducibility<<<1, 1>>>();
}

TEST(CUDAGeneratorImpl, TestPhiloxEngineReproducibility) {
  // Test Description:
  //   Tests if same inputs give same results.
  //   launch one thread and create two engines.
  //   Given same seed, idx and offset, assert that the engines
  //   should be aligned and have the same sequence.
  if (!at::cuda::is_available()) return;
  test_engine_reproducibility();
  cudaError_t err = cudaDeviceSynchronize();
  bool isEQ = err == cudaSuccess;
  ASSERT_TRUE(isEQ);
}

__global__ void testEngineOffset1(){
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
  assert(engine1() == engine2());
}

void test_engine_offset1(){
  testEngineOffset1<<<1, 1>>>();
}

TEST(CUDAGeneratorImpl, TestPhiloxEngineOffset1) {
  // Test Description:
  //   Tests offsetting in same thread.
  //   launch one thread and create two engines.
  //   make one engine skip the first 8 values and
  //   make another engine increment to until the
  //   first 8 values. Assert that the first call
  //   of engine2 and the 9th call of engine1 are equal.
  if (!at::cuda::is_available()) return;
  test_engine_offset1();
  cudaError_t err = cudaDeviceSynchronize();
  bool isEQ = err == cudaSuccess;
  ASSERT_TRUE(isEQ);
}

__global__ void testEngineOffset2(){
  unsigned long long increment_val = ::ldexp(1.0, 64);
  at::Philox4_32_10 engine1(123, 0, increment_val);
  at::Philox4_32_10 engine2(123, increment_val, increment_val);

  engine2.incr_n(increment_val);
  engine2.incr();
  assert(engine1() == engine2());
}

void test_engine_offset2(){
  testEngineOffset2<<<1, 1>>>();
}

TEST(CUDAGeneratorImpl, TestPhiloxEngineOffset2) {
  // Test Description:
  //   Tests edge case at the end of the 2^190th value of the generator.
  //   launch one thread and create two engines
  //   make engine1 skip to the 2^64th 128 bit while being at thread 0
  //   make engine2 skip to the 2^64th 128 bit while being at 2^64th thread
  //   Assert that engine2 should be increment_val+1 steps behind engine1.
  if (!at::cuda::is_available()) return;
  test_engine_offset2();
  cudaDeviceSynchronize();
  bool isEQ = cudaGetLastError() == cudaSuccess;
  ASSERT_TRUE(isEQ);
}

__global__ void testEngineOffset3(){
  unsigned long long increment_val = ::ldexp(1.0, 64);
  at::Philox4_32_10 engine1(123, 0, increment_val);
  at::Philox4_32_10 engine2(123, 1, 0);
  engine1.incr();
  assert(engine1() == engine2());
}

void test_engine_offset3(){
  testEngineOffset2<<<1, 1>>>();
}

TEST(CUDAGeneratorImpl, TestPhiloxEngineOffset3) {
  // Test Description:
  //   Tests edge case in between threads.
  //   launch one thread and create two engines
  //   make engine1 skip to the 2^64th 128 bit while being at thread 0
  //   start engine2 at thread 1, with offset 0
  //   Assert that engine1 is 1 step behind engine2.
  if (!at::cuda::is_available()) return;
  test_engine_offset3();
  cudaDeviceSynchronize();
  bool isEQ = cudaGetLastError() == cudaSuccess;
  ASSERT_TRUE(isEQ);
}

__global__ void testEngineThreadIndex(){
  at::Philox4_32_10 engine1(123456, 0, 4);
  at::Philox4_32_10 engine2(123456, 1, 4);
  assert(engine1() != engine2());
}

void test_engine_thread_index(){
  testEngineThreadIndex<<<1, 1>>>();
}

TEST(CUDAGeneratorImpl, TestPhiloxEngineIndex) {
  // Test Description:
  //   Tests if thread indexing is working properly.
  //   launch one thread and create two engines
  //   with different thread index but same offset.
  //   Assert that the engines have different sequences.
  if (!at::cuda::is_available()) return;
  test_engine_thread_index();
  cudaDeviceSynchronize();
  bool isEQ = cudaGetLastError() == cudaSuccess;
  ASSERT_TRUE(isEQ);
}

/*
* CUDA Generator Tests
*/

TEST(CUDAGeneratorImpl, TestGeneratorDynamicCast) {
  //  Test Description: Check dynamic cast for CUDA
  if (!at::cuda::is_available()) return;
  auto foo = at::cuda::detail::createCUDAGenerator();
  auto result = foo.get<CUDAGeneratorImpl>();
  ASSERT_EQ(typeid(at::CUDAGeneratorImpl*).hash_code(), typeid(result).hash_code());
}

TEST(CUDAGeneratorImpl, TestDefaultGenerator) {
  // Test Description:
  // Check if default generator state is created only once
  // address of generator should be same in all calls
  if (!at::cuda::is_available()) return;
  auto foo = at::cuda::detail::getDefaultCUDAGenerator();
  auto bar = at::cuda::detail::getDefaultCUDAGenerator();
  ASSERT_EQ(foo, bar);

  if (c10::cuda::device_count() >= 2) {
    foo = at::cuda::detail::getDefaultCUDAGenerator(1);
    bar = at::cuda::detail::getDefaultCUDAGenerator(1);
    ASSERT_EQ(foo, bar);

    foo = at::cuda::detail::getDefaultCUDAGenerator(0);
    bar = at::cuda::detail::getDefaultCUDAGenerator(1);
    ASSERT_NE(foo, bar);
  }
}

TEST(CUDAGeneratorImpl, TestCloning) {
  // Test Description:
  // Check cloning of new generators.
  // Note that we don't allow cloning of other
  // generator states into default generators.
  if (!at::cuda::is_available()) return;
  auto gen1 = at::cuda::detail::createCUDAGenerator();
  gen1.set_current_seed(123); // modify gen1 state
  auto cuda_gen1 = check_generator<CUDAGeneratorImpl>(gen1);
  cuda_gen1->set_philox_offset_per_thread(4);
  auto gen2 = at::cuda::detail::createCUDAGenerator();
  gen2 = gen1.clone();
  auto cuda_gen2 = check_generator<CUDAGeneratorImpl>(gen2);
  ASSERT_EQ(gen1.current_seed(), gen2.current_seed());
  ASSERT_EQ(
    cuda_gen1->philox_offset_per_thread(),
    cuda_gen2->philox_offset_per_thread()
  );
}

void thread_func_get_set_current_seed(Generator generator) {
  std::lock_guard<std::mutex> lock(generator.mutex());
  auto current_seed = generator.current_seed();
  current_seed++;
  generator.set_current_seed(current_seed);
}

TEST(CUDAGeneratorImpl, TestMultithreadingGetSetCurrentSeed) {
  // Test Description:
  // Test current seed getter and setter are thread safe
  // See Note [Acquire lock when using random generators]
  if (!at::cuda::is_available()) return;
  auto gen1 = at::cuda::detail::getDefaultCUDAGenerator();
  auto initial_seed = gen1.current_seed();
  std::thread t0{thread_func_get_set_current_seed, gen1};
  std::thread t1{thread_func_get_set_current_seed, gen1};
  std::thread t2{thread_func_get_set_current_seed, gen1};
  t0.join();
  t1.join();
  t2.join();
  ASSERT_EQ(gen1.current_seed(), initial_seed+3);
}

TEST(CUDAGeneratorImpl, TestRNGForking) {
  // Test Description:
  // Test that state of a generator can be frozen and
  // restored
  // See Note [Acquire lock when using random generators]
  if (!at::cuda::is_available()) return;
  auto default_gen = at::cuda::detail::getDefaultCUDAGenerator();
  auto current_gen = at::cuda::detail::createCUDAGenerator();
  {
    std::lock_guard<std::mutex> lock(default_gen.mutex());
    current_gen = default_gen.clone(); // capture the current state of default generator
  }
  auto target_value = at::randn({1000}, at::kCUDA);
  // Dramatically alter the internal state of the main generator
  auto x = at::randn({100000}, at::kCUDA);
  auto forked_value = at::randn({1000}, current_gen, at::kCUDA);
  ASSERT_EQ(target_value.sum().item<double>(), forked_value.sum().item<double>());
}

void makeRandomNumber() {
  cudaSetDevice(std::rand() % 2);
  auto x = at::randn({1000});
}

void testCudaRNGMultithread() {
  auto threads = std::vector<std::thread>();
  for (auto i = 0; i < 1000; i++) {
    threads.emplace_back(makeRandomNumber);
  }
  for (auto& t : threads) {
    t.join();
  }
};

TEST(CUDAGeneratorImpl, TestMultithreadRNG) {
  if (!at::cuda::is_available()) return;
  testCudaRNGMultithread();
}
