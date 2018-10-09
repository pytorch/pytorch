#include "gtest/gtest.h"

#include "ATen/ATen.h"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/cuda/PhiloxRNGEngine.h"
#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"

#include <assert.h>
#include <thread>

using namespace at;

/*
* Philox Engine Tests
*/

__global__ void testEngineReproducibility(){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  at::cuda::Philox4_32_10 engine1(0, idx, 4);
  at::cuda::Philox4_32_10 engine2(0, idx, 4);
  assert(engine1() == engine2());
}

void test_engine_reproducibility(){
  testEngineReproducibility<<<1, 1>>>();
}

TEST(CUDAGenerator, TestPhiloxEngineReproducibility) {
  // Test Description:
  //   Tests if same inputs give same results.
  //   launch one thread and create two engines.
  //   Given same seed, idx and offset, assert that the engines
  //   should be aligned and have the same sequence.
  test_engine_reproducibility();
  cudaError_t err = cudaDeviceSynchronize();
  bool isEQ = err == cudaSuccess;
  ASSERT_TRUE(isEQ);
}

__global__ void testEngineOffset1(){
  at::cuda::Philox4_32_10 engine1(123, 1, 0);
  // Note: offset is a multiple of 4.
  // So if you want to skip 8 values, offset would
  // be 2, since 2*4=8.
  at::cuda::Philox4_32_10 engine2(123, 1, 2);
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

TEST(CUDAGenerator, TestPhiloxEngineOffset1) {
  // Test Description:
  //   Tests offsetting in same thread.
  //   launch one thread and create two engines.
  //   make one engine skip the first 8 values and
  //   make another engine increment to until the
  //   first 8 values. Assert that the first call
  //   of engine2 and the 9th call of engine1 are equal.
  test_engine_offset1();
  cudaError_t err = cudaDeviceSynchronize();
  bool isEQ = err == cudaSuccess;
  ASSERT_TRUE(isEQ);
}

__global__ void testEngineOffset2(){
  unsigned long long increment_val = ::ldexp(1.0, 64);
  at::cuda::Philox4_32_10 engine1(123, 0, increment_val);
  at::cuda::Philox4_32_10 engine2(123, increment_val, increment_val);
  
  engine2.incr_n(increment_val);
  engine2.incr();
  assert(engine1() == engine2());
}

void test_engine_offset2(){
  testEngineOffset2<<<1, 1>>>();
}

TEST(CUDAGenerator, TestPhiloxEngineOffset2) {
  // Test Description:
  //   Tests edge case at the end of the 2^190th value of the generator.
  //   launch one thread and create two engines
  //   make engine1 skip to the 2^64th 128 bit while being at thread 0
  //   make engine2 skip to the 2^64th 128 bit while being at 2^64th thread
  //   Assert that engine2 should be increment_val+1 steps behind engine1.
  test_engine_offset2();
  cudaDeviceSynchronize();
  bool isEQ = cudaGetLastError() == cudaSuccess;
  ASSERT_TRUE(isEQ);
}

__global__ void testEngineOffset3(){
  unsigned long long increment_val = ::ldexp(1.0, 64);
  at::cuda::Philox4_32_10 engine1(123, 0, increment_val);
  at::cuda::Philox4_32_10 engine2(123, 1, 0);
  engine1.incr();
  assert(engine1() == engine2());
}

void test_engine_offset3(){
  testEngineOffset2<<<1, 1>>>();
}

TEST(CUDAGenerator, TestPhiloxEngineOffset3) {
  // Test Description:
  //   Tests edge case in between threads.
  //   launch one thread and create two engines
  //   make engine1 skip to the 2^64th 128 bit while being at thread 0
  //   start engine2 at thread 1, with offset 0
  //   Assert that engine1 is 1 step behind engine2.
  test_engine_offset3();
  cudaDeviceSynchronize();
  bool isEQ = cudaGetLastError() == cudaSuccess;
  ASSERT_TRUE(isEQ);
}

__global__ void testEngineThreadIndex(){
  at::cuda::Philox4_32_10 engine1(123456, 0, 4);
  at::cuda::Philox4_32_10 engine2(123456, 1, 4);
  assert(engine1() != engine2());
}

void test_engine_thread_index(){
  testEngineThreadIndex<<<1, 1>>>();
}

TEST(CUDAGenerator, TestPhiloxEngineIndex) {
  // Test Description:
  //   Tests if thread indexing is working properly.
  //   launch one thread and create two engines
  //   with different thread index but same offset.
  //   Assert that the engines have different sequences.
  test_engine_thread_index();
  cudaDeviceSynchronize();
  bool isEQ = cudaGetLastError() == cudaSuccess;
  ASSERT_TRUE(isEQ);
}

/*
* CUDA Generator Tests
*/

TEST(CUDAGenerator, TestGeneratorDynamicCast) {
  // Check dynamic cast for CUDA
  auto foo = at::globalContext().createGenerator(at::kCUDA);
  auto result = dynamic_cast<at::Generator*>(&foo);
  ASSERT_EQ(typeid(at::Generator*).hash_code(), typeid(result).hash_code());
}

TEST(CUDAGenerator, TestDefaultGenerator) {
  // Check if default generator state is created only once
  // address of generator should be same in all calls
  auto foo = &at::globalContext().getDefaultGenerator(at::kCUDA);
  auto bar = &at::globalContext().getDefaultGenerator(at::kCUDA);
  ASSERT_EQ(foo, bar);

  if (at::cuda::getNumGPUs() >= 2) {
    foo = &at::globalContext().getDefaultGenerator(at::kCUDA, 1);
    bar = &at::globalContext().getDefaultGenerator(at::kCUDA, 1);
    ASSERT_EQ(foo, bar);

    foo = &at::globalContext().getDefaultGenerator(at::kCUDA, 0);
    bar = &at::globalContext().getDefaultGenerator(at::kCUDA, 1);
    ASSERT_NE(foo, bar);
  }
}

TEST(CUDAGenerator, TestGetSetDefaultGenerator) {
  // check setting of state for default generator
  auto new_gen = at::globalContext().createGenerator(at::kCUDA);
  new_gen.setCurrentSeed(123);
  auto& default_gen = at::globalContext().getDefaultGenerator(at::kCUDA);
  default_gen.setState(new_gen.getState());
  ASSERT_EQ(new_gen.getCurrentSeed(), default_gen.getCurrentSeed());
}

TEST(CUDAGenerator, TestSeeding) {
  auto& foo = at::globalContext().getDefaultGenerator(at::kCUDA);
  foo.setCurrentSeed(123);
  auto current_seed = foo.getCurrentSeed();
  ASSERT_EQ(current_seed, 123);
}

TEST(CUDAGenerator, TestCallingCPUGeneratorMethod) {
  auto& default_gen = at::globalContext().getDefaultGenerator(at::kCUDA);
  ASSERT_THROW(default_gen.getCPUEngine(), c10::Error);
  std::mt19937_64 engine;
  ASSERT_THROW(default_gen.setCPUEngine(engine), c10::Error);
  ASSERT_THROW(default_gen.random64(), c10::Error);
}

template <typename scalar_t>
__global__ void testIncrementPhiloxOffset1(scalar_t* ret, std::pair<uint64_t, uint64_t> seeds){
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  at::cuda::Philox4_32_10 engine(seeds.first, idx, seeds.second);
  ret[idx] = at::cuda::standard_uniform_distribution(engine);
}

template <typename scalar_t>
__global__ void testIncrementPhiloxOffset1Assert(scalar_t* ret, uint64_t offset){
  for(int i = 0; i < 512; i++){
    at::cuda::Philox4_32_10 engine(123, i, offset);
    assert(ret[i] == at::cuda::standard_uniform_distribution(engine));
  }
}

TEST(CUDAGenerator, TestIncrementPhiloxOffset1) {
  // Test Description:
  //   Tests that when yielding one element per thread and one engine call, 
  //   philox_offset_per_thread increments by 1. Also demonstrates the running state of
  //   a generator.
  //   Launch 32 blocks of 16 threads processing 512 elements.
  //   Then manually produce those elements in a single thread and assert
  //   for equality in the accumulated tensor.
  at::Tensor ret = at::empty({512}, at::device(at::kCUDA).dtype(at::kFloat));
  
  // demonstrating step 1 of philox offset calculation
  uint64_t numel = ret.numel();
  uint64_t block_size = 16;
  unsigned int blocks_per_sm = at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor/block_size;
  dim3 grid((numel + block_size -1)/block_size);
  grid.x = std::min((unsigned int)at::cuda::getCurrentDeviceProperties()->multiProcessorCount * blocks_per_sm, grid.x);
  // note that the default generator keeps a running sum of offset increments
  // hence, you need to re-seed an engine, if you want reproducibility.
  auto& gen = at::globalContext().getDefaultGenerator(at::kCUDA);
  gen.setCurrentSeed(123);
  // doing no loop unrolling and calling only uniform function
  auto seeds = gen.incrementPhiloxOffset(numel, grid.x, block_size, 1);
  // assert starting offset is 0 and current offset is 1 (1 element per thread and 1 engine call)
  ASSERT_EQ(seeds.second, 0);
  ASSERT_EQ(gen.getState()->philox_offset_per_thread, 1);

  // get a tensor filled with uniformly distributed samples
  AT_DISPATCH_ALL_TYPES_AND_HALF(ret.type(), "TestIncrementPhiloxOffset1", [&] {
    testIncrementPhiloxOffset1<<<grid, block_size>>>(ret.data<scalar_t>(), seeds);
  });

  // check if the samples are correctly produced
  AT_DISPATCH_ALL_TYPES_AND_HALF(ret.type(), "TestIncrementPhiloxOffset1", [&] {
    testIncrementPhiloxOffset1Assert<<<1, 1>>>(ret.data<scalar_t>(), 0);
  });

  // now repeat the same thing and notice the nature of philox_offset_per_thread
  seeds = gen.incrementPhiloxOffset(numel, grid.x, block_size, 1);
  ASSERT_EQ(seeds.second, 1); // see how we are using 1 increment by the previous kernel launch
  ASSERT_EQ(gen.getState()->philox_offset_per_thread, 2); // 2 is for the next kernel launch

  // get a tensor filled with uniformly distributed samples
  AT_DISPATCH_ALL_TYPES_AND_HALF(ret.type(), "TestIncrementPhiloxOffset1", [&] {
    testIncrementPhiloxOffset1<<<grid, block_size>>>(ret.data<scalar_t>(), seeds);
  });

  // check if the samples are correctly produced
  AT_DISPATCH_ALL_TYPES_AND_HALF(ret.type(), "TestIncrementPhiloxOffset1", [&] {
    testIncrementPhiloxOffset1Assert<<<1, 1>>>(ret.data<scalar_t>(), 1);
  });
  

  cudaDeviceSynchronize();
  bool isEQ = cudaGetLastError() == cudaSuccess;
  ASSERT_TRUE(isEQ);
}

template <typename scalar_t>
__global__ void testIncrementPhiloxOffset2(scalar_t* ret, std::pair<uint64_t, uint64_t> seeds){
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  at::cuda::Philox4_32_10 engine(seeds.first, idx, seeds.second);
  for(int ii = 0; ii < 4; ii+=2){
    // demonstrating normal returns two values at once
    float2 result = at::cuda::normal_distribution(engine);
    int li = idx + blockDim.x * gridDim.x * ii;
    ret[li] = result.x;
    li = idx + blockDim.x * gridDim.x * (ii+1);
    ret[li] = result.y;
  }
}

template <typename scalar_t>
__global__ void testIncrementPhiloxOffset2Assert(scalar_t* ret, uint64_t offset){
  for(int i = 0; i < 512; i++){
    at::cuda::Philox4_32_10 engine(123, i, offset);
    for(int ii = 0; ii < 4; ii+=2){
      // demonstrating normal returns two values at once
      float2 result = at::cuda::normal_distribution(engine);
      int li = i + 16 * 32 * ii;
      assert(ret[li] == result.x);
      li = i + 16 * 32 * (ii+1);
      assert(ret[li] == result.y);
    }
  }
}

TEST(CUDAGenerator, TestIncrementPhiloxOffset2) {
  // Test Description:
  //   Tests that when yielding 4 element per thread with 2 engine calls,
  //   number of randoms needed is 4*2 = 8 and hence, philox_offset_per_thread increments by 2. 
  //   Also demonstrates the running state of a generator.
  //   Launch 32 blocks of 16 threads processing 2048 elements.
  //   Then manually produce those elements in a single thread and assert
  //   for equality in the accumulated tensor.
  at::Tensor ret = at::empty({2048}, at::device(at::kCUDA).dtype(at::kFloat));
  uint64_t numel = ret.numel();
  uint64_t block_size = 16;
  dim3 grid(32);
  auto& gen = at::globalContext().getDefaultGenerator(at::kCUDA);
  gen.setCurrentSeed(123);
  // doing no loop unrolling and calling only normal function which has 2
  // engine() calls
  auto seeds = gen.incrementPhiloxOffset(numel, grid.x, block_size, 2);
  // assert starting offset is 0 and current offset is 2 (4 element per thread and 2 engine calls)
  ASSERT_EQ(seeds.second, 0);
  ASSERT_EQ(gen.getState()->philox_offset_per_thread, 2);

  // get a tensor filled with normally distributed samples
  AT_DISPATCH_ALL_TYPES_AND_HALF(ret.type(), "TestIncrementPhiloxOffset2", [&] {
    testIncrementPhiloxOffset2<<<grid, block_size>>>(ret.data<scalar_t>(), seeds);
  });

  // check if the samples are correctly produced
  AT_DISPATCH_ALL_TYPES_AND_HALF(ret.type(), "TestIncrementPhiloxOffset2", [&] {
    testIncrementPhiloxOffset2Assert<<<1, 1>>>(ret.data<scalar_t>(), 0);
  });

  // now repeat the same thing and notice the nature of philox_offset_per_thread
  seeds = gen.incrementPhiloxOffset(numel, grid.x, block_size, 2);
  ASSERT_EQ(seeds.second, 2); // see how we are using 2 increment by the previous kernel launch
  ASSERT_EQ(gen.getState()->philox_offset_per_thread, 4); // 4 is for the next kernel launch

  // get a tensor filled with normally distributed samples
  AT_DISPATCH_ALL_TYPES_AND_HALF(ret.type(), "TestIncrementPhiloxOffset2", [&] {
    testIncrementPhiloxOffset2<<<grid, block_size>>>(ret.data<scalar_t>(), seeds);
  });

  // check if the samples are correctly produced
  AT_DISPATCH_ALL_TYPES_AND_HALF(ret.type(), "TestIncrementPhiloxOffset2", [&] {
    testIncrementPhiloxOffset2Assert<<<1, 1>>>(ret.data<scalar_t>(), 2);
  });

  cudaDeviceSynchronize();
  bool isEQ = cudaGetLastError() == cudaSuccess;
  ASSERT_TRUE(isEQ);
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

TEST(CUDAGenerator, TestMultithreadRNG) {
  testCudaRNGMultithread();
}