#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>

__device__ void random_float(float* x) {
  for(int i=0; i < 4; i++) {
    curandStatePhilox4_32_10_t state;
    curand_init(
            123,
            i,
            4,
            &state);
    auto ret = curand_uniform4(&state);
    x[i] = ret.x;
  }
}


__global__ void myKernel(float* x) {
  random_float(x);
}

TEST(DistributionsTest, TestPhiloxIncrement) {
  // Test Description:
  //   In Distributions.cu we mentioned that philox increment
  //   should be at least the number of curand() random numbers used in
  //   each thread. In this test, we make sure that uniform_ correctly
  //   increments philox and doesn't reuse randoms from previous calls.
  //    - We check that by first getting 4 randoms from uniform_.
  //      Once we get these 4 randoms, that would mean that philox counter for
  //      thread 0, 1, 2 and 3, was incremented by 4 (check calc_execution_policy
  //      function for details).
  //    - Now get 4 randoms with offset=4 for thread {0,1,2,3} from myKernel above.
  //    - Now get 4 more randoms from uniform_ (note thread {0,1,2,3} for this call would
  //      start from a philox_offset value of 1)
  //    - the 4 randoms from myKernel and the 4 randoms from the previous call
  //      of uniform_ should match, signifying that the philox offset was 
  //      incremented properly and no randoms are being reused from previous calls

  // if cuda not available, return
  if (!at::cuda::is_available()) return;

  // manual seed to 123
  at::manual_seed(123);

  // get 4 randoms from uniform_()
  auto self = at::empty({4}, at::TensorOptions(at::kCUDA));
  self.uniform_();

  // allocate 4 float on host memory
  float *x;
  cudaMallocManaged(&x, 4*sizeof(float));

  myKernel<<<1, 1>>>(x);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  
  // get 4 new float from uniform_()
  self.uniform_();
  
  // check randoms from myKernel are equal to the randoms from the second
  // call of uniform_()
  for (int i = 0; i < 4; i++) {
    ASSERT_EQ(self[i].item().to<float>(), x[i]);
  }

  // Free memory
  cudaFree(x);
}
