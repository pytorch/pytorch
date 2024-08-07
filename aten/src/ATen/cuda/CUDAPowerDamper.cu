// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <iostream>

#include <ATen/cuda/CUDAPowerDamper.h>
#include <ATen/cuda/CUDAPowerDamperKernel.cuh>

using namespace facebook::gpu_power_damper;
using namespace std;

inline void checkCudaStatus(cudaError_t status) {
  if (status != cudaSuccess) {
    std::cout << "cuda API failed with status " << status << ":"
              << cudaGetErrorString(status) << std::endl;
    exit(-1);
  }
}

void cuda_power_damper::initialize_power_gen_params() {
  // Get the number of streaming multiprocessors on the device
  checkCudaStatus(cudaSetDevice(this->_gpu_idx));
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  this->_num_multiprocessors = deviceProp.multiProcessorCount;
  std::cout << "Number of multiprocessors on GPU" << this->_gpu_idx << " = "
            << this->_num_multiprocessors << std::endl;

  // Run the power virus on all available multiprocessors in the beginning
  uint32_t numBlocks = this->_num_multiprocessors * this->_tdp_pct_start / 100;

  // Use maximum thread count per block to generate maximum occupancy
  // per SM
  this->_block_size = 1024;

  // Start with 20 iterations first to measure kernel execution time
  uint32_t iters = 20;

  // We should tune this if we see large power fluctiations. This parameter
  // controls the kernel duration
  uint32_t kernelIters = 800;

  // Each warp computes a F_M x F_N matrix (fragmen sizes in tensor cores)
  uint32_t numWarps = this->_block_size / WARP_SIZE * numBlocks;
  this->_buffer_elements = numWarps * F_M * F_N;

  // Setting the target device
  checkCudaStatus(cudaSetDevice(this->_gpu_idx));

  // Create a low priority CUDA stream
  int leastPriority = 3;
  int greatestPriority = 0;
  checkCudaStatus(
      cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
  checkCudaStatus(cudaStreamCreateWithPriority(
      &this->_damper_cuda_stream, cudaStreamDefault, leastPriority));

  // Allocate memory on device
  checkCudaStatus(
      cudaMalloc((void**)&this->_dev_buffer, sizeof(float) * this->_buffer_elements));

  // Grid launch parameters
  dim3 dimBlock(this->_block_size);
  dim3 dimGridInit(numBlocks);

  // Initialize buffers on the device as it is faster
  set_array<<<dimGridInit, dimBlock>>>(this->_dev_buffer, .5f, this->_buffer_elements);

  // Measure kernel execution time in order to create the damping plan
  std::cout << "Measuring kernel execution time ..." << std::endl;

  cudaEvent_t ev_start, ev_stop;
  checkCudaStatus(cudaEventCreate(&ev_start));
  checkCudaStatus(cudaEventCreate(&ev_stop));
  checkCudaStatus(cudaEventRecord(ev_start, 0));
  for (uint32_t i = 0; i < iters; i++) {
    float val_init = (float)i / (float)iters;
    tensor_cores_rng<<<dimGridInit, dimBlock, 0, this->_damper_cuda_stream>>>(
        this->_dev_buffer, val_init, kernelIters);
  }
  checkCudaStatus(cudaEventRecord(ev_stop, 0));
  checkCudaStatus(cudaDeviceSynchronize());

  float t_elapsed_ms = 0.0f;
  checkCudaStatus(cudaEventElapsedTime(&t_elapsed_ms, ev_start, ev_stop));
  this->_power_gen_kernel_us = (uint32_t)(t_elapsed_ms / iters * 1000.0);
  std::cout << "Kernel execution time (us) on GPU " << this->_gpu_idx << " = "
            << this->_power_gen_kernel_us << std::endl;

  // Free memory on device
  // cudaFree(this->_dev_buffer);
}

// Host function that generates a GPU power usage close enouch to the TDP
// percentage given as argument and then slowly generates decreasing power
// so that after "drain_seconds" have passed, the GPU is in active idle and
// this function exits. The drainage is done in a number of steps given as
// input argument.

int cuda_power_damper::gen_and_drain_power() {
  // Run the power virus on all available multiprocessors in the beginning
  uint32_t numBlocksStart =
      this->_num_multiprocessors * this->_tdp_pct_start / 100;
  uint32_t numBlocksStop =
      this->_num_multiprocessors * this->_tdp_pct_stop / 100;

  // We should tune this if we see large power fluctiations. This parameter
  // controls the kernel duration
  uint32_t kernelIters = 800;

  // Setting the target device
  checkCudaStatus(cudaSetDevice(this->_gpu_idx));

  // Allocate memory on device
  // float* d_c;
  // checkCudaStatus(
  //    cudaMalloc((void**)&d_c, sizeof(float) * this->_buffer_elements));

  // Grid launch parameters
  dim3 dimBlock(this->_block_size);
  uint32_t numBlocks = numBlocksStart;
  dim3 dimGridInit(numBlocks);

  // Initialize buffers on the device as it is faster
  set_array<<<dimGridInit, dimBlock>>>(this->_dev_buffer, .5f, this->_buffer_elements);

  // Calculate plan
  uint32_t kernels_per_drain_time =
      this->_drain_milliseconds * 1000 / this->_power_gen_kernel_us;
  uint32_t kernels_per_step = kernels_per_drain_time / this->_steps;
  uint32_t blocksDecrement = (numBlocksStart - numBlocksStop) / this->_steps;

  // Need to remeasure kernel execution to detect if it's slower
  // That means the GPU is running stuff again
  cudaEvent_t ev_start, ev_stop;
  checkCudaStatus(cudaEventCreate(&ev_start));
  checkCudaStatus(cudaEventCreate(&ev_stop));
  uint32_t num_slowdowns = 0;

  // Run power damper on the stream that we just created in the init phase
  for (int i = 0; i < this->_steps; ++i) {
    dim3 dimGrid(numBlocks);

    checkCudaStatus(cudaEventRecord(ev_start, 0));
    for (uint32_t iter = 0; iter < kernels_per_step; ++iter) {
      float val_init = (float)iter / (float)kernels_per_step;
      tensor_cores_rng<<<dimGrid, dimBlock, 0, this->_damper_cuda_stream>>>(
          this->_dev_buffer, val_init, kernelIters);
    }
    checkCudaStatus(cudaEventRecord(ev_stop, 0));
    checkCudaStatus(cudaDeviceSynchronize());

    // We need to calculate kernel execution time to see if it's much slower
    // If it is, we stop the power drainage because it likely means that the
    // GPU has picked up work.

    float t_elapsed_ms = 0.0f;
    checkCudaStatus(cudaEventElapsedTime(&t_elapsed_ms, ev_start, ev_stop));
    float current_kernel_us = (float)(t_elapsed_ms / kernels_per_step * 1000.0);
    float slowdown = current_kernel_us / (float)this->_power_gen_kernel_us;

    // std::cout << "num_blocks = " << numBlocks << " slowdown = " << slowdown
    //          << std::endl;

    if (slowdown >= this->_tolerated_slowdown) {
      num_slowdowns++;

      if (num_slowdowns == this->_strike_slowdowns) {
        std::cout << "The power virus is running slower by " << slowdown
                  << "X. ";
        std::cout << "We are stopping the power drainage event." << std::endl;
        break;
      }
    }

    // Lower power usage a step
    numBlocks = max(1, numBlocks - blocksDecrement);
  }

  // Wait for everything to finish on the stream
  checkCudaStatus(cudaDeviceSynchronize());

  // Download the results on the host and compute a checksum. This is to prevent
  // nvcc from considering all the above as being dead code

  float checksum = 0.0;
  float* h_c = (float*)malloc(sizeof(float) * this->_buffer_elements);
  checkCudaStatus(cudaMemcpy(
      h_c,
      this->_dev_buffer,
      sizeof(float) * this->_buffer_elements,
      cudaMemcpyDeviceToHost));
  for (uint32_t i = 0; i < this->_buffer_elements; i += 1024) {
    checksum += h_c[i];
  }

  // Free memory on device
  // cudaFree(this->_dev_buffer);

  // Return something to avoid weird compiler optimizations
  return checksum;
}
