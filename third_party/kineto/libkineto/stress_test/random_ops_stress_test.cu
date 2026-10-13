/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdio.h>
#include <unistd.h>

#include "random_ops_stress_test.cuh"
#include "tensor_cache.cuh"

namespace kineto_stress_test {

#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#define RNG_SEED 2049

// NCCL variables buffers
ncclUniqueId nccl_id;
ncclComm_t nccl_communicator;
float* pBuffNCCLSend;
float* pBuffNCCLRecv;

// C = A + B kernel where A and B are generated using a linear
// congruential generator. If the number of iterations is small
// the kernel is memory bandwidth bound. If iterations count is
// high, the kernel is compute bound.

// We use the template call to be able to change the kernel name with
// a simple hardcoded constant number

template <uint32_t offset_seed_a, uint32_t offset_seed_b>
__global__ void iterative_lcg_3_buffers(lcg_kernel_input input) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;
  while (idx < input.len) {
    uint32_t seed_a = (uint32_t)input.d_a[idx] + offset_seed_a;
    uint32_t seed_b = (uint32_t)input.d_b[idx] + offset_seed_b;
    uint32_t xna = 0;
    uint32_t xnb = 0;

    for (int i = 0; i < input.iters; ++i) {
      xna = (LCG_A * seed_a + LCG_C) % LCG_M;
      xnb = (LCG_A * seed_b + LCG_C) % LCG_M;
      seed_a = xna;
      seed_b = xnb;
    }

    input.d_c[idx] = 0.25 + (float)((xna + xnb) % 1000) / 1000.0;
    idx += stride;
  }
}

template <uint32_t offset_seed_a, uint32_t offset_seed_b>
__global__ void iterative_lcg_3_with_uvm(lcg_kernel_input input) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;
  uint64_t uvm_idx = idx;

  float val_uvm_a = 0.0f;
  float val_uvm_b = 0.0f;

  // Fetch data from UVM
  for (int i = 0; i < 5; ++i) {
    val_uvm_a += input.uvm_a[uvm_idx % input.uvm_len];
    val_uvm_b += input.uvm_b[uvm_idx % input.uvm_len];
    uvm_idx += 4096;
  }

  // Load device buffers, do some compute, save to device buffer
  while (idx < input.len) {
    uint32_t seed_a = (uint32_t)(input.d_a[idx] + val_uvm_a) + offset_seed_a;
    uint32_t seed_b = (uint32_t)(input.d_b[idx] + val_uvm_b) + offset_seed_b;
    uint32_t xna = 0;
    uint32_t xnb = 0;

    for (int i = 0; i < input.iters; ++i) {
      xna = (LCG_A * seed_a + LCG_C) % LCG_M;
      xnb = (LCG_A * seed_b + LCG_C) % LCG_M;
      seed_a = xna;
      seed_b = xnb;
    }

    input.d_c[idx] = 0.25 + (float)((xna + xnb) % 1000) / 1000.0;
    idx += stride;
  }
}

void run_stress_test(
    uint32_t thread_id,
    uint32_t num_workers,
    stress_test_args test_args) {
  // We need to print an output to avoid making the compiler believe
  // that the following is a bunch of dead code.
  float checksum = 0.0;

  // Use a fixed random seed to be deterministic
  uint32_t rng_state = RNG_SEED + thread_id;

  // Check memory usage
  size_t mem_free = 0;
  size_t mem_total = 0;
  size_t mem_used_before = 0;
  size_t mem_used_during = 0;
  checkCudaStatus(cudaMemGetInfo(&mem_free, &mem_total), __LINE__);
  mem_used_before = (mem_total - mem_free) / 1024 / 1024;

  // Create multiple streams
  cudaStream_t* v_streams = NULL;
  cudaStream_t memcpy_stream = NULL;
  cudaStream_t uvm_stream = NULL;

  // Allocate streams within the test function as this can run on multiple
  // threads and we want to see the effect of parallel stream creation

  if (test_args.pre_alloc_streams) {
    v_streams =
        test_args.compute_streams + (thread_id * test_args.num_cuda_streams);

    if (test_args.use_memcpy_stream) {
      memcpy_stream = test_args.memcpy_streams[thread_id];
    }

    if (test_args.use_uvm_stream) {
      uvm_stream = test_args.uvm_streams[thread_id];
    }
  } else {
    v_streams = (cudaStream_t*)malloc(
        test_args.num_cuda_streams * sizeof(cudaStream_t));
    for (uint32_t i = 0; i < test_args.num_cuda_streams; ++i) {
      checkCudaStatus(
          cudaStreamCreateWithFlags(v_streams + i, cudaStreamNonBlocking),
          __LINE__);
    }

    if (test_args.use_memcpy_stream) {
      checkCudaStatus(
          cudaStreamCreateWithFlags(&memcpy_stream, cudaStreamNonBlocking),
          __LINE__);
    }

    if (test_args.use_uvm_stream) {
      checkCudaStatus(
          cudaStreamCreateWithFlags(&uvm_stream, cudaStreamNonBlocking),
          __LINE__);
    }
  }

  // Create output buffer for async downloads
  float* h_output = (float*)malloc(sizeof(float) * test_args.num_operations);
  memset(h_output, 0, test_args.num_operations * sizeof(float));

  // Measure time
  float t_wall_ms = 0.0;
  clock_t begin = clock();
  uint32_t num_pairs_per_worker = num_tensor_pairs / num_workers;

  // Start running the benchmark
  for (uint32_t i = 0; i < test_args.num_operations; ++i) {
    // All good things start with a break. In our case some GPU idle time
    if (test_args.simulate_host_time) {
      uint32_t gpu_idle_us =
          rand_r(&rng_state) % (test_args.max_idle_us - test_args.min_idle_us) +
          test_args.min_idle_us;
      usleep(gpu_idle_us);
    }

    // Generate stream ID and tensor pair index
    uint32_t local_pair_idx = i % num_pairs_per_worker;
    uint32_t pair_idx = thread_id * num_pairs_per_worker + local_pair_idx;

    // Select the stream to run the operation on
    uint32_t stream_idx = local_pair_idx % test_args.num_cuda_streams;
    cudaStream_t current_stream = v_streams[stream_idx];
    cudaStream_t current_memcpy_stream = v_streams[stream_idx];
    if (test_args.use_memcpy_stream) {
      current_memcpy_stream = memcpy_stream;
    }
    cudaStream_t current_uvm_stream = v_streams[stream_idx];
    if (test_args.use_uvm_stream) {
      current_uvm_stream = uvm_stream;
    }

    // Check if we do a CUDA malloc
    if (((float)(rand_r(&rng_state) % 32767) / 32767.0) <
        test_args.prob_cuda_malloc) {
      checkCudaStatus(cudaDeviceSynchronize(), __LINE__);
      free_and_realloc_tensor_pairs(p_memory_pool + pair_idx, current_stream);

      // Initialize device buffers with random values
      simple_rng_lcg<<<
          test_args.thread_blocks_per_kernel,
          test_args.kernel_block_size,
          0,
          current_stream>>>(
          p_memory_pool[pair_idx].d_A, p_memory_pool[pair_idx].n_elements);
      CUDA_KERNEL_LAUNCH_CHECK();
      simple_rng_lcg<<<
          test_args.thread_blocks_per_kernel,
          test_args.kernel_block_size,
          0,
          current_stream>>>(
          p_memory_pool[pair_idx].d_B, p_memory_pool[pair_idx].n_elements);
      CUDA_KERNEL_LAUNCH_CHECK();
      simple_rng_lcg<<<
          test_args.thread_blocks_per_kernel,
          test_args.kernel_block_size,
          0,
          current_stream>>>(
          p_memory_pool[pair_idx].d_C, p_memory_pool[pair_idx].n_elements);
      CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Do a CUDA memcpy if needed
    if (p_memory_pool[pair_idx].b_copy_h2d) {
      checkCudaStatus(
          cudaMemcpyAsync(
              p_memory_pool[pair_idx].d_A,
              p_memory_pool[pair_idx].h_A,
              p_memory_pool[pair_idx].n_elements * sizeof(float),
              cudaMemcpyHostToDevice,
              current_memcpy_stream),
          __LINE__);
      checkCudaStatus(
          cudaMemcpyAsync(
              p_memory_pool[pair_idx].d_B,
              p_memory_pool[pair_idx].h_B,
              p_memory_pool[pair_idx].n_elements * sizeof(float),
              cudaMemcpyHostToDevice,
              current_memcpy_stream),
          __LINE__);
    }

    // Launch kernel
    uint32_t num_iters_stream = rand_r(&rng_state) %
            (test_args.max_iters_kernel - test_args.min_iters_kernel) +
        test_args.min_iters_kernel;
    lcg_kernel_input kernel_args;
    kernel_args.d_a = p_memory_pool[pair_idx].d_A;
    kernel_args.d_b = p_memory_pool[pair_idx].d_B;
    kernel_args.d_c = p_memory_pool[pair_idx].d_C;
    kernel_args.len = p_memory_pool[pair_idx].n_elements;
    kernel_args.iters = num_iters_stream;

    if (test_args.use_uvm_buffers) {
      kernel_args.uvm_a = test_args.uvm_a;
      kernel_args.uvm_b = test_args.uvm_b;
      kernel_args.uvm_len = test_args.uvm_len;
    } else {
      kernel_args.uvm_a = NULL;
      kernel_args.uvm_b = NULL;
      kernel_args.uvm_len = 0;
    }

    bool b_do_memset =
        ((float)(rand_r(&rng_state) % 32767) / 32767.0) < test_args.memset_prob;
    bool b_uvm_kernel = ((float)(rand_r(&rng_state) % 32767) / 32767.0) <
            test_args.uvm_kernel_prob
        ? true
        : false;
    if ((kernel_args.uvm_len > 0) && (b_uvm_kernel)) {
      if (b_do_memset) {
        memset((void*)test_args.uvm_a, 42, kernel_args.len * sizeof(float));
        memset((void*)test_args.uvm_a, 42, kernel_args.len * sizeof(float));
        // checkCudaStatus(cudaMemset((void*)test_args.uvm_a, 42,
        // kernel_args.len * sizeof(float)));
        // checkCudaStatus(cudaMemset((void*)test_args.uvm_a, 42,
        // kernel_args.len * sizeof(float)));
      } else {
        iterative_lcg_3_with_uvm<113, 119>
            <<<test_args.thread_blocks_per_kernel,
               test_args.kernel_block_size,
               0,
               current_uvm_stream>>>(kernel_args);
        CUDA_KERNEL_LAUNCH_CHECK();
      }
    } else {
      // Check to see if we do a simple kernel call or a memset
      if (b_do_memset) {
        checkCudaStatus(cudaMemset(
            (void*)kernel_args.d_a, 42, kernel_args.len * sizeof(float)));
        checkCudaStatus(cudaMemset(
            (void*)kernel_args.d_b, 42, kernel_args.len * sizeof(float)));
        checkCudaStatus(cudaMemset(
            (void*)kernel_args.d_c, 42, kernel_args.len * sizeof(float)));
      } else {
        call_compute_kernel(
            test_args.thread_blocks_per_kernel,
            test_args.kernel_block_size,
            0,
            current_stream,
            kernel_args,
            i);
      }
    }

    // Simulate NCCL communication
    // printf("PID %d pair index %d.\n", getpid(), pair_idx);
    if ((i % test_args.num_iters_nccl_sync == 0) && (test_args.is_multi_rank)) {
      uint32_t n_elements = p_memory_pool[pair_idx].n_elements;
      size_t szTransfer = n_elements * sizeof(float);
      checkCudaStatus(
          cudaMemcpyAsync(
              pBuffNCCLSend,
              p_memory_pool[pair_idx].d_C,
              szTransfer,
              cudaMemcpyDeviceToDevice,
              current_stream),
          __LINE__);
      NCCLCHECK(ncclAllReduce(
          (const void*)pBuffNCCLSend,
          (void*)pBuffNCCLRecv,
          n_elements,
          ncclFloat,
          ncclAvg,
          nccl_communicator,
          current_stream));
      // checkCudaStatus(cudaStreamSynchronize(current_stream), __LINE__);
      checkCudaStatus(
          cudaMemcpyAsync(
              p_memory_pool[pair_idx].d_C,
              pBuffNCCLRecv,
              szTransfer,
              cudaMemcpyDeviceToDevice,
              current_stream),
          __LINE__);
    }

    // Simulate checkpoint download. The odd workers will have higher stream
    // priorities but lower number of transactions
    bool enable_d2h_copy = p_memory_pool[pair_idx].b_copy_d2h;
    if (thread_id % 2 != 0) {
      if (rand_r(&rng_state) % 100 < 97) {
        enable_d2h_copy = false;
      }
    }

    // Tehchnically we should wait for the kernels to complete before
    // downloading using a stream synchronization on the compute stream. But if
    // we want to generate multiple overlapping transfers, we need to remove the
    // synchronization. That means the downloaded tensors may not have correct
    // data.

    if (enable_d2h_copy) {
      // checkCudaStatus(cudaStreamSynchronize(current_stream), __LINE__);
      uint32_t rand_index =
          rand_r(&rng_state) % p_memory_pool[pair_idx].n_elements;
      checkCudaStatus(
          cudaMemcpyAsync(
              p_memory_pool[pair_idx].h_C,
              p_memory_pool[pair_idx].d_C,
              p_memory_pool[pair_idx].n_elements * sizeof(float),
              cudaMemcpyDeviceToHost,
              current_memcpy_stream),
          __LINE__);
      uint32_t rand_idx_out = rand_r(&rng_state) % test_args.num_operations;
      // checkCudaStatus(cudaStreamSynchronize(current_memcpy_stream),
      // __LINE__);
      h_output[rand_idx_out] = p_memory_pool[pair_idx].h_C[rand_index];
    }

    // Get memory during execution
    if (test_args.monitor_mem_usage) {
      if (i % 10000 == 0) {
        checkCudaStatus(cudaMemGetInfo(&mem_free, &mem_total), __LINE__);
        size_t mem_crnt = (mem_total - mem_free) / 1024 / 1024;
        if (mem_crnt >= mem_used_during) {
          mem_used_during = mem_crnt;
        }
      }
    }
  }

  // Synchronize all streams
  for (int i = 0; i < test_args.num_cuda_streams; ++i) {
    checkCudaStatus(cudaStreamSynchronize(v_streams[i]), __LINE__);
  }

  if (test_args.use_memcpy_stream) {
    checkCudaStatus(cudaStreamSynchronize(memcpy_stream), __LINE__);
  }

  if (test_args.use_uvm_stream) {
    checkCudaStatus(cudaStreamSynchronize(uvm_stream), __LINE__);
  }

  // Measure execution time only until the streams are synchronized.
  // If we measure the time it takes to destroy them we may get high
  // run to run variation.

  clock_t end = clock();
  t_wall_ms = (double)(end - begin) / 1e+3;

  // Destroy the streams to avoid memory leaks
  if (!test_args.pre_alloc_streams) {
    for (int i = 0; i < test_args.num_cuda_streams; ++i) {
      checkCudaStatus(cudaStreamDestroy(v_streams[i]), __LINE__);
    }

    if (test_args.use_memcpy_stream) {
      checkCudaStatus(cudaStreamDestroy(memcpy_stream), __LINE__);
    }

    if (test_args.use_uvm_stream) {
      checkCudaStatus(cudaStreamDestroy(uvm_stream), __LINE__);
    }

    if (v_streams) {
      free(v_streams);
    }
  }

  // Compute a checksum to have some value as an output of the function
  for (int i = 0; i < test_args.num_operations; ++i) {
    checksum += h_output[i];
  }
  // checksum /= (float)test_args.num_operations;
  free(h_output);

  // Check how much memory we are using
  checkCudaStatus(cudaMemGetInfo(&mem_free, &mem_total), __LINE__);
  size_t mem_used_after = (mem_total - mem_free) / 1024 / 1024;

  printf(
      "Thread Index: %4d; GPU MB at Start: %6zu; Max GPU MB During Run: %6zu; GPU MB at Stop: %6zu; Runtime (ms): %6.3f; Checksum: %.5f\n",
      thread_id,
      mem_used_before,
      mem_used_during,
      mem_used_after,
      t_wall_ms,
      checksum);

  checkCudaStatus(cudaDeviceSynchronize(), __LINE__);
}

// In case CUPTI compresses data using kernel name as a key to a hash map
// we want to see what happens in the case where we have lots of unique
// kernel names. This will make the trace to look like a rainbow.

void call_compute_kernel(
    uint32_t thread_blocks,
    uint32_t threads_per_block,
    uint32_t shmem_sz,
    cudaStream_t stream,
    lcg_kernel_input kernel_args,
    uint32_t op_id) {
  switch (op_id % 20) {
    case 0:
      iterative_lcg_3_buffers<0, 1>
          <<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 1:
      iterative_lcg_3_buffers<1, 2>
          <<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 2:
      iterative_lcg_3_buffers<2, 3>
          <<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 3:
      iterative_lcg_3_buffers<3, 4>
          <<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 4:
      iterative_lcg_3_buffers<4, 5>
          <<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 5:
      iterative_lcg_3_buffers<5, 6>
          <<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 6:
      iterative_lcg_3_buffers<6, 7>
          <<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 7:
      iterative_lcg_3_buffers<7, 8>
          <<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 8:
      iterative_lcg_3_buffers<8, 9>
          <<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 9:
      iterative_lcg_3_buffers<9, 10>
          <<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 10:
      iterative_lcg_3_buffers<10, 11>
          <<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 11:
      iterative_lcg_3_buffers<11, 12>
          <<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 12:
      iterative_lcg_3_buffers<12, 13>
          <<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 13:
      iterative_lcg_3_buffers<13, 14>
          <<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 14:
      iterative_lcg_3_buffers<14, 15>
          <<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 15:
      iterative_lcg_3_buffers<15, 16>
          <<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 16:
      iterative_lcg_3_buffers<16, 17>
          <<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 17:
      iterative_lcg_3_buffers<17, 18>
          <<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 18:
      iterative_lcg_3_buffers<18, 19>
          <<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 19:
      iterative_lcg_3_buffers<19, 20>
          <<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 20:
      iterative_lcg_3_buffers<20, 1>
          <<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 21:
      iterative_lcg_3_buffers<21, 2>
          <<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 22:
      iterative_lcg_3_buffers<22, 3>
          <<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 23:
      iterative_lcg_3_buffers<23, 4>
          <<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 24:
      iterative_lcg_3_buffers<24, 5>
          <<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 25:
      iterative_lcg_3_buffers<25, 6>
          <<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 26:
      iterative_lcg_3_buffers<26, 7>
          <<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 27:
      iterative_lcg_3_buffers<27, 8>
          <<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 28:
      iterative_lcg_3_buffers<28, 9>
          <<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 29:
      iterative_lcg_3_buffers<29, 10>
          <<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 30:
      iterative_lcg_3_buffers<30, 11>
          <<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 31:
      iterative_lcg_3_buffers<31, 12>
          <<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 32:
      iterative_lcg_3_buffers<32, 13>
          <<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 33:
      iterative_lcg_3_buffers<33, 14>
          <<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 34:
      iterative_lcg_3_buffers<34, 15>
          <<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 35:
      iterative_lcg_3_buffers<35, 16>
          <<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 36:
      iterative_lcg_3_buffers<36, 17>
          <<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 37:
      iterative_lcg_3_buffers<37, 18>
          <<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 38:
      iterative_lcg_3_buffers<38, 19>
          <<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    case 39:
      iterative_lcg_3_buffers<39, 20>
          <<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
    default:
      iterative_lcg_3_buffers<0, 0>
          <<<thread_blocks, threads_per_block, 0, stream>>>(kernel_args);
      break;
  }
  CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace kineto_stress_test
