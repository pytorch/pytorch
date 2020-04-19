#pragma once

#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/cuda/Exceptions.h>
#include <cstdint>
#include <mutex>

namespace torch {
namespace custom_prng {

/*
  Generates `block_t_size`-bytes random key Tensor on CPU 
  using `generator`, which must be an instance of `at::CPUGeneratorImpl`
  and passes it to `device`.
*/
at::Tensor key_tensor(c10::optional<at::Generator> generator, int block_t_size, at::Device device) {
  std::lock_guard<std::mutex> lock(generator->mutex());
  auto gen = at::check_generator<at::CPUGeneratorImpl>(generator);
  auto t = torch::empty({block_t_size}, torch::kUInt8);
  for (int i = 0; i < block_t_size; i++) {
    t[i] = static_cast<uint8_t>(gen->random());
  }
  return t.to(device);
}

template<size_t size>
struct DummyRNG {
  __device__ DummyRNG(uint64_t* vals) {
    memcpy(&vals_, vals, size * sizeof(uint64_t));
  }
  uint32_t __device__ random() { return static_cast<uint32_t>(vals_[index++]); }
  uint64_t __device__ random64() { return vals_[index++]; }
private:
  uint64_t vals_[size];
  int index = 0;
};

template<typename scalar_t, typename uint_t, size_t N = 1, typename cipher_t, typename transform_t>
__global__ static void block_cipher_contiguous_kernel(scalar_t* data, int numel, int block_t_size, cipher_t cipher, transform_t transform_func) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto unroll_factor = block_t_size / sizeof(uint_t) / N;
  if (unroll_factor * idx < numel) {
    auto block = cipher(idx);
    #pragma unroll
    for (auto i = 0; i < unroll_factor; ++i) {
      const auto li = unroll_factor * idx + i;
      if (li < numel) {
        uint64_t vals[N];
        #pragma unroll
        for (auto j = 0; j < N; j++) {
          vals[j] = (reinterpret_cast<uint_t*>(&block))[N * i + j];
        }
        DummyRNG<N> rng(vals);
        data[li] = transform_func(&rng);
      }
    }
  }
}

template<typename scalar_t, typename uint_t, size_t N = 1, typename cipher_t, typename transform_t>
__global__ static void block_cipher_kernel(scalar_t* data, int numel, int block_t_size, cipher_t cipher, transform_t transform_func, OffsetCalculator<1> offset_calc) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto unroll_factor = block_t_size / sizeof(uint_t) / N;
  if (unroll_factor * idx < numel) {
    auto block = cipher(idx);
    #pragma unroll
    for (auto i = 0; i < unroll_factor; ++i) {
      const auto li = unroll_factor * idx + i;
      if (li < numel) {
        uint64_t vals[N];
        #pragma unroll
        for (auto j = 0; j < N; j++) {
          vals[j] = (reinterpret_cast<uint_t*>(&block))[N * i + j];
        }
        DummyRNG<N> rng(vals);
        data[offset_calc.get(li)[0] / sizeof(scalar_t)] = transform_func(&rng);
      }
    }
  }
}

template<typename scalar_t, typename uint_t, size_t N = 1, typename cipher_t, typename transform_t>
void block_cipher_ctr_mode(at::TensorIterator& iter, int block_t_size, cipher_t cipher, transform_t transform_func) {
  const auto numel = iter.numel();
  if (numel == 0) {
    return;
  }
  const auto unroll_factor = block_t_size / sizeof(uint_t) / N;
  const auto block = 256;
  const auto grid = (numel + (block * unroll_factor) - 1) / (block * unroll_factor);
  scalar_t* data = (scalar_t*)iter.data_ptr(0);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (iter.output(0).is_contiguous()) {
    block_cipher_contiguous_kernel<scalar_t, uint_t, N, cipher_t, transform_t><<<grid, block, 0, stream>>>(data, numel, block_t_size, cipher, transform_func);
  } else {
    auto offset_calc = make_offset_calculator<1>(iter);
    block_cipher_kernel<scalar_t, uint_t, N, cipher_t, transform_t><<<grid, block, 0, stream>>>(data, numel, block_t_size, cipher, transform_func, offset_calc);
  }
  AT_CUDA_CHECK(cudaGetLastError());
}

}}
