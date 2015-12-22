// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_REDUCTION_H
#define EIGEN_CXX11_TENSOR_TENSOR_REDUCTION_H

namespace Eigen {
namespace internal {


#if defined(EIGEN_USE_GPU) && defined(__CUDACC__)
// Full reducers for GPU, don't vectorize for now

// Reducer function that enables multiple cuda thread to safely accumulate at the same
// output address. It basically reads the current value of the output variable, and
// attempts to update it with the new value. If in the meantime another cuda thread
// updated the content of the output address it will try again.
template <typename T, typename R>
__device__ EIGEN_ALWAYS_INLINE void atomicReduce(T* output, T accum, R& reducer) {
#if __CUDA_ARCH__ >= 300
  if (sizeof(T) == 4)
  {
    unsigned int oldval = *reinterpret_cast<unsigned int*>(output);
    unsigned int newval = oldval;
    reducer.reduce(accum, reinterpret_cast<T*>(&newval));
    if (newval == oldval) {
      return;
    }
    unsigned int readback;
    while ((readback = atomicCAS((unsigned int*)output, oldval, newval)) != oldval) {
      oldval = readback;
      newval = oldval;
      reducer.reduce(accum, reinterpret_cast<T*>(&newval));
      if (newval == oldval) {
        return;
      }
    }
  }
  else if (sizeof(T) == 8) {
    unsigned long long oldval = *reinterpret_cast<unsigned long long*>(output);
    unsigned long long newval = oldval;
    reducer.reduce(accum, reinterpret_cast<T*>(&newval));
    if (newval == oldval) {
      return;
    }
    unsigned long long readback;
    while ((readback = atomicCAS((unsigned long long*)output, oldval, newval)) != oldval) {
      oldval = readback;
      newval = oldval;
      reducer.reduce(accum, reinterpret_cast<T*>(&newval));
      if (newval == oldval) {
        return;
      }
    }
  }
  else {
    assert(0 && "Wordsize not supported");
  }
#else
  assert(0 && "Shouldn't be called on unsupported device");
#endif
}

template <typename T>
__device__ inline void atomicReduce(T* output, T accum, SumReducer<T>&) {
#if __CUDA_ARCH__ >= 300
  atomicAdd(output, accum);
#else
  assert(0 && "Shouldn't be called on unsupported device");
#endif
}

template <int BlockSize, int NumPerThread, typename Self,
          typename Reducer, typename Index>
__global__ void FullReductionKernel(Reducer reducer, const Self input, Index num_coeffs,
                                    typename Self::CoeffReturnType* output) {
  const Index first_index = blockIdx.x * BlockSize * NumPerThread + threadIdx.x;

  if (first_index == 0) {
    *output = reducer.initialize();
  }

  typename Self::CoeffReturnType accum = reducer.initialize();
  for (Index i = 0; i < NumPerThread; ++i) {
    const Index index = first_index + i * BlockSize;
    if (index >= num_coeffs) {
      break;
    }
    typename Self::CoeffReturnType val = input.m_impl.coeff(index);
    reducer.reduce(val, &accum);
  }

  for (int offset = warpSize/2; offset > 0; offset /= 2) {
    reducer.reduce(__shfl_down(accum, offset), &accum);
  }

  if ((threadIdx.x & (warpSize - 1)) == 0) {
    atomicReduce(output, accum, reducer);
  }
}


template <typename Self, typename Op, bool Vectorizable>
struct FullReducer<Self, Op, GpuDevice, Vectorizable> {
  // Unfortunately nvidia doesn't support well exotic types such as complex,
  // so reduce the scope of the optimized version of the code to the simple case
  // of floats.
  static const bool HasOptimizedImplementation = !Op::IsStateful &&
                                                 internal::is_same<typename Self::CoeffReturnType, float>::value;

  template <typename OutputType>
  EIGEN_DEVICE_FUNC static void run(const Self& self, Op& reducer, const GpuDevice& device, OutputType* output) {
    assert(false && "Should only be called on floats");
  }

  EIGEN_DEVICE_FUNC static void run(const Self& self, Op& reducer, const GpuDevice& device, float* output) {
    typedef typename Self::Index Index;

    const Index num_coeffs = array_prod(self.m_impl.dimensions());
    const int block_size = 256;
    const int num_per_thread = 128;
    const int num_blocks = std::ceil(static_cast<float>(num_coeffs) / (block_size * num_per_thread));
    LAUNCH_CUDA_KERNEL((FullReductionKernel<block_size, num_per_thread>),
                       num_blocks, block_size, 0, device, reducer, self, num_coeffs, output);
  }
};

#endif


} // end namespace internal
} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_REDUCTION_H
