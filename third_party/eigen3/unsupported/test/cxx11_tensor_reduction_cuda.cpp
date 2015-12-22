// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_TEST_NO_LONGDOUBLE
#define EIGEN_TEST_NO_COMPLEX
#define EIGEN_TEST_FUNC cxx11_tensor_reduction_cuda
#define EIGEN_USE_GPU

#include "main.h"
#include <unsupported/Eigen/CXX11/Tensor>


template<int DataLayout>
static void test_full_reductions() {

  Eigen::CudaStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  const int num_rows = internal::random<int>(1024, 5*1024);
  const int num_cols = internal::random<int>(1024, 5*1024);

  Tensor<float, 2, DataLayout> in(num_rows, num_cols);
  in.setRandom();

  Tensor<float, 0, DataLayout> full_redux;
  full_redux = in.sum();

  std::size_t in_bytes = in.size() * sizeof(float);
  std::size_t out_bytes = full_redux.size() * sizeof(float);
  float* gpu_in_ptr = static_cast<float*>(gpu_device.allocate(in_bytes));
  float* gpu_out_ptr = static_cast<float*>(gpu_device.allocate(out_bytes));
  gpu_device.memcpyHostToDevice(gpu_in_ptr, in.data(), in_bytes);

  TensorMap<Tensor<float, 2, DataLayout> > in_gpu(gpu_in_ptr, num_rows, num_cols);
  TensorMap<Tensor<float, 0, DataLayout> > out_gpu(gpu_out_ptr);

  out_gpu.device(gpu_device) = in_gpu.sum();

  Tensor<float, 0, DataLayout> full_redux_gpu;
  gpu_device.memcpyDeviceToHost(full_redux_gpu.data(), gpu_out_ptr, out_bytes);
  gpu_device.synchronize();

  // Check that the CPU and GPU reductions return the same result.
  VERIFY_IS_APPROX(full_redux(), full_redux_gpu());
}

void test_cxx11_tensor_reduction_cuda() {
  CALL_SUBTEST(test_full_reductions<ColMajor>());
  CALL_SUBTEST(test_full_reductions<RowMajor>());
}
