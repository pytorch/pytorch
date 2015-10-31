// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// TODO(mdevin): Free the cuda memory.

#define EIGEN_TEST_FUNC cxx11_tensor_cuda
#define EIGEN_USE_GPU

#include "main.h"
#include <unsupported/Eigen/CXX11/Tensor>

using Eigen::Tensor;

template <int Layout>
void test_cuda_simple_argmax()
{
  Tensor<double, 3, Layout> in(Eigen::array<DenseIndex, 3>(72,53,97));
  Tensor<DenseIndex, 1, Layout> out_max(Eigen::array<DenseIndex, 1>(1));
  Tensor<DenseIndex, 1, Layout> out_min(Eigen::array<DenseIndex, 1>(1));
  in.setRandom();
  in *= in.constant(100.0);
  in(0, 0, 0) = -1000.0;
  in(71, 52, 96) = 1000.0;

  std::size_t in_bytes = in.size() * sizeof(double);
  std::size_t out_bytes = out_max.size() * sizeof(DenseIndex);

  double* d_in;
  DenseIndex* d_out_max;
  DenseIndex* d_out_min;
  cudaMalloc((void**)(&d_in), in_bytes);
  cudaMalloc((void**)(&d_out_max), out_bytes);
  cudaMalloc((void**)(&d_out_min), out_bytes);

  cudaMemcpy(d_in, in.data(), in_bytes, cudaMemcpyHostToDevice);

  Eigen::CudaStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<double, 3, Layout>, Aligned > gpu_in(d_in, Eigen::array<DenseIndex, 3>(72,53,97));
  Eigen::TensorMap<Eigen::Tensor<DenseIndex, 1, Layout>, Aligned > gpu_out_max(d_out_max, Eigen::array<DenseIndex, 1>(1));
  Eigen::TensorMap<Eigen::Tensor<DenseIndex, 1, Layout>, Aligned > gpu_out_min(d_out_min, Eigen::array<DenseIndex, 1>(1));

  gpu_out_max.device(gpu_device) = gpu_in.argmax();
  gpu_out_min.device(gpu_device) = gpu_in.argmin();

  assert(cudaMemcpyAsync(out_max.data(), d_out_max, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
  assert(cudaMemcpyAsync(out_min.data(), d_out_min, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
  assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);

  VERIFY_IS_EQUAL(out_max(Eigen::array<DenseIndex, 1>(0)), 72*53*97 - 1);
  VERIFY_IS_EQUAL(out_min(Eigen::array<DenseIndex, 1>(0)), 0);
}

template <int DataLayout>
void test_cuda_argmax_dim()
{
  Tensor<float, 4, DataLayout> tensor(2,3,5,7);
  std::vector<int> dims;
  dims.push_back(2); dims.push_back(3); dims.push_back(5); dims.push_back(7);

  for (int dim = 0; dim < 4; ++dim) {
    tensor.setRandom();
    tensor = (tensor + tensor.constant(0.5)).log();

    array<DenseIndex, 3> out_shape;
    for (int d = 0; d < 3; ++d) out_shape[d] = (d < dim) ? dims[d] : dims[d+1];

    Tensor<DenseIndex, 3, DataLayout> tensor_arg(out_shape);

    array<DenseIndex, 4> ix;
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 5; ++k) {
          for (int l = 0; l < 7; ++l) {
            ix[0] = i; ix[1] = j; ix[2] = k; ix[3] = l;
            if (ix[dim] != 0) continue;
            // suppose dim == 1, then for all i, k, l, set tensor(i, 0, k, l) = 10.0
            tensor(ix) = 10.0;
          }
        }
      }
    }

    std::size_t in_bytes = tensor.size() * sizeof(float);
    std::size_t out_bytes = tensor_arg.size() * sizeof(DenseIndex);

    float* d_in;
    DenseIndex* d_out;
    cudaMalloc((void**)(&d_in), in_bytes);
    cudaMalloc((void**)(&d_out), out_bytes);

    cudaMemcpy(d_in, tensor.data(), in_bytes, cudaMemcpyHostToDevice);

    Eigen::CudaStreamDevice stream;
    Eigen::GpuDevice gpu_device(&stream);

    Eigen::TensorMap<Eigen::Tensor<float, 4, DataLayout>, Aligned > gpu_in(d_in, Eigen::array<DenseIndex, 4>(2, 3, 5, 7));
    Eigen::TensorMap<Eigen::Tensor<DenseIndex, 3, DataLayout>, Aligned > gpu_out(d_out, out_shape);

    gpu_out.device(gpu_device) = gpu_in.argmax(dim);

    assert(cudaMemcpyAsync(tensor_arg.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
    assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);

    VERIFY_IS_EQUAL(tensor_arg.dimensions().TotalSize(),
                    size_t(2*3*5*7 / tensor.dimension(dim)));

    for (size_t n = 0; n < tensor_arg.dimensions().TotalSize(); ++n) {
      // Expect max to be in the first index of the reduced dimension
      VERIFY_IS_EQUAL(tensor_arg.data()[n], 0);
    }

    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 5; ++k) {
          for (int l = 0; l < 7; ++l) {
            ix[0] = i; ix[1] = j; ix[2] = k; ix[3] = l;
            if (ix[dim] != tensor.dimension(dim) - 1) continue;
            // suppose dim == 1, then for all i, k, l, set tensor(i, 2, k, l) = 20.0
            tensor(ix) = 20.0;
          }
        }
      }
    }

    cudaMemcpy(d_in, tensor.data(), in_bytes, cudaMemcpyHostToDevice);

    gpu_out.device(gpu_device) = gpu_in.argmax(dim);

    assert(cudaMemcpyAsync(tensor_arg.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
    assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);

    for (size_t n = 0; n < tensor_arg.dimensions().TotalSize(); ++n) {
      // Expect max to be in the last index of the reduced dimension
      VERIFY_IS_EQUAL(tensor_arg.data()[n], tensor.dimension(dim) - 1);
    }
  }
}

template <int DataLayout>
void test_cuda_argmin_dim()
{
  Tensor<float, 4, DataLayout> tensor(2,3,5,7);
  std::vector<int> dims;
  dims.push_back(2); dims.push_back(3); dims.push_back(5); dims.push_back(7);

  for (int dim = 0; dim < 4; ++dim) {
    tensor.setRandom();
    tensor = (tensor + tensor.constant(0.5)).log();

    array<DenseIndex, 3> out_shape;
    for (int d = 0; d < 3; ++d) out_shape[d] = (d < dim) ? dims[d] : dims[d+1];

    Tensor<DenseIndex, 3, DataLayout> tensor_arg(out_shape);

    array<DenseIndex, 4> ix;
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 5; ++k) {
          for (int l = 0; l < 7; ++l) {
            ix[0] = i; ix[1] = j; ix[2] = k; ix[3] = l;
            if (ix[dim] != 0) continue;
            // suppose dim == 1, then for all i, k, l, set tensor(i, 0, k, l) = 10.0
            tensor(ix) = -10.0;
          }
        }
      }
    }

    std::size_t in_bytes = tensor.size() * sizeof(float);
    std::size_t out_bytes = tensor_arg.size() * sizeof(DenseIndex);

    float* d_in;
    DenseIndex* d_out;
    cudaMalloc((void**)(&d_in), in_bytes);
    cudaMalloc((void**)(&d_out), out_bytes);

    cudaMemcpy(d_in, tensor.data(), in_bytes, cudaMemcpyHostToDevice);

    Eigen::CudaStreamDevice stream;
    Eigen::GpuDevice gpu_device(&stream);

    Eigen::TensorMap<Eigen::Tensor<float, 4, DataLayout>, Aligned > gpu_in(d_in, Eigen::array<DenseIndex, 4>(2, 3, 5, 7));
    Eigen::TensorMap<Eigen::Tensor<DenseIndex, 3, DataLayout>, Aligned > gpu_out(d_out, out_shape);

    gpu_out.device(gpu_device) = gpu_in.argmin(dim);

    assert(cudaMemcpyAsync(tensor_arg.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
    assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);

    VERIFY_IS_EQUAL(tensor_arg.dimensions().TotalSize(),
                    size_t(2*3*5*7 / tensor.dimension(dim)));

    for (size_t n = 0; n < tensor_arg.dimensions().TotalSize(); ++n) {
      // Expect min to be in the first index of the reduced dimension
      VERIFY_IS_EQUAL(tensor_arg.data()[n], 0);
    }

    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 5; ++k) {
          for (int l = 0; l < 7; ++l) {
            ix[0] = i; ix[1] = j; ix[2] = k; ix[3] = l;
            if (ix[dim] != tensor.dimension(dim) - 1) continue;
            // suppose dim == 1, then for all i, k, l, set tensor(i, 2, k, l) = 20.0
            tensor(ix) = -20.0;
          }
        }
      }
    }

    cudaMemcpy(d_in, tensor.data(), in_bytes, cudaMemcpyHostToDevice);

    gpu_out.device(gpu_device) = gpu_in.argmin(dim);

    assert(cudaMemcpyAsync(tensor_arg.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, gpu_device.stream()) == cudaSuccess);
    assert(cudaStreamSynchronize(gpu_device.stream()) == cudaSuccess);

    for (size_t n = 0; n < tensor_arg.dimensions().TotalSize(); ++n) {
      // Expect max to be in the last index of the reduced dimension
      VERIFY_IS_EQUAL(tensor_arg.data()[n], tensor.dimension(dim) - 1);
    }
  }
}

void test_cxx11_tensor_cuda()
{
  CALL_SUBTEST(test_cuda_simple_argmax<RowMajor>());
  CALL_SUBTEST(test_cuda_simple_argmax<ColMajor>());
  CALL_SUBTEST(test_cuda_argmax_dim<RowMajor>());
  CALL_SUBTEST(test_cuda_argmax_dim<ColMajor>());
  CALL_SUBTEST(test_cuda_argmin_dim<RowMajor>());
  CALL_SUBTEST(test_cuda_argmin_dim<ColMajor>());
}
