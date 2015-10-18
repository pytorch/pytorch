// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_TEST_NO_LONGDOUBLE
#define EIGEN_TEST_NO_COMPLEX
#define EIGEN_TEST_FUNC cxx11_tensor_device
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU


#include "main.h"
#include <unsupported/Eigen/CXX11/Tensor>

using Eigen::Tensor;
using Eigen::RowMajor;

// Context for evaluation on cpu
struct CPUContext {
  CPUContext(const Eigen::Tensor<float, 3>& in1, Eigen::Tensor<float, 3>& in2, Eigen::Tensor<float, 3>& out) : in1_(in1), in2_(in2), out_(out), kernel_1d_(2), kernel_2d_(2,2), kernel_3d_(2,2,2) {
    kernel_1d_(0) = 3.14f;
    kernel_1d_(1) = 2.7f;

    kernel_2d_(0,0) = 3.14f;
    kernel_2d_(1,0) = 2.7f;
    kernel_2d_(0,1) = 0.2f;
    kernel_2d_(1,1) = 7.0f;

    kernel_3d_(0,0,0) = 3.14f;
    kernel_3d_(0,1,0) = 2.7f;
    kernel_3d_(0,0,1) = 0.2f;
    kernel_3d_(0,1,1) = 7.0f;
    kernel_3d_(1,0,0) = -1.0f;
    kernel_3d_(1,1,0) = -0.3f;
    kernel_3d_(1,0,1) = -0.7f;
    kernel_3d_(1,1,1) = -0.5f;
  }

  const Eigen::DefaultDevice& device() const { return cpu_device_; }

  const Eigen::Tensor<float, 3>& in1() const { return in1_; }
  const Eigen::Tensor<float, 3>& in2() const { return in2_; }
  Eigen::Tensor<float, 3>& out() { return out_; }
  const Eigen::Tensor<float, 1>& kernel1d() const { return kernel_1d_; }
  const Eigen::Tensor<float, 2>& kernel2d() const { return kernel_2d_; }
  const Eigen::Tensor<float, 3>& kernel3d() const { return kernel_3d_; }

 private:
  const Eigen::Tensor<float, 3>& in1_;
  const Eigen::Tensor<float, 3>& in2_;
  Eigen::Tensor<float, 3>& out_;

  Eigen::Tensor<float, 1> kernel_1d_;
  Eigen::Tensor<float, 2> kernel_2d_;
  Eigen::Tensor<float, 3> kernel_3d_;

  Eigen::DefaultDevice cpu_device_;
};


// Context for evaluation on GPU
struct GPUContext {
  GPUContext(const Eigen::TensorMap<Eigen::Tensor<float, 3> >& in1, Eigen::TensorMap<Eigen::Tensor<float, 3> >& in2, Eigen::TensorMap<Eigen::Tensor<float, 3> >& out) : in1_(in1), in2_(in2), out_(out), gpu_device_(&stream_) {
    assert(cudaMalloc((void**)(&kernel_1d_), 2*sizeof(float)) == cudaSuccess);
    float kernel_1d_val[] = {3.14f, 2.7f};
    assert(cudaMemcpy(kernel_1d_, kernel_1d_val, 2*sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);

    assert(cudaMalloc((void**)(&kernel_2d_), 4*sizeof(float)) == cudaSuccess);
    float kernel_2d_val[] = {3.14f, 2.7f, 0.2f, 7.0f};
    assert(cudaMemcpy(kernel_2d_, kernel_2d_val, 4*sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);

    assert(cudaMalloc((void**)(&kernel_3d_), 8*sizeof(float)) == cudaSuccess);
    float kernel_3d_val[] = {3.14f, -1.0f, 2.7f, -0.3f, 0.2f, -0.7f, 7.0f, -0.5f};
    assert(cudaMemcpy(kernel_3d_, kernel_3d_val, 8*sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
  }
  ~GPUContext() {
    assert(cudaFree(kernel_1d_) == cudaSuccess);
    assert(cudaFree(kernel_2d_) == cudaSuccess);
    assert(cudaFree(kernel_3d_) == cudaSuccess);
  }

  const Eigen::GpuDevice& device() const { return gpu_device_; }

  const Eigen::TensorMap<Eigen::Tensor<float, 3> >& in1() const { return in1_; }
  const Eigen::TensorMap<Eigen::Tensor<float, 3> >& in2() const { return in2_; }
  Eigen::TensorMap<Eigen::Tensor<float, 3> >& out() { return out_; }
  Eigen::TensorMap<Eigen::Tensor<float, 1> > kernel1d() const { return Eigen::TensorMap<Eigen::Tensor<float, 1> >(kernel_1d_, 2); }
  Eigen::TensorMap<Eigen::Tensor<float, 2> > kernel2d() const { return Eigen::TensorMap<Eigen::Tensor<float, 2> >(kernel_2d_, 2, 2); }
  Eigen::TensorMap<Eigen::Tensor<float, 3> > kernel3d() const { return Eigen::TensorMap<Eigen::Tensor<float, 3> >(kernel_3d_, 2, 2, 2); }

 private:
  const Eigen::TensorMap<Eigen::Tensor<float, 3> >& in1_;
  const Eigen::TensorMap<Eigen::Tensor<float, 3> >& in2_;
  Eigen::TensorMap<Eigen::Tensor<float, 3> >& out_;

  float* kernel_1d_;
  float* kernel_2d_;
  float* kernel_3d_;

  Eigen::CudaStreamDevice stream_;
  Eigen::GpuDevice gpu_device_;
};


// The actual expression to evaluate
template <typename Context>
static void test_contextual_eval(Context* context)
{
  context->out().device(context->device()) = context->in1() + context->in2() * 3.14f + context->in1().constant(2.718f);
}

template <typename Context>
static void test_forced_contextual_eval(Context* context)
{
  context->out().device(context->device()) = (context->in1() + context->in2()).eval() * 3.14f + context->in1().constant(2.718f);
}

template <typename Context>
static void test_compound_assignment(Context* context)
{
  context->out().device(context->device()) = context->in1().constant(2.718f);
  context->out().device(context->device()) += context->in1() + context->in2() * 3.14f;
}


template <typename Context>
static void test_contraction(Context* context)
{
  Eigen::array<std::pair<int, int>, 2> dims;
  dims[0] = std::make_pair(1, 1);
  dims[1] = std::make_pair(2, 2);

  Eigen::array<int, 2> shape(40, 50*70);

  Eigen::DSizes<int, 2> indices(0,0);
  Eigen::DSizes<int, 2> sizes(40,40);

  context->out().reshape(shape).slice(indices, sizes).device(context->device()) = context->in1().contract(context->in2(), dims);
}


template <typename Context>
static void test_1d_convolution(Context* context)
{
  Eigen::DSizes<int, 3> indices(0,0,0);
  Eigen::DSizes<int, 3> sizes(40,49,70);

  Eigen::array<int, 1> dims(1);
  context->out().slice(indices, sizes).device(context->device()) = context->in1().convolve(context->kernel1d(), dims);
}

template <typename Context>
static void test_2d_convolution(Context* context)
{
  Eigen::DSizes<int, 3> indices(0,0,0);
  Eigen::DSizes<int, 3> sizes(40,49,69);

  Eigen::array<int, 2> dims(1,2);
  context->out().slice(indices, sizes).device(context->device()) = context->in1().convolve(context->kernel2d(), dims);
}

template <typename Context>
static void test_3d_convolution(Context* context)
{
  Eigen::DSizes<int, 3> indices(0,0,0);
  Eigen::DSizes<int, 3> sizes(39,49,69);

  Eigen::array<int, 3> dims(0,1,2);
  context->out().slice(indices, sizes).device(context->device()) = context->in1().convolve(context->kernel3d(), dims);
}


static void test_cpu() {
  Eigen::Tensor<float, 3> in1(40,50,70);
  Eigen::Tensor<float, 3> in2(40,50,70);
  Eigen::Tensor<float, 3> out(40,50,70);

  in1 = in1.random() + in1.constant(10.0f);
  in2 = in2.random() + in2.constant(10.0f);

  CPUContext context(in1, in2, out);
  test_contextual_eval(&context);
  for (int i = 0; i < 40; ++i) {
    for (int j = 0; j < 50; ++j) {
      for (int k = 0; k < 70; ++k) {
        VERIFY_IS_APPROX(out(i,j,k), in1(i,j,k) + in2(i,j,k) * 3.14f + 2.718f);
      }
    }
  }

  test_forced_contextual_eval(&context);
  for (int i = 0; i < 40; ++i) {
    for (int j = 0; j < 50; ++j) {
      for (int k = 0; k < 70; ++k) {
        VERIFY_IS_APPROX(out(i,j,k), (in1(i,j,k) + in2(i,j,k)) * 3.14f + 2.718f);
      }
    }
  }

  test_compound_assignment(&context);
  for (int i = 0; i < 40; ++i) {
    for (int j = 0; j < 50; ++j) {
      for (int k = 0; k < 70; ++k) {
        VERIFY_IS_APPROX(out(i,j,k), in1(i,j,k) + in2(i,j,k) * 3.14f + 2.718f);
      }
    }
  }

  test_contraction(&context);
  for (int i = 0; i < 40; ++i) {
    for (int j = 0; j < 40; ++j) {
      const float result = out(i,j,0);
      float expected = 0;
      for (int k = 0; k < 50; ++k) {
        for (int l = 0; l < 70; ++l) {
          expected += in1(i, k, l) * in2(j, k, l);
        }
      }
      VERIFY_IS_APPROX(expected, result);
    }
  }

  test_1d_convolution(&context);
  for (int i = 0; i < 40; ++i) {
    for (int j = 0; j < 49; ++j) {
      for (int k = 0; k < 70; ++k) {
        VERIFY_IS_APPROX(out(i,j,k), (in1(i,j,k) * 3.14f + in1(i,j+1,k) * 2.7f));
      }
    }
  }

  test_2d_convolution(&context);
  for (int i = 0; i < 40; ++i) {
    for (int j = 0; j < 49; ++j) {
      for (int k = 0; k < 69; ++k) {
        const float result = out(i,j,k);
        const float expected = (in1(i,j,k) * 3.14f + in1(i,j+1,k) * 2.7f) +
                               (in1(i,j,k+1) * 0.2f + in1(i,j+1,k+1) * 7.0f);
        if (fabs(expected) < 1e-4 && fabs(result) < 1e-4) {
          continue;
        }
        VERIFY_IS_APPROX(expected, result);
      }
    }
  }

  test_3d_convolution(&context);
  for (int i = 0; i < 39; ++i) {
    for (int j = 0; j < 49; ++j) {
      for (int k = 0; k < 69; ++k) {
        const float result = out(i,j,k);
        const float expected = (in1(i,j,k) * 3.14f + in1(i,j+1,k) * 2.7f +
                                in1(i,j,k+1) * 0.2f + in1(i,j+1,k+1) * 7.0f) +
                               (in1(i+1,j,k) * -1.0f + in1(i+1,j+1,k) * -0.3f +
                                in1(i+1,j,k+1) * -0.7f + in1(i+1,j+1,k+1) * -0.5f);
        if (fabs(expected) < 1e-4 && fabs(result) < 1e-4) {
          continue;
        }
        VERIFY_IS_APPROX(expected, result);
      }
    }
  }
}

static void test_gpu() {
  Eigen::Tensor<float, 3> in1(40,50,70);
  Eigen::Tensor<float, 3> in2(40,50,70);
  Eigen::Tensor<float, 3> out(40,50,70);
  in1 = in1.random() + in1.constant(10.0f);
  in2 = in2.random() + in2.constant(10.0f);

  std::size_t in1_bytes = in1.size() * sizeof(float);
  std::size_t in2_bytes = in2.size() * sizeof(float);
  std::size_t out_bytes = out.size() * sizeof(float);

  float* d_in1;
  float* d_in2;
  float* d_out;
  cudaMalloc((void**)(&d_in1), in1_bytes);
  cudaMalloc((void**)(&d_in2), in2_bytes);
  cudaMalloc((void**)(&d_out), out_bytes);

  cudaMemcpy(d_in1, in1.data(), in1_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_in2, in2.data(), in2_bytes, cudaMemcpyHostToDevice);

  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in1(d_in1, 40,50,70);
  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_in2(d_in2, 40,50,70);
  Eigen::TensorMap<Eigen::Tensor<float, 3> > gpu_out(d_out, 40,50,70);

  GPUContext context(gpu_in1, gpu_in2, gpu_out);
  test_contextual_eval(&context);
  assert(cudaMemcpy(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
  for (int i = 0; i < 40; ++i) {
    for (int j = 0; j < 50; ++j) {
      for (int k = 0; k < 70; ++k) {
        VERIFY_IS_APPROX(out(i,j,k), in1(i,j,k) + in2(i,j,k) * 3.14f + 2.718f);
      }
    }
  }

  test_forced_contextual_eval(&context);
  assert(cudaMemcpy(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
  for (int i = 0; i < 40; ++i) {
    for (int j = 0; j < 50; ++j) {
      for (int k = 0; k < 70; ++k) {
        VERIFY_IS_APPROX(out(i,j,k), (in1(i,j,k) + in2(i,j,k)) * 3.14f + 2.718f);
      }
    }
  }

  test_compound_assignment(&context);
  assert(cudaMemcpy(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
  for (int i = 0; i < 40; ++i) {
    for (int j = 0; j < 50; ++j) {
      for (int k = 0; k < 70; ++k) {
        VERIFY_IS_APPROX(out(i,j,k), in1(i,j,k) + in2(i,j,k) * 3.14f + 2.718f);
      }
    }
  }

  test_contraction(&context);
  assert(cudaMemcpy(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
  for (int i = 0; i < 40; ++i) {
    for (int j = 0; j < 40; ++j) {
      const float result = out(i,j,0);
      float expected = 0;
      for (int k = 0; k < 50; ++k) {
        for (int l = 0; l < 70; ++l) {
          expected += in1(i, k, l) * in2(j, k, l);
        }
      }
      VERIFY_IS_APPROX(expected, result);
    }
  }

  test_1d_convolution(&context);
  assert(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, context.device().stream()) == cudaSuccess);
  assert(cudaStreamSynchronize(context.device().stream()) == cudaSuccess);
  for (int i = 0; i < 40; ++i) {
    for (int j = 0; j < 49; ++j) {
      for (int k = 0; k < 70; ++k) {
        VERIFY_IS_APPROX(out(i,j,k), (in1(i,j,k) * 3.14f + in1(i,j+1,k) * 2.7f));
      }
    }
  }

  test_2d_convolution(&context);
  assert(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, context.device().stream()) == cudaSuccess);
  assert(cudaStreamSynchronize(context.device().stream()) == cudaSuccess);
  for (int i = 0; i < 40; ++i) {
    for (int j = 0; j < 49; ++j) {
      for (int k = 0; k < 69; ++k) {
        const float result = out(i,j,k);
        const float expected = (in1(i,j,k) * 3.14f + in1(i,j+1,k) * 2.7f +
                                in1(i,j,k+1) * 0.2f + in1(i,j+1,k+1) * 7.0f);
        VERIFY_IS_APPROX(expected, result);
      }
    }
  }

  test_3d_convolution(&context);
  assert(cudaMemcpyAsync(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost, context.device().stream()) == cudaSuccess);
  assert(cudaStreamSynchronize(context.device().stream()) == cudaSuccess);
  for (int i = 0; i < 39; ++i) {
    for (int j = 0; j < 49; ++j) {
      for (int k = 0; k < 69; ++k) {
       const float result = out(i,j,k);
        const float expected = (in1(i,j,k) * 3.14f + in1(i,j+1,k) * 2.7f +
                                in1(i,j,k+1) * 0.2f + in1(i,j+1,k+1) * 7.0f +
                                in1(i+1,j,k) * -1.0f + in1(i+1,j+1,k) * -0.3f +
                                in1(i+1,j,k+1) * -0.7f + in1(i+1,j+1,k+1) * -0.5f);
        VERIFY_IS_APPROX(expected, result);
      }
    }
  }
}


void test_cxx11_tensor_device()
{
  CALL_SUBTEST(test_cpu());
  CALL_SUBTEST(test_gpu());
}
