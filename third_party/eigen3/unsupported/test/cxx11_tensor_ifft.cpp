// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Jianwei Cui <thucjw@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <complex>
#include <cmath>
#include <Eigen/CXX11/Tensor>

using Eigen::Tensor;

template <int DataLayout>
static void test_1D_fft_ifft_invariant(int sequence_length) {
  Tensor<double, 1, DataLayout> tensor(sequence_length);
  tensor.setRandom();

  array<int, 1> fft;
  fft[0] = 0;

  Tensor<std::complex<double>, 1, DataLayout> tensor_after_fft;
  Tensor<std::complex<double>, 1, DataLayout> tensor_after_fft_ifft;

  tensor_after_fft = tensor.template fft<Eigen::BothParts, Eigen::FFT_FORWARD>(fft);
  tensor_after_fft_ifft = tensor_after_fft.template fft<Eigen::BothParts, Eigen::FFT_REVERSE>(fft);

  VERIFY_IS_EQUAL(tensor_after_fft.dimension(0), sequence_length);
  VERIFY_IS_EQUAL(tensor_after_fft_ifft.dimension(0), sequence_length);

  for (int i = 0; i < sequence_length; ++i) {
    VERIFY_IS_APPROX(static_cast<float>(tensor(i)), static_cast<float>(std::real(tensor_after_fft_ifft(i))));
  }
}

template <int DataLayout>
static void test_2D_fft_ifft_invariant(int dim0, int dim1) {
  Tensor<double, 2, DataLayout> tensor(dim0, dim1);
  tensor.setRandom();

  array<int, 2> fft;
  fft[0] = 0;
  fft[1] = 1;

  Tensor<std::complex<double>, 2, DataLayout> tensor_after_fft;
  Tensor<std::complex<double>, 2, DataLayout> tensor_after_fft_ifft;

  tensor_after_fft = tensor.template fft<Eigen::BothParts, Eigen::FFT_FORWARD>(fft);
  tensor_after_fft_ifft = tensor_after_fft.template fft<Eigen::BothParts, Eigen::FFT_REVERSE>(fft);

  VERIFY_IS_EQUAL(tensor_after_fft.dimension(0), dim0);
  VERIFY_IS_EQUAL(tensor_after_fft.dimension(1), dim1);
  VERIFY_IS_EQUAL(tensor_after_fft_ifft.dimension(0), dim0);
  VERIFY_IS_EQUAL(tensor_after_fft_ifft.dimension(1), dim1);

  for (int i = 0; i < dim0; ++i) {
    for (int j = 0; j < dim1; ++j) {
      //std::cout << "[" << i << "][" << j << "]" <<  "  Original data: " << tensor(i,j) << " Transformed data:" << tensor_after_fft_ifft(i,j) << std::endl;
      VERIFY_IS_APPROX(static_cast<float>(tensor(i,j)), static_cast<float>(std::real(tensor_after_fft_ifft(i,j))));
    }
  }
}

template <int DataLayout>
static void test_3D_fft_ifft_invariant(int dim0, int dim1, int dim2) {
  Tensor<double, 3, DataLayout> tensor(dim0, dim1, dim2);
  tensor.setRandom();

  array<int, 3> fft;
  fft[0] = 0;
  fft[1] = 1;
  fft[2] = 2;

  Tensor<std::complex<double>, 3, DataLayout> tensor_after_fft;
  Tensor<std::complex<double>, 3, DataLayout> tensor_after_fft_ifft;

  tensor_after_fft = tensor.template fft<Eigen::BothParts, Eigen::FFT_FORWARD>(fft);
  tensor_after_fft_ifft = tensor_after_fft.template fft<Eigen::BothParts, Eigen::FFT_REVERSE>(fft);

  VERIFY_IS_EQUAL(tensor_after_fft.dimension(0), dim0);
  VERIFY_IS_EQUAL(tensor_after_fft.dimension(1), dim1);
  VERIFY_IS_EQUAL(tensor_after_fft.dimension(2), dim2);
  VERIFY_IS_EQUAL(tensor_after_fft_ifft.dimension(0), dim0);
  VERIFY_IS_EQUAL(tensor_after_fft_ifft.dimension(1), dim1);
  VERIFY_IS_EQUAL(tensor_after_fft_ifft.dimension(2), dim2);

  for (int i = 0; i < dim0; ++i) {
    for (int j = 0; j < dim1; ++j) {
      for (int k = 0; k < dim2; ++k) {
        VERIFY_IS_APPROX(static_cast<float>(tensor(i,j,k)), static_cast<float>(std::real(tensor_after_fft_ifft(i,j,k))));
      }
    }
  }
}

template <int DataLayout>
static void test_sub_fft_ifft_invariant(int dim0, int dim1, int dim2, int dim3) {
  Tensor<double, 4, DataLayout> tensor(dim0, dim1, dim2, dim3);
  tensor.setRandom();

  array<int, 2> fft;
  fft[0] = 2;
  fft[1] = 0;

  Tensor<std::complex<double>, 4, DataLayout> tensor_after_fft;
  Tensor<double, 4, DataLayout> tensor_after_fft_ifft;

  tensor_after_fft = tensor.template fft<Eigen::BothParts, Eigen::FFT_FORWARD>(fft);
  tensor_after_fft_ifft = tensor_after_fft.template fft<Eigen::RealPart, Eigen::FFT_REVERSE>(fft);

  VERIFY_IS_EQUAL(tensor_after_fft.dimension(0), dim0);
  VERIFY_IS_EQUAL(tensor_after_fft.dimension(1), dim1);
  VERIFY_IS_EQUAL(tensor_after_fft.dimension(2), dim2);
  VERIFY_IS_EQUAL(tensor_after_fft.dimension(3), dim3);
  VERIFY_IS_EQUAL(tensor_after_fft_ifft.dimension(0), dim0);
  VERIFY_IS_EQUAL(tensor_after_fft_ifft.dimension(1), dim1);
  VERIFY_IS_EQUAL(tensor_after_fft_ifft.dimension(2), dim2);
  VERIFY_IS_EQUAL(tensor_after_fft_ifft.dimension(3), dim3);

  for (int i = 0; i < dim0; ++i) {
    for (int j = 0; j < dim1; ++j) {
      for (int k = 0; k < dim2; ++k) {
        for (int l = 0; l < dim3; ++l) {
          VERIFY_IS_APPROX(static_cast<float>(tensor(i,j,k,l)), static_cast<float>(tensor_after_fft_ifft(i,j,k,l)));
        }
      }
    }
  }
}

void test_cxx11_tensor_ifft() {
  CALL_SUBTEST(test_1D_fft_ifft_invariant<ColMajor>(4));
  CALL_SUBTEST(test_1D_fft_ifft_invariant<ColMajor>(16));
  CALL_SUBTEST(test_1D_fft_ifft_invariant<ColMajor>(32));
  CALL_SUBTEST(test_1D_fft_ifft_invariant<ColMajor>(1024*1024));

  CALL_SUBTEST(test_2D_fft_ifft_invariant<ColMajor>(4,4));
  CALL_SUBTEST(test_2D_fft_ifft_invariant<ColMajor>(8,16));
  CALL_SUBTEST(test_2D_fft_ifft_invariant<ColMajor>(16,32));
  CALL_SUBTEST(test_2D_fft_ifft_invariant<ColMajor>(1024,1024));

  CALL_SUBTEST(test_3D_fft_ifft_invariant<ColMajor>(4,4,4));
  CALL_SUBTEST(test_3D_fft_ifft_invariant<ColMajor>(8,16,32));
  CALL_SUBTEST(test_3D_fft_ifft_invariant<ColMajor>(16,4,8));
  CALL_SUBTEST(test_3D_fft_ifft_invariant<ColMajor>(256,256,256));

  CALL_SUBTEST(test_sub_fft_ifft_invariant<ColMajor>(4,4,4,4));
  CALL_SUBTEST(test_sub_fft_ifft_invariant<ColMajor>(8,16,32,64));
  CALL_SUBTEST(test_sub_fft_ifft_invariant<ColMajor>(16,4,8,12));
  CALL_SUBTEST(test_sub_fft_ifft_invariant<ColMajor>(64,64,64,64));
}
