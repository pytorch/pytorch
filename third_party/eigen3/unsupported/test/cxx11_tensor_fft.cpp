// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Jianwei Cui <thucjw@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <Eigen/CXX11/Tensor>

using Eigen::Tensor;

template <int DataLayout>
static void test_fft_2D_golden() {
  Tensor<float, 2, DataLayout, long> input(2, 3);
  input(0, 0) = 1;
  input(0, 1) = 2;
  input(0, 2) = 3;
  input(1, 0) = 4;
  input(1, 1) = 5;
  input(1, 2) = 6;

  array<int, 2> fft;
  fft[0] = 0;
  fft[1] = 1;

  Tensor<std::complex<float>, 2, DataLayout, long> output = input.template fft<Eigen::BothParts, Eigen::FFT_FORWARD>(fft);

  std::complex<float> output_golden[6]; // in ColMajor order
  output_golden[0] = std::complex<float>(21, 0);
  output_golden[1] = std::complex<float>(-9, 0);
  output_golden[2] = std::complex<float>(-3, 1.73205);
  output_golden[3] = std::complex<float>( 0, 0);
  output_golden[4] = std::complex<float>(-3, -1.73205);
  output_golden[5] = std::complex<float>(0 ,0);

  std::complex<float> c_offset = std::complex<float>(1.0, 1.0);

  if (DataLayout == ColMajor) {
    VERIFY_IS_APPROX(output(0) + c_offset, output_golden[0] + c_offset);
    VERIFY_IS_APPROX(output(1) + c_offset, output_golden[1] + c_offset);
    VERIFY_IS_APPROX(output(2) + c_offset, output_golden[2] + c_offset);
    VERIFY_IS_APPROX(output(3) + c_offset, output_golden[3] + c_offset);
    VERIFY_IS_APPROX(output(4) + c_offset, output_golden[4] + c_offset);
    VERIFY_IS_APPROX(output(5) + c_offset, output_golden[5] + c_offset);
  }
  else {
    VERIFY_IS_APPROX(output(0)+ c_offset, output_golden[0]+ c_offset);
    VERIFY_IS_APPROX(output(1)+ c_offset, output_golden[2]+ c_offset);
    VERIFY_IS_APPROX(output(2)+ c_offset, output_golden[4]+ c_offset);
    VERIFY_IS_APPROX(output(3)+ c_offset, output_golden[1]+ c_offset);
    VERIFY_IS_APPROX(output(4)+ c_offset, output_golden[3]+ c_offset);
    VERIFY_IS_APPROX(output(5)+ c_offset, output_golden[5]+ c_offset);
  }
}

static void test_fft_complex_input_golden() {
  Tensor<std::complex<float>, 1, ColMajor, long> input(5);
  input(0) = std::complex<float>(1, 1);
  input(1) = std::complex<float>(2, 2);
  input(2) = std::complex<float>(3, 3);
  input(3) = std::complex<float>(4, 4);
  input(4) = std::complex<float>(5, 5);

  array<int, 1> fft;
  fft[0] = 0;

  Tensor<std::complex<float>, 1, ColMajor, long> forward_output_both_parts = input.fft<BothParts, FFT_FORWARD>(fft);
  Tensor<std::complex<float>, 1, ColMajor, long> reverse_output_both_parts = input.fft<BothParts, FFT_REVERSE>(fft);

  Tensor<float, 1, ColMajor, long> forward_output_real_part = input.fft<RealPart, FFT_FORWARD>(fft);
  Tensor<float, 1, ColMajor, long> reverse_output_real_part = input.fft<RealPart, FFT_REVERSE>(fft);

  Tensor<float, 1, ColMajor, long> forward_output_imag_part = input.fft<ImagPart, FFT_FORWARD>(fft);
  Tensor<float, 1, ColMajor, long> reverse_output_imag_part = input.fft<ImagPart, FFT_REVERSE>(fft);

  VERIFY_IS_EQUAL(forward_output_both_parts.dimension(0), input.dimension(0));
  VERIFY_IS_EQUAL(reverse_output_both_parts.dimension(0), input.dimension(0));

  VERIFY_IS_EQUAL(forward_output_real_part.dimension(0), input.dimension(0));
  VERIFY_IS_EQUAL(reverse_output_real_part.dimension(0), input.dimension(0));

  VERIFY_IS_EQUAL(forward_output_imag_part.dimension(0), input.dimension(0));
  VERIFY_IS_EQUAL(reverse_output_imag_part.dimension(0), input.dimension(0));

  std::complex<float> forward_golden_result[5];
  std::complex<float> reverse_golden_result[5];

  forward_golden_result[0] = std::complex<float>(15.000000000000000,+15.000000000000000);
  forward_golden_result[1] = std::complex<float>(-5.940954801177935, +0.940954801177934);
  forward_golden_result[2] = std::complex<float>(-3.312299240582266, -1.687700759417735);
  forward_golden_result[3] = std::complex<float>(-1.687700759417735, -3.312299240582266);
  forward_golden_result[4] = std::complex<float>( 0.940954801177934, -5.940954801177935);

  reverse_golden_result[0] = std::complex<float>( 3.000000000000000, + 3.000000000000000);
  reverse_golden_result[1] = std::complex<float>( 0.188190960235587, - 1.188190960235587);
  reverse_golden_result[2] = std::complex<float>(-0.337540151883547, - 0.662459848116453);
  reverse_golden_result[3] = std::complex<float>(-0.662459848116453, - 0.337540151883547);
  reverse_golden_result[4] = std::complex<float>(-1.188190960235587, + 0.188190960235587);

  for(int i = 0; i < 5; ++i) {
    VERIFY_IS_APPROX(forward_output_both_parts(i), forward_golden_result[i]);
    VERIFY_IS_APPROX(forward_output_real_part(i), forward_golden_result[i].real());
    VERIFY_IS_APPROX(forward_output_imag_part(i), forward_golden_result[i].imag());
  }

  for(int i = 0; i < 5; ++i) {
    VERIFY_IS_APPROX(reverse_output_both_parts(i), reverse_golden_result[i]);
    VERIFY_IS_APPROX(reverse_output_real_part(i), reverse_golden_result[i].real());
    VERIFY_IS_APPROX(reverse_output_imag_part(i), reverse_golden_result[i].imag());
  }
}

static void test_fft_real_input_golden() {
  Tensor<float, 1, ColMajor, long> input(5);
  input(0) = 1.0;
  input(1) = 2.0;
  input(2) = 3.0;
  input(3) = 4.0;
  input(4) = 5.0;

  array<int, 1> fft;
  fft[0] = 0;

  Tensor<std::complex<float>, 1, ColMajor, long> forward_output_both_parts = input.fft<BothParts, FFT_FORWARD>(fft);
  Tensor<std::complex<float>, 1, ColMajor, long> reverse_output_both_parts = input.fft<BothParts, FFT_REVERSE>(fft);

  Tensor<float, 1, ColMajor, long> forward_output_real_part = input.fft<RealPart, FFT_FORWARD>(fft);
  Tensor<float, 1, ColMajor, long> reverse_output_real_part = input.fft<RealPart, FFT_REVERSE>(fft);

  Tensor<float, 1, ColMajor, long> forward_output_imag_part = input.fft<ImagPart, FFT_FORWARD>(fft);
  Tensor<float, 1, ColMajor, long> reverse_output_imag_part = input.fft<ImagPart, FFT_REVERSE>(fft);

  VERIFY_IS_EQUAL(forward_output_both_parts.dimension(0), input.dimension(0));
  VERIFY_IS_EQUAL(reverse_output_both_parts.dimension(0), input.dimension(0));

  VERIFY_IS_EQUAL(forward_output_real_part.dimension(0), input.dimension(0));
  VERIFY_IS_EQUAL(reverse_output_real_part.dimension(0), input.dimension(0));

  VERIFY_IS_EQUAL(forward_output_imag_part.dimension(0), input.dimension(0));
  VERIFY_IS_EQUAL(reverse_output_imag_part.dimension(0), input.dimension(0));

  std::complex<float> forward_golden_result[5];
  std::complex<float> reverse_golden_result[5];


  forward_golden_result[0] = std::complex<float>(  15, 0);
  forward_golden_result[1] = std::complex<float>(-2.5, +3.44095480117793);
  forward_golden_result[2] = std::complex<float>(-2.5, +0.81229924058227);
  forward_golden_result[3] = std::complex<float>(-2.5, -0.81229924058227);
  forward_golden_result[4] = std::complex<float>(-2.5, -3.44095480117793);

  reverse_golden_result[0] = std::complex<float>( 3.0, 0);
  reverse_golden_result[1] = std::complex<float>(-0.5, -0.688190960235587);
  reverse_golden_result[2] = std::complex<float>(-0.5, -0.162459848116453);
  reverse_golden_result[3] = std::complex<float>(-0.5, +0.162459848116453);
  reverse_golden_result[4] = std::complex<float>(-0.5, +0.688190960235587);

  std::complex<float> c_offset(1.0, 1.0);
  float r_offset = 1.0;

  for(int i = 0; i < 5; ++i) {
    VERIFY_IS_APPROX(forward_output_both_parts(i) + c_offset, forward_golden_result[i] + c_offset);
    VERIFY_IS_APPROX(forward_output_real_part(i)  + r_offset, forward_golden_result[i].real() + r_offset);
    VERIFY_IS_APPROX(forward_output_imag_part(i)  + r_offset, forward_golden_result[i].imag() + r_offset);
  }

  for(int i = 0; i < 5; ++i) {
    VERIFY_IS_APPROX(reverse_output_both_parts(i) + c_offset, reverse_golden_result[i] + c_offset);
    VERIFY_IS_APPROX(reverse_output_real_part(i)  + r_offset, reverse_golden_result[i].real() + r_offset);
    VERIFY_IS_APPROX(reverse_output_imag_part(i)  + r_offset, reverse_golden_result[i].imag() + r_offset);
  }
}


template <int DataLayout, typename RealScalar, bool isComplexInput, int FFTResultType, int FFTDirection, int TensorRank>
static void test_fft_real_input_energy() {

  Eigen::DSizes<long, TensorRank> dimensions;
  int total_size = 1;
  for (int i = 0; i < TensorRank; ++i) {
    dimensions[i] = rand() % 20 + 1;
    total_size *= dimensions[i];
  }
  const DSizes<long, TensorRank> arr = dimensions;

  typedef typename internal::conditional<isComplexInput == true, std::complex<RealScalar>, RealScalar>::type InputScalar;

  Tensor<InputScalar, TensorRank, DataLayout, long> input;
  input.resize(arr);
  input.setRandom();

  array<int, TensorRank> fft;
  for (int i = 0; i < TensorRank; ++i) {
    fft[i] = i;
  }

  typedef typename internal::conditional<FFTResultType == Eigen::BothParts, std::complex<RealScalar>, RealScalar>::type OutputScalar;
  Tensor<OutputScalar, TensorRank, DataLayout> output;
  output = input.template fft<FFTResultType, FFTDirection>(fft);

  for (int i = 0; i < TensorRank; ++i) {
    VERIFY_IS_EQUAL(output.dimension(i), input.dimension(i));
  }

  float energy_original = 0.0;
  float energy_after_fft = 0.0;

  for (int i = 0; i < total_size; ++i) {
    energy_original += pow(std::abs(input(i)), 2);
  }

  for (int i = 0; i < total_size; ++i) {
    energy_after_fft += pow(std::abs(output(i)), 2);
  }

  if(FFTDirection == FFT_FORWARD) {
    VERIFY_IS_APPROX(energy_original, energy_after_fft / total_size);
  }
  else {
    VERIFY_IS_APPROX(energy_original, energy_after_fft * total_size);
  }
}

void test_cxx11_tensor_fft() {
    test_fft_complex_input_golden();
    test_fft_real_input_golden();

    test_fft_2D_golden<ColMajor>();
    test_fft_2D_golden<RowMajor>();

    test_fft_real_input_energy<ColMajor, float,  true,  Eigen::BothParts, FFT_FORWARD, 1>();
    test_fft_real_input_energy<ColMajor, double, true,  Eigen::BothParts, FFT_FORWARD, 1>();
    test_fft_real_input_energy<ColMajor, float,  false,  Eigen::BothParts, FFT_FORWARD, 1>();
    test_fft_real_input_energy<ColMajor, double, false,  Eigen::BothParts, FFT_FORWARD, 1>();

    test_fft_real_input_energy<ColMajor, float,  true,  Eigen::BothParts, FFT_FORWARD, 2>();
    test_fft_real_input_energy<ColMajor, double, true,  Eigen::BothParts, FFT_FORWARD, 2>();
    test_fft_real_input_energy<ColMajor, float,  false,  Eigen::BothParts, FFT_FORWARD, 2>();
    test_fft_real_input_energy<ColMajor, double, false,  Eigen::BothParts, FFT_FORWARD, 2>();

    test_fft_real_input_energy<ColMajor, float,  true,  Eigen::BothParts, FFT_FORWARD, 3>();
    test_fft_real_input_energy<ColMajor, double, true,  Eigen::BothParts, FFT_FORWARD, 3>();
    test_fft_real_input_energy<ColMajor, float,  false,  Eigen::BothParts, FFT_FORWARD, 3>();
    test_fft_real_input_energy<ColMajor, double, false,  Eigen::BothParts, FFT_FORWARD, 3>();

    test_fft_real_input_energy<ColMajor, float,  true,  Eigen::BothParts, FFT_FORWARD, 4>();
    test_fft_real_input_energy<ColMajor, double, true,  Eigen::BothParts, FFT_FORWARD, 4>();
    test_fft_real_input_energy<ColMajor, float,  false,  Eigen::BothParts, FFT_FORWARD, 4>();
    test_fft_real_input_energy<ColMajor, double, false,  Eigen::BothParts, FFT_FORWARD, 4>();

    test_fft_real_input_energy<RowMajor, float,  true,  Eigen::BothParts, FFT_FORWARD, 1>();
    test_fft_real_input_energy<RowMajor, double, true,  Eigen::BothParts, FFT_FORWARD, 1>();
    test_fft_real_input_energy<RowMajor, float,  false,  Eigen::BothParts, FFT_FORWARD, 1>();
    test_fft_real_input_energy<RowMajor, double, false,  Eigen::BothParts, FFT_FORWARD, 1>();

    test_fft_real_input_energy<RowMajor, float,  true,  Eigen::BothParts, FFT_FORWARD, 2>();
    test_fft_real_input_energy<RowMajor, double, true,  Eigen::BothParts, FFT_FORWARD, 2>();
    test_fft_real_input_energy<RowMajor, float,  false,  Eigen::BothParts, FFT_FORWARD, 2>();
    test_fft_real_input_energy<RowMajor, double, false,  Eigen::BothParts, FFT_FORWARD, 2>();

    test_fft_real_input_energy<RowMajor, float,  true,  Eigen::BothParts, FFT_FORWARD, 3>();
    test_fft_real_input_energy<RowMajor, double, true,  Eigen::BothParts, FFT_FORWARD, 3>();
    test_fft_real_input_energy<RowMajor, float,  false,  Eigen::BothParts, FFT_FORWARD, 3>();
    test_fft_real_input_energy<RowMajor, double, false,  Eigen::BothParts, FFT_FORWARD, 3>();

    test_fft_real_input_energy<RowMajor, float,  true,  Eigen::BothParts, FFT_FORWARD, 4>();
    test_fft_real_input_energy<RowMajor, double, true,  Eigen::BothParts, FFT_FORWARD, 4>();
    test_fft_real_input_energy<RowMajor, float,  false,  Eigen::BothParts, FFT_FORWARD, 4>();
    test_fft_real_input_energy<RowMajor, double, false,  Eigen::BothParts, FFT_FORWARD, 4>();
}
