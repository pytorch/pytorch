// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2013 Christian Seiler <christian@iwakd.de>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#include <Eigen/CXX11/Tensor>

using Eigen::Tensor;
using Eigen::RowMajor;

static void test_0d()
{
  Tensor<int, 0> scalar1;
  Tensor<int, 0, RowMajor> scalar2;
  Tensor<int, 0> scalar3;
  Tensor<int, 0, RowMajor> scalar4;

  scalar3.resize();
  scalar4.resize();

  scalar1() = 7;
  scalar2() = 13;
  scalar3.setValues(17);
  scalar4.setZero();

  VERIFY_IS_EQUAL(scalar1.rank(), 0);
  VERIFY_IS_EQUAL(scalar1.size(), 1);

  VERIFY_IS_EQUAL(scalar1(), 7);
  VERIFY_IS_EQUAL(scalar2(), 13);
  VERIFY_IS_EQUAL(scalar3(), 17);
  VERIFY_IS_EQUAL(scalar4(), 0);

  Tensor<int, 0> scalar5(scalar1);

  VERIFY_IS_EQUAL(scalar5(), 7);
  VERIFY_IS_EQUAL(scalar5.data()[0], 7);
}

static void test_1d()
{
  Tensor<int, 1> vec1(6);
  Tensor<int, 1, RowMajor> vec2(6);
  Tensor<int, 1> vec3;
  Tensor<int, 1, RowMajor> vec4;

  vec3.resize(6);
  vec4.resize(6);

  vec1(0) = 4;  vec2(0) = 0; vec3(0) = 5;
  vec1(1) = 8;  vec2(1) = 1; vec3(1) = 4;
  vec1(2) = 15; vec2(2) = 2; vec3(2) = 3;
  vec1(3) = 16; vec2(3) = 3; vec3(3) = 2;
  vec1(4) = 23; vec2(4) = 4; vec3(4) = 1;
  vec1(5) = 42; vec2(5) = 5; vec3(5) = 0;
  vec4.setZero();

  VERIFY_IS_EQUAL((vec1.rank()), 1);
  VERIFY_IS_EQUAL((vec1.size()), 6);
  VERIFY_IS_EQUAL((vec1.dimensions()[0]), 6);

  VERIFY_IS_EQUAL((vec1[0]), 4);
  VERIFY_IS_EQUAL((vec1[1]), 8);
  VERIFY_IS_EQUAL((vec1[2]), 15);
  VERIFY_IS_EQUAL((vec1[3]), 16);
  VERIFY_IS_EQUAL((vec1[4]), 23);
  VERIFY_IS_EQUAL((vec1[5]), 42);

  VERIFY_IS_EQUAL((vec2[0]), 0);
  VERIFY_IS_EQUAL((vec2[1]), 1);
  VERIFY_IS_EQUAL((vec2[2]), 2);
  VERIFY_IS_EQUAL((vec2[3]), 3);
  VERIFY_IS_EQUAL((vec2[4]), 4);
  VERIFY_IS_EQUAL((vec2[5]), 5);

  VERIFY_IS_EQUAL((vec3[0]), 5);
  VERIFY_IS_EQUAL((vec3[1]), 4);
  VERIFY_IS_EQUAL((vec3[2]), 3);
  VERIFY_IS_EQUAL((vec3[3]), 2);
  VERIFY_IS_EQUAL((vec3[4]), 1);
  VERIFY_IS_EQUAL((vec3[5]), 0);

  VERIFY_IS_EQUAL((vec4[0]), 0);
  VERIFY_IS_EQUAL((vec4[1]), 0);
  VERIFY_IS_EQUAL((vec4[2]), 0);
  VERIFY_IS_EQUAL((vec4[3]), 0);
  VERIFY_IS_EQUAL((vec4[4]), 0);
  VERIFY_IS_EQUAL((vec4[5]), 0);

  Tensor<int, 1> vec5(vec1);

  VERIFY_IS_EQUAL((vec5(0)), 4);
  VERIFY_IS_EQUAL((vec5(1)), 8);
  VERIFY_IS_EQUAL((vec5(2)), 15);
  VERIFY_IS_EQUAL((vec5(3)), 16);
  VERIFY_IS_EQUAL((vec5(4)), 23);
  VERIFY_IS_EQUAL((vec5(5)), 42);

  VERIFY_IS_EQUAL((vec5.data()[0]), 4);
  VERIFY_IS_EQUAL((vec5.data()[1]), 8);
  VERIFY_IS_EQUAL((vec5.data()[2]), 15);
  VERIFY_IS_EQUAL((vec5.data()[3]), 16);
  VERIFY_IS_EQUAL((vec5.data()[4]), 23);
  VERIFY_IS_EQUAL((vec5.data()[5]), 42);
}

static void test_2d()
{
  Tensor<int, 2> mat1(2,3);
  Tensor<int, 2, RowMajor> mat2(2,3);

  mat1(0,0) = 0;
  mat1(0,1) = 1;
  mat1(0,2) = 2;
  mat1(1,0) = 3;
  mat1(1,1) = 4;
  mat1(1,2) = 5;

  mat2(0,0) = 0;
  mat2(0,1) = 1;
  mat2(0,2) = 2;
  mat2(1,0) = 3;
  mat2(1,1) = 4;
  mat2(1,2) = 5;

  VERIFY_IS_EQUAL((mat1.rank()), 2);
  VERIFY_IS_EQUAL((mat1.size()), 6);
  VERIFY_IS_EQUAL((mat1.dimensions()[0]), 2);
  VERIFY_IS_EQUAL((mat1.dimensions()[1]), 3);

  VERIFY_IS_EQUAL((mat2.rank()), 2);
  VERIFY_IS_EQUAL((mat2.size()), 6);
  VERIFY_IS_EQUAL((mat2.dimensions()[0]), 2);
  VERIFY_IS_EQUAL((mat2.dimensions()[1]), 3);

  VERIFY_IS_EQUAL((mat1.data()[0]), 0);
  VERIFY_IS_EQUAL((mat1.data()[1]), 3);
  VERIFY_IS_EQUAL((mat1.data()[2]), 1);
  VERIFY_IS_EQUAL((mat1.data()[3]), 4);
  VERIFY_IS_EQUAL((mat1.data()[4]), 2);
  VERIFY_IS_EQUAL((mat1.data()[5]), 5);

  VERIFY_IS_EQUAL((mat2.data()[0]), 0);
  VERIFY_IS_EQUAL((mat2.data()[1]), 1);
  VERIFY_IS_EQUAL((mat2.data()[2]), 2);
  VERIFY_IS_EQUAL((mat2.data()[3]), 3);
  VERIFY_IS_EQUAL((mat2.data()[4]), 4);
  VERIFY_IS_EQUAL((mat2.data()[5]), 5);
}

static void test_3d()
{
  Tensor<int, 3> epsilon(3,3,3);
  epsilon.setZero();
  epsilon(0,1,2) = epsilon(2,0,1) = epsilon(1,2,0) = 1;
  epsilon(2,1,0) = epsilon(0,2,1) = epsilon(1,0,2) = -1;

  VERIFY_IS_EQUAL((epsilon.size()), 27);
  VERIFY_IS_EQUAL((epsilon.dimensions()[0]), 3);
  VERIFY_IS_EQUAL((epsilon.dimensions()[1]), 3);
  VERIFY_IS_EQUAL((epsilon.dimensions()[2]), 3);

  VERIFY_IS_EQUAL((epsilon(0,0,0)), 0);
  VERIFY_IS_EQUAL((epsilon(0,0,1)), 0);
  VERIFY_IS_EQUAL((epsilon(0,0,2)), 0);
  VERIFY_IS_EQUAL((epsilon(0,1,0)), 0);
  VERIFY_IS_EQUAL((epsilon(0,1,1)), 0);
  VERIFY_IS_EQUAL((epsilon(0,2,0)), 0);
  VERIFY_IS_EQUAL((epsilon(0,2,2)), 0);
  VERIFY_IS_EQUAL((epsilon(1,0,0)), 0);
  VERIFY_IS_EQUAL((epsilon(1,0,1)), 0);
  VERIFY_IS_EQUAL((epsilon(1,1,0)), 0);
  VERIFY_IS_EQUAL((epsilon(1,1,1)), 0);
  VERIFY_IS_EQUAL((epsilon(1,1,2)), 0);
  VERIFY_IS_EQUAL((epsilon(1,2,1)), 0);
  VERIFY_IS_EQUAL((epsilon(1,2,2)), 0);
  VERIFY_IS_EQUAL((epsilon(2,0,0)), 0);
  VERIFY_IS_EQUAL((epsilon(2,0,2)), 0);
  VERIFY_IS_EQUAL((epsilon(2,1,1)), 0);
  VERIFY_IS_EQUAL((epsilon(2,1,2)), 0);
  VERIFY_IS_EQUAL((epsilon(2,2,0)), 0);
  VERIFY_IS_EQUAL((epsilon(2,2,1)), 0);
  VERIFY_IS_EQUAL((epsilon(2,2,2)), 0);

  VERIFY_IS_EQUAL((epsilon(0,1,2)), 1);
  VERIFY_IS_EQUAL((epsilon(2,0,1)), 1);
  VERIFY_IS_EQUAL((epsilon(1,2,0)), 1);
  VERIFY_IS_EQUAL((epsilon(2,1,0)), -1);
  VERIFY_IS_EQUAL((epsilon(0,2,1)), -1);
  VERIFY_IS_EQUAL((epsilon(1,0,2)), -1);

  array<Eigen::DenseIndex, 3> dims{{2,3,4}};
  Tensor<int, 3> t1(dims);
  Tensor<int, 3, RowMajor> t2(dims);

  VERIFY_IS_EQUAL((t1.size()), 24);
  VERIFY_IS_EQUAL((t1.dimensions()[0]), 2);
  VERIFY_IS_EQUAL((t1.dimensions()[1]), 3);
  VERIFY_IS_EQUAL((t1.dimensions()[2]), 4);

  VERIFY_IS_EQUAL((t2.size()), 24);
  VERIFY_IS_EQUAL((t2.dimensions()[0]), 2);
  VERIFY_IS_EQUAL((t2.dimensions()[1]), 3);
  VERIFY_IS_EQUAL((t2.dimensions()[2]), 4);

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 4; k++) {
        t1(i, j, k) = 100 * i + 10 * j + k;
        t2(i, j, k) = 100 * i + 10 * j + k;
      }
    }
  }

  VERIFY_IS_EQUAL((t1.data()[0]),    0);
  VERIFY_IS_EQUAL((t1.data()[1]),  100);
  VERIFY_IS_EQUAL((t1.data()[2]),   10);
  VERIFY_IS_EQUAL((t1.data()[3]),  110);
  VERIFY_IS_EQUAL((t1.data()[4]),   20);
  VERIFY_IS_EQUAL((t1.data()[5]),  120);
  VERIFY_IS_EQUAL((t1.data()[6]),    1);
  VERIFY_IS_EQUAL((t1.data()[7]),  101);
  VERIFY_IS_EQUAL((t1.data()[8]),   11);
  VERIFY_IS_EQUAL((t1.data()[9]),  111);
  VERIFY_IS_EQUAL((t1.data()[10]),  21);
  VERIFY_IS_EQUAL((t1.data()[11]), 121);
  VERIFY_IS_EQUAL((t1.data()[12]),   2);
  VERIFY_IS_EQUAL((t1.data()[13]), 102);
  VERIFY_IS_EQUAL((t1.data()[14]),  12);
  VERIFY_IS_EQUAL((t1.data()[15]), 112);
  VERIFY_IS_EQUAL((t1.data()[16]),  22);
  VERIFY_IS_EQUAL((t1.data()[17]), 122);
  VERIFY_IS_EQUAL((t1.data()[18]),   3);
  VERIFY_IS_EQUAL((t1.data()[19]), 103);
  VERIFY_IS_EQUAL((t1.data()[20]),  13);
  VERIFY_IS_EQUAL((t1.data()[21]), 113);
  VERIFY_IS_EQUAL((t1.data()[22]),  23);
  VERIFY_IS_EQUAL((t1.data()[23]), 123);

  VERIFY_IS_EQUAL((t2.data()[0]),    0);
  VERIFY_IS_EQUAL((t2.data()[1]),    1);
  VERIFY_IS_EQUAL((t2.data()[2]),    2);
  VERIFY_IS_EQUAL((t2.data()[3]),    3);
  VERIFY_IS_EQUAL((t2.data()[4]),   10);
  VERIFY_IS_EQUAL((t2.data()[5]),   11);
  VERIFY_IS_EQUAL((t2.data()[6]),   12);
  VERIFY_IS_EQUAL((t2.data()[7]),   13);
  VERIFY_IS_EQUAL((t2.data()[8]),   20);
  VERIFY_IS_EQUAL((t2.data()[9]),   21);
  VERIFY_IS_EQUAL((t2.data()[10]),  22);
  VERIFY_IS_EQUAL((t2.data()[11]),  23);
  VERIFY_IS_EQUAL((t2.data()[12]), 100);
  VERIFY_IS_EQUAL((t2.data()[13]), 101);
  VERIFY_IS_EQUAL((t2.data()[14]), 102);
  VERIFY_IS_EQUAL((t2.data()[15]), 103);
  VERIFY_IS_EQUAL((t2.data()[16]), 110);
  VERIFY_IS_EQUAL((t2.data()[17]), 111);
  VERIFY_IS_EQUAL((t2.data()[18]), 112);
  VERIFY_IS_EQUAL((t2.data()[19]), 113);
  VERIFY_IS_EQUAL((t2.data()[20]), 120);
  VERIFY_IS_EQUAL((t2.data()[21]), 121);
  VERIFY_IS_EQUAL((t2.data()[22]), 122);
  VERIFY_IS_EQUAL((t2.data()[23]), 123);
}

static void test_simple_assign()
{
  Tensor<int, 3> epsilon(3,3,3);
  epsilon.setZero();
  epsilon(0,1,2) = epsilon(2,0,1) = epsilon(1,2,0) = 1;
  epsilon(2,1,0) = epsilon(0,2,1) = epsilon(1,0,2) = -1;

  Tensor<int, 3> e2(3,3,3);
  e2.setZero();
  VERIFY_IS_EQUAL((e2(1,2,0)), 0);

  e2 = epsilon;
  VERIFY_IS_EQUAL((e2(1,2,0)), 1);
  VERIFY_IS_EQUAL((e2(0,1,2)), 1);
  VERIFY_IS_EQUAL((e2(2,0,1)), 1);
  VERIFY_IS_EQUAL((e2(2,1,0)), -1);
  VERIFY_IS_EQUAL((e2(0,2,1)), -1);
  VERIFY_IS_EQUAL((e2(1,0,2)), -1);
}

static void test_resize()
{
  Tensor<int, 3> epsilon;
  epsilon.resize(2,3,7);
  VERIFY_IS_EQUAL(epsilon.dimension(0), 2);
  VERIFY_IS_EQUAL(epsilon.dimension(1), 3);
  VERIFY_IS_EQUAL(epsilon.dimension(2), 7);
  VERIFY_IS_EQUAL(epsilon.dimensions().TotalSize(), 2*3*7);

  const int* old_data = epsilon.data();
  epsilon.resize(3,2,7);
  VERIFY_IS_EQUAL(epsilon.dimension(0), 3);
  VERIFY_IS_EQUAL(epsilon.dimension(1), 2);
  VERIFY_IS_EQUAL(epsilon.dimension(2), 7);
  VERIFY_IS_EQUAL(epsilon.dimensions().TotalSize(), 2*3*7);
  VERIFY_IS_EQUAL(epsilon.data(), old_data);

  epsilon.resize(3,5,7);
  VERIFY_IS_EQUAL(epsilon.dimension(0), 3);
  VERIFY_IS_EQUAL(epsilon.dimension(1), 5);
  VERIFY_IS_EQUAL(epsilon.dimension(2), 7);
  VERIFY_IS_EQUAL(epsilon.dimensions().TotalSize(), 3*5*7);
  VERIFY_IS_NOT_EQUAL(epsilon.data(), old_data);
}

void test_cxx11_tensor_simple()
{
  CALL_SUBTEST(test_0d());
  CALL_SUBTEST(test_1d());
  CALL_SUBTEST(test_2d());
  CALL_SUBTEST(test_3d());
  CALL_SUBTEST(test_simple_assign());
  CALL_SUBTEST(test_resize());
}
