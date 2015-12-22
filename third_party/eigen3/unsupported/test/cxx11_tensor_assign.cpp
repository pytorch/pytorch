// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#include <Eigen/CXX11/Tensor>

using Eigen::Tensor;
using Eigen::RowMajor;

static void test_1d()
{
  Tensor<int, 1> vec1(6);
  Tensor<int, 1, RowMajor> vec2(6);
  vec1(0) = 4;  vec2(0) = 0;
  vec1(1) = 8;  vec2(1) = 1;
  vec1(2) = 15; vec2(2) = 2;
  vec1(3) = 16; vec2(3) = 3;
  vec1(4) = 23; vec2(4) = 4;
  vec1(5) = 42; vec2(5) = 5;

  int col_major[6];
  int row_major[6];
  memset(col_major, 0, 6*sizeof(int));
  memset(row_major, 0, 6*sizeof(int));
  TensorMap<Tensor<int, 1> > vec3(col_major, 6);
  TensorMap<Tensor<int, 1, RowMajor> > vec4(row_major, 6);

  vec3 = vec1;
  vec4 = vec2;

  VERIFY_IS_EQUAL(vec3(0), 4);
  VERIFY_IS_EQUAL(vec3(1), 8);
  VERIFY_IS_EQUAL(vec3(2), 15);
  VERIFY_IS_EQUAL(vec3(3), 16);
  VERIFY_IS_EQUAL(vec3(4), 23);
  VERIFY_IS_EQUAL(vec3(5), 42);

  VERIFY_IS_EQUAL(vec4(0), 0);
  VERIFY_IS_EQUAL(vec4(1), 1);
  VERIFY_IS_EQUAL(vec4(2), 2);
  VERIFY_IS_EQUAL(vec4(3), 3);
  VERIFY_IS_EQUAL(vec4(4), 4);
  VERIFY_IS_EQUAL(vec4(5), 5);

  vec1.setZero();
  vec2.setZero();
  vec1 = vec3;
  vec2 = vec4;

  VERIFY_IS_EQUAL(vec1(0), 4);
  VERIFY_IS_EQUAL(vec1(1), 8);
  VERIFY_IS_EQUAL(vec1(2), 15);
  VERIFY_IS_EQUAL(vec1(3), 16);
  VERIFY_IS_EQUAL(vec1(4), 23);
  VERIFY_IS_EQUAL(vec1(5), 42);

  VERIFY_IS_EQUAL(vec2(0), 0);
  VERIFY_IS_EQUAL(vec2(1), 1);
  VERIFY_IS_EQUAL(vec2(2), 2);
  VERIFY_IS_EQUAL(vec2(3), 3);
  VERIFY_IS_EQUAL(vec2(4), 4);
  VERIFY_IS_EQUAL(vec2(5), 5);
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

  int col_major[6];
  int row_major[6];
  memset(col_major, 0, 6*sizeof(int));
  memset(row_major, 0, 6*sizeof(int));
  TensorMap<Tensor<int, 2> > mat3(row_major, 2, 3);
  TensorMap<Tensor<int, 2, RowMajor> > mat4(col_major, 2, 3);

  mat3 = mat1;
  mat4 = mat2;

  VERIFY_IS_EQUAL(mat3(0,0), 0);
  VERIFY_IS_EQUAL(mat3(0,1), 1);
  VERIFY_IS_EQUAL(mat3(0,2), 2);
  VERIFY_IS_EQUAL(mat3(1,0), 3);
  VERIFY_IS_EQUAL(mat3(1,1), 4);
  VERIFY_IS_EQUAL(mat3(1,2), 5);

  VERIFY_IS_EQUAL(mat4(0,0), 0);
  VERIFY_IS_EQUAL(mat4(0,1), 1);
  VERIFY_IS_EQUAL(mat4(0,2), 2);
  VERIFY_IS_EQUAL(mat4(1,0), 3);
  VERIFY_IS_EQUAL(mat4(1,1), 4);
  VERIFY_IS_EQUAL(mat4(1,2), 5);

  mat1.setZero();
  mat2.setZero();
  mat1 = mat3;
  mat2 = mat4;

  VERIFY_IS_EQUAL(mat1(0,0), 0);
  VERIFY_IS_EQUAL(mat1(0,1), 1);
  VERIFY_IS_EQUAL(mat1(0,2), 2);
  VERIFY_IS_EQUAL(mat1(1,0), 3);
  VERIFY_IS_EQUAL(mat1(1,1), 4);
  VERIFY_IS_EQUAL(mat1(1,2), 5);

  VERIFY_IS_EQUAL(mat2(0,0), 0);
  VERIFY_IS_EQUAL(mat2(0,1), 1);
  VERIFY_IS_EQUAL(mat2(0,2), 2);
  VERIFY_IS_EQUAL(mat2(1,0), 3);
  VERIFY_IS_EQUAL(mat2(1,1), 4);
  VERIFY_IS_EQUAL(mat2(1,2), 5);
}

static void test_3d()
{
  Tensor<int, 3> mat1(2,3,7);
  Tensor<int, 3, RowMajor> mat2(2,3,7);

  int val = 0;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 7; ++k) {
        mat1(i,j,k) = val;
        mat2(i,j,k) = val;
        val++;
      }
    }
  }

  int col_major[2*3*7];
  int row_major[2*3*7];
  memset(col_major, 0, 2*3*7*sizeof(int));
  memset(row_major, 0, 2*3*7*sizeof(int));
  TensorMap<Tensor<int, 3> > mat3(col_major, 2, 3, 7);
  TensorMap<Tensor<int, 3, RowMajor> > mat4(row_major, 2, 3, 7);

  mat3 = mat1;
  mat4 = mat2;

  val = 0;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 7; ++k) {
        VERIFY_IS_EQUAL(mat3(i,j,k), val);
        VERIFY_IS_EQUAL(mat4(i,j,k), val);
        val++;
      }
    }
  }

  mat1.setZero();
  mat2.setZero();
  mat1 = mat3;
  mat2 = mat4;

  val = 0;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 7; ++k) {
        VERIFY_IS_EQUAL(mat1(i,j,k), val);
        VERIFY_IS_EQUAL(mat2(i,j,k), val);
        val++;
      }
    }
  }
}

static void test_same_type()
{
  Tensor<int, 1> orig_tensor(5);
  Tensor<int, 1> dest_tensor(5);
  orig_tensor.setRandom();
  dest_tensor.setRandom();
  int* orig_data = orig_tensor.data();
  int* dest_data = dest_tensor.data();
  dest_tensor = orig_tensor;
  VERIFY_IS_EQUAL(orig_tensor.data(), orig_data);
  VERIFY_IS_EQUAL(dest_tensor.data(), dest_data);
  for (int i = 0; i < 5; ++i) {
    VERIFY_IS_EQUAL(dest_tensor(i), orig_tensor(i));
  }

  TensorFixedSize<int, Sizes<5> > orig_array;
  TensorFixedSize<int, Sizes<5> > dest_array;
  orig_array.setRandom();
  dest_array.setRandom();
  orig_data = orig_array.data();
  dest_data = dest_array.data();
  dest_array = orig_array;
  VERIFY_IS_EQUAL(orig_array.data(), orig_data);
  VERIFY_IS_EQUAL(dest_array.data(), dest_data);
  for (int i = 0; i < 5; ++i) {
    VERIFY_IS_EQUAL(dest_array(i), orig_array(i));
  }

  int orig[5] = {1, 2, 3, 4, 5};
  int dest[5] = {6, 7, 8, 9, 10};
  TensorMap<Tensor<int, 1> > orig_map(orig, 5);
  TensorMap<Tensor<int, 1> > dest_map(dest, 5);
  orig_data = orig_map.data();
  dest_data = dest_map.data();
  dest_map = orig_map;
  VERIFY_IS_EQUAL(orig_map.data(), orig_data);
  VERIFY_IS_EQUAL(dest_map.data(), dest_data);
  for (int i = 0; i < 5; ++i) {
    VERIFY_IS_EQUAL(dest[i], i+1);
  }
}

static void test_auto_resize()
{
  Tensor<int, 1> tensor1;
  Tensor<int, 1> tensor2(3);
  Tensor<int, 1> tensor3(5);
  Tensor<int, 1> tensor4(7);

  Tensor<int, 1> new_tensor(5);
  new_tensor.setRandom();

  tensor1 = tensor2 = tensor3 = tensor4 = new_tensor;

  VERIFY_IS_EQUAL(tensor1.dimension(0), new_tensor.dimension(0));
  VERIFY_IS_EQUAL(tensor2.dimension(0), new_tensor.dimension(0));
  VERIFY_IS_EQUAL(tensor3.dimension(0), new_tensor.dimension(0));
  VERIFY_IS_EQUAL(tensor4.dimension(0), new_tensor.dimension(0));
  for (int i = 0; i < new_tensor.dimension(0); ++i) {
    VERIFY_IS_EQUAL(tensor1(i), new_tensor(i));
    VERIFY_IS_EQUAL(tensor2(i), new_tensor(i));
    VERIFY_IS_EQUAL(tensor3(i), new_tensor(i));
    VERIFY_IS_EQUAL(tensor4(i), new_tensor(i));
  }
}


static void test_compound_assign()
{
  Tensor<int, 1> start_tensor(10);
  Tensor<int, 1> offset_tensor(10);
  start_tensor.setRandom();
  offset_tensor.setRandom();

  Tensor<int, 1> tensor = start_tensor;
  tensor += offset_tensor;
  for (int i = 0; i < 10; ++i) {
    VERIFY_IS_EQUAL(tensor(i), start_tensor(i) + offset_tensor(i));
  }

  tensor = start_tensor;
  tensor -= offset_tensor;
  for (int i = 0; i < 10; ++i) {
    VERIFY_IS_EQUAL(tensor(i), start_tensor(i) - offset_tensor(i));
  }

  tensor = start_tensor;
  tensor *= offset_tensor;
  for (int i = 0; i < 10; ++i) {
    VERIFY_IS_EQUAL(tensor(i), start_tensor(i) * offset_tensor(i));
  }

  tensor = start_tensor;
  tensor /= offset_tensor;
  for (int i = 0; i < 10; ++i) {
    VERIFY_IS_EQUAL(tensor(i), start_tensor(i) / offset_tensor(i));
  }
}

static void test_std_initializers_tensor() {
#ifdef EIGEN_HAS_VARIADIC_TEMPLATES
  Tensor<int, 1> a(3);
  a.setValues({0, 1, 2});
  VERIFY_IS_EQUAL(a(0), 0);
  VERIFY_IS_EQUAL(a(1), 1);
  VERIFY_IS_EQUAL(a(2), 2);

  // It fills the top-left slice.
  a.setValues({10, 20});
  VERIFY_IS_EQUAL(a(0), 10);
  VERIFY_IS_EQUAL(a(1), 20);
  VERIFY_IS_EQUAL(a(2), 2);

  // Chaining.
  Tensor<int, 1> a2(3);
  a2 = a.setValues({100, 200, 300});
  VERIFY_IS_EQUAL(a(0), 100);
  VERIFY_IS_EQUAL(a(1), 200);
  VERIFY_IS_EQUAL(a(2), 300);
  VERIFY_IS_EQUAL(a2(0), 100);
  VERIFY_IS_EQUAL(a2(1), 200);
  VERIFY_IS_EQUAL(a2(2), 300);

  Tensor<int, 2> b(2, 3);
  b.setValues({{0, 1, 2}, {3, 4, 5}});
  VERIFY_IS_EQUAL(b(0, 0), 0);
  VERIFY_IS_EQUAL(b(0, 1), 1);
  VERIFY_IS_EQUAL(b(0, 2), 2);
  VERIFY_IS_EQUAL(b(1, 0), 3);
  VERIFY_IS_EQUAL(b(1, 1), 4);
  VERIFY_IS_EQUAL(b(1, 2), 5);

  // It fills the top-left slice.
  b.setValues({{10, 20}, {30}});
  VERIFY_IS_EQUAL(b(0, 0), 10);
  VERIFY_IS_EQUAL(b(0, 1), 20);
  VERIFY_IS_EQUAL(b(0, 2), 2);
  VERIFY_IS_EQUAL(b(1, 0), 30);
  VERIFY_IS_EQUAL(b(1, 1), 4);
  VERIFY_IS_EQUAL(b(1, 2), 5);

  Eigen::Tensor<int, 3> c(3, 2, 4);
  c.setValues({{{0, 1, 2, 3}, {4, 5, 6, 7}},
               {{10, 11, 12, 13}, {14, 15, 16, 17}},
               {{20, 21, 22, 23}, {24, 25, 26, 27}}});
  VERIFY_IS_EQUAL(c(0, 0, 0), 0);
  VERIFY_IS_EQUAL(c(0, 0, 1), 1);
  VERIFY_IS_EQUAL(c(0, 0, 2), 2);
  VERIFY_IS_EQUAL(c(0, 0, 3), 3);
  VERIFY_IS_EQUAL(c(0, 1, 0), 4);
  VERIFY_IS_EQUAL(c(0, 1, 1), 5);
  VERIFY_IS_EQUAL(c(0, 1, 2), 6);
  VERIFY_IS_EQUAL(c(0, 1, 3), 7);
  VERIFY_IS_EQUAL(c(1, 0, 0), 10);
  VERIFY_IS_EQUAL(c(1, 0, 1), 11);
  VERIFY_IS_EQUAL(c(1, 0, 2), 12);
  VERIFY_IS_EQUAL(c(1, 0, 3), 13);
  VERIFY_IS_EQUAL(c(1, 1, 0), 14);
  VERIFY_IS_EQUAL(c(1, 1, 1), 15);
  VERIFY_IS_EQUAL(c(1, 1, 2), 16);
  VERIFY_IS_EQUAL(c(1, 1, 3), 17);
  VERIFY_IS_EQUAL(c(2, 0, 0), 20);
  VERIFY_IS_EQUAL(c(2, 0, 1), 21);
  VERIFY_IS_EQUAL(c(2, 0, 2), 22);
  VERIFY_IS_EQUAL(c(2, 0, 3), 23);
  VERIFY_IS_EQUAL(c(2, 1, 0), 24);
  VERIFY_IS_EQUAL(c(2, 1, 1), 25);
  VERIFY_IS_EQUAL(c(2, 1, 2), 26);
  VERIFY_IS_EQUAL(c(2, 1, 3), 27);
#endif  // EIGEN_HAS_VARIADIC_TEMPLATES
}

void test_cxx11_tensor_assign()
{
  CALL_SUBTEST(test_1d());
  CALL_SUBTEST(test_2d());
  CALL_SUBTEST(test_3d());
  CALL_SUBTEST(test_same_type());
  CALL_SUBTEST(test_auto_resize());
  CALL_SUBTEST(test_compound_assign());
  CALL_SUBTEST(test_std_initializers_tensor());
}
