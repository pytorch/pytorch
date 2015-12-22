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

static void test_simple_reshape()
{
  Tensor<float, 5> tensor1(2,3,1,7,1);
  tensor1.setRandom();

  Tensor<float, 3> tensor2(2,3,7);
  Tensor<float, 2> tensor3(6,7);
  Tensor<float, 2> tensor4(2,21);

  Tensor<float, 3>::Dimensions dim1(2,3,7);
  tensor2 = tensor1.reshape(dim1);
  Tensor<float, 2>::Dimensions dim2(6,7);
  tensor3 = tensor1.reshape(dim2);
  Tensor<float, 2>::Dimensions dim3(2,21);
  tensor4 = tensor1.reshape(dim1).reshape(dim3);

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 7; ++k) {
        VERIFY_IS_EQUAL(tensor1(i,j,0,k,0), tensor2(i,j,k));
        VERIFY_IS_EQUAL(tensor1(i,j,0,k,0), tensor3(i+2*j,k));
        VERIFY_IS_EQUAL(tensor1(i,j,0,k,0), tensor4(i,j+3*k));
      }
    }
  }
}


static void test_reshape_in_expr() {
  MatrixXf m1(2,3*5*7*11);
  MatrixXf m2(3*5*7*11,13);
  m1.setRandom();
  m2.setRandom();
  MatrixXf m3 = m1 * m2;

  TensorMap<Tensor<float, 5>> tensor1(m1.data(), 2,3,5,7,11);
  TensorMap<Tensor<float, 5>> tensor2(m2.data(), 3,5,7,11,13);
  Tensor<float, 2>::Dimensions newDims1(2,3*5*7*11);
  Tensor<float, 2>::Dimensions newDims2(3*5*7*11,13);
  typedef Tensor<float, 1>::DimensionPair DimPair;
  array<DimPair, 1> contract_along{{DimPair(1, 0)}};
  Tensor<float, 2> tensor3(2,13);
  tensor3 = tensor1.reshape(newDims1).contract(tensor2.reshape(newDims2), contract_along);

  Map<MatrixXf> res(tensor3.data(), 2, 13);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 13; ++j) {
      VERIFY_IS_APPROX(res(i,j), m3(i,j));
    }
  }
}


static void test_reshape_as_lvalue()
{
  Tensor<float, 3> tensor(2,3,7);
  tensor.setRandom();

  Tensor<float, 2> tensor2d(6,7);
  Tensor<float, 3>::Dimensions dim(2,3,7);
  tensor2d.reshape(dim) = tensor;

  float scratch[2*3*1*7*1];
  TensorMap<Tensor<float, 5>> tensor5d(scratch, 2,3,1,7,1);
  tensor5d.reshape(dim).device(Eigen::DefaultDevice()) = tensor;

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 7; ++k) {
        VERIFY_IS_EQUAL(tensor2d(i+2*j,k), tensor(i,j,k));
        VERIFY_IS_EQUAL(tensor5d(i,j,0,k,0), tensor(i,j,k));
      }
    }
  }
}

template<int DataLayout>
static void test_simple_slice()
{
  Tensor<float, 5, DataLayout> tensor(2,3,5,7,11);
  tensor.setRandom();

  Tensor<float, 5, DataLayout> slice1(1,1,1,1,1);
  Eigen::DSizes<ptrdiff_t, 5> indices(1,2,3,4,5);
  Eigen::DSizes<ptrdiff_t, 5> sizes(1,1,1,1,1);
  slice1 = tensor.slice(indices, sizes);
  VERIFY_IS_EQUAL(slice1(0,0,0,0,0), tensor(1,2,3,4,5));

  Tensor<float, 5, DataLayout> slice2(1,1,2,2,3);
  Eigen::DSizes<ptrdiff_t, 5> indices2(1,1,3,4,5);
  Eigen::DSizes<ptrdiff_t, 5> sizes2(1,1,2,2,3);
  slice2 = tensor.slice(indices2, sizes2);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 3; ++k) {
        VERIFY_IS_EQUAL(slice2(0,0,i,j,k), tensor(1,1,3+i,4+j,5+k));
      }
    }
  }
}

static void test_const_slice()
{
  const float b[1] = {42};
  TensorMap<Tensor<const float, 1> > m(b, 1);
  DSizes<DenseIndex, 1> offsets;
  offsets[0] = 0;
  TensorRef<Tensor<const float, 1> > slice_ref(m.slice(offsets, m.dimensions()));
  VERIFY_IS_EQUAL(slice_ref(0), 42);
}

template<int DataLayout>
static void test_slice_in_expr() {
  typedef Matrix<float, Dynamic, Dynamic, DataLayout> Mtx;
  Mtx m1(7,7);
  Mtx m2(3,3);
  m1.setRandom();
  m2.setRandom();

  Mtx m3 = m1.block(1, 2, 3, 3) * m2.block(0, 2, 3, 1);

  TensorMap<Tensor<float, 2, DataLayout>> tensor1(m1.data(), 7, 7);
  TensorMap<Tensor<float, 2, DataLayout>> tensor2(m2.data(), 3, 3);
  Tensor<float, 2, DataLayout> tensor3(3,1);
  typedef Tensor<float, 1>::DimensionPair DimPair;
  array<DimPair, 1> contract_along{{DimPair(1, 0)}};

  Eigen::DSizes<ptrdiff_t, 2> indices1(1,2);
  Eigen::DSizes<ptrdiff_t, 2> sizes1(3,3);
  Eigen::DSizes<ptrdiff_t, 2> indices2(0,2);
  Eigen::DSizes<ptrdiff_t, 2> sizes2(3,1);
  tensor3 = tensor1.slice(indices1, sizes1).contract(tensor2.slice(indices2, sizes2), contract_along);

  Map<Mtx> res(tensor3.data(), 3, 1);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 1; ++j) {
      VERIFY_IS_APPROX(res(i,j), m3(i,j));
    }
  }

  // Take an arbitrary slice of an arbitrarily sized tensor.
  TensorMap<Tensor<const float, 2, DataLayout>> tensor4(m1.data(), 7, 7);
  Tensor<float, 1, DataLayout> tensor6 = tensor4.reshape(DSizes<ptrdiff_t, 1>(7*7)).exp().slice(DSizes<ptrdiff_t, 1>(0), DSizes<ptrdiff_t, 1>(35));
  for (int i = 0; i < 35; ++i) {
    VERIFY_IS_APPROX(tensor6(i), expf(tensor4.data()[i]));
  }
}

template<int DataLayout>
static void test_slice_as_lvalue()
{
  Tensor<float, 3, DataLayout> tensor1(2,2,7);
  tensor1.setRandom();
  Tensor<float, 3, DataLayout> tensor2(2,2,7);
  tensor2.setRandom();
  Tensor<float, 3, DataLayout> tensor3(4,3,5);
  tensor3.setRandom();
  Tensor<float, 3, DataLayout> tensor4(4,3,2);
  tensor4.setRandom();
  Tensor<float, 3, DataLayout> tensor5(10,13,12);
  tensor5.setRandom();

  Tensor<float, 3, DataLayout> result(4,5,7);
  Eigen::DSizes<ptrdiff_t, 3> sizes12(2,2,7);
  Eigen::DSizes<ptrdiff_t, 3> first_slice(0,0,0);
  result.slice(first_slice, sizes12) = tensor1;
  Eigen::DSizes<ptrdiff_t, 3> second_slice(2,0,0);
  result.slice(second_slice, sizes12).device(Eigen::DefaultDevice()) = tensor2;

  Eigen::DSizes<ptrdiff_t, 3> sizes3(4,3,5);
  Eigen::DSizes<ptrdiff_t, 3> third_slice(0,2,0);
  result.slice(third_slice, sizes3) = tensor3;

  Eigen::DSizes<ptrdiff_t, 3> sizes4(4,3,2);
  Eigen::DSizes<ptrdiff_t, 3> fourth_slice(0,2,5);
  result.slice(fourth_slice, sizes4) = tensor4;

  for (int j = 0; j < 2; ++j) {
    for (int k = 0; k < 7; ++k) {
      for (int i = 0; i < 2; ++i) {
        VERIFY_IS_EQUAL(result(i,j,k), tensor1(i,j,k));
        VERIFY_IS_EQUAL(result(i+2,j,k), tensor2(i,j,k));
      }
    }
  }
  for (int i = 0; i < 4; ++i) {
    for (int j = 2; j < 5; ++j) {
      for (int k = 0; k < 5; ++k) {
        VERIFY_IS_EQUAL(result(i,j,k), tensor3(i,j-2,k));
      }
      for (int k = 5; k < 7; ++k) {
        VERIFY_IS_EQUAL(result(i,j,k), tensor4(i,j-2,k-5));
      }
    }
  }

  Eigen::DSizes<ptrdiff_t, 3> sizes5(4,5,7);
  Eigen::DSizes<ptrdiff_t, 3> fifth_slice(0,0,0);
  result.slice(fifth_slice, sizes5) = tensor5.slice(fifth_slice, sizes5);
  for (int i = 0; i < 4; ++i) {
    for (int j = 2; j < 5; ++j) {
      for (int k = 0; k < 7; ++k) {
        VERIFY_IS_EQUAL(result(i,j,k), tensor5(i,j,k));
      }
    }
  }
}

template<int DataLayout>
static void test_slice_raw_data()
{
  Tensor<float, 4, DataLayout> tensor(3,5,7,11);
  tensor.setRandom();

  Eigen::DSizes<ptrdiff_t, 4> offsets(1,2,3,4);
  Eigen::DSizes<ptrdiff_t, 4> extents(1,1,1,1);
  typedef TensorEvaluator<decltype(tensor.slice(offsets, extents)), DefaultDevice> SliceEvaluator;
  auto slice1 = SliceEvaluator(tensor.slice(offsets, extents), DefaultDevice());
  VERIFY_IS_EQUAL(slice1.dimensions().TotalSize(), 1);
  VERIFY_IS_EQUAL(slice1.data()[0], tensor(1,2,3,4));

  if (DataLayout == ColMajor) {
    extents = Eigen::DSizes<ptrdiff_t, 4>(2,1,1,1);
    auto slice2 = SliceEvaluator(tensor.slice(offsets, extents), DefaultDevice());
    VERIFY_IS_EQUAL(slice2.dimensions().TotalSize(), 2);
    VERIFY_IS_EQUAL(slice2.data()[0], tensor(1,2,3,4));
    VERIFY_IS_EQUAL(slice2.data()[1], tensor(2,2,3,4));
  } else {
    extents = Eigen::DSizes<ptrdiff_t, 4>(1,1,1,2);
    auto slice2 = SliceEvaluator(tensor.slice(offsets, extents), DefaultDevice());
    VERIFY_IS_EQUAL(slice2.dimensions().TotalSize(), 2);
    VERIFY_IS_EQUAL(slice2.data()[0], tensor(1,2,3,4));
    VERIFY_IS_EQUAL(slice2.data()[1], tensor(1,2,3,5));
  }

  extents = Eigen::DSizes<ptrdiff_t, 4>(1,2,1,1);
  auto slice3 = SliceEvaluator(tensor.slice(offsets, extents), DefaultDevice());
  VERIFY_IS_EQUAL(slice3.dimensions().TotalSize(), 2);
  VERIFY_IS_EQUAL(slice3.data(), static_cast<float*>(0));

  if (DataLayout == ColMajor) {
    offsets = Eigen::DSizes<ptrdiff_t, 4>(0,2,3,4);
    extents = Eigen::DSizes<ptrdiff_t, 4>(3,2,1,1);
    auto slice4 = SliceEvaluator(tensor.slice(offsets, extents), DefaultDevice());
    VERIFY_IS_EQUAL(slice4.dimensions().TotalSize(), 6);
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 2; ++j) {
        VERIFY_IS_EQUAL(slice4.data()[i+3*j], tensor(i,2+j,3,4));
      }
    }
  } else {
    offsets = Eigen::DSizes<ptrdiff_t, 4>(1,2,3,0);
    extents = Eigen::DSizes<ptrdiff_t, 4>(1,1,2,11);
    auto slice4 = SliceEvaluator(tensor.slice(offsets, extents), DefaultDevice());
    VERIFY_IS_EQUAL(slice4.dimensions().TotalSize(), 22);
    for (int l = 0; l < 11; ++l) {
      for (int k = 0; k < 2; ++k) {
        VERIFY_IS_EQUAL(slice4.data()[l+11*k], tensor(1,2,3+k,l));
      }
    }
  }

  if (DataLayout == ColMajor) {
    offsets = Eigen::DSizes<ptrdiff_t, 4>(0,0,0,4);
    extents = Eigen::DSizes<ptrdiff_t, 4>(3,5,7,2);
    auto slice5 = SliceEvaluator(tensor.slice(offsets, extents), DefaultDevice());
    VERIFY_IS_EQUAL(slice5.dimensions().TotalSize(), 210);
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 5; ++j) {
        for (int k = 0; k < 7; ++k) {
          for (int l = 0; l < 2; ++l) {
            int slice_index = i + 3 * (j + 5 * (k + 7 * l));
            VERIFY_IS_EQUAL(slice5.data()[slice_index], tensor(i,j,k,l+4));
          }
        }
      }
    }
  } else {
    offsets = Eigen::DSizes<ptrdiff_t, 4>(1,0,0,0);
    extents = Eigen::DSizes<ptrdiff_t, 4>(2,5,7,11);
    auto slice5 = SliceEvaluator(tensor.slice(offsets, extents), DefaultDevice());
    VERIFY_IS_EQUAL(slice5.dimensions().TotalSize(), 770);
    for (int l = 0; l < 11; ++l) {
      for (int k = 0; k < 7; ++k) {
        for (int j = 0; j < 5; ++j) {
          for (int i = 0; i < 2; ++i) {
            int slice_index = l + 11 * (k + 7 * (j + 5 * i));
            VERIFY_IS_EQUAL(slice5.data()[slice_index], tensor(i+1,j,k,l));
          }
        }
      }
    }

  }

  offsets = Eigen::DSizes<ptrdiff_t, 4>(0,0,0,0);
  extents = Eigen::DSizes<ptrdiff_t, 4>(3,5,7,11);
  auto slice6 = SliceEvaluator(tensor.slice(offsets, extents), DefaultDevice());
  VERIFY_IS_EQUAL(slice6.dimensions().TotalSize(), 3*5*7*11);
  VERIFY_IS_EQUAL(slice6.data(), tensor.data());
}

template<int DataLayout>
static void test_composition()
{
  Eigen::Tensor<float, 2, DataLayout> matrix(7, 11);
  matrix.setRandom();

  const DSizes<ptrdiff_t, 3> newDims(1, 1, 11);
  Eigen::Tensor<float, 3, DataLayout> tensor =
      matrix.slice(DSizes<ptrdiff_t, 2>(2, 0), DSizes<ptrdiff_t, 2>(1, 11)).reshape(newDims);

  VERIFY_IS_EQUAL(tensor.dimensions().TotalSize(), 11);
  VERIFY_IS_EQUAL(tensor.dimension(0), 1);
  VERIFY_IS_EQUAL(tensor.dimension(1), 1);
  VERIFY_IS_EQUAL(tensor.dimension(2), 11);
  for (int i = 0; i < 11; ++i) {
    VERIFY_IS_EQUAL(tensor(0,0,i), matrix(2,i));
  }
}


void test_cxx11_tensor_morphing()
{
  CALL_SUBTEST(test_simple_reshape());
  CALL_SUBTEST(test_reshape_in_expr());
  CALL_SUBTEST(test_reshape_as_lvalue());

  CALL_SUBTEST(test_simple_slice<ColMajor>());
  CALL_SUBTEST(test_simple_slice<RowMajor>());
  CALL_SUBTEST(test_const_slice());
  CALL_SUBTEST(test_slice_in_expr<ColMajor>());
  CALL_SUBTEST(test_slice_in_expr<RowMajor>());
  CALL_SUBTEST(test_slice_as_lvalue<ColMajor>());
  CALL_SUBTEST(test_slice_as_lvalue<RowMajor>());
  CALL_SUBTEST(test_slice_raw_data<ColMajor>());
  CALL_SUBTEST(test_slice_raw_data<RowMajor>());

  CALL_SUBTEST(test_composition<ColMajor>());
  CALL_SUBTEST(test_composition<RowMajor>());
}
