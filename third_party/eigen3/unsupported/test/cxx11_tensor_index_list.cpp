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

#ifdef EIGEN_HAS_INDEX_LIST

static void test_static_index_list()
{
  Tensor<float, 4> tensor(2,3,5,7);
  tensor.setRandom();

  constexpr auto reduction_axis = make_index_list(0, 1, 2);
  VERIFY_IS_EQUAL(internal::array_get<0>(reduction_axis), 0);
  VERIFY_IS_EQUAL(internal::array_get<1>(reduction_axis), 1);
  VERIFY_IS_EQUAL(internal::array_get<2>(reduction_axis), 2);
  VERIFY_IS_EQUAL(static_cast<DenseIndex>(reduction_axis[0]), 0);
  VERIFY_IS_EQUAL(static_cast<DenseIndex>(reduction_axis[1]), 1);
  VERIFY_IS_EQUAL(static_cast<DenseIndex>(reduction_axis[2]), 2);

  EIGEN_STATIC_ASSERT((internal::array_get<0>(reduction_axis) == 0), YOU_MADE_A_PROGRAMMING_MISTAKE);
  EIGEN_STATIC_ASSERT((internal::array_get<1>(reduction_axis) == 1), YOU_MADE_A_PROGRAMMING_MISTAKE);
  EIGEN_STATIC_ASSERT((internal::array_get<2>(reduction_axis) == 2), YOU_MADE_A_PROGRAMMING_MISTAKE);

  Tensor<float, 1> result = tensor.sum(reduction_axis);
  for (int i = 0; i < result.size(); ++i) {
    float expected = 0.0f;
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 5; ++l) {
          expected += tensor(j,k,l,i);
        }
      }
    }
    VERIFY_IS_APPROX(result(i), expected);
  }
}


static void test_type2index_list()
{
  Tensor<float, 5> tensor(2,3,5,7,11);
  tensor.setRandom();
  tensor += tensor.constant(10.0f);

  typedef Eigen::IndexList<Eigen::type2index<0>> Dims0;
  typedef Eigen::IndexList<Eigen::type2index<0>, Eigen::type2index<1>> Dims1;
  typedef Eigen::IndexList<Eigen::type2index<0>, Eigen::type2index<1>, Eigen::type2index<2>> Dims2;
  typedef Eigen::IndexList<Eigen::type2index<0>, Eigen::type2index<1>, Eigen::type2index<2>, Eigen::type2index<3>> Dims3;
  typedef Eigen::IndexList<Eigen::type2index<0>, Eigen::type2index<1>, Eigen::type2index<2>, Eigen::type2index<3>, Eigen::type2index<4>> Dims4;

#if 0
  EIGEN_STATIC_ASSERT((internal::indices_statically_known_to_increase<Dims0>() == true), YOU_MADE_A_PROGRAMMING_MISTAKE);
  EIGEN_STATIC_ASSERT((internal::indices_statically_known_to_increase<Dims1>() == true), YOU_MADE_A_PROGRAMMING_MISTAKE);
  EIGEN_STATIC_ASSERT((internal::indices_statically_known_to_increase<Dims2>() == true), YOU_MADE_A_PROGRAMMING_MISTAKE);
  EIGEN_STATIC_ASSERT((internal::indices_statically_known_to_increase<Dims3>() == true), YOU_MADE_A_PROGRAMMING_MISTAKE);
  EIGEN_STATIC_ASSERT((internal::indices_statically_known_to_increase<Dims4>() == true), YOU_MADE_A_PROGRAMMING_MISTAKE);
#endif

  EIGEN_STATIC_ASSERT((internal::are_inner_most_dims<Dims0, 1, ColMajor>::value == true), YOU_MADE_A_PROGRAMMING_MISTAKE);
  EIGEN_STATIC_ASSERT((internal::are_inner_most_dims<Dims1, 2, ColMajor>::value == true), YOU_MADE_A_PROGRAMMING_MISTAKE);
  EIGEN_STATIC_ASSERT((internal::are_inner_most_dims<Dims2, 3, ColMajor>::value == true), YOU_MADE_A_PROGRAMMING_MISTAKE);
  EIGEN_STATIC_ASSERT((internal::are_inner_most_dims<Dims3, 4, ColMajor>::value == true), YOU_MADE_A_PROGRAMMING_MISTAKE);
  EIGEN_STATIC_ASSERT((internal::are_inner_most_dims<Dims4, 5, ColMajor>::value == true), YOU_MADE_A_PROGRAMMING_MISTAKE);

  EIGEN_STATIC_ASSERT((internal::are_inner_most_dims<Dims0, 1, RowMajor>::value == true), YOU_MADE_A_PROGRAMMING_MISTAKE);
  EIGEN_STATIC_ASSERT((internal::are_inner_most_dims<Dims1, 2, RowMajor>::value == true), YOU_MADE_A_PROGRAMMING_MISTAKE);
  EIGEN_STATIC_ASSERT((internal::are_inner_most_dims<Dims2, 3, RowMajor>::value == true), YOU_MADE_A_PROGRAMMING_MISTAKE);
  EIGEN_STATIC_ASSERT((internal::are_inner_most_dims<Dims3, 4, RowMajor>::value == true), YOU_MADE_A_PROGRAMMING_MISTAKE);
  EIGEN_STATIC_ASSERT((internal::are_inner_most_dims<Dims4, 5, RowMajor>::value == true), YOU_MADE_A_PROGRAMMING_MISTAKE);

  const Dims0 reduction_axis0;
  Tensor<float, 4> result0 = tensor.sum(reduction_axis0);
  for (int m = 0; m < 11; ++m) {
    for (int l = 0; l < 7; ++l) {
      for (int k = 0; k < 5; ++k) {
        for (int j = 0; j < 3; ++j) {
          float expected = 0.0f;
          for (int i = 0; i < 2; ++i) {
            expected += tensor(i,j,k,l,m);
          }
          VERIFY_IS_APPROX(result0(j,k,l,m), expected);
        }
      }
    }
  }

  const Dims1 reduction_axis1;
  Tensor<float, 3> result1 = tensor.sum(reduction_axis1);
  for (int m = 0; m < 11; ++m) {
    for (int l = 0; l < 7; ++l) {
      for (int k = 0; k < 5; ++k) {
        float expected = 0.0f;
        for (int j = 0; j < 3; ++j) {
          for (int i = 0; i < 2; ++i) {
            expected += tensor(i,j,k,l,m);
          }
        }
        VERIFY_IS_APPROX(result1(k,l,m), expected);
      }
    }
  }

  const Dims2 reduction_axis2;
  Tensor<float, 2> result2 = tensor.sum(reduction_axis2);
  for (int m = 0; m < 11; ++m) {
    for (int l = 0; l < 7; ++l) {
      float expected = 0.0f;
      for (int k = 0; k < 5; ++k) {
        for (int j = 0; j < 3; ++j) {
          for (int i = 0; i < 2; ++i) {
            expected += tensor(i,j,k,l,m);
          }
        }
      }
      VERIFY_IS_APPROX(result2(l,m), expected);
    }
  }

  const Dims3 reduction_axis3;
  Tensor<float, 1> result3 = tensor.sum(reduction_axis3);
  for (int m = 0; m < 11; ++m) {
    float expected = 0.0f;
    for (int l = 0; l < 7; ++l) {
      for (int k = 0; k < 5; ++k) {
        for (int j = 0; j < 3; ++j) {
          for (int i = 0; i < 2; ++i) {
            expected += tensor(i,j,k,l,m);
          }
        }
      }
    }
    VERIFY_IS_APPROX(result3(m), expected);
  }

  const Dims4 reduction_axis4;
  Tensor<float, 0> result4 = tensor.sum(reduction_axis4);
  float expected = 0.0f;
  for (int m = 0; m < 11; ++m) {
    for (int l = 0; l < 7; ++l) {
      for (int k = 0; k < 5; ++k) {
        for (int j = 0; j < 3; ++j) {
          for (int i = 0; i < 2; ++i) {
            expected += tensor(i,j,k,l,m);
          }
        }
      }
    }
  }
  VERIFY_IS_APPROX(result4(), expected);
}


static void test_dynamic_index_list()
{
  Tensor<float, 4> tensor(2,3,5,7);
  tensor.setRandom();

  int dim1 = 2;
  int dim2 = 1;
  int dim3 = 0;

  auto reduction_axis = make_index_list(dim1, dim2, dim3);

  VERIFY_IS_EQUAL(internal::array_get<0>(reduction_axis), 2);
  VERIFY_IS_EQUAL(internal::array_get<1>(reduction_axis), 1);
  VERIFY_IS_EQUAL(internal::array_get<2>(reduction_axis), 0);
  VERIFY_IS_EQUAL(static_cast<DenseIndex>(reduction_axis[0]), 2);
  VERIFY_IS_EQUAL(static_cast<DenseIndex>(reduction_axis[1]), 1);
  VERIFY_IS_EQUAL(static_cast<DenseIndex>(reduction_axis[2]), 0);

  Tensor<float, 1> result = tensor.sum(reduction_axis);
  for (int i = 0; i < result.size(); ++i) {
    float expected = 0.0f;
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 5; ++l) {
          expected += tensor(j,k,l,i);
        }
      }
    }
    VERIFY_IS_APPROX(result(i), expected);
  }
}

static void test_mixed_index_list()
{
  Tensor<float, 4> tensor(2,3,5,7);
  tensor.setRandom();

  int dim2 = 1;
  int dim4 = 3;

  auto reduction_axis = make_index_list(0, dim2, 2, dim4);

  VERIFY_IS_EQUAL(internal::array_get<0>(reduction_axis), 0);
  VERIFY_IS_EQUAL(internal::array_get<1>(reduction_axis), 1);
  VERIFY_IS_EQUAL(internal::array_get<2>(reduction_axis), 2);
  VERIFY_IS_EQUAL(internal::array_get<3>(reduction_axis), 3);
  VERIFY_IS_EQUAL(static_cast<DenseIndex>(reduction_axis[0]), 0);
  VERIFY_IS_EQUAL(static_cast<DenseIndex>(reduction_axis[1]), 1);
  VERIFY_IS_EQUAL(static_cast<DenseIndex>(reduction_axis[2]), 2);
  VERIFY_IS_EQUAL(static_cast<DenseIndex>(reduction_axis[3]), 3);

  typedef IndexList<type2index<0>, int, type2index<2>, int> ReductionIndices;
  ReductionIndices reduction_indices;
  reduction_indices.set(1, 1);
  reduction_indices.set(3, 3);
  EIGEN_STATIC_ASSERT((internal::array_get<0>(reduction_indices) == 0), YOU_MADE_A_PROGRAMMING_MISTAKE);
  EIGEN_STATIC_ASSERT((internal::array_get<2>(reduction_indices) == 2), YOU_MADE_A_PROGRAMMING_MISTAKE);
  EIGEN_STATIC_ASSERT((internal::index_known_statically<ReductionIndices>(0) == true), YOU_MADE_A_PROGRAMMING_MISTAKE);
  EIGEN_STATIC_ASSERT((internal::index_known_statically<ReductionIndices>(2) == true), YOU_MADE_A_PROGRAMMING_MISTAKE);
  EIGEN_STATIC_ASSERT((internal::index_statically_eq<ReductionIndices>(0, 0) == true), YOU_MADE_A_PROGRAMMING_MISTAKE);
  EIGEN_STATIC_ASSERT((internal::index_statically_eq<ReductionIndices>(2, 2) == true), YOU_MADE_A_PROGRAMMING_MISTAKE);
#if 0
  EIGEN_STATIC_ASSERT((internal::all_indices_known_statically<ReductionIndices>() == false), YOU_MADE_A_PROGRAMMING_MISTAKE);
  EIGEN_STATIC_ASSERT((internal::indices_statically_known_to_increase<ReductionIndices>() == false), YOU_MADE_A_PROGRAMMING_MISTAKE);
#endif

  typedef IndexList<type2index<0>, type2index<1>, type2index<2>, type2index<3>> ReductionList;
  ReductionList reduction_list;
  EIGEN_STATIC_ASSERT((internal::index_statically_eq<ReductionList>(0, 0) == true), YOU_MADE_A_PROGRAMMING_MISTAKE);
  EIGEN_STATIC_ASSERT((internal::index_statically_eq<ReductionList>(1, 1) == true), YOU_MADE_A_PROGRAMMING_MISTAKE);
  EIGEN_STATIC_ASSERT((internal::index_statically_eq<ReductionList>(2, 2) == true), YOU_MADE_A_PROGRAMMING_MISTAKE);
  EIGEN_STATIC_ASSERT((internal::index_statically_eq<ReductionList>(3, 3) == true), YOU_MADE_A_PROGRAMMING_MISTAKE);
#if 0
  EIGEN_STATIC_ASSERT((internal::all_indices_known_statically<ReductionList>() == true), YOU_MADE_A_PROGRAMMING_MISTAKE);
  EIGEN_STATIC_ASSERT((internal::indices_statically_known_to_increase<ReductionList>() == true), YOU_MADE_A_PROGRAMMING_MISTAKE);
#endif

  Tensor<float, 0> result1 = tensor.sum(reduction_axis);
  Tensor<float, 0> result2 = tensor.sum(reduction_indices);
  Tensor<float, 0> result3 = tensor.sum(reduction_list);

  float expected = 0.0f;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 5; ++k) {
        for (int l = 0; l < 7; ++l) {
          expected += tensor(i,j,k,l);
        }
      }
    }
  }
  VERIFY_IS_APPROX(result1(), expected);
  VERIFY_IS_APPROX(result2(), expected);
  VERIFY_IS_APPROX(result3(), expected);
}


static void test_dim_check()
{
  Eigen::IndexList<Eigen::type2index<1>, int> dim1;
  dim1.set(1, 2);
  Eigen::IndexList<Eigen::type2index<1>, int> dim2;
  dim2.set(1, 2);
  VERIFY(dimensions_match(dim1, dim2));
}


#endif

void test_cxx11_tensor_index_list()
{
#ifdef EIGEN_HAS_INDEX_LIST
  CALL_SUBTEST(test_static_index_list());
  CALL_SUBTEST(test_type2index_list());
  CALL_SUBTEST(test_dynamic_index_list());
  CALL_SUBTEST(test_mixed_index_list());
  CALL_SUBTEST(test_dim_check());
#endif
}
