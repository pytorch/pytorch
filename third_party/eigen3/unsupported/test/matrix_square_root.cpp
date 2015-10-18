// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "matrix_functions.h"

template<typename MatrixType>
void testMatrixSqrt(const MatrixType& m)
{
  MatrixType A;
  generateTestMatrix<MatrixType>::run(A, m.rows());
  MatrixType sqrtA = A.sqrt();
  VERIFY_IS_APPROX(sqrtA * sqrtA, A);
}

void test_matrix_square_root()
{
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(testMatrixSqrt(Matrix3cf()));
    CALL_SUBTEST_2(testMatrixSqrt(MatrixXcd(12,12)));
    CALL_SUBTEST_3(testMatrixSqrt(Matrix4f()));
    CALL_SUBTEST_4(testMatrixSqrt(Matrix<double,Dynamic,Dynamic,RowMajor>(9, 9)));
    CALL_SUBTEST_5(testMatrixSqrt(Matrix<float,1,1>()));
    CALL_SUBTEST_5(testMatrixSqrt(Matrix<std::complex<float>,1,1>()));
  }
}
