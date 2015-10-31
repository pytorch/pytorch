// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "matrix_functions.h"

double binom(int n, int k)
{
  double res = 1;
  for (int i=0; i<k; i++)
    res = res * (n-k+i+1) / (i+1);
  return res;
}

template <typename T>
T expfn(T x, int)
{
  return std::exp(x);
}

template <typename T>
void test2dRotation(double tol)
{
  Matrix<T,2,2> A, B, C;
  T angle;

  A << 0, 1, -1, 0;
  for (int i=0; i<=20; i++)
  {
    angle = static_cast<T>(pow(10, i / 5. - 2));
    B << std::cos(angle), std::sin(angle), -std::sin(angle), std::cos(angle);

    C = (angle*A).matrixFunction(expfn);
    std::cout << "test2dRotation: i = " << i << "   error funm = " << relerr(C, B);
    VERIFY(C.isApprox(B, static_cast<T>(tol)));

    C = (angle*A).exp();
    std::cout << "   error expm = " << relerr(C, B) << "\n";
    VERIFY(C.isApprox(B, static_cast<T>(tol)));
  }
}

template <typename T>
void test2dHyperbolicRotation(double tol)
{
  Matrix<std::complex<T>,2,2> A, B, C;
  std::complex<T> imagUnit(0,1);
  T angle, ch, sh;

  for (int i=0; i<=20; i++)
  {
    angle = static_cast<T>((i-10) / 2.0);
    ch = std::cosh(angle);
    sh = std::sinh(angle);
    A << 0, angle*imagUnit, -angle*imagUnit, 0;
    B << ch, sh*imagUnit, -sh*imagUnit, ch;

    C = A.matrixFunction(expfn);
    std::cout << "test2dHyperbolicRotation: i = " << i << "   error funm = " << relerr(C, B);
    VERIFY(C.isApprox(B, static_cast<T>(tol)));

    C = A.exp();
    std::cout << "   error expm = " << relerr(C, B) << "\n";
    VERIFY(C.isApprox(B, static_cast<T>(tol)));
  }
}

template <typename T>
void testPascal(double tol)
{
  for (int size=1; size<20; size++)
  {
    Matrix<T,Dynamic,Dynamic> A(size,size), B(size,size), C(size,size);
    A.setZero();
    for (int i=0; i<size-1; i++)
      A(i+1,i) = static_cast<T>(i+1);
    B.setZero();
    for (int i=0; i<size; i++)
      for (int j=0; j<=i; j++)
    B(i,j) = static_cast<T>(binom(i,j));

    C = A.matrixFunction(expfn);
    std::cout << "testPascal: size = " << size << "   error funm = " << relerr(C, B);
    VERIFY(C.isApprox(B, static_cast<T>(tol)));

    C = A.exp();
    std::cout << "   error expm = " << relerr(C, B) << "\n";
    VERIFY(C.isApprox(B, static_cast<T>(tol)));
  }
}

template<typename MatrixType>
void randomTest(const MatrixType& m, double tol)
{
  /* this test covers the following files:
     Inverse.h
  */
  typename MatrixType::Index rows = m.rows();
  typename MatrixType::Index cols = m.cols();
  MatrixType m1(rows, cols), m2(rows, cols), identity = MatrixType::Identity(rows, cols);

  typedef typename NumTraits<typename internal::traits<MatrixType>::Scalar>::Real RealScalar;

  for(int i = 0; i < g_repeat; i++) {
    m1 = MatrixType::Random(rows, cols);

    m2 = m1.matrixFunction(expfn) * (-m1).matrixFunction(expfn);
    std::cout << "randomTest: error funm = " << relerr(identity, m2);
    VERIFY(identity.isApprox(m2, static_cast<RealScalar>(tol)));

    m2 = m1.exp() * (-m1).exp();
    std::cout << "   error expm = " << relerr(identity, m2) << "\n";
    VERIFY(identity.isApprox(m2, static_cast<RealScalar>(tol)));
  }
}

void test_matrix_exponential()
{
  CALL_SUBTEST_2(test2dRotation<double>(1e-13));
  CALL_SUBTEST_1(test2dRotation<float>(2e-5));  // was 1e-5, relaxed for clang 2.8 / linux / x86-64
  CALL_SUBTEST_8(test2dRotation<long double>(1e-13)); 
  CALL_SUBTEST_2(test2dHyperbolicRotation<double>(1e-14));
  CALL_SUBTEST_1(test2dHyperbolicRotation<float>(1e-5));
  CALL_SUBTEST_8(test2dHyperbolicRotation<long double>(1e-14));
  CALL_SUBTEST_6(testPascal<float>(1e-6));
  CALL_SUBTEST_5(testPascal<double>(1e-15));
  CALL_SUBTEST_2(randomTest(Matrix2d(), 1e-13));
  CALL_SUBTEST_7(randomTest(Matrix<double,3,3,RowMajor>(), 1e-13));
  CALL_SUBTEST_3(randomTest(Matrix4cd(), 1e-13));
  CALL_SUBTEST_4(randomTest(MatrixXd(8,8), 1e-13));
  CALL_SUBTEST_1(randomTest(Matrix2f(), 1e-4));
  CALL_SUBTEST_5(randomTest(Matrix3cf(), 1e-4));
  CALL_SUBTEST_1(randomTest(Matrix4f(), 1e-4));
  CALL_SUBTEST_6(randomTest(MatrixXf(8,8), 1e-4));
  CALL_SUBTEST_9(randomTest(Matrix<long double,Dynamic,Dynamic>(7,7), 1e-13));
}
