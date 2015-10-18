// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012, 2013 Chen-Pang He <jdh8@ms63.hinet.net>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "matrix_functions.h"

template<typename T>
void test2dRotation(double tol)
{
  Matrix<T,2,2> A, B, C;
  T angle, c, s;

  A << 0, 1, -1, 0;
  MatrixPower<Matrix<T,2,2> > Apow(A);

  for (int i=0; i<=20; ++i) {
    angle = pow(10, (i-10) / 5.);
    c = std::cos(angle);
    s = std::sin(angle);
    B << c, s, -s, c;

    C = Apow(std::ldexp(angle,1) / M_PI);
    std::cout << "test2dRotation: i = " << i << "   error powerm = " << relerr(C,B) << '\n';
    VERIFY(C.isApprox(B, tol));
  }
}

template<typename T>
void test2dHyperbolicRotation(double tol)
{
  Matrix<std::complex<T>,2,2> A, B, C;
  T angle, ch = std::cosh((T)1);
  std::complex<T> ish(0, std::sinh((T)1));

  A << ch, ish, -ish, ch;
  MatrixPower<Matrix<std::complex<T>,2,2> > Apow(A);

  for (int i=0; i<=20; ++i) {
    angle = std::ldexp(static_cast<T>(i-10), -1);
    ch = std::cosh(angle);
    ish = std::complex<T>(0, std::sinh(angle));
    B << ch, ish, -ish, ch;

    C = Apow(angle);
    std::cout << "test2dHyperbolicRotation: i = " << i << "   error powerm = " << relerr(C,B) << '\n';
    VERIFY(C.isApprox(B, tol));
  }
}

template<typename T>
void test3dRotation(double tol)
{
  Matrix<T,3,1> v;
  T angle;

  for (int i=0; i<=20; ++i) {
    v = Matrix<T,3,1>::Random();
    v.normalize();
    angle = pow(10, (i-10) / 5.);
    VERIFY(AngleAxis<T>(angle, v).matrix().isApprox(AngleAxis<T>(1,v).matrix().pow(angle), tol));
  }
}

template<typename MatrixType>
void testGeneral(const MatrixType& m, double tol)
{
  typedef typename MatrixType::RealScalar RealScalar;
  MatrixType m1, m2, m3, m4, m5;
  RealScalar x, y;

  for (int i=0; i < g_repeat; ++i) {
    generateTestMatrix<MatrixType>::run(m1, m.rows());
    MatrixPower<MatrixType> mpow(m1);

    x = internal::random<RealScalar>();
    y = internal::random<RealScalar>();
    m2 = mpow(x);
    m3 = mpow(y);

    m4 = mpow(x+y);
    m5.noalias() = m2 * m3;
    VERIFY(m4.isApprox(m5, tol));

    m4 = mpow(x*y);
    m5 = m2.pow(y);
    VERIFY(m4.isApprox(m5, tol));

    m4 = (std::abs(x) * m1).pow(y);
    m5 = std::pow(std::abs(x), y) * m3;
    VERIFY(m4.isApprox(m5, tol));
  }
}

template<typename MatrixType>
void testSingular(const MatrixType& m_const, double tol)
{
  // we need to pass by reference in order to prevent errors with
  // MSVC for aligned data types ...
  MatrixType& m = const_cast<MatrixType&>(m_const);

  const int IsComplex = NumTraits<typename internal::traits<MatrixType>::Scalar>::IsComplex;
  typedef typename internal::conditional<IsComplex, TriangularView<MatrixType,Upper>, const MatrixType&>::type TriangularType;
  typename internal::conditional< IsComplex, ComplexSchur<MatrixType>, RealSchur<MatrixType> >::type schur;
  MatrixType T;

  for (int i=0; i < g_repeat; ++i) {
    m.setRandom();
    m.col(0).fill(0);

    schur.compute(m);
    T = schur.matrixT();
    const MatrixType& U = schur.matrixU();
    processTriangularMatrix<MatrixType>::run(m, T, U);
    MatrixPower<MatrixType> mpow(m);

    T = T.sqrt();
    VERIFY(mpow(0.5).isApprox(U * (TriangularType(T) * U.adjoint()), tol));

    T = T.sqrt();
    VERIFY(mpow(0.25).isApprox(U * (TriangularType(T) * U.adjoint()), tol));

    T = T.sqrt();
    VERIFY(mpow(0.125).isApprox(U * (TriangularType(T) * U.adjoint()), tol));
  }
}

template<typename MatrixType>
void testLogThenExp(const MatrixType& m_const, double tol)
{
  // we need to pass by reference in order to prevent errors with
  // MSVC for aligned data types ...
  MatrixType& m = const_cast<MatrixType&>(m_const);

  typedef typename MatrixType::Scalar Scalar;
  Scalar x;

  for (int i=0; i < g_repeat; ++i) {
    generateTestMatrix<MatrixType>::run(m, m.rows());
    x = internal::random<Scalar>();
    VERIFY(m.pow(x).isApprox((x * m.log()).exp(), tol));
  }
}

typedef Matrix<double,3,3,RowMajor>         Matrix3dRowMajor;
typedef Matrix<long double,3,3>             Matrix3e;
typedef Matrix<long double,Dynamic,Dynamic> MatrixXe;
 
void test_matrix_power()
{
  CALL_SUBTEST_2(test2dRotation<double>(1e-13));
  CALL_SUBTEST_1(test2dRotation<float>(2e-5));  // was 1e-5, relaxed for clang 2.8 / linux / x86-64
  CALL_SUBTEST_9(test2dRotation<long double>(1e-13)); 
  CALL_SUBTEST_2(test2dHyperbolicRotation<double>(1e-14));
  CALL_SUBTEST_1(test2dHyperbolicRotation<float>(1e-5));
  CALL_SUBTEST_9(test2dHyperbolicRotation<long double>(1e-14));

  CALL_SUBTEST_10(test3dRotation<double>(1e-13));
  CALL_SUBTEST_11(test3dRotation<float>(1e-5));
  CALL_SUBTEST_12(test3dRotation<long double>(1e-13));

  CALL_SUBTEST_2(testGeneral(Matrix2d(),         1e-13));
  CALL_SUBTEST_7(testGeneral(Matrix3dRowMajor(), 1e-13));
  CALL_SUBTEST_3(testGeneral(Matrix4cd(),        1e-13));
  CALL_SUBTEST_4(testGeneral(MatrixXd(8,8),      2e-12));
  CALL_SUBTEST_1(testGeneral(Matrix2f(),         1e-4));
  CALL_SUBTEST_5(testGeneral(Matrix3cf(),        1e-4));
  CALL_SUBTEST_8(testGeneral(Matrix4f(),         1e-4));
  CALL_SUBTEST_6(testGeneral(MatrixXf(2,2),      1e-3)); // see bug 614
  CALL_SUBTEST_9(testGeneral(MatrixXe(7,7),      1e-13));
  CALL_SUBTEST_10(testGeneral(Matrix3d(),        1e-13));
  CALL_SUBTEST_11(testGeneral(Matrix3f(),        1e-4));
  CALL_SUBTEST_12(testGeneral(Matrix3e(),        1e-13));

  CALL_SUBTEST_2(testSingular(Matrix2d(),         1e-13));
  CALL_SUBTEST_7(testSingular(Matrix3dRowMajor(), 1e-13));
  CALL_SUBTEST_3(testSingular(Matrix4cd(),        1e-13));
  CALL_SUBTEST_4(testSingular(MatrixXd(8,8),      2e-12));
  CALL_SUBTEST_1(testSingular(Matrix2f(),         1e-4));
  CALL_SUBTEST_5(testSingular(Matrix3cf(),        1e-4));
  CALL_SUBTEST_8(testSingular(Matrix4f(),         1e-4));
  CALL_SUBTEST_6(testSingular(MatrixXf(2,2),      1e-3));
  CALL_SUBTEST_9(testSingular(MatrixXe(7,7),      1e-13));
  CALL_SUBTEST_10(testSingular(Matrix3d(),        1e-13));
  CALL_SUBTEST_11(testSingular(Matrix3f(),        1e-4));
  CALL_SUBTEST_12(testSingular(Matrix3e(),        1e-13));

  CALL_SUBTEST_2(testLogThenExp(Matrix2d(),         1e-13));
  CALL_SUBTEST_7(testLogThenExp(Matrix3dRowMajor(), 1e-13));
  CALL_SUBTEST_3(testLogThenExp(Matrix4cd(),        1e-13));
  CALL_SUBTEST_4(testLogThenExp(MatrixXd(8,8),      2e-12));
  CALL_SUBTEST_1(testLogThenExp(Matrix2f(),         1e-4));
  CALL_SUBTEST_5(testLogThenExp(Matrix3cf(),        1e-4));
  CALL_SUBTEST_8(testLogThenExp(Matrix4f(),         1e-4));
  CALL_SUBTEST_6(testLogThenExp(MatrixXf(2,2),      1e-3));
  CALL_SUBTEST_9(testLogThenExp(MatrixXe(7,7),      1e-13));
  CALL_SUBTEST_10(testLogThenExp(Matrix3d(),        1e-13));
  CALL_SUBTEST_11(testLogThenExp(Matrix3f(),        1e-4));
  CALL_SUBTEST_12(testLogThenExp(Matrix3e(),        1e-13));
}
