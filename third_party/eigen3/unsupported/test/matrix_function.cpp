// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <unsupported/Eigen/MatrixFunctions>

// Variant of VERIFY_IS_APPROX which uses absolute error instead of
// relative error.
#define VERIFY_IS_APPROX_ABS(a, b) VERIFY(test_isApprox_abs(a, b))

template<typename Type1, typename Type2>
inline bool test_isApprox_abs(const Type1& a, const Type2& b)
{
  return ((a-b).array().abs() < test_precision<typename Type1::RealScalar>()).all();
}


// Returns a matrix with eigenvalues clustered around 0, 1 and 2.
template<typename MatrixType>
MatrixType randomMatrixWithRealEivals(const typename MatrixType::Index size)
{
  typedef typename MatrixType::Index Index;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  MatrixType diag = MatrixType::Zero(size, size);
  for (Index i = 0; i < size; ++i) {
    diag(i, i) = Scalar(RealScalar(internal::random<int>(0,2)))
      + internal::random<Scalar>() * Scalar(RealScalar(0.01));
  }
  MatrixType A = MatrixType::Random(size, size);
  HouseholderQR<MatrixType> QRofA(A);
  return QRofA.householderQ().inverse() * diag * QRofA.householderQ();
}

template <typename MatrixType, int IsComplex = NumTraits<typename internal::traits<MatrixType>::Scalar>::IsComplex>
struct randomMatrixWithImagEivals
{
  // Returns a matrix with eigenvalues clustered around 0 and +/- i.
  static MatrixType run(const typename MatrixType::Index size);
};

// Partial specialization for real matrices
template<typename MatrixType>
struct randomMatrixWithImagEivals<MatrixType, 0>
{
  static MatrixType run(const typename MatrixType::Index size)
  {
    typedef typename MatrixType::Index Index;
    typedef typename MatrixType::Scalar Scalar;
    MatrixType diag = MatrixType::Zero(size, size);
    Index i = 0;
    while (i < size) {
      Index randomInt = internal::random<Index>(-1, 1);
      if (randomInt == 0 || i == size-1) {
        diag(i, i) = internal::random<Scalar>() * Scalar(0.01);
        ++i;
      } else {
        Scalar alpha = Scalar(randomInt) + internal::random<Scalar>() * Scalar(0.01);
        diag(i, i+1) = alpha;
        diag(i+1, i) = -alpha;
        i += 2;
      }
    }
    MatrixType A = MatrixType::Random(size, size);
    HouseholderQR<MatrixType> QRofA(A);
    return QRofA.householderQ().inverse() * diag * QRofA.householderQ();
  }
};

// Partial specialization for complex matrices
template<typename MatrixType>
struct randomMatrixWithImagEivals<MatrixType, 1>
{
  static MatrixType run(const typename MatrixType::Index size)
  {
    typedef typename MatrixType::Index Index;
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::RealScalar RealScalar;
    const Scalar imagUnit(0, 1);
    MatrixType diag = MatrixType::Zero(size, size);
    for (Index i = 0; i < size; ++i) {
      diag(i, i) = Scalar(RealScalar(internal::random<Index>(-1, 1))) * imagUnit
        + internal::random<Scalar>() * Scalar(RealScalar(0.01));
    }
    MatrixType A = MatrixType::Random(size, size);
    HouseholderQR<MatrixType> QRofA(A);
    return QRofA.householderQ().inverse() * diag * QRofA.householderQ();
  }
};


template<typename MatrixType>
void testMatrixExponential(const MatrixType& A)
{
  typedef typename internal::traits<MatrixType>::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef std::complex<RealScalar> ComplexScalar;

  VERIFY_IS_APPROX(A.exp(), A.matrixFunction(internal::stem_function_exp<ComplexScalar>));
}

template<typename MatrixType>
void testMatrixLogarithm(const MatrixType& A)
{
  typedef typename internal::traits<MatrixType>::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;

  MatrixType scaledA;
  RealScalar maxImagPartOfSpectrum = A.eigenvalues().imag().cwiseAbs().maxCoeff();
  if (maxImagPartOfSpectrum >= 0.9 * M_PI)
    scaledA = A * 0.9 * M_PI / maxImagPartOfSpectrum;
  else
    scaledA = A;

  // identity X.exp().log() = X only holds if Im(lambda) < pi for all eigenvalues of X
  MatrixType expA = scaledA.exp();
  MatrixType logExpA = expA.log();
  VERIFY_IS_APPROX(logExpA, scaledA);
}

template<typename MatrixType>
void testHyperbolicFunctions(const MatrixType& A)
{
  // Need to use absolute error because of possible cancellation when
  // adding/subtracting expA and expmA.
  VERIFY_IS_APPROX_ABS(A.sinh(), (A.exp() - (-A).exp()) / 2);
  VERIFY_IS_APPROX_ABS(A.cosh(), (A.exp() + (-A).exp()) / 2);
}

template<typename MatrixType>
void testGonioFunctions(const MatrixType& A)
{
  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef std::complex<RealScalar> ComplexScalar;
  typedef Matrix<ComplexScalar, MatrixType::RowsAtCompileTime, 
                 MatrixType::ColsAtCompileTime, MatrixType::Options> ComplexMatrix;

  ComplexScalar imagUnit(0,1);
  ComplexScalar two(2,0);

  ComplexMatrix Ac = A.template cast<ComplexScalar>();
  
  ComplexMatrix exp_iA = (imagUnit * Ac).exp();
  ComplexMatrix exp_miA = (-imagUnit * Ac).exp();
  
  ComplexMatrix sinAc = A.sin().template cast<ComplexScalar>();
  VERIFY_IS_APPROX_ABS(sinAc, (exp_iA - exp_miA) / (two*imagUnit));
  
  ComplexMatrix cosAc = A.cos().template cast<ComplexScalar>();
  VERIFY_IS_APPROX_ABS(cosAc, (exp_iA + exp_miA) / 2);
}

template<typename MatrixType>
void testMatrix(const MatrixType& A)
{
  testMatrixExponential(A);
  testMatrixLogarithm(A);
  testHyperbolicFunctions(A);
  testGonioFunctions(A);
}

template<typename MatrixType>
void testMatrixType(const MatrixType& m)
{
  // Matrices with clustered eigenvalue lead to different code paths
  // in MatrixFunction.h and are thus useful for testing.
  typedef typename MatrixType::Index Index;

  const Index size = m.rows();
  for (int i = 0; i < g_repeat; i++) {
    testMatrix(MatrixType::Random(size, size).eval());
    testMatrix(randomMatrixWithRealEivals<MatrixType>(size));
    testMatrix(randomMatrixWithImagEivals<MatrixType>::run(size));
  }
}

void test_matrix_function()
{
  CALL_SUBTEST_1(testMatrixType(Matrix<float,1,1>()));
  CALL_SUBTEST_2(testMatrixType(Matrix3cf()));
  CALL_SUBTEST_3(testMatrixType(MatrixXf(8,8)));
  CALL_SUBTEST_4(testMatrixType(Matrix2d()));
  CALL_SUBTEST_5(testMatrixType(Matrix<double,5,5,RowMajor>()));
  CALL_SUBTEST_6(testMatrixType(Matrix4cd()));
  CALL_SUBTEST_7(testMatrixType(MatrixXd(13,13)));
}
