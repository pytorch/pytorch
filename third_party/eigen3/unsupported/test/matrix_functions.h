// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009-2011 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <unsupported/Eigen/MatrixFunctions>

// For complex matrices, any matrix is fine.
template<typename MatrixType, int IsComplex = NumTraits<typename internal::traits<MatrixType>::Scalar>::IsComplex>
struct processTriangularMatrix
{
  static void run(MatrixType&, MatrixType&, const MatrixType&)
  { }
};

// For real matrices, make sure none of the eigenvalues are negative.
template<typename MatrixType>
struct processTriangularMatrix<MatrixType,0>
{
  static void run(MatrixType& m, MatrixType& T, const MatrixType& U)
  {
    const Index size = m.cols();

    for (Index i=0; i < size; ++i) {
      if (i == size - 1 || T.coeff(i+1,i) == 0)
        T.coeffRef(i,i) = std::abs(T.coeff(i,i));
      else
        ++i;
    }
    m = U * T * U.transpose();
  }
};

template <typename MatrixType, int IsComplex = NumTraits<typename internal::traits<MatrixType>::Scalar>::IsComplex>
struct generateTestMatrix;

template <typename MatrixType>
struct generateTestMatrix<MatrixType,0>
{
  static void run(MatrixType& result, typename MatrixType::Index size)
  {
    result = MatrixType::Random(size, size);
    RealSchur<MatrixType> schur(result);
    MatrixType T = schur.matrixT();
    processTriangularMatrix<MatrixType>::run(result, T, schur.matrixU());
  }
};

template <typename MatrixType>
struct generateTestMatrix<MatrixType,1>
{
  static void run(MatrixType& result, typename MatrixType::Index size)
  {
    result = MatrixType::Random(size, size);
  }
};

template <typename Derived, typename OtherDerived>
double relerr(const MatrixBase<Derived>& A, const MatrixBase<OtherDerived>& B)
{
  return std::sqrt((A - B).cwiseAbs2().sum() / (std::min)(A.cwiseAbs2().sum(), B.cwiseAbs2().sum()));
}
