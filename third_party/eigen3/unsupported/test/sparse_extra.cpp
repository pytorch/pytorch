// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2010 Gael Guennebaud <g.gael@free.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


// import basic and product tests for deprectaed DynamicSparseMatrix
#define EIGEN_NO_DEPRECATED_WARNING
#include "sparse_basic.cpp"
#include "sparse_product.cpp"
#include <Eigen/SparseExtra>

template<typename SetterType,typename DenseType, typename Scalar, int Options>
bool test_random_setter(SparseMatrix<Scalar,Options>& sm, const DenseType& ref, const std::vector<Vector2i>& nonzeroCoords)
{
  {
    sm.setZero();
    SetterType w(sm);
    std::vector<Vector2i> remaining = nonzeroCoords;
    while(!remaining.empty())
    {
      int i = internal::random<int>(0,static_cast<int>(remaining.size())-1);
      w(remaining[i].x(),remaining[i].y()) = ref.coeff(remaining[i].x(),remaining[i].y());
      remaining[i] = remaining.back();
      remaining.pop_back();
    }
  }
  return sm.isApprox(ref);
}

template<typename SetterType,typename DenseType, typename T>
bool test_random_setter(DynamicSparseMatrix<T>& sm, const DenseType& ref, const std::vector<Vector2i>& nonzeroCoords)
{
  sm.setZero();
  std::vector<Vector2i> remaining = nonzeroCoords;
  while(!remaining.empty())
  {
    int i = internal::random<int>(0,static_cast<int>(remaining.size())-1);
    sm.coeffRef(remaining[i].x(),remaining[i].y()) = ref.coeff(remaining[i].x(),remaining[i].y());
    remaining[i] = remaining.back();
    remaining.pop_back();
  }
  return sm.isApprox(ref);
}

template<typename SparseMatrixType> void sparse_extra(const SparseMatrixType& ref)
{
  const Index rows = ref.rows();
  const Index cols = ref.cols();
  typedef typename SparseMatrixType::Scalar Scalar;
  enum { Flags = SparseMatrixType::Flags };

  double density = (std::max)(8./(rows*cols), 0.01);
  typedef Matrix<Scalar,Dynamic,Dynamic> DenseMatrix;
  typedef Matrix<Scalar,Dynamic,1> DenseVector;
  Scalar eps = 1e-6;

  SparseMatrixType m(rows, cols);
  DenseMatrix refMat = DenseMatrix::Zero(rows, cols);
  DenseVector vec1 = DenseVector::Random(rows);

  std::vector<Vector2i> zeroCoords;
  std::vector<Vector2i> nonzeroCoords;
  initSparse<Scalar>(density, refMat, m, 0, &zeroCoords, &nonzeroCoords);

  if (zeroCoords.size()==0 || nonzeroCoords.size()==0)
    return;

  // test coeff and coeffRef
  for (int i=0; i<(int)zeroCoords.size(); ++i)
  {
    VERIFY_IS_MUCH_SMALLER_THAN( m.coeff(zeroCoords[i].x(),zeroCoords[i].y()), eps );
    if(internal::is_same<SparseMatrixType,SparseMatrix<Scalar,Flags> >::value)
      VERIFY_RAISES_ASSERT( m.coeffRef(zeroCoords[0].x(),zeroCoords[0].y()) = 5 );
  }
  VERIFY_IS_APPROX(m, refMat);

  m.coeffRef(nonzeroCoords[0].x(), nonzeroCoords[0].y()) = Scalar(5);
  refMat.coeffRef(nonzeroCoords[0].x(), nonzeroCoords[0].y()) = Scalar(5);

  VERIFY_IS_APPROX(m, refMat);

  // random setter
//   {
//     m.setZero();
//     VERIFY_IS_NOT_APPROX(m, refMat);
//     SparseSetter<SparseMatrixType, RandomAccessPattern> w(m);
//     std::vector<Vector2i> remaining = nonzeroCoords;
//     while(!remaining.empty())
//     {
//       int i = internal::random<int>(0,remaining.size()-1);
//       w->coeffRef(remaining[i].x(),remaining[i].y()) = refMat.coeff(remaining[i].x(),remaining[i].y());
//       remaining[i] = remaining.back();
//       remaining.pop_back();
//     }
//   }
//   VERIFY_IS_APPROX(m, refMat);

    VERIFY(( test_random_setter<RandomSetter<SparseMatrixType, StdMapTraits> >(m,refMat,nonzeroCoords) ));
    #ifdef EIGEN_UNORDERED_MAP_SUPPORT
    VERIFY(( test_random_setter<RandomSetter<SparseMatrixType, StdUnorderedMapTraits> >(m,refMat,nonzeroCoords) ));
    #endif
    #ifdef _DENSE_HASH_MAP_H_
    VERIFY(( test_random_setter<RandomSetter<SparseMatrixType, GoogleDenseHashMapTraits> >(m,refMat,nonzeroCoords) ));
    #endif
    #ifdef _SPARSE_HASH_MAP_H_
    VERIFY(( test_random_setter<RandomSetter<SparseMatrixType, GoogleSparseHashMapTraits> >(m,refMat,nonzeroCoords) ));
    #endif


  // test RandomSetter
  /*{
    SparseMatrixType m1(rows,cols), m2(rows,cols);
    DenseMatrix refM1 = DenseMatrix::Zero(rows, rows);
    initSparse<Scalar>(density, refM1, m1);
    {
      Eigen::RandomSetter<SparseMatrixType > setter(m2);
      for (int j=0; j<m1.outerSize(); ++j)
        for (typename SparseMatrixType::InnerIterator i(m1,j); i; ++i)
          setter(i.index(), j) = i.value();
    }
    VERIFY_IS_APPROX(m1, m2);
  }*/


}

void test_sparse_extra()
{
  for(int i = 0; i < g_repeat; i++) {
    int s = Eigen::internal::random<int>(1,50);
    CALL_SUBTEST_1( sparse_extra(SparseMatrix<double>(8, 8)) );
    CALL_SUBTEST_2( sparse_extra(SparseMatrix<std::complex<double> >(s, s)) );
    CALL_SUBTEST_1( sparse_extra(SparseMatrix<double>(s, s)) );

    CALL_SUBTEST_3( sparse_extra(DynamicSparseMatrix<double>(s, s)) );
//    CALL_SUBTEST_3(( sparse_basic(DynamicSparseMatrix<double>(s, s)) ));
//    CALL_SUBTEST_3(( sparse_basic(DynamicSparseMatrix<double,ColMajor,long int>(s, s)) ));

    CALL_SUBTEST_3( (sparse_product<DynamicSparseMatrix<float, ColMajor> >()) );
    CALL_SUBTEST_3( (sparse_product<DynamicSparseMatrix<float, RowMajor> >()) );
  }
}
