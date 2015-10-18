// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// #define EIGEN_DONT_VECTORIZE
// #define EIGEN_MAX_ALIGN_BYTES 0
#include "sparse_solver.h"
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

template<typename T, typename I> void test_incomplete_cholesky_T()
{
  typedef SparseMatrix<T,0,I> SparseMatrixType;
  ConjugateGradient<SparseMatrixType, Lower, IncompleteCholesky<T, Lower, AMDOrdering<I> > >     cg_illt_lower_amd;
  ConjugateGradient<SparseMatrixType, Lower, IncompleteCholesky<T, Lower, NaturalOrdering<I> > > cg_illt_lower_nat;
  ConjugateGradient<SparseMatrixType, Upper, IncompleteCholesky<T, Upper, AMDOrdering<I> > >     cg_illt_upper_amd;
  ConjugateGradient<SparseMatrixType, Upper, IncompleteCholesky<T, Upper, AMDOrdering<I> > >     cg_illt_upper_nat;
  

  CALL_SUBTEST( check_sparse_spd_solving(cg_illt_lower_amd) );
  CALL_SUBTEST( check_sparse_spd_solving(cg_illt_lower_nat) );
  CALL_SUBTEST( check_sparse_spd_solving(cg_illt_upper_amd) );
  CALL_SUBTEST( check_sparse_spd_solving(cg_illt_upper_nat) );
}

void test_incomplete_cholesky()
{
  CALL_SUBTEST_1(( test_incomplete_cholesky_T<double,int>() ));
  CALL_SUBTEST_2(( test_incomplete_cholesky_T<std::complex<double>, int>() ));
  CALL_SUBTEST_3(( test_incomplete_cholesky_T<double,long int>() ));
}
