// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Giacomo Po <gpo@ucla.edu>
// Copyright (C) 2011 Gael Guennebaud <g.gael@free.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
#include <cmath>

#include "../../test/sparse_solver.h"
#include <Eigen/IterativeSolvers>

template<typename T> void test_minres_T()
{
  // Identity preconditioner
  MINRES<SparseMatrix<T>, Lower, IdentityPreconditioner    > minres_colmajor_lower_I;
  MINRES<SparseMatrix<T>, Upper, IdentityPreconditioner    > minres_colmajor_upper_I;

  // Diagonal preconditioner
  MINRES<SparseMatrix<T>, Lower, DiagonalPreconditioner<T> > minres_colmajor_lower_diag;
  MINRES<SparseMatrix<T>, Upper, DiagonalPreconditioner<T> > minres_colmajor_upper_diag;
  MINRES<SparseMatrix<T>, Lower|Upper, DiagonalPreconditioner<T> > minres_colmajor_uplo_diag;
  
  // call tests for SPD matrix
  CALL_SUBTEST( check_sparse_spd_solving(minres_colmajor_lower_I) );
  CALL_SUBTEST( check_sparse_spd_solving(minres_colmajor_upper_I) );
    
  CALL_SUBTEST( check_sparse_spd_solving(minres_colmajor_lower_diag)  );
  CALL_SUBTEST( check_sparse_spd_solving(minres_colmajor_upper_diag)  );
  CALL_SUBTEST( check_sparse_spd_solving(minres_colmajor_uplo_diag)  );
    
  // TO DO: symmetric semi-definite matrix
  // TO DO: symmetric indefinite matrix

}

void test_minres()
{
  CALL_SUBTEST_1(test_minres_T<double>());
//  CALL_SUBTEST_2(test_minres_T<std::compex<double> >());

}
