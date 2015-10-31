// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Gael Guennebaud <g.gael@free.fr>
// Copyright (C) 2012 Kolja Brix <brix@igpm.rwth-aaachen.de>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "../../test/sparse_solver.h"
#include <Eigen/IterativeSolvers>

template<typename T> void test_gmres_T()
{
  GMRES<SparseMatrix<T>, DiagonalPreconditioner<T> > gmres_colmajor_diag;
  GMRES<SparseMatrix<T>, IdentityPreconditioner    > gmres_colmajor_I;
  GMRES<SparseMatrix<T>, IncompleteLUT<T> >           gmres_colmajor_ilut;
  //GMRES<SparseMatrix<T>, SSORPreconditioner<T> >     gmres_colmajor_ssor;

  CALL_SUBTEST( check_sparse_square_solving(gmres_colmajor_diag)  );
//   CALL_SUBTEST( check_sparse_square_solving(gmres_colmajor_I)     );
  CALL_SUBTEST( check_sparse_square_solving(gmres_colmajor_ilut)     );
  //CALL_SUBTEST( check_sparse_square_solving(gmres_colmajor_ssor)     );
}

void test_gmres()
{
  CALL_SUBTEST_1(test_gmres_T<double>());
  CALL_SUBTEST_2(test_gmres_T<std::complex<double> >());
}
