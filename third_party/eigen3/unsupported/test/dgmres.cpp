// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Gael Guennebaud <g.gael@free.fr>
// Copyright (C) 2012 desire Nuentsa <desire.nuentsa_wakam@inria.fr
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "../../test/sparse_solver.h"
#include <Eigen/src/IterativeSolvers/DGMRES.h>

template<typename T> void test_dgmres_T()
{
  DGMRES<SparseMatrix<T>, DiagonalPreconditioner<T> > dgmres_colmajor_diag;
  DGMRES<SparseMatrix<T>, IdentityPreconditioner    > dgmres_colmajor_I;
  DGMRES<SparseMatrix<T>, IncompleteLUT<T> >           dgmres_colmajor_ilut;
  //GMRES<SparseMatrix<T>, SSORPreconditioner<T> >     dgmres_colmajor_ssor;

  CALL_SUBTEST( check_sparse_square_solving(dgmres_colmajor_diag)  );
//   CALL_SUBTEST( check_sparse_square_solving(dgmres_colmajor_I)     );
  CALL_SUBTEST( check_sparse_square_solving(dgmres_colmajor_ilut)     );
  //CALL_SUBTEST( check_sparse_square_solving(dgmres_colmajor_ssor)     );
}

void test_dgmres()
{
  CALL_SUBTEST_1(test_dgmres_T<double>());
  CALL_SUBTEST_2(test_dgmres_T<std::complex<double> >());
}
