// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2013 Christoph Hertzberg <chtz@informatik.uni-bremen.de>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <unsupported/Eigen/AutoDiff>

/*
 * In this file scalar derivations are tested for correctness.
 * TODO add more tests!
 */

template<typename Scalar> void check_atan2()
{
  typedef Matrix<Scalar, 1, 1> Deriv1;
  typedef AutoDiffScalar<Deriv1> AD;
  
  AD x(internal::random<Scalar>(-3.0, 3.0), Deriv1::UnitX());
  
  using std::exp;
  Scalar r = exp(internal::random<Scalar>(-10, 10));
  
  AD s = sin(x), c = cos(x);
  AD res = atan2(r*s, r*c);
  
  VERIFY_IS_APPROX(res.value(), x.value());
  VERIFY_IS_APPROX(res.derivatives(), x.derivatives());
}




void test_autodiff_scalar()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( check_atan2<float>() );
    CALL_SUBTEST_2( check_atan2<double>() );
  }
}
