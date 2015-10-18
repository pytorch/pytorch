// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Manuel Yguel <manuel.yguel@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <unsupported/Eigen/Polynomials>
#include <iostream>

using namespace std;

namespace Eigen {
namespace internal {
template<int Size>
struct increment_if_fixed_size
{
  enum {
    ret = (Size == Dynamic) ? Dynamic : Size+1
  };
};
}
}

template<typename _Scalar, int _Deg>
void realRoots_to_monicPolynomial_test(int deg)
{
  typedef internal::increment_if_fixed_size<_Deg>            Dim;
  typedef Matrix<_Scalar,Dim::ret,1>                  PolynomialType;
  typedef Matrix<_Scalar,_Deg,1>                      EvalRootsType;

  PolynomialType pols(deg+1);
  EvalRootsType roots = EvalRootsType::Random(deg);
  roots_to_monicPolynomial( roots, pols );

  EvalRootsType evr( deg );
  for( int i=0; i<roots.size(); ++i ){
    evr[i] = std::abs( poly_eval( pols, roots[i] ) ); }

  bool evalToZero = evr.isZero( test_precision<_Scalar>() );
  if( !evalToZero ){
    cerr << evr.transpose() << endl; }
  VERIFY( evalToZero );
}

template<typename _Scalar> void realRoots_to_monicPolynomial_scalar()
{
  CALL_SUBTEST_2( (realRoots_to_monicPolynomial_test<_Scalar,2>(2)) );
  CALL_SUBTEST_3( (realRoots_to_monicPolynomial_test<_Scalar,3>(3)) );
  CALL_SUBTEST_4( (realRoots_to_monicPolynomial_test<_Scalar,4>(4)) );
  CALL_SUBTEST_5( (realRoots_to_monicPolynomial_test<_Scalar,5>(5)) );
  CALL_SUBTEST_6( (realRoots_to_monicPolynomial_test<_Scalar,6>(6)) );
  CALL_SUBTEST_7( (realRoots_to_monicPolynomial_test<_Scalar,7>(7)) );
  CALL_SUBTEST_8( (realRoots_to_monicPolynomial_test<_Scalar,17>(17)) );

  CALL_SUBTEST_9( (realRoots_to_monicPolynomial_test<_Scalar,Dynamic>(
          internal::random<int>(18,26) )) );
}




template<typename _Scalar, int _Deg>
void CauchyBounds(int deg)
{
  typedef internal::increment_if_fixed_size<_Deg>            Dim;
  typedef Matrix<_Scalar,Dim::ret,1>                  PolynomialType;
  typedef Matrix<_Scalar,_Deg,1>                      EvalRootsType;

  PolynomialType pols(deg+1);
  EvalRootsType roots = EvalRootsType::Random(deg);
  roots_to_monicPolynomial( roots, pols );
  _Scalar M = cauchy_max_bound( pols );
  _Scalar m = cauchy_min_bound( pols );
  _Scalar Max = roots.array().abs().maxCoeff();
  _Scalar min = roots.array().abs().minCoeff();
  bool eval = (M >= Max) && (m <= min);
  if( !eval )
  {
    cerr << "Roots: " << roots << endl;
    cerr << "Bounds: (" << m << ", " << M << ")" << endl;
    cerr << "Min,Max: (" << min << ", " << Max << ")" << endl;
  }
  VERIFY( eval );
}

template<typename _Scalar> void CauchyBounds_scalar()
{
  CALL_SUBTEST_2( (CauchyBounds<_Scalar,2>(2)) );
  CALL_SUBTEST_3( (CauchyBounds<_Scalar,3>(3)) );
  CALL_SUBTEST_4( (CauchyBounds<_Scalar,4>(4)) );
  CALL_SUBTEST_5( (CauchyBounds<_Scalar,5>(5)) );
  CALL_SUBTEST_6( (CauchyBounds<_Scalar,6>(6)) );
  CALL_SUBTEST_7( (CauchyBounds<_Scalar,7>(7)) );
  CALL_SUBTEST_8( (CauchyBounds<_Scalar,17>(17)) );

  CALL_SUBTEST_9( (CauchyBounds<_Scalar,Dynamic>(
          internal::random<int>(18,26) )) );
}

void test_polynomialutils()
{
  for(int i = 0; i < g_repeat; i++)
  {
    realRoots_to_monicPolynomial_scalar<double>();
    realRoots_to_monicPolynomial_scalar<float>();
    CauchyBounds_scalar<double>();
    CauchyBounds_scalar<float>();
  }
}
