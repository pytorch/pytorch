// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Eugene Brevdo <ebrevdo@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SPECIAL_FUNCTIONS_H
#define EIGEN_SPECIAL_FUNCTIONS_H

namespace Eigen {
namespace internal {

/****************************************************************************
 * Implementation of lgamma                                                 *
 ****************************************************************************/

template<typename Scalar>
struct lgamma_impl
{
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar run(const Scalar&)
  {
    EIGEN_STATIC_ASSERT((internal::is_same<Scalar, Scalar>::value == false),
                        THIS_TYPE_IS_NOT_SUPPORTED);
    return Scalar(0);
  }
};

template<typename Scalar>
struct lgamma_retval
{
  typedef Scalar type;
};

#ifdef EIGEN_HAS_C99_MATH
template<>
struct lgamma_impl<float>
{
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE double run(const float& x) { return ::lgammaf(x); }
};

template<>
struct lgamma_impl<double>
{
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE double run(const double& x) { return ::lgamma(x); }
};
#endif

/****************************************************************************
 * Implementation of erf                                                    *
 ****************************************************************************/

template<typename Scalar>
struct erf_impl
{
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar run(const Scalar&)
  {
    EIGEN_STATIC_ASSERT((internal::is_same<Scalar, Scalar>::value == false),
                        THIS_TYPE_IS_NOT_SUPPORTED);
    return Scalar(0);
  }
};

template<typename Scalar>
struct erf_retval
{
  typedef Scalar type;
};

#ifdef EIGEN_HAS_C99_MATH
template<>
struct erf_impl<float>
{
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE float run(const float& x) { return ::erff(x); }
};

template<>
struct erf_impl<double>
{
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE double run(const double& x) { return ::erf(x); }
};
#endif  // EIGEN_HAS_C99_MATH

/***************************************************************************
* Implementation of erfc                                                   *
****************************************************************************/

template<typename Scalar>
struct erfc_impl
{
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar run(const Scalar&)
  {
    EIGEN_STATIC_ASSERT((internal::is_same<Scalar, Scalar>::value == false),
                        THIS_TYPE_IS_NOT_SUPPORTED);
    return Scalar(0);
  }
};

template<typename Scalar>
struct erfc_retval
{
  typedef Scalar type;
};

#ifdef EIGEN_HAS_C99_MATH
template<>
struct erfc_impl<float>
{
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE float run(const float x) { return ::erfcf(x); }
};

template<>
struct erfc_impl<double>
{
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE double run(const double x) { return ::erfc(x); }
};
#endif  // EIGEN_HAS_C99_MATH

}  // end namespace internal


namespace numext {

template<typename Scalar>
EIGEN_DEVICE_FUNC
inline EIGEN_MATHFUNC_RETVAL(lgamma, Scalar) lgamma(const Scalar& x)
{
  return EIGEN_MATHFUNC_IMPL(lgamma, Scalar)::run(x);
}

template<typename Scalar>
EIGEN_DEVICE_FUNC
inline EIGEN_MATHFUNC_RETVAL(erf, Scalar) erf(const Scalar& x)
{
  return EIGEN_MATHFUNC_IMPL(erf, Scalar)::run(x);
}

template<typename Scalar>
EIGEN_DEVICE_FUNC
inline EIGEN_MATHFUNC_RETVAL(erfc, Scalar) erfc(const Scalar& x)
{
  return EIGEN_MATHFUNC_IMPL(erfc, Scalar)::run(x);
}

}  // end namespace numext

}  // end namespace Eigen

#endif  // EIGEN_SPECIAL_FUNCTIONS_H
