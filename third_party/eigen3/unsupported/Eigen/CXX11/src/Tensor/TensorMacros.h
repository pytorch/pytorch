// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_META_MACROS_H
#define EIGEN_CXX11_TENSOR_TENSOR_META_MACROS_H


/** use this macro in sfinae selection in templated functions
 *
 *   template<typename T,
 *            typename std::enable_if< isBanana<T>::value , int >::type = 0
 *   >
 *   void foo(){}
 *
 *   becomes =>
 *
 *   template<typename TopoType,
 *           SFINAE_ENABLE_IF( isBanana<T>::value )
 *   >
 *   void foo(){}
 */

// SFINAE requires variadic templates
#ifndef __CUDACC__
#ifdef EIGEN_HAS_VARIADIC_TEMPLATES
  // SFINAE doesn't work for gcc <= 4.7
  #ifdef EIGEN_COMP_GNUC
    #if EIGEN_GNUC_AT_LEAST(4,8)
      #define EIGEN_HAS_SFINAE
    #endif
  #else
    #define EIGEN_HAS_SFINAE
  #endif
#endif
#endif

#define EIGEN_SFINAE_ENABLE_IF( __condition__ ) \
    typename internal::enable_if< ( __condition__ ) , int >::type = 0


#if defined(EIGEN_HAS_CONSTEXPR)
#define EIGEN_CONSTEXPR constexpr
#else
#define EIGEN_CONSTEXPR
#endif


#endif
