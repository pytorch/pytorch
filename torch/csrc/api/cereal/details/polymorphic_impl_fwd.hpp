/*! \file polymorphic_impl_fwd.hpp
    \brief Internal polymorphism support forward declarations
    \ingroup Internal */
/*
  Copyright (c) 2014, Randolph Voorhies, Shane Grant
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:
      * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
      * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
      * Neither the name of cereal nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL RANDOLPH VOORHIES OR SHANE GRANT BE LIABLE FOR ANY
  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/* This code is heavily inspired by the boost serialization implementation by the following authors

   (C) Copyright 2002 Robert Ramey - http://www.rrsd.com .
   Use, modification and distribution is subject to the Boost Software
   License, Version 1.0. (See http://www.boost.org/LICENSE_1_0.txt)

    See http://www.boost.org for updates, documentation, and revision history.

   (C) Copyright 2006 David Abrahams - http://www.boost.org.

   See /boost/serialization/export.hpp and /boost/archive/detail/register_archive.hpp for their
   implementation.
*/

#ifndef CEREAL_DETAILS_POLYMORPHIC_IMPL_FWD_HPP_
#define CEREAL_DETAILS_POLYMORPHIC_IMPL_FWD_HPP_

namespace cereal
{
  namespace detail
  {
    //! Forward declaration, see polymorphic_impl.hpp for more information
    template <class Base, class Derived>
    struct RegisterPolymorphicCaster;

    //! Forward declaration, see polymorphic_impl.hpp for more information
    struct PolymorphicCasters;

    //! Forward declaration, see polymorphic_impl.hpp for more information
    template <class Base, class Derived>
    struct PolymorphicRelation;
  } // namespace detail
} // namespace cereal

#endif // CEREAL_DETAILS_POLYMORPHIC_IMPL_FWD_HPP_
