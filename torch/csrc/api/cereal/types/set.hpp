/*! \file set.hpp
    \brief Support for types found in \<set\>
    \ingroup STLSupport */
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
#ifndef CEREAL_TYPES_SET_HPP_
#define CEREAL_TYPES_SET_HPP_

#include "cereal/cereal.hpp"
#include <set>

namespace cereal
{
  namespace set_detail
  {
    //! @internal
    template <class Archive, class SetT> inline
    void save( Archive & ar, SetT const & set )
    {
      ar( make_size_tag( static_cast<size_type>(set.size()) ) );

      for( const auto & i : set )
        ar( i );
    }

    //! @internal
    template <class Archive, class SetT> inline
    void load( Archive & ar, SetT & set )
    {
      size_type size;
      ar( make_size_tag( size ) );

      set.clear();

      auto hint = set.begin();
      for( size_type i = 0; i < size; ++i )
      {
        typename SetT::key_type key;

        ar( key );
        #ifdef CEREAL_OLDER_GCC
        hint = set.insert( hint, std::move( key ) );
        #else // NOT CEREAL_OLDER_GCC
        hint = set.emplace_hint( hint, std::move( key ) );
        #endif // NOT CEREAL_OLDER_GCC
      }
    }
  }

  //! Saving for std::set
  template <class Archive, class K, class C, class A> inline
  void CEREAL_SAVE_FUNCTION_NAME( Archive & ar, std::set<K, C, A> const & set )
  {
    set_detail::save( ar, set );
  }

  //! Loading for std::set
  template <class Archive, class K, class C, class A> inline
  void CEREAL_LOAD_FUNCTION_NAME( Archive & ar, std::set<K, C, A> & set )
  {
    set_detail::load( ar, set );
  }

  //! Saving for std::multiset
  template <class Archive, class K, class C, class A> inline
  void CEREAL_SAVE_FUNCTION_NAME( Archive & ar, std::multiset<K, C, A> const & multiset )
  {
    set_detail::save( ar, multiset );
  }

  //! Loading for std::multiset
  template <class Archive, class K, class C, class A> inline
  void CEREAL_LOAD_FUNCTION_NAME( Archive & ar, std::multiset<K, C, A> & multiset )
  {
    set_detail::load( ar, multiset );
  }
} // namespace cereal

#endif // CEREAL_TYPES_SET_HPP_
