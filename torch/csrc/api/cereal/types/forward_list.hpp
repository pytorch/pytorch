/*! \file forward_list.hpp
    \brief Support for types found in \<forward_list\>
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
#ifndef CEREAL_TYPES_FORWARD_LIST_HPP_
#define CEREAL_TYPES_FORWARD_LIST_HPP_

#include "cereal/cereal.hpp"
#include <forward_list>

namespace cereal
{
  //! Saving for std::forward_list all other types
  template <class Archive, class T, class A> inline
  void CEREAL_SAVE_FUNCTION_NAME( Archive & ar, std::forward_list<T, A> const & forward_list )
  {
    // write the size - note that this is slow because we need to traverse
    // the entire list. there are ways we could avoid this but this was chosen
    // since it works in the most general fashion with any archive type
    size_type const size = std::distance( forward_list.begin(), forward_list.end() );

    ar( make_size_tag( size ) );

    // write the list
    for( const auto & i : forward_list )
      ar( i );
  }

  //! Loading for std::forward_list all other types from
  template <class Archive, class T, class A>
  void CEREAL_LOAD_FUNCTION_NAME( Archive & ar, std::forward_list<T, A> & forward_list )
  {
    size_type size;
    ar( make_size_tag( size ) );

    forward_list.resize( static_cast<size_t>( size ) );

    for( auto & i : forward_list )
      ar( i );
  }
} // namespace cereal

#endif // CEREAL_TYPES_FORWARD_LIST_HPP_
