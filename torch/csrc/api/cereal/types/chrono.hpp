/*! \file chrono.hpp
    \brief Support for types found in \<chrono\>
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
#ifndef CEREAL_TYPES_CHRONO_HPP_
#define CEREAL_TYPES_CHRONO_HPP_

#include <chrono>

namespace cereal
{
  //! Saving std::chrono::duration
  template <class Archive, class R, class P> inline
  void CEREAL_SAVE_FUNCTION_NAME( Archive & ar, std::chrono::duration<R, P> const & dur )
  {
    ar( CEREAL_NVP_("count", dur.count()) );
  }

  //! Loading std::chrono::duration
  template <class Archive, class R, class P> inline
  void CEREAL_LOAD_FUNCTION_NAME( Archive & ar, std::chrono::duration<R, P> & dur )
  {
    R count;
    ar( CEREAL_NVP_("count", count) );

    dur = std::chrono::duration<R, P>{count};
  }

  //! Saving std::chrono::time_point
  template <class Archive, class C, class D> inline
  void CEREAL_SAVE_FUNCTION_NAME( Archive & ar, std::chrono::time_point<C, D> const & dur )
  {
    ar( CEREAL_NVP_("time_since_epoch", dur.time_since_epoch()) );
  }

  //! Loading std::chrono::time_point
  template <class Archive, class C, class D> inline
  void CEREAL_LOAD_FUNCTION_NAME( Archive & ar, std::chrono::time_point<C, D> & dur )
  {
    D elapsed;
    ar( CEREAL_NVP_("time_since_epoch", elapsed) );

    dur = std::chrono::time_point<C, D>{elapsed};
  }
} // namespace cereal

#endif // CEREAL_TYPES_CHRONO_HPP_
