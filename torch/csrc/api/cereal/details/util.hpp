/*! \file util.hpp
    \brief Internal misc utilities
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
#ifndef CEREAL_DETAILS_UTIL_HPP_
#define CEREAL_DETAILS_UTIL_HPP_

#include <typeinfo>
#include <string>

#ifdef _MSC_VER
namespace cereal
{
  namespace util
  {
    //! Demangles the type encoded in a string
    /*! @internal */
    inline std::string demangle( std::string const & name )
    { return name; }

    //! Gets the demangled name of a type
    /*! @internal */
    template <class T> inline
    std::string demangledName()
    { return typeid( T ).name(); }
  } // namespace util
} // namespace cereal
#else // clang or gcc
#include <cxxabi.h>
#include <cstdlib>
namespace cereal
{
  namespace util
  {
    //! Demangles the type encoded in a string
    /*! @internal */
    inline std::string demangle(std::string mangledName)
    {
      int status = 0;
      char *demangledName = nullptr;
      std::size_t len;

      demangledName = abi::__cxa_demangle(mangledName.c_str(), 0, &len, &status);

      std::string retName(demangledName);
      free(demangledName);

      return retName;
    }

    //! Gets the demangled name of a type
    /*! @internal */
    template<class T> inline
    std::string demangledName()
    { return demangle(typeid(T).name()); }
  }
} // namespace cereal
#endif // clang or gcc branch of _MSC_VER
#endif // CEREAL_DETAILS_UTIL_HPP_
