// Protocol Buffers - Google's data interchange format
// Copyright 2008 Google Inc.  All rights reserved.
// https://developers.google.com/protocol-buffers/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Author: kenton@google.com (Kenton Varda)
//
// Deals with the fact that hash_map is not defined everywhere.

#ifndef GOOGLE_PROTOBUF_STUBS_HASH_H__
#define GOOGLE_PROTOBUF_STUBS_HASH_H__

#include <string.h>
#include <google/protobuf/stubs/common.h>

#define GOOGLE_PROTOBUF_HAVE_HASH_MAP 1
#define GOOGLE_PROTOBUF_HAVE_HASH_SET 1

// Android
#if defined(__ANDROID__)
# undef GOOGLE_PROTOBUF_HAVE_HASH_MAP
# undef GOOGLE_PROTOBUF_HAVE_HASH_MAP

// Use C++11 unordered_{map|set} if available.
#elif ((_LIBCPP_STD_VER >= 11) || \
      (((__cplusplus >= 201103L) || defined(__GXX_EXPERIMENTAL_CXX0X)) && \
      (__GLIBCXX__ > 20090421)))
# define GOOGLE_PROTOBUF_HAS_CXX11_HASH

// For XCode >= 4.6:  the compiler is clang with libc++.
// For earlier XCode version: the compiler is gcc-4.2.1 with libstdc++.
// libc++ provides <unordered_map> and friends even in non C++11 mode,
// and it does not provide the tr1 library. Therefore the following macro
// checks against this special case.
// Note that we should not test the __APPLE_CC__ version number or the
// __clang__ macro, since the new compiler can still use -stdlib=libstdc++, in
// which case <unordered_map> is not compilable without -std=c++11
#elif defined(__APPLE_CC__)
# if __GNUC__ >= 4
#  define GOOGLE_PROTOBUF_HAS_TR1
# else
// Not tested for gcc < 4... These setting can compile under 4.2.1 though.
#  define GOOGLE_PROTOBUF_HASH_NAMESPACE __gnu_cxx
#  include <ext/hash_map>
#  define GOOGLE_PROTOBUF_HASH_MAP_CLASS hash_map
#  include <ext/hash_set>
#  define GOOGLE_PROTOBUF_HASH_SET_CLASS hash_set
# endif

// Version checks for gcc.
#elif defined(__GNUC__)
// For GCC 4.x+, use tr1::unordered_map/set; otherwise, follow the
// instructions from:
// https://gcc.gnu.org/onlinedocs/libstdc++/manual/backwards.html
# if __GNUC__ >= 4
#  define GOOGLE_PROTOBUF_HAS_TR1
# elif __GNUC__ >= 3
#  include <backward/hash_map>
#  define GOOGLE_PROTOBUF_HASH_MAP_CLASS hash_map
#  include <backward/hash_set>
#  define GOOGLE_PROTOBUF_HASH_SET_CLASS hash_set
#  if __GNUC__ == 3 && __GNUC_MINOR__ == 0
#   define GOOGLE_PROTOBUF_HASH_NAMESPACE std       // GCC 3.0
#  else
#   define GOOGLE_PROTOBUF_HASH_NAMESPACE __gnu_cxx // GCC 3.1 and later
#  endif
# else
#  define GOOGLE_PROTOBUF_HASH_NAMESPACE
#  include <hash_map>
#  define GOOGLE_PROTOBUF_HASH_MAP_CLASS hash_map
#  include <hash_set>
#  define GOOGLE_PROTOBUF_HASH_SET_CLASS hash_set
# endif

// Version checks for MSC.
// Apparently Microsoft decided to move hash_map *back* to the std namespace in
// MSVC 2010:
// http://blogs.msdn.com/vcblog/archive/2009/05/25/stl-breaking-changes-in-visual-studio-2010-beta-1.aspx
// And.. they are moved back to stdext in MSVC 2013 (haven't checked 2012). That
// said, use unordered_map for MSVC 2010 and beyond is our safest bet.
#elif defined(_MSC_VER)
# if _MSC_VER >= 1600  // Since Visual Studio 2010
#  define GOOGLE_PROTOBUF_HAS_CXX11_HASH
#  define GOOGLE_PROTOBUF_HASH_COMPARE std::hash_compare
# elif _MSC_VER >= 1500  // Since Visual Studio 2008
#  undef GOOGLE_PROTOBUF_HAVE_HASH_MAP
#  undef GOOGLE_PROTOBUF_HAVE_HASH_SET
# elif _MSC_VER >= 1310
#  define GOOGLE_PROTOBUF_HASH_NAMESPACE stdext
#  include <hash_map>
#  define GOOGLE_PROTOBUF_HASH_MAP_CLASS hash_map
#  include <hash_set>
#  define GOOGLE_PROTOBUF_HASH_SET_CLASS hash_set
#  define GOOGLE_PROTOBUF_HASH_COMPARE stdext::hash_compare
# else
#  define GOOGLE_PROTOBUF_HASH_NAMESPACE std
#  include <hash_map>
#  define GOOGLE_PROTOBUF_HASH_MAP_CLASS hash_map
#  include <hash_set>
#  define GOOGLE_PROTOBUF_HASH_SET_CLASS hash_set
#  define GOOGLE_PROTOBUF_HASH_COMPARE stdext::hash_compare
# endif

// **ADD NEW COMPILERS SUPPORT HERE.**
// For other compilers, undefine the macro and fallback to use std::map, in
// google/protobuf/stubs/hash.h
#else
# undef GOOGLE_PROTOBUF_HAVE_HASH_MAP
# undef GOOGLE_PROTOBUF_HAVE_HASH_SET
#endif

#if defined(GOOGLE_PROTOBUF_HAS_CXX11_HASH)
# define GOOGLE_PROTOBUF_HASH_NAMESPACE std
# include <unordered_map>
# define GOOGLE_PROTOBUF_HASH_MAP_CLASS unordered_map
# include <unordered_set>
# define GOOGLE_PROTOBUF_HASH_SET_CLASS unordered_set
#elif defined(GOOGLE_PROTOBUF_HAS_TR1)
# define GOOGLE_PROTOBUF_HASH_NAMESPACE std::tr1
# include <tr1/unordered_map>
# define GOOGLE_PROTOBUF_HASH_MAP_CLASS unordered_map
# include <tr1/unordered_set>
# define GOOGLE_PROTOBUF_HASH_SET_CLASS unordered_set
#endif

# define GOOGLE_PROTOBUF_HASH_NAMESPACE_DECLARATION_START \
  namespace google {                                      \
  namespace protobuf {
# define GOOGLE_PROTOBUF_HASH_NAMESPACE_DECLARATION_END }}

#undef GOOGLE_PROTOBUF_HAS_CXX11_HASH
#undef GOOGLE_PROTOBUF_HAS_TR1

#if defined(GOOGLE_PROTOBUF_HAVE_HASH_MAP) && \
    defined(GOOGLE_PROTOBUF_HAVE_HASH_SET)
#else
#define GOOGLE_PROTOBUF_MISSING_HASH
#include <map>
#include <set>
#endif

namespace google {
namespace protobuf {

#ifdef GOOGLE_PROTOBUF_MISSING_HASH
#undef GOOGLE_PROTOBUF_MISSING_HASH

// This system doesn't have hash_map or hash_set.  Emulate them using map and
// set.

// Make hash<T> be the same as less<T>.  Note that everywhere where custom
// hash functions are defined in the protobuf code, they are also defined such
// that they can be used as "less" functions, which is required by MSVC anyway.
template <typename Key>
struct hash {
  // Dummy, just to make derivative hash functions compile.
  int operator()(const Key& key) {
    GOOGLE_LOG(FATAL) << "Should never be called.";
    return 0;
  }

  inline bool operator()(const Key& a, const Key& b) const {
    return a < b;
  }
};

// Make sure char* is compared by value.
template <>
struct hash<const char*> {
  // Dummy, just to make derivative hash functions compile.
  int operator()(const char* key) {
    GOOGLE_LOG(FATAL) << "Should never be called.";
    return 0;
  }

  inline bool operator()(const char* a, const char* b) const {
    return strcmp(a, b) < 0;
  }
};

template <typename Key, typename Data,
          typename HashFcn = hash<Key>,
          typename EqualKey = std::equal_to<Key>,
          typename Alloc = std::allocator< std::pair<const Key, Data> > >
class hash_map : public std::map<Key, Data, HashFcn, Alloc> {
  typedef std::map<Key, Data, HashFcn, Alloc> BaseClass;

 public:
  hash_map(int a = 0, const HashFcn& b = HashFcn(),
           const EqualKey& c = EqualKey(),
           const Alloc& d = Alloc()) : BaseClass(b, d) {}
};

template <typename Key,
          typename HashFcn = hash<Key>,
          typename EqualKey = std::equal_to<Key> >
class hash_set : public std::set<Key, HashFcn> {
 public:
  hash_set(int = 0) {}
};

#elif defined(_MSC_VER) && !defined(_STLPORT_VERSION)

template <typename Key>
struct hash : public GOOGLE_PROTOBUF_HASH_COMPARE<Key> {
};

// MSVC's hash_compare<const char*> hashes based on the string contents but
// compares based on the string pointer.  WTF?
class CstringLess {
 public:
  inline bool operator()(const char* a, const char* b) const {
    return strcmp(a, b) < 0;
  }
};

template <>
struct hash<const char*>
    : public GOOGLE_PROTOBUF_HASH_COMPARE<const char*, CstringLess> {};

template <typename Key, typename Data,
          typename HashFcn = hash<Key>,
          typename EqualKey = std::equal_to<Key>,
          typename Alloc = std::allocator< std::pair<const Key, Data> > >
class hash_map
    : public GOOGLE_PROTOBUF_HASH_NAMESPACE::GOOGLE_PROTOBUF_HASH_MAP_CLASS<
          Key, Data, HashFcn, EqualKey, Alloc> {
  typedef GOOGLE_PROTOBUF_HASH_NAMESPACE::GOOGLE_PROTOBUF_HASH_MAP_CLASS<
      Key, Data, HashFcn, EqualKey, Alloc> BaseClass;

 public:
  hash_map(int a = 0, const HashFcn& b = HashFcn(),
           const EqualKey& c = EqualKey(),
           const Alloc& d = Alloc()) : BaseClass(a, b, c, d) {}
};

template <typename Key, typename HashFcn = hash<Key>,
          typename EqualKey = std::equal_to<Key> >
class hash_set
    : public GOOGLE_PROTOBUF_HASH_NAMESPACE::GOOGLE_PROTOBUF_HASH_SET_CLASS<
          Key, HashFcn, EqualKey> {
 public:
  hash_set(int = 0) {}
};

#else

template <typename Key>
struct hash : public GOOGLE_PROTOBUF_HASH_NAMESPACE::hash<Key> {
};

template <typename Key>
struct hash<const Key*> {
  inline size_t operator()(const Key* key) const {
    return reinterpret_cast<size_t>(key);
  }
};

// Unlike the old SGI version, the TR1 "hash" does not special-case char*.  So,
// we go ahead and provide our own implementation.
template <>
struct hash<const char*> {
  inline size_t operator()(const char* str) const {
    size_t result = 0;
    for (; *str != '\0'; str++) {
      result = 5 * result + *str;
    }
    return result;
  }
};

template<>
struct hash<bool> {
  size_t operator()(bool x) const {
    return static_cast<size_t>(x);
  }
};

template <typename Key, typename Data,
          typename HashFcn = hash<Key>,
          typename EqualKey = std::equal_to<Key>,
          typename Alloc = std::allocator< std::pair<const Key, Data> > >
class hash_map
    : public GOOGLE_PROTOBUF_HASH_NAMESPACE::GOOGLE_PROTOBUF_HASH_MAP_CLASS<
          Key, Data, HashFcn, EqualKey, Alloc> {
  typedef GOOGLE_PROTOBUF_HASH_NAMESPACE::GOOGLE_PROTOBUF_HASH_MAP_CLASS<
      Key, Data, HashFcn, EqualKey, Alloc> BaseClass;

 public:
  hash_map(int a = 0, const HashFcn& b = HashFcn(),
           const EqualKey& c = EqualKey(),
           const Alloc& d = Alloc()) : BaseClass(a, b, c, d) {}
};

template <typename Key, typename HashFcn = hash<Key>,
          typename EqualKey = std::equal_to<Key> >
class hash_set
    : public GOOGLE_PROTOBUF_HASH_NAMESPACE::GOOGLE_PROTOBUF_HASH_SET_CLASS<
          Key, HashFcn, EqualKey> {
 public:
  hash_set(int = 0) {}
};

#endif  // !GOOGLE_PROTOBUF_MISSING_HASH

template <>
struct hash<string> {
  inline size_t operator()(const string& key) const {
    return hash<const char*>()(key.c_str());
  }

  static const size_t bucket_size = 4;
  static const size_t min_buckets = 8;
  inline bool operator()(const string& a, const string& b) const {
    return a < b;
  }
};

template <typename First, typename Second>
struct hash<pair<First, Second> > {
  inline size_t operator()(const pair<First, Second>& key) const {
    size_t first_hash = hash<First>()(key.first);
    size_t second_hash = hash<Second>()(key.second);

    // FIXME(kenton):  What is the best way to compute this hash?  I have
    // no idea!  This seems a bit better than an XOR.
    return first_hash * ((1 << 16) - 1) + second_hash;
  }

  static const size_t bucket_size = 4;
  static const size_t min_buckets = 8;
  inline bool operator()(const pair<First, Second>& a,
                           const pair<First, Second>& b) const {
    return a < b;
  }
};

// Used by GCC/SGI STL only.  (Why isn't this provided by the standard
// library?  :( )
struct streq {
  inline bool operator()(const char* a, const char* b) const {
    return strcmp(a, b) == 0;
  }
};

}  // namespace protobuf
}  // namespace google

#endif  // GOOGLE_PROTOBUF_STUBS_HASH_H__
