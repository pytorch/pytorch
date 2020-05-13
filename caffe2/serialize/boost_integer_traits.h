/* boost integer_traits.hpp header file
 *
 * Copyright Jens Maurer 2000
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * $Id$
 *
 * Idea by Beman Dawes, Ed Brey, Steve Cleary, and Nathan Myers
 */
//  See http://www.boost.org/libs/integer for documentation.
#pragma once
#include <climits>
#include <limits>
#include <cwchar>
//
// We simply cannot include this header on gcc without getting copious warnings of the kind:
//
// ../../../boost/integer_traits.hpp:164:66: warning: use of C99 long long integer constant
//
// And yet there is no other reasonable implementation, so we declare this a system header
// to suppress these warnings.
//
#if defined(__GNUC__) && (__GNUC__ >= 4)
#pragma GCC system_header
#endif
namespace boost {
template<class T>
class integer_traits : public std::numeric_limits<T>
{
public:
  static bool constexpr is_integral = false;
};
namespace detail {
template<class T, T min_val, T max_val>
class integer_traits_base
{
public:
    static bool constexpr is_integral = true;
    static T constexpr const_min = min_val;
    static T constexpr const_max = max_val;
};
#ifndef BOOST_NO_INCLASS_MEMBER_INITIALIZATION
//  A definition is required even for integral static constants
template<class T, T min_val, T max_val>
const bool integer_traits_base<T, min_val, max_val>::is_integral;
template<class T, T min_val, T max_val>
const T integer_traits_base<T, min_val, max_val>::const_min;
template<class T, T min_val, T max_val>
const T integer_traits_base<T, min_val, max_val>::const_max;
#endif
} // namespace detail
template<>
class integer_traits<bool>
  : public std::numeric_limits<bool>,
    public detail::integer_traits_base<bool, false, true>
{ };
template<>
class integer_traits<char>
  : public std::numeric_limits<char>,
    public detail::integer_traits_base<char, CHAR_MIN, CHAR_MAX>
{ };
template<>
class integer_traits<signed char>
  : public std::numeric_limits<signed char>,
    public detail::integer_traits_base<signed char, SCHAR_MIN, SCHAR_MAX>
{ };
template<>
class integer_traits<unsigned char>
  : public std::numeric_limits<unsigned char>,
    public detail::integer_traits_base<unsigned char, 0, UCHAR_MAX>
{ };
#ifndef BOOST_NO_INTRINSIC_WCHAR_T
template<>
class integer_traits<wchar_t>
  : public std::numeric_limits<wchar_t>,
    // Don't trust WCHAR_MIN and WCHAR_MAX with Mac OS X's native
    // library: they are wrong!
#if defined(WCHAR_MIN) && defined(WCHAR_MAX) && !defined(__APPLE__)
    public detail::integer_traits_base<wchar_t, WCHAR_MIN, WCHAR_MAX>
#elif defined(__BORLANDC__) || defined(__CYGWIN__) || defined(__MINGW32__) || (defined(__BEOS__) && defined(__GNUC__))
    // No WCHAR_MIN and WCHAR_MAX, whar_t is short and unsigned:
    public detail::integer_traits_base<wchar_t, 0, 0xffff>
#elif (defined(__sgi) && (!defined(__SGI_STL_PORT) || __SGI_STL_PORT < 0x400))\
    || (defined __APPLE__)\
    || (defined(__OpenBSD__) && defined(__GNUC__))\
    || (defined(__NetBSD__) && defined(__GNUC__))\
    || (defined(__FreeBSD__) && defined(__GNUC__))\
    || (defined(__DragonFly__) && defined(__GNUC__))\
    || (defined(__hpux) && defined(__GNUC__) && (__GNUC__ == 3) && !defined(__SGI_STL_PORT))
    // No WCHAR_MIN and WCHAR_MAX, wchar_t has the same range as int.
    //  - SGI MIPSpro with native library
    //  - gcc 3.x on HP-UX
    //  - Mac OS X with native library
    //  - gcc on FreeBSD, OpenBSD and NetBSD
    public detail::integer_traits_base<wchar_t, INT_MIN, INT_MAX>
#else
#error No WCHAR_MIN and WCHAR_MAX present, please adjust integer_traits<> for your compiler.
#endif
{ };
#endif // BOOST_NO_INTRINSIC_WCHAR_T
template<>
class integer_traits<short>
  : public std::numeric_limits<short>,
    public detail::integer_traits_base<short, SHRT_MIN, SHRT_MAX>
{ };
template<>
class integer_traits<unsigned short>
  : public std::numeric_limits<unsigned short>,
    public detail::integer_traits_base<unsigned short, 0, USHRT_MAX>
{ };
template<>
class integer_traits<int>
  : public std::numeric_limits<int>,
    public detail::integer_traits_base<int, INT_MIN, INT_MAX>
{ };
template<>
class integer_traits<unsigned int>
  : public std::numeric_limits<unsigned int>,
    public detail::integer_traits_base<unsigned int, 0, UINT_MAX>
{ };
template<>
class integer_traits<long>
  : public std::numeric_limits<long>,
    public detail::integer_traits_base<long, LONG_MIN, LONG_MAX>
{ };
template<>
class integer_traits<unsigned long>
  : public std::numeric_limits<unsigned long>,
    public detail::integer_traits_base<unsigned long, 0, ULONG_MAX>
{ };
#if !defined(BOOST_NO_INTEGRAL_INT64_T) && !defined(BOOST_NO_INT64_T)
#if defined(ULLONG_MAX) && defined(BOOST_HAS_LONG_LONG)
template<>
class integer_traits< ::boost::long_long_type>
  : public std::numeric_limits< ::boost::long_long_type>,
    public detail::integer_traits_base< ::boost::long_long_type, LLONG_MIN, LLONG_MAX>
{ };
template<>
class integer_traits< ::boost::ulong_long_type>
  : public std::numeric_limits< ::boost::ulong_long_type>,
    public detail::integer_traits_base< ::boost::ulong_long_type, 0, ULLONG_MAX>
{ };
#elif defined(ULONG_LONG_MAX) && defined(BOOST_HAS_LONG_LONG)
template<>
class integer_traits< ::boost::long_long_type>  : public std::numeric_limits< ::boost::long_long_type>,    public detail::integer_traits_base< ::boost::long_long_type, LONG_LONG_MIN, LONG_LONG_MAX>{ };
template<>
class integer_traits< ::boost::ulong_long_type>
  : public std::numeric_limits< ::boost::ulong_long_type>,
    public detail::integer_traits_base< ::boost::ulong_long_type, 0, ULONG_LONG_MAX>
{ };
#elif defined(ULONGLONG_MAX) && defined(BOOST_HAS_LONG_LONG)
template<>
class integer_traits< ::boost::long_long_type>
  : public std::numeric_limits< ::boost::long_long_type>,
    public detail::integer_traits_base< ::boost::long_long_type, LONGLONG_MIN, LONGLONG_MAX>
{ };
template<>
class integer_traits< ::boost::ulong_long_type>
  : public std::numeric_limits< ::boost::ulong_long_type>,
    public detail::integer_traits_base< ::boost::ulong_long_type, 0, ULONGLONG_MAX>
{ };
#elif defined(_LLONG_MAX) && defined(_C2) && defined(BOOST_HAS_LONG_LONG)
template<>
class integer_traits< ::boost::long_long_type>
  : public std::numeric_limits< ::boost::long_long_type>,
    public detail::integer_traits_base< ::boost::long_long_type, -_LLONG_MAX - _C2, _LLONG_MAX>
{ };
template<>
class integer_traits< ::boost::ulong_long_type>
  : public std::numeric_limits< ::boost::ulong_long_type>,
    public detail::integer_traits_base< ::boost::ulong_long_type, 0, _ULLONG_MAX>
{ };
#elif defined(BOOST_HAS_LONG_LONG)
//
// we have long long but no constants, this happens for example with gcc in -ansi mode,
// we'll just have to work out the values for ourselves (assumes 2's compliment representation):
//
template<>
class integer_traits< ::boost::long_long_type>
  : public std::numeric_limits< ::boost::long_long_type>,
    public detail::integer_traits_base< ::boost::long_long_type, (1LL << (sizeof(::boost::long_long_type) * CHAR_BIT - 1)), ~(1LL << (sizeof(::boost::long_long_type) * CHAR_BIT - 1))>
{ };
template<>
class integer_traits< ::boost::ulong_long_type>
  : public std::numeric_limits< ::boost::ulong_long_type>,
    public detail::integer_traits_base< ::boost::ulong_long_type, 0, ~0uLL>
{ };
#elif defined(BOOST_HAS_MS_INT64)
template<>
class integer_traits< __int64>
  : public std::numeric_limits< __int64>,
    public detail::integer_traits_base< __int64, _I64_MIN, _I64_MAX>
{ };
template<>
class integer_traits< unsigned __int64>
  : public std::numeric_limits< unsigned __int64>,
    public detail::integer_traits_base< unsigned __int64, 0, _UI64_MAX>
{ };
#endif
#endif
} // namespace boost
