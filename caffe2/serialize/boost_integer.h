#pragma once

#include "caffe2/serialize/boost_integer_fwd.h" // self include
#include <climits>
#include <cstdint>

#include "caffe2/serialize/boost_integer_traits.h"

//
// We simply cannot include this header on gcc without getting copious warnings of the kind:
//
// boost/integer.hpp:77:30: warning: use of C99 long long integer constant
//
// And yet there is no other reasonable implementation, so we declare this a system header
// to suppress these warnings.
//
#if defined(__GNUC__) && (__GNUC__ >= 4)
#pragma GCC system_header
#endif

namespace boost
{

  //  Helper templates  ------------------------------------------------------//

  //  fast integers from least integers
  //  int_fast_t<> works correctly for unsigned too, in spite of the name.
  template< typename LeastInt >
  struct int_fast_t
  {
     typedef LeastInt fast;
     typedef fast     type;
  }; // imps may specialize

  namespace detail{

  //  convert category to type
  template< int Category > struct int_least_helper {}; // default is empty
  template< int Category > struct uint_least_helper {}; // default is empty

  //  specializatons: 1=long, 2=int, 3=short, 4=signed char,
  //     6=unsigned long, 7=unsigned int, 8=unsigned short, 9=unsigned char
  //  no specializations for 0 and 5: requests for a type > long are in error
#ifdef BOOST_HAS_LONG_LONG
  template<> struct int_least_helper<1> { typedef ::boost::long_long_type least; };
#elif defined(BOOST_HAS_MS_INT64)
  template<> struct int_least_helper<1> { typedef __int64 least; };
#endif
  template<> struct int_least_helper<2> { typedef long least; };
  template<> struct int_least_helper<3> { typedef int least; };
  template<> struct int_least_helper<4> { typedef short least; };
  template<> struct int_least_helper<5> { typedef signed char least; };
#ifdef BOOST_HAS_LONG_LONG
  template<> struct uint_least_helper<1> { typedef ::boost::ulong_long_type least; };
#elif defined(BOOST_HAS_MS_INT64)
  template<> struct uint_least_helper<1> { typedef unsigned __int64 least; };
#endif
  template<> struct uint_least_helper<2> { typedef unsigned long least; };
  template<> struct uint_least_helper<3> { typedef unsigned int least; };
  template<> struct uint_least_helper<4> { typedef unsigned short least; };
  template<> struct uint_least_helper<5> { typedef unsigned char least; };

  template <int Bits>
  struct exact_signed_base_helper{};
  template <int Bits>
  struct exact_unsigned_base_helper{};

  template <> struct exact_signed_base_helper<sizeof(signed char)* CHAR_BIT> { typedef signed char exact; };
  template <> struct exact_unsigned_base_helper<sizeof(unsigned char)* CHAR_BIT> { typedef unsigned char exact; };
#if USHRT_MAX != UCHAR_MAX
  template <> struct exact_signed_base_helper<sizeof(short)* CHAR_BIT> { typedef short exact; };
  template <> struct exact_unsigned_base_helper<sizeof(unsigned short)* CHAR_BIT> { typedef unsigned short exact; };
#endif
#if UINT_MAX != USHRT_MAX
  template <> struct exact_signed_base_helper<sizeof(int)* CHAR_BIT> { typedef int exact; };
  template <> struct exact_unsigned_base_helper<sizeof(unsigned int)* CHAR_BIT> { typedef unsigned int exact; };
#endif
#if ULONG_MAX != UINT_MAX && ( !defined __TI_COMPILER_VERSION__ || \
    ( __TI_COMPILER_VERSION__ >= 7000000 && !defined __TI_40BIT_LONG__ ) )
  template <> struct exact_signed_base_helper<sizeof(long)* CHAR_BIT> { typedef long exact; };
  template <> struct exact_unsigned_base_helper<sizeof(unsigned long)* CHAR_BIT> { typedef unsigned long exact; };
#endif
#if defined(BOOST_HAS_LONG_LONG) &&\
   ((defined(ULLONG_MAX) && (ULLONG_MAX != ULONG_MAX)) ||\
    (defined(ULONG_LONG_MAX) && (ULONG_LONG_MAX != ULONG_MAX)) ||\
    (defined(ULONGLONG_MAX) && (ULONGLONG_MAX != ULONG_MAX)) ||\
    (defined(_ULLONG_MAX) && (_ULLONG_MAX != ULONG_MAX)))
  template <> struct exact_signed_base_helper<sizeof(long_long_type)* CHAR_BIT> { typedef long_long_type exact; };
  template <> struct exact_unsigned_base_helper<sizeof(ulong_long_type)* CHAR_BIT> { typedef ulong_long_type exact; };
#endif


  } // namespace detail

  //  integer templates specifying number of bits  ---------------------------//

  //  signed
  template< int Bits >   // bits (including sign) required
  struct int_t : public boost::detail::exact_signed_base_helper<Bits>
  {
    //   BOOST_STATIC_ASSERT_MSG(Bits <= (int)(sizeof(boost::intmax_t) * CHAR_BIT),
        //  "No suitable signed integer type with the requested number of bits is available.");
      typedef typename boost::detail::int_least_helper
        <
#ifdef BOOST_HAS_LONG_LONG
          (Bits <= (int)(sizeof(boost::long_long_type) * CHAR_BIT)) +
#else
           1 +
#endif
          (Bits-1 <= ::std::numeric_limits<long>::digits) +
          (Bits-1 <= ::std::numeric_limits<int>::digits) +
          (Bits-1 <= ::std::numeric_limits<short>::digits) +
          (Bits-1 <= ::std::numeric_limits<signed char>::digits)
        >::least  least;
      typedef typename int_fast_t<least>::type  fast;
  };

  //  unsigned
  template< int Bits >   // bits required
  struct uint_t : public boost::detail::exact_unsigned_base_helper<Bits>
  {
    //  BOOST_STATIC_ASSERT_MSG(Bits <= (int)(sizeof(boost::uintmax_t) * CHAR_BIT),
        //  "No suitable unsigned integer type with the requested number of bits is available.");
#if (defined(__BORLANDC__) || defined(__CODEGEAR__)) && defined(BOOST_NO_INTEGRAL_INT64_T)
     // It's really not clear why this workaround should be needed... shrug I guess!  JM
     BOOST_STATIC_CONSTANT(int, s =
           6 +
          (Bits <= ::std::numeric_limits<unsigned long>::digits) +
          (Bits <= ::std::numeric_limits<unsigned int>::digits) +
          (Bits <= ::std::numeric_limits<unsigned short>::digits) +
          (Bits <= ::std::numeric_limits<unsigned char>::digits));
     typedef typename detail::int_least_helper< ::boost::uint_t<Bits>::s>::least least;
#else
      typedef typename boost::detail::uint_least_helper
        <
#ifdef BOOST_HAS_LONG_LONG
          (Bits <= (int)(sizeof(boost::long_long_type) * CHAR_BIT)) +
#else
           1 +
#endif
          (Bits <= ::std::numeric_limits<unsigned long>::digits) +
          (Bits <= ::std::numeric_limits<unsigned int>::digits) +
          (Bits <= ::std::numeric_limits<unsigned short>::digits) +
          (Bits <= ::std::numeric_limits<unsigned char>::digits)
        >::least  least;
#endif
      typedef typename int_fast_t<least>::type  fast;
      // int_fast_t<> works correctly for unsigned too, in spite of the name.
  };

  //  integer templates specifying extreme value  ----------------------------//

  //  signed
#if !defined(BOOST_NO_INTEGRAL_INT64_T) && !defined(BOOST_NO_INT64_T) && defined(BOOST_HAS_LONG_LONG)
  template< boost::long_long_type MaxValue >   // maximum value to require support
#else
  template< long MaxValue >   // maximum value to require support
#endif
  struct int_max_value_t
  {
      typedef typename boost::detail::int_least_helper
        <
#if !defined(BOOST_NO_INTEGRAL_INT64_T) && !defined(BOOST_NO_INT64_T) && defined(BOOST_HAS_LONG_LONG)
          (MaxValue <= ::boost::integer_traits<boost::long_long_type>::const_max) +
#else
           1 +
#endif
          (MaxValue <= ::boost::integer_traits<long>::const_max) +
          (MaxValue <= ::boost::integer_traits<int>::const_max) +
          (MaxValue <= ::boost::integer_traits<short>::const_max) +
          (MaxValue <= ::boost::integer_traits<signed char>::const_max)
        >::least  least;
      typedef typename int_fast_t<least>::type  fast;
  };

#if !defined(BOOST_NO_INTEGRAL_INT64_T) && !defined(BOOST_NO_INT64_T) && defined(BOOST_HAS_LONG_LONG)
  template< boost::long_long_type MinValue >   // minimum value to require support
#else
  template< long MinValue >   // minimum value to require support
#endif
  struct int_min_value_t
  {
      typedef typename boost::detail::int_least_helper
        <
#if !defined(BOOST_NO_INTEGRAL_INT64_T) && !defined(BOOST_NO_INT64_T) && defined(BOOST_HAS_LONG_LONG)
          (MinValue >= ::boost::integer_traits<boost::long_long_type>::const_min) +
#else
           1 +
#endif
          (MinValue >= ::boost::integer_traits<long>::const_min) +
          (MinValue >= ::boost::integer_traits<int>::const_min) +
          (MinValue >= ::boost::integer_traits<short>::const_min) +
          (MinValue >= ::boost::integer_traits<signed char>::const_min)
        >::least  least;
      typedef typename int_fast_t<least>::type  fast;
  };

  //  unsigned
#if !defined(BOOST_NO_INTEGRAL_INT64_T) && defined(BOOST_HAS_LONG_LONG)
  template< boost::ulong_long_type MaxValue >   // minimum value to require support
#else
  template< unsigned long MaxValue >   // minimum value to require support
#endif
  struct uint_value_t
  {
#if (defined(__BORLANDC__) || defined(__CODEGEAR__))
     // It's really not clear why this workaround should be needed... shrug I guess!  JM
#if defined(BOOST_NO_INTEGRAL_INT64_T)
      BOOST_STATIC_CONSTANT(unsigned, which =
           1 +
          (MaxValue <= ::boost::integer_traits<unsigned long>::const_max) +
          (MaxValue <= ::boost::integer_traits<unsigned int>::const_max) +
          (MaxValue <= ::boost::integer_traits<unsigned short>::const_max) +
          (MaxValue <= ::boost::integer_traits<unsigned char>::const_max));
      typedef typename detail::int_least_helper< ::boost::uint_value_t<MaxValue>::which>::least least;
#else // BOOST_NO_INTEGRAL_INT64_T
      BOOST_STATIC_CONSTANT(unsigned, which =
           1 +
          (MaxValue <= ::boost::integer_traits<boost::ulong_long_type>::const_max) +
          (MaxValue <= ::boost::integer_traits<unsigned long>::const_max) +
          (MaxValue <= ::boost::integer_traits<unsigned int>::const_max) +
          (MaxValue <= ::boost::integer_traits<unsigned short>::const_max) +
          (MaxValue <= ::boost::integer_traits<unsigned char>::const_max));
      typedef typename detail::uint_least_helper< ::boost::uint_value_t<MaxValue>::which>::least least;
#endif // BOOST_NO_INTEGRAL_INT64_T
#else
      typedef typename boost::detail::uint_least_helper
        <
#if !defined(BOOST_NO_INTEGRAL_INT64_T) && defined(BOOST_HAS_LONG_LONG)
          (MaxValue <= ::boost::integer_traits<boost::ulong_long_type>::const_max) +
#else
           1 +
#endif
          (MaxValue <= ::boost::integer_traits<unsigned long>::const_max) +
          (MaxValue <= ::boost::integer_traits<unsigned int>::const_max) +
          (MaxValue <= ::boost::integer_traits<unsigned short>::const_max) +
          (MaxValue <= ::boost::integer_traits<unsigned char>::const_max)
        >::least  least;
#endif
      typedef typename int_fast_t<least>::type  fast;
  };


} // namespace boost