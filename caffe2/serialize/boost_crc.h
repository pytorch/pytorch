//  Boost CRC library crc.hpp header file  -----------------------------------//

//  Copyright 2001, 2004 Daryle Walker.  Use, modification, and distribution are
//  subject to the Boost Software License, Version 1.0.  (See accompanying file
#pragma once

#include <climits>  // for CHAR_BIT, etc.
#include <cstddef>  // for std::size_t
#include "caffe2/serialize/boost_integer.h"
#include "caffe2/serialize/boost_integer_traits.h"
#include <limits> 

// The type of CRC parameters that can go in a template should be related
// on the CRC's bit count.  This macro expresses that type in a compact
// form, but also allows an alternate type for compilers that don't support
// dependent types (in template value-parameters).
#if !(defined(BOOST_NO_DEPENDENT_TYPES_IN_TEMPLATE_VALUE_PARAMETERS) || (defined(BOOST_MSVC) && (BOOST_MSVC <= 1300)))
#define CRC_PARM_TYPE  typename ::boost::uint_t<Bits>::fast
// #else
// #define CRC_PARM_TYPE  unsigned long
// #endif

// Some compilers [MS VC++ 6] cannot correctly set up several versions of a
// function template unless every template argument can be unambiguously
// deduced from the function arguments.  (The bug is hidden if only one version
// is needed.)  Since all of the CRC function templates have this problem, the
// workaround is to make up a dummy function argument that encodes the template
// arguments.  Calls to such template functions need all their template
// arguments explicitly specified.  At least one compiler that needs this
// workaround also needs the default value for the dummy argument to be
// specified in the definition.
#if defined(__GNUC__) || !defined(BOOST_NO_EXPLICIT_FUNCTION_TEMPLATE_ARGUMENTS)
#define CRC_DUMMY_PARM_TYPE
#define CRC_DUMMY_INIT
#define ACRC_DUMMY_PARM_TYPE
#define ACRC_DUMMY_INIT
#else
namespace boost { namespace detail {
    template < std::size_t Bits, CRC_PARM_TYPE TruncPoly,
     CRC_PARM_TYPE InitRem, CRC_PARM_TYPE FinalXor,
     bool ReflectIn, bool ReflectRem >
    struct dummy_crc_argument  { };
} }
#define CRC_DUMMY_PARM_TYPE   , detail::dummy_crc_argument<Bits, \
 TruncPoly, InitRem, FinalXor, ReflectIn, ReflectRem> *p_
#define CRC_DUMMY_INIT        CRC_DUMMY_PARM_TYPE = 0
#define ACRC_DUMMY_PARM_TYPE  , detail::dummy_crc_argument<Bits, \
 TruncPoly, 0, 0, false, false> *p_
#define ACRC_DUMMY_INIT       ACRC_DUMMY_PARM_TYPE = 0
#endif


namespace boost
{


//  Forward declarations  ----------------------------------------------------//

template < std::size_t Bits >
    class crc_basic;

template < std::size_t Bits, CRC_PARM_TYPE TruncPoly = 0u,
           CRC_PARM_TYPE InitRem = 0u,
           CRC_PARM_TYPE FinalXor = 0u, bool ReflectIn = false,
           bool ReflectRem = false >
    class crc_optimal;

template < std::size_t Bits, CRC_PARM_TYPE TruncPoly,
           CRC_PARM_TYPE InitRem, CRC_PARM_TYPE FinalXor,
           bool ReflectIn, bool ReflectRem >
    typename uint_t<Bits>::fast  crc( void const *buffer,
     std::size_t byte_count
     CRC_DUMMY_PARM_TYPE );

template < std::size_t Bits, CRC_PARM_TYPE TruncPoly >
    typename uint_t<Bits>::fast  augmented_crc( void const *buffer,
     std::size_t byte_count, typename uint_t<Bits>::fast initial_remainder
     ACRC_DUMMY_PARM_TYPE );

template < std::size_t Bits, CRC_PARM_TYPE TruncPoly >
    typename uint_t<Bits>::fast  augmented_crc( void const *buffer,
     std::size_t byte_count
     ACRC_DUMMY_PARM_TYPE );

typedef crc_optimal<16, 0x8005, 0, 0, true, true>         crc_16_type;
typedef crc_optimal<16, 0x1021, 0xFFFF, 0, false, false>  crc_ccitt_type;
typedef crc_optimal<16, 0x8408, 0, 0, true, true>         crc_xmodem_type;

typedef crc_optimal<32, 0x04C11DB7, 0xFFFFFFFF, 0xFFFFFFFF, true, true>
  crc_32_type;


//  Forward declarations for implementation detail stuff  --------------------//
//  (Just for the stuff that will be needed for the next two sections)

namespace detail
{
    template < std::size_t Bits >
        struct mask_uint_t;

    template <  >
        struct mask_uint_t< std::numeric_limits<unsigned char>::digits >;

    #if USHRT_MAX > UCHAR_MAX
    template <  >
        struct mask_uint_t< std::numeric_limits<unsigned short>::digits >;
    #endif

    #if UINT_MAX > USHRT_MAX
    template <  >
        struct mask_uint_t< std::numeric_limits<unsigned int>::digits >;
    #endif

    #if ULONG_MAX > UINT_MAX
    template <  >
        struct mask_uint_t< std::numeric_limits<unsigned long>::digits >;
    #endif

    template < std::size_t Bits, CRC_PARM_TYPE TruncPoly, bool Reflect >
        struct crc_table_t;

    template < std::size_t Bits, bool DoReflect >
        class crc_helper;

    #ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
    template < std::size_t Bits >
        class crc_helper< Bits, false >;
    #endif

}  // namespace detail


//  Simple cyclic redundancy code (CRC) class declaration  -------------------//

template < std::size_t Bits >
class crc_basic
{
    // Implementation type
    typedef detail::mask_uint_t<Bits>  masking_type;

public:
    // Type
    typedef typename masking_type::least  value_type;

    // Constant for the template parameter
    static std::size_t constexpr bit_count = Bits;

    // Constructor
    explicit  crc_basic( value_type truncated_polynominal,
               value_type initial_remainder = 0, value_type final_xor_value = 0,
               bool reflect_input = false, bool reflect_remainder = false );

    // Internal Operations
    value_type  get_truncated_polynominal() const;
    value_type  get_initial_remainder() const;
    value_type  get_final_xor_value() const;
    bool        get_reflect_input() const;
    bool        get_reflect_remainder() const;

    value_type  get_interim_remainder() const;
    void        reset( value_type new_rem );
    void        reset();

    // External Operations
    void  process_bit( bool bit );
    void  process_bits( unsigned char bits, std::size_t bit_count );
    void  process_byte( unsigned char byte );
    void  process_block( void const *bytes_begin, void const *bytes_end );
    void  process_bytes( void const *buffer, std::size_t byte_count );

    value_type  checksum() const;

private:
    // Member data
    value_type  rem_;
    value_type  poly_, init_, final_;  // non-const to allow assignability
    bool        rft_in_, rft_out_;     // non-const to allow assignability

};  // boost::crc_basic


//  Optimized cyclic redundancy code (CRC) class declaration  ----------------//

template < std::size_t Bits, CRC_PARM_TYPE TruncPoly,
           CRC_PARM_TYPE InitRem, CRC_PARM_TYPE FinalXor,
           bool ReflectIn, bool ReflectRem >
class crc_optimal
{
    // Implementation type
    typedef detail::mask_uint_t<Bits>  masking_type;

public:
    // Type
    typedef typename masking_type::fast  value_type;

    // Constants for the template parameters
    static std::size_t constexpr bit_count = Bits;
    static value_type constexpr truncated_polynominal = TruncPoly;
    static value_type constexpr initial_remainder = InitRem;
    static value_type constexpr final_xor_value = FinalXor;
    static bool constexpr reflect_input = ReflectIn;
    static bool constexpr reflect_remainder = ReflectRem;

    // Constructor
    explicit  crc_optimal( value_type init_rem = InitRem );

    // Internal Operations
    value_type  get_truncated_polynominal() const;
    value_type  get_initial_remainder() const;
    value_type  get_final_xor_value() const;
    bool        get_reflect_input() const;
    bool        get_reflect_remainder() const;

    value_type  get_interim_remainder() const;
    void        reset( value_type new_rem = InitRem );

    // External Operations
    void  process_byte( unsigned char byte );
    void  process_block( void const *bytes_begin, void const *bytes_end );
    void  process_bytes( void const *buffer, std::size_t byte_count );

    value_type  checksum() const;

    // Operators
    void        operator ()( unsigned char byte );
    value_type  operator ()() const;

private:
    // The implementation of output reflection depends on both reflect states.
    static bool constexpr reflect_output = (ReflectRem != ReflectIn);

    #ifndef __BORLANDC__
    #define BOOST_CRC_REF_OUT_VAL  reflect_output
    #else
    typedef crc_optimal  self_type;
    #define BOOST_CRC_REF_OUT_VAL  (self_type::reflect_output)
    #endif

    // More implementation types
    typedef detail::crc_table_t<Bits, TruncPoly, ReflectIn>  crc_table_type;
    typedef detail::crc_helper<Bits, ReflectIn>              helper_type;
    typedef detail::crc_helper<Bits, BOOST_CRC_REF_OUT_VAL>  reflect_out_type;

    #undef BOOST_CRC_REF_OUT_VAL

    // Member data
    value_type  rem_;

};  // boost::crc_optimal


//  Implementation detail stuff  ---------------------------------------------//

namespace detail
{
    // Forward declarations for more implementation details
    template < std::size_t Bits >
        struct high_uint_t;

    template < std::size_t Bits >
        struct reflector;


    // Traits class for mask; given the bit number
    // (1-based), get the mask for that bit by itself.
    template < std::size_t Bits >
    struct high_uint_t
        : boost::uint_t< Bits >
    {
        typedef boost::uint_t<Bits>        base_type;
        typedef typename base_type::least  least;
        typedef typename base_type::fast   fast;

#if defined(__EDG_VERSION__) && __EDG_VERSION__ <= 243
        static const least high_bit = 1ul << ( Bits - 1u );
        static const fast high_bit_fast = 1ul << ( Bits - 1u );
#else
        static least constexpr high_bit = (least( 1u ) << ( Bits
         - 1u ));
        static fast constexpr high_bit_fast = (fast( 1u ) << ( Bits
         - 1u ));
#endif

    };  // boost::detail::high_uint_t


    // Reflection routine class wrapper
    // (since MS VC++ 6 couldn't handle the unwrapped version)
    template < std::size_t Bits >
    struct reflector
    {
        typedef typename boost::uint_t<Bits>::fast  value_type;

        static  value_type  reflect( value_type x );

    };  // boost::detail::reflector

    // Function that reflects its argument
    template < std::size_t Bits >
    typename reflector<Bits>::value_type
    reflector<Bits>::reflect
    (
        typename reflector<Bits>::value_type  x
    )
    {
        value_type        reflection = 0;
        value_type const  one = 1;

        for ( std::size_t i = 0 ; i < Bits ; ++i, x >>= 1 )
        {
            if ( x & one )
            {
                reflection |= ( one << (Bits - 1u - i) );
            }
        }

        return reflection;
    }


    // Traits class for masks; given the bit number (1-based),
    // get the mask for that bit and its lower bits.
    template < std::size_t Bits >
    struct mask_uint_t
        : high_uint_t< Bits >
    {
        typedef high_uint_t<Bits>          base_type;
        typedef typename base_type::least  least;
        typedef typename base_type::fast   fast;

        #ifndef __BORLANDC__
        using base_type::high_bit;
        using base_type::high_bit_fast;
        #else
        static least constexpr high_bit = base_type::high_bit;
        static fast, high_bit_fast constexpr = base_type::high_bit_fast;
        #endif

#if defined(__EDG_VERSION__) && __EDG_VERSION__ <= 243
        static least constexpr sig_bits = (~( ~( 0ul ) << Bits ));
#else
        static least constexpr sig_bits = (~( ~(least( 0u )) << Bits ));
#endif
#if defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ == 0 && __GNUC_PATCHLEVEL__ == 2
        // Work around a weird bug that ICEs the compiler in build_c_cast
        static fast constexpr sig_bits_fast = static_cast<fast>(sig_bits);
#else
        static fast sig_bits_fast = fast(sig_bits);
#endif
    };  // boost::detail::mask_uint_t

    template <  >
    struct mask_uint_t< std::numeric_limits<unsigned char>::digits >
        : high_uint_t< std::numeric_limits<unsigned char>::digits >
    {
        typedef high_uint_t<std::numeric_limits<unsigned char>::digits>
          base_type;
        typedef base_type::least  least;
        typedef base_type::fast   fast;

        #ifndef __BORLANDC__
        using base_type::high_bit;
        using base_type::high_bit_fast;
        #else
        static least constexpr high_bit = base_type::high_bit;
        static fast constexpr high_bit_fast = base_type::high_bit_fast;
        #endif

        static least constexpr sig_bits = (~( least(0u) ));
        static fast constexpr sig_bits_fast = fast(sig_bits);
    };  // boost::detail::mask_uint_t

    #if USHRT_MAX > UCHAR_MAX
    template <  >
    struct mask_uint_t< std::numeric_limits<unsigned short>::digits >
        : high_uint_t< std::numeric_limits<unsigned short>::digits >
    {
        typedef high_uint_t<std::numeric_limits<unsigned short>::digits>
          base_type;
        typedef base_type::least  least;
        typedef base_type::fast   fast;

        #ifndef __BORLANDC__
        using base_type::high_bit;
        using base_type::high_bit_fast;
        #else
        static least constexpr high_bit = base_type::high_bit;
        static fast constexpr high_bit_fast = base_type::high_bit_fast;
        #endif

        static least constexpr sig_bits = (~( least(0u) ));
        static fast constexpr sig_bits_fast = fast(sig_bits);

    };  // boost::detail::mask_uint_t
    #endif

    #if UINT_MAX > USHRT_MAX
    template <  >
    struct mask_uint_t< std::numeric_limits<unsigned int>::digits >
        : high_uint_t< std::numeric_limits<unsigned int>::digits >
    {
        typedef high_uint_t<std::numeric_limits<unsigned int>::digits>
          base_type;
        typedef base_type::least  least;
        typedef base_type::fast   fast;

        #ifndef __BORLANDC__
        using base_type::high_bit;
        using base_type::high_bit_fast;
        #else
        static least constexpr high_bit = base_type::high_bit;
        static fast constexpr high_bit_fast = base_type::high_bit_fast;
        #endif

        static least constexpr sig_bits = (~( least(0u) ));
        static fast constexpr sig_bits_fast = fast(sig_bits);

    };  // boost::detail::mask_uint_t
    #endif

    #if ULONG_MAX > UINT_MAX
    template <  >
    struct mask_uint_t< std::numeric_limits<unsigned long>::digits >
        : high_uint_t< std::numeric_limits<unsigned long>::digits >
    {
        typedef high_uint_t<std::numeric_limits<unsigned long>::digits>
          base_type;
        typedef base_type::least  least;
        typedef base_type::fast   fast;

        #ifndef __BORLANDC__
        using base_type::high_bit;
        using base_type::high_bit_fast;
        #else
        static least constexpr high_bit = base_type::high_bit;
        static fast constexpr high_bit_fast = base_type::high_bit_fast;
        #endif
        static least constexpr sig_bits = (~( least(0u) ));
        static fast constexpr sig_bits_fast = fast(sig_bits);

    };  // boost::detail::mask_uint_t
    #endif


    // CRC table generator
    template < std::size_t Bits, CRC_PARM_TYPE TruncPoly, bool Reflect >
    struct crc_table_t
    {
        static std::size_t constexpr byte_combos = (1ul << CHAR_BIT);

        typedef mask_uint_t<Bits>            masking_type;
        typedef typename masking_type::fast  value_type;
#if defined(__BORLANDC__) && defined(_M_IX86) && (__BORLANDC__ == 0x560)
        // for some reason Borland's command line compiler (version 0x560)
        // chokes over this unless we do the calculation for it:
        typedef value_type                   table_type[ 0x100 ];
#elif defined(__GNUC__)
        // old versions of GCC (before 4.0.2) choke on using byte_combos
        // as a constant expression when compiling with -pedantic.
        typedef value_type                   table_type[1ul << CHAR_BIT];
#else
        typedef value_type                   table_type[ byte_combos ];
#endif

        static  void  init_table();

        static  table_type  table_;

    };  // boost::detail::crc_table_t

    // CRC table generator static data member definition
    // (Some compilers [Borland C++] require the initializer to be present.)
    template < std::size_t Bits, CRC_PARM_TYPE TruncPoly, bool Reflect >
    typename crc_table_t<Bits, TruncPoly, Reflect>::table_type
    crc_table_t<Bits, TruncPoly, Reflect>::table_
     = { 0 };

    // Populate CRC lookup table
    template < std::size_t Bits, CRC_PARM_TYPE TruncPoly, bool Reflect >
    void
    crc_table_t<Bits, TruncPoly, Reflect>::init_table
    (
    )
    {
        // compute table only on the first run
        static  bool  did_init = false;
        if ( did_init )  return;

        // factor-out constants to avoid recalculation
        value_type const     fast_hi_bit = masking_type::high_bit_fast;
        unsigned char const  byte_hi_bit = 1u << (CHAR_BIT - 1u);

        // loop over every possible dividend value
        unsigned char  dividend = 0;
        do
        {
            value_type  remainder = 0;

            // go through all the dividend's bits
            for ( unsigned char mask = byte_hi_bit ; mask ; mask >>= 1 )
            {
                // check if divisor fits
                if ( dividend & mask )
                {
                    remainder ^= fast_hi_bit;
                }

                // do polynominal division
                if ( remainder & fast_hi_bit )
                {
                    remainder <<= 1;
                    remainder ^= TruncPoly;
                }
                else
                {
                    remainder <<= 1;
                }
            }

            table_[ crc_helper<CHAR_BIT, Reflect>::reflect(dividend) ]
             = crc_helper<Bits, Reflect>::reflect( remainder );
        }
        while ( ++dividend );

        did_init = true;
    }

    #ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
    // Align the msb of the remainder to a byte
    template < std::size_t Bits, bool RightShift >
    class remainder
    {
    public:
        typedef typename uint_t<Bits>::fast  value_type;

        static unsigned char align_msb( value_type rem )
            { return rem >> (Bits - CHAR_BIT); }
    };

    // Specialization for the case that the remainder has less
    // bits than a byte: align the remainder msb to the byte msb
    template < std::size_t Bits >
    class remainder< Bits, false >
    {
    public:
        typedef typename uint_t<Bits>::fast  value_type;

        static unsigned char align_msb( value_type rem )
            { return rem << (CHAR_BIT - Bits); }
    };
    #endif

    // CRC helper routines
    template < std::size_t Bits, bool DoReflect >
    class crc_helper
    {
    public:
        // Type
        typedef typename uint_t<Bits>::fast  value_type;

    #ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
        // Possibly reflect a remainder
        static  value_type  reflect( value_type x )
            { return detail::reflector<Bits>::reflect( x ); }

        // Compare a byte to the remainder's highest byte
        static  unsigned char  index( value_type rem, unsigned char x )
            { return x ^ rem; }

        // Shift out the remainder's highest byte
        static  value_type  shift( value_type rem )
            { return rem >> CHAR_BIT; }
    #else
        // Possibly reflect a remainder
        static  value_type  reflect( value_type x )
            { return DoReflect ? detail::reflector<Bits>::reflect( x ) : x; }

        // Compare a byte to the remainder's highest byte
        static  unsigned char  index( value_type rem, unsigned char x )
            { return x ^ ( DoReflect ? rem :
                                ((Bits>CHAR_BIT)?( rem >> (Bits - CHAR_BIT) ) :
                                    ( rem << (CHAR_BIT - Bits) ))); }

        // Shift out the remainder's highest byte
        static  value_type  shift( value_type rem )
            { return DoReflect ? rem >> CHAR_BIT : rem << CHAR_BIT; }
    #endif

    };  // boost::detail::crc_helper

    #ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
    template < std::size_t Bits >
    class crc_helper<Bits, false>
    {
    public:
        // Type
        typedef typename uint_t<Bits>::fast  value_type;

        // Possibly reflect a remainder
        static  value_type  reflect( value_type x )
            { return x; }

        // Compare a byte to the remainder's highest byte
        static  unsigned char  index( value_type rem, unsigned char x )
            { return x ^ remainder<Bits,(Bits>CHAR_BIT)>::align_msb( rem ); }

        // Shift out the remainder's highest byte
        static  value_type  shift( value_type rem )
            { return rem << CHAR_BIT; }

    };  // boost::detail::crc_helper
    #endif


}  // namespace detail


//  Simple CRC class function definitions  -----------------------------------//

template < std::size_t Bits >
inline
crc_basic<Bits>::crc_basic
(
    typename crc_basic<Bits>::value_type  truncated_polynominal,
    typename crc_basic<Bits>::value_type  initial_remainder,      // = 0
    typename crc_basic<Bits>::value_type  final_xor_value,        // = 0
    bool                                  reflect_input,          // = false
    bool                                  reflect_remainder       // = false
)
    : rem_( initial_remainder ), poly_( truncated_polynominal )
    , init_( initial_remainder ), final_( final_xor_value )
    , rft_in_( reflect_input ), rft_out_( reflect_remainder )
{
}

template < std::size_t Bits >
inline
typename crc_basic<Bits>::value_type
crc_basic<Bits>::get_truncated_polynominal
(
) const
{
    return poly_;
}

template < std::size_t Bits >
inline
typename crc_basic<Bits>::value_type
crc_basic<Bits>::get_initial_remainder
(
) const
{
    return init_;
}

template < std::size_t Bits >
inline
typename crc_basic<Bits>::value_type
crc_basic<Bits>::get_final_xor_value
(
) const
{
    return final_;
}

template < std::size_t Bits >
inline
bool
crc_basic<Bits>::get_reflect_input
(
) const
{
    return rft_in_;
}

template < std::size_t Bits >
inline
bool
crc_basic<Bits>::get_reflect_remainder
(
) const
{
    return rft_out_;
}

template < std::size_t Bits >
inline
typename crc_basic<Bits>::value_type
crc_basic<Bits>::get_interim_remainder
(
) const
{
    return rem_ & masking_type::sig_bits;
}

template < std::size_t Bits >
inline
void
crc_basic<Bits>::reset
(
    typename crc_basic<Bits>::value_type  new_rem
)
{
    rem_ = new_rem;
}

template < std::size_t Bits >
inline
void
crc_basic<Bits>::reset
(
)
{
    this->reset( this->get_initial_remainder() );
}

template < std::size_t Bits >
inline
void
crc_basic<Bits>::process_bit
(
    bool  bit
)
{
    value_type const  high_bit_mask = masking_type::high_bit;

    // compare the new bit with the remainder's highest
    rem_ ^= ( bit ? high_bit_mask : 0u );

    // a full polynominal division step is done when the highest bit is one
    bool const  do_poly_div = static_cast<bool>( rem_ & high_bit_mask );

    // shift out the highest bit
    rem_ <<= 1;

    // carry out the division, if needed
    if ( do_poly_div )
    {
        rem_ ^= poly_;
    }
}

template < std::size_t Bits >
void
crc_basic<Bits>::process_bits
(
    unsigned char  bits,
    std::size_t    bit_count
)
{
    // ignore the bits above the ones we want
    bits <<= CHAR_BIT - bit_count;

    // compute the CRC for each bit, starting with the upper ones
    unsigned char const  high_bit_mask = 1u << ( CHAR_BIT - 1u );
    for ( std::size_t i = bit_count ; i > 0u ; --i, bits <<= 1u )
    {
        process_bit( static_cast<bool>(bits & high_bit_mask) );
    }
}

template < std::size_t Bits >
inline
void
crc_basic<Bits>::process_byte
(
    unsigned char  byte
)
{
    process_bits( (rft_in_ ? detail::reflector<CHAR_BIT>::reflect(byte)
     : byte), CHAR_BIT );
}

template < std::size_t Bits >
void
crc_basic<Bits>::process_block
(
    void const *  bytes_begin,
    void const *  bytes_end
)
{
    for ( unsigned char const * p
     = static_cast<unsigned char const *>(bytes_begin) ; p < bytes_end ; ++p )
    {
        process_byte( *p );
    }
}

template < std::size_t Bits >
inline
void
crc_basic<Bits>::process_bytes
(
    void const *  buffer,
    std::size_t   byte_count
)
{
    unsigned char const * const  b = static_cast<unsigned char const *>(
     buffer );

    process_block( b, b + byte_count );
}

template < std::size_t Bits >
inline
typename crc_basic<Bits>::value_type
crc_basic<Bits>::checksum
(
) const
{
    return ( (rft_out_ ? detail::reflector<Bits>::reflect( rem_ ) : rem_)
     ^ final_ ) & masking_type::sig_bits;
}


//  Optimized CRC class function definitions  --------------------------------//

// Macro to compact code
#define BOOST_CRC_OPTIMAL_NAME  crc_optimal<Bits, TruncPoly, InitRem, \
 FinalXor, ReflectIn, ReflectRem>

template < std::size_t Bits, CRC_PARM_TYPE TruncPoly,
           CRC_PARM_TYPE InitRem, CRC_PARM_TYPE FinalXor,
           bool ReflectIn, bool ReflectRem >
inline
BOOST_CRC_OPTIMAL_NAME::crc_optimal
(
    typename BOOST_CRC_OPTIMAL_NAME::value_type  init_rem  // = InitRem
)
    : rem_( helper_type::reflect(init_rem) )
{
    crc_table_type::init_table();
}

template < std::size_t Bits, CRC_PARM_TYPE TruncPoly,
           CRC_PARM_TYPE InitRem, CRC_PARM_TYPE FinalXor,
           bool ReflectIn, bool ReflectRem >
inline
typename BOOST_CRC_OPTIMAL_NAME::value_type
BOOST_CRC_OPTIMAL_NAME::get_truncated_polynominal
(
) const
{
    return TruncPoly;
}

template < std::size_t Bits, CRC_PARM_TYPE TruncPoly,
           CRC_PARM_TYPE InitRem, CRC_PARM_TYPE FinalXor,
           bool ReflectIn, bool ReflectRem >
inline
typename BOOST_CRC_OPTIMAL_NAME::value_type
BOOST_CRC_OPTIMAL_NAME::get_initial_remainder
(
) const
{
    return InitRem;
}

template < std::size_t Bits, CRC_PARM_TYPE TruncPoly,
           CRC_PARM_TYPE InitRem, CRC_PARM_TYPE FinalXor,
           bool ReflectIn, bool ReflectRem >
inline
typename BOOST_CRC_OPTIMAL_NAME::value_type
BOOST_CRC_OPTIMAL_NAME::get_final_xor_value
(
) const
{
    return FinalXor;
}

template < std::size_t Bits, CRC_PARM_TYPE TruncPoly,
           CRC_PARM_TYPE InitRem, CRC_PARM_TYPE FinalXor,
           bool ReflectIn, bool ReflectRem >
inline
bool
BOOST_CRC_OPTIMAL_NAME::get_reflect_input
(
) const
{
    return ReflectIn;
}

template < std::size_t Bits, CRC_PARM_TYPE TruncPoly,
           CRC_PARM_TYPE InitRem, CRC_PARM_TYPE FinalXor,
           bool ReflectIn, bool ReflectRem >
inline
bool
BOOST_CRC_OPTIMAL_NAME::get_reflect_remainder
(
) const
{
    return ReflectRem;
}

template < std::size_t Bits, CRC_PARM_TYPE TruncPoly,
           CRC_PARM_TYPE InitRem, CRC_PARM_TYPE FinalXor,
           bool ReflectIn, bool ReflectRem >
inline
typename BOOST_CRC_OPTIMAL_NAME::value_type
BOOST_CRC_OPTIMAL_NAME::get_interim_remainder
(
) const
{
    // Interim remainder should be _un_-reflected, so we have to undo it.
    return helper_type::reflect( rem_ ) & masking_type::sig_bits_fast;
}

template < std::size_t Bits, CRC_PARM_TYPE TruncPoly,
           CRC_PARM_TYPE InitRem, CRC_PARM_TYPE FinalXor,
           bool ReflectIn, bool ReflectRem >
inline
void
BOOST_CRC_OPTIMAL_NAME::reset
(
    typename BOOST_CRC_OPTIMAL_NAME::value_type  new_rem  // = InitRem
)
{
    rem_ = helper_type::reflect( new_rem );
}

template < std::size_t Bits, CRC_PARM_TYPE TruncPoly,
           CRC_PARM_TYPE InitRem, CRC_PARM_TYPE FinalXor,
           bool ReflectIn, bool ReflectRem >
inline
void
BOOST_CRC_OPTIMAL_NAME::process_byte
(
    unsigned char  byte
)
{
    process_bytes( &byte, sizeof(byte) );
}

template < std::size_t Bits, CRC_PARM_TYPE TruncPoly,
           CRC_PARM_TYPE InitRem, CRC_PARM_TYPE FinalXor,
           bool ReflectIn, bool ReflectRem >
void
BOOST_CRC_OPTIMAL_NAME::process_block
(
    void const *  bytes_begin,
    void const *  bytes_end
)
{
    // Recompute the CRC for each byte passed
    for ( unsigned char const * p
     = static_cast<unsigned char const *>(bytes_begin) ; p < bytes_end ; ++p )
    {
        // Compare the new byte with the remainder's higher bits to
        // get the new bits, shift out the remainder's current higher
        // bits, and update the remainder with the polynominal division
        // of the new bits.
        unsigned char const  byte_index = helper_type::index( rem_, *p );
        rem_ = helper_type::shift( rem_ );
        rem_ ^= crc_table_type::table_[ byte_index ];
    }
}

template < std::size_t Bits, CRC_PARM_TYPE TruncPoly,
           CRC_PARM_TYPE InitRem, CRC_PARM_TYPE FinalXor,
           bool ReflectIn, bool ReflectRem >
inline
void
BOOST_CRC_OPTIMAL_NAME::process_bytes
(
    void const *   buffer,
    std::size_t  byte_count
)
{
    unsigned char const * const  b = static_cast<unsigned char const *>(
     buffer );
    process_block( b, b + byte_count );
}

template < std::size_t Bits, CRC_PARM_TYPE TruncPoly,
           CRC_PARM_TYPE InitRem, CRC_PARM_TYPE FinalXor,
           bool ReflectIn, bool ReflectRem >
inline
typename BOOST_CRC_OPTIMAL_NAME::value_type
BOOST_CRC_OPTIMAL_NAME::checksum
(
) const
{
    return ( reflect_out_type::reflect(rem_) ^ get_final_xor_value() )
     & masking_type::sig_bits_fast;
}

template < std::size_t Bits, CRC_PARM_TYPE TruncPoly,
           CRC_PARM_TYPE InitRem, CRC_PARM_TYPE FinalXor,
           bool ReflectIn, bool ReflectRem >
inline
void
BOOST_CRC_OPTIMAL_NAME::operator ()
(
    unsigned char  byte
)
{
    process_byte( byte );
}

template < std::size_t Bits, CRC_PARM_TYPE TruncPoly,
           CRC_PARM_TYPE InitRem, CRC_PARM_TYPE FinalXor,
           bool ReflectIn, bool ReflectRem >
inline
typename BOOST_CRC_OPTIMAL_NAME::value_type
BOOST_CRC_OPTIMAL_NAME::operator ()
(
) const
{
    return checksum();
}


//  CRC computation function definition  -------------------------------------//

template < std::size_t Bits, CRC_PARM_TYPE TruncPoly,
           CRC_PARM_TYPE InitRem, CRC_PARM_TYPE FinalXor,
           bool ReflectIn, bool ReflectRem >
inline
typename uint_t<Bits>::fast
crc
(
    void const *  buffer,
    std::size_t   byte_count
    CRC_DUMMY_INIT
)
{
    BOOST_CRC_OPTIMAL_NAME  computer;
    computer.process_bytes( buffer, byte_count );
    return computer.checksum();
}


//  Augmented-message CRC computation function definitions  ------------------//

template < std::size_t Bits, CRC_PARM_TYPE TruncPoly >
typename uint_t<Bits>::fast
augmented_crc
(
    void const *                 buffer,
    std::size_t                  byte_count,
    typename uint_t<Bits>::fast  initial_remainder
    ACRC_DUMMY_INIT
)
{
    typedef unsigned char                                byte_type;
    typedef detail::mask_uint_t<Bits>                    masking_type;
    typedef detail::crc_table_t<Bits, TruncPoly, false>  crc_table_type;

    typename masking_type::fast  rem = initial_remainder;
    byte_type const * const      b = static_cast<byte_type const *>( buffer );
    byte_type const * const      e = b + byte_count;

    crc_table_type::init_table();
    for ( byte_type const * p = b ; p < e ; ++p )
    {
        // Use the current top byte as the table index to the next
        // "partial product."  Shift out that top byte, shifting in
        // the next augmented-message byte.  Complete the division.
        byte_type const  byte_index = rem >> ( Bits - CHAR_BIT );
        rem <<= CHAR_BIT;
        rem |= *p;
        rem ^= crc_table_type::table_[ byte_index ];
    }

    return rem & masking_type::sig_bits_fast;
}

template < std::size_t Bits, CRC_PARM_TYPE TruncPoly >
inline
typename uint_t<Bits>::fast
augmented_crc
(
    void const *  buffer,
    std::size_t   byte_count
    ACRC_DUMMY_INIT
)
{
   // The last function argument has its type specified so the other version of
   // augmented_crc will be called.  If the cast wasn't in place, and the
   // ACRC_DUMMY_INIT added a third argument (for a workaround), the "0"
   // would match as that third argument, leading to infinite recursion.
   return augmented_crc<Bits, TruncPoly>( buffer, byte_count,
    static_cast<typename uint_t<Bits>::fast>(0) );
}


}  // namespace boost


// Undo header-private macros
#undef BOOST_CRC_OPTIMAL_NAME
#undef ACRC_DUMMY_INIT
#undef ACRC_DUMMY_PARM_TYPE
#undef CRC_DUMMY_INIT
#undef CRC_DUMMY_PARM_TYPE
#undef CRC_PARM_TYPE


#endif  // BOOST_CRC_HPP