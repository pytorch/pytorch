/*
    MPFR C++: Multi-precision floating point number class for C++. 
    Based on MPFR library:    http://mpfr.org

    Project homepage:    http://www.holoborodko.com/pavel/mpfr
    Contact e-mail:      pavel@holoborodko.com

    Copyright (c) 2008-2014 Pavel Holoborodko

    Contributors:
    Dmitriy Gubanov, Konstantin Holoborodko, Brian Gladman, 
    Helmut Jarausch, Fokko Beekhof, Ulrich Mutze, Heinz van Saanen, 
    Pere Constans, Peter van Hoof, Gael Guennebaud, Tsai Chia Cheng, 
    Alexei Zubanov, Jauhien Piatlicki, Victor Berger, John Westwood,
    Petr Aleksandrov, Orion Poplawski, Charles Karney.

    Licensing:
    (A) MPFR C++ is under GNU General Public License ("GPL").
    
    (B) Non-free licenses may also be purchased from the author, for users who 
        do not want their programs protected by the GPL.

        The non-free licenses are for users that wish to use MPFR C++ in 
        their products but are unwilling to release their software 
        under the GPL (which would require them to release source code 
        and allow free redistribution).

        Such users can purchase an unlimited-use license from the author.
        Contact us for more details.
    
    GNU General Public License ("GPL") copyright permissions statement:
    **************************************************************************
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __MPREAL_H__
#define __MPREAL_H__

#include <string>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <limits>

// Options
// FIXME HAVE_INT64_SUPPORT leads to clashes with long int and int64_t on some systems.
//#define MPREAL_HAVE_INT64_SUPPORT               // Enable int64_t support if possible. Available only for MSVC 2010 & GCC.
#define MPREAL_HAVE_MSVC_DEBUGVIEW              // Enable Debugger Visualizer for "Debug" builds in MSVC.
#define MPREAL_HAVE_DYNAMIC_STD_NUMERIC_LIMITS  // Enable extended std::numeric_limits<mpfr::mpreal> specialization.
                                                // Meaning that "digits", "round_style" and similar members are defined as functions, not constants.
                                                // See std::numeric_limits<mpfr::mpreal> at the end of the file for more information.

// Library version
#define MPREAL_VERSION_MAJOR 3
#define MPREAL_VERSION_MINOR 5
#define MPREAL_VERSION_PATCHLEVEL 9
#define MPREAL_VERSION_STRING "3.5.9"

// Detect compiler using signatures from http://predef.sourceforge.net/
#if defined(__GNUC__) && defined(__INTEL_COMPILER)
    #define IsInf(x) (isinf)(x)                   // Intel ICC compiler on Linux 

#elif defined(_MSC_VER)                         // Microsoft Visual C++ 
    #define IsInf(x) (!_finite(x))                           

#else
    #define IsInf(x) (std::isinf)(x)              // GNU C/C++ (and/or other compilers), just hope for C99 conformance
#endif

// A Clang feature extension to determine compiler features.
#ifndef __has_feature
    #define __has_feature(x) 0
#endif

// Detect support for r-value references (move semantic). Borrowed from Eigen.
#if (__has_feature(cxx_rvalue_references) || \
       defined(__GXX_EXPERIMENTAL_CXX0X__) || __cplusplus >= 201103L || \
      (defined(_MSC_VER) && _MSC_VER >= 1600))

    #define MPREAL_HAVE_MOVE_SUPPORT

    // Use fields in mpfr_t structure to check if it was initialized / set dummy initialization 
    #define mpfr_is_initialized(x)      (0 != (x)->_mpfr_d)
    #define mpfr_set_uninitialized(x)   ((x)->_mpfr_d = 0 )
#endif

// Detect support for explicit converters. 
#if (__has_feature(cxx_explicit_conversions) || \
       defined(__GXX_EXPERIMENTAL_CXX0X__) || __cplusplus >= 201103L || \
      (defined(_MSC_VER) && _MSC_VER >= 1800))

    #define MPREAL_HAVE_EXPLICIT_CONVERTERS
#endif

// Detect available 64-bit capabilities
#if defined(MPREAL_HAVE_INT64_SUPPORT)
    
    #define MPFR_USE_INTMAX_T                   // Should be defined before mpfr.h

    #if defined(_MSC_VER)                       // MSVC + Windows
        #if (_MSC_VER >= 1600)                    
            #include <stdint.h>                 // <stdint.h> is available only in msvc2010!

        #else                                   // MPFR relies on intmax_t which is available only in msvc2010
            #undef MPREAL_HAVE_INT64_SUPPORT    // Besides, MPFR & MPIR have to be compiled with msvc2010
            #undef MPFR_USE_INTMAX_T            // Since we cannot detect this, disable x64 by default
                                                // Someone should change this manually if needed.
        #endif

    #elif defined (__GNUC__) && defined(__linux__)
        #if defined(__amd64__) || defined(__amd64) || defined(__x86_64__) || defined(__x86_64) || defined(__ia64) || defined(__itanium__) || defined(_M_IA64) || defined (__PPC64__)
            #undef MPREAL_HAVE_INT64_SUPPORT    // Remove all shaman dances for x64 builds since
            #undef MPFR_USE_INTMAX_T            // GCC already supports x64 as of "long int" is 64-bit integer, nothing left to do
        #else
            #include <stdint.h>                 // use int64_t, uint64_t otherwise
        #endif

    #else
        #include <stdint.h>                     // rely on int64_t, uint64_t in all other cases, Mac OSX, etc.
    #endif

#endif 

#if defined(MPREAL_HAVE_MSVC_DEBUGVIEW) && defined(_MSC_VER) && defined(_DEBUG)
    #define MPREAL_MSVC_DEBUGVIEW_CODE     DebugView = toString();
    #define MPREAL_MSVC_DEBUGVIEW_DATA     std::string DebugView;
#else
    #define MPREAL_MSVC_DEBUGVIEW_CODE 
    #define MPREAL_MSVC_DEBUGVIEW_DATA 
#endif

#include <mpfr.h>

#if (MPFR_VERSION < MPFR_VERSION_NUM(3,0,0))
    #include <cstdlib>                          // Needed for random()
#endif

// Less important options
#define MPREAL_DOUBLE_BITS_OVERFLOW -1          // Triggers overflow exception during conversion to double if mpreal 
                                                // cannot fit in MPREAL_DOUBLE_BITS_OVERFLOW bits
                                                // = -1 disables overflow checks (default)
#if defined(__GNUC__)
  #define MPREAL_PERMISSIVE_EXPR __extension__
#else
  #define MPREAL_PERMISSIVE_EXPR
#endif

namespace mpfr {

class mpreal {
private:
    mpfr_t mp;
    
public:
    
    // Get default rounding mode & precision
    inline static mp_rnd_t   get_default_rnd()    {    return (mp_rnd_t)(mpfr_get_default_rounding_mode());       }
    inline static mp_prec_t  get_default_prec()   {    return mpfr_get_default_prec();                            }

    // Constructors && type conversions
    mpreal();
    mpreal(const mpreal& u);
    mpreal(const mpf_t u);    
    mpreal(const mpz_t u,             mp_prec_t prec = mpreal::get_default_prec(), mp_rnd_t mode = mpreal::get_default_rnd());    
    mpreal(const mpq_t u,             mp_prec_t prec = mpreal::get_default_prec(), mp_rnd_t mode = mpreal::get_default_rnd());    
    mpreal(const double u,            mp_prec_t prec = mpreal::get_default_prec(), mp_rnd_t mode = mpreal::get_default_rnd());
    mpreal(const long double u,       mp_prec_t prec = mpreal::get_default_prec(), mp_rnd_t mode = mpreal::get_default_rnd());
    mpreal(const unsigned long int u, mp_prec_t prec = mpreal::get_default_prec(), mp_rnd_t mode = mpreal::get_default_rnd());
    mpreal(const unsigned int u,      mp_prec_t prec = mpreal::get_default_prec(), mp_rnd_t mode = mpreal::get_default_rnd());
    mpreal(const long int u,          mp_prec_t prec = mpreal::get_default_prec(), mp_rnd_t mode = mpreal::get_default_rnd());
    mpreal(const int u,               mp_prec_t prec = mpreal::get_default_prec(), mp_rnd_t mode = mpreal::get_default_rnd());
    
    // Construct mpreal from mpfr_t structure.
    // shared = true allows to avoid deep copy, so that mpreal and 'u' share the same data & pointers.    
    mpreal(const mpfr_t  u, bool shared = false);   

#if defined (MPREAL_HAVE_INT64_SUPPORT)
    mpreal(const uint64_t u,          mp_prec_t prec = mpreal::get_default_prec(),  mp_rnd_t mode = mpreal::get_default_rnd());
    mpreal(const int64_t u,           mp_prec_t prec = mpreal::get_default_prec(),  mp_rnd_t mode = mpreal::get_default_rnd());
#endif

    mpreal(const char* s,             mp_prec_t prec = mpreal::get_default_prec(), int base = 10, mp_rnd_t mode = mpreal::get_default_rnd());
    mpreal(const std::string& s,      mp_prec_t prec = mpreal::get_default_prec(), int base = 10, mp_rnd_t mode = mpreal::get_default_rnd());

    ~mpreal();                           

#ifdef MPREAL_HAVE_MOVE_SUPPORT
    mpreal& operator=(mpreal&& v);
    mpreal(mpreal&& u);
#endif

    // Operations
    // =
    // +, -, *, /, ++, --, <<, >> 
    // *=, +=, -=, /=,
    // <, >, ==, <=, >=

    // =
    mpreal& operator=(const mpreal& v);
    mpreal& operator=(const mpf_t v);
    mpreal& operator=(const mpz_t v);
    mpreal& operator=(const mpq_t v);
    mpreal& operator=(const long double v);
    mpreal& operator=(const double v);        
    mpreal& operator=(const unsigned long int v);
    mpreal& operator=(const unsigned int v);
    mpreal& operator=(const long int v);
    mpreal& operator=(const int v);
    mpreal& operator=(const char* s);
    mpreal& operator=(const std::string& s);

    // +
    mpreal& operator+=(const mpreal& v);
    mpreal& operator+=(const mpf_t v);
    mpreal& operator+=(const mpz_t v);
    mpreal& operator+=(const mpq_t v);
    mpreal& operator+=(const long double u);
    mpreal& operator+=(const double u);
    mpreal& operator+=(const unsigned long int u);
    mpreal& operator+=(const unsigned int u);
    mpreal& operator+=(const long int u);
    mpreal& operator+=(const int u);

#if defined (MPREAL_HAVE_INT64_SUPPORT)
    mpreal& operator+=(const int64_t  u);
    mpreal& operator+=(const uint64_t u);
    mpreal& operator-=(const int64_t  u);
    mpreal& operator-=(const uint64_t u);
    mpreal& operator*=(const int64_t  u);
    mpreal& operator*=(const uint64_t u);
    mpreal& operator/=(const int64_t  u);
    mpreal& operator/=(const uint64_t u);
#endif 

    const mpreal operator+() const;
    mpreal& operator++ ();
    const mpreal  operator++ (int); 

    // -
    mpreal& operator-=(const mpreal& v);
    mpreal& operator-=(const mpz_t v);
    mpreal& operator-=(const mpq_t v);
    mpreal& operator-=(const long double u);
    mpreal& operator-=(const double u);
    mpreal& operator-=(const unsigned long int u);
    mpreal& operator-=(const unsigned int u);
    mpreal& operator-=(const long int u);
    mpreal& operator-=(const int u);
    const mpreal operator-() const;
    friend const mpreal operator-(const unsigned long int b, const mpreal& a);
    friend const mpreal operator-(const unsigned int b,      const mpreal& a);
    friend const mpreal operator-(const long int b,          const mpreal& a);
    friend const mpreal operator-(const int b,               const mpreal& a);
    friend const mpreal operator-(const double b,            const mpreal& a);
    mpreal& operator-- ();    
    const mpreal  operator-- (int);

    // *
    mpreal& operator*=(const mpreal& v);
    mpreal& operator*=(const mpz_t v);
    mpreal& operator*=(const mpq_t v);
    mpreal& operator*=(const long double v);
    mpreal& operator*=(const double v);
    mpreal& operator*=(const unsigned long int v);
    mpreal& operator*=(const unsigned int v);
    mpreal& operator*=(const long int v);
    mpreal& operator*=(const int v);
    
    // /
    mpreal& operator/=(const mpreal& v);
    mpreal& operator/=(const mpz_t v);
    mpreal& operator/=(const mpq_t v);
    mpreal& operator/=(const long double v);
    mpreal& operator/=(const double v);
    mpreal& operator/=(const unsigned long int v);
    mpreal& operator/=(const unsigned int v);
    mpreal& operator/=(const long int v);
    mpreal& operator/=(const int v);
    friend const mpreal operator/(const unsigned long int b, const mpreal& a);
    friend const mpreal operator/(const unsigned int b,      const mpreal& a);
    friend const mpreal operator/(const long int b,          const mpreal& a);
    friend const mpreal operator/(const int b,               const mpreal& a);
    friend const mpreal operator/(const double b,            const mpreal& a);

    //<<= Fast Multiplication by 2^u
    mpreal& operator<<=(const unsigned long int u);
    mpreal& operator<<=(const unsigned int u);
    mpreal& operator<<=(const long int u);
    mpreal& operator<<=(const int u);

    //>>= Fast Division by 2^u
    mpreal& operator>>=(const unsigned long int u);
    mpreal& operator>>=(const unsigned int u);
    mpreal& operator>>=(const long int u);
    mpreal& operator>>=(const int u);

    // Boolean Operators
    friend bool operator >  (const mpreal& a, const mpreal& b);
    friend bool operator >= (const mpreal& a, const mpreal& b);
    friend bool operator <  (const mpreal& a, const mpreal& b);
    friend bool operator <= (const mpreal& a, const mpreal& b);
    friend bool operator == (const mpreal& a, const mpreal& b);
    friend bool operator != (const mpreal& a, const mpreal& b);

    // Optimized specializations for boolean operators
    friend bool operator == (const mpreal& a, const unsigned long int b);
    friend bool operator == (const mpreal& a, const unsigned int b);
    friend bool operator == (const mpreal& a, const long int b);
    friend bool operator == (const mpreal& a, const int b);
    friend bool operator == (const mpreal& a, const long double b);
    friend bool operator == (const mpreal& a, const double b);

    // Type Conversion operators
    bool            toBool      (mp_rnd_t mode = GMP_RNDZ)    const;
    long            toLong      (mp_rnd_t mode = GMP_RNDZ)    const;
    unsigned long   toULong     (mp_rnd_t mode = GMP_RNDZ)    const;
    float           toFloat     (mp_rnd_t mode = GMP_RNDN)    const;
    double          toDouble    (mp_rnd_t mode = GMP_RNDN)    const;
    long double     toLDouble   (mp_rnd_t mode = GMP_RNDN)    const;

#if defined (MPREAL_HAVE_EXPLICIT_CONVERTERS)
    explicit operator bool               () const { return toBool();       }
    explicit operator int                () const { return int(toLong());  }
    explicit operator long               () const { return toLong();       }
    explicit operator long long          () const { return toLong();       }
    explicit operator unsigned           () const { return unsigned(toULong()); }
    explicit operator unsigned long      () const { return toULong();      }
    explicit operator unsigned long long () const { return toULong();      }
    explicit operator float              () const { return toFloat();      }
    explicit operator double             () const { return toDouble();     }
    explicit operator long double        () const { return toLDouble();    }
#endif

#if defined (MPREAL_HAVE_INT64_SUPPORT)
    int64_t         toInt64     (mp_rnd_t mode = GMP_RNDZ)    const;
    uint64_t        toUInt64    (mp_rnd_t mode = GMP_RNDZ)    const;

    #if defined (MPREAL_HAVE_EXPLICIT_CONVERTERS)
    explicit operator int64_t   () const { return toInt64();      }
    explicit operator uint64_t  () const { return toUInt64();     }
    #endif
#endif

    // Get raw pointers so that mpreal can be directly used in raw mpfr_* functions
    ::mpfr_ptr    mpfr_ptr();
    ::mpfr_srcptr mpfr_ptr()    const;
    ::mpfr_srcptr mpfr_srcptr() const;

    // Convert mpreal to string with n significant digits in base b
    // n = -1 -> convert with the maximum available digits
    std::string toString(int n = -1, int b = 10, mp_rnd_t mode = mpreal::get_default_rnd()) const;

#if (MPFR_VERSION >= MPFR_VERSION_NUM(2,4,0))
    std::string toString(const std::string& format) const;
#endif

    std::ostream& output(std::ostream& os) const;

    // Math Functions
    friend const mpreal sqr (const mpreal& v, mp_rnd_t rnd_mode);
    friend const mpreal sqrt(const mpreal& v, mp_rnd_t rnd_mode);
    friend const mpreal sqrt(const unsigned long int v, mp_rnd_t rnd_mode);
    friend const mpreal cbrt(const mpreal& v, mp_rnd_t rnd_mode);
    friend const mpreal root(const mpreal& v, unsigned long int k, mp_rnd_t rnd_mode);
    friend const mpreal pow (const mpreal& a, const mpreal& b, mp_rnd_t rnd_mode);
    friend const mpreal pow (const mpreal& a, const mpz_t b, mp_rnd_t rnd_mode);
    friend const mpreal pow (const mpreal& a, const unsigned long int b, mp_rnd_t rnd_mode);
    friend const mpreal pow (const mpreal& a, const long int b, mp_rnd_t rnd_mode);
    friend const mpreal pow (const unsigned long int a, const mpreal& b, mp_rnd_t rnd_mode);
    friend const mpreal pow (const unsigned long int a, const unsigned long int b, mp_rnd_t rnd_mode);
    friend const mpreal fabs(const mpreal& v, mp_rnd_t rnd_mode);

    friend const mpreal abs(const mpreal& v, mp_rnd_t rnd_mode);
    friend const mpreal dim(const mpreal& a, const mpreal& b, mp_rnd_t rnd_mode);
    friend inline const mpreal mul_2ui(const mpreal& v, unsigned long int k, mp_rnd_t rnd_mode);
    friend inline const mpreal mul_2si(const mpreal& v, long int k, mp_rnd_t rnd_mode);
    friend inline const mpreal div_2ui(const mpreal& v, unsigned long int k, mp_rnd_t rnd_mode);
    friend inline const mpreal div_2si(const mpreal& v, long int k, mp_rnd_t rnd_mode);
    friend int cmpabs(const mpreal& a,const mpreal& b);
    
    friend const mpreal log  (const mpreal& v, mp_rnd_t rnd_mode);
    friend const mpreal log2 (const mpreal& v, mp_rnd_t rnd_mode);
    friend const mpreal log10(const mpreal& v, mp_rnd_t rnd_mode);
    friend const mpreal exp  (const mpreal& v, mp_rnd_t rnd_mode); 
    friend const mpreal exp2 (const mpreal& v, mp_rnd_t rnd_mode);
    friend const mpreal exp10(const mpreal& v, mp_rnd_t rnd_mode);
    friend const mpreal log1p(const mpreal& v, mp_rnd_t rnd_mode);
    friend const mpreal expm1(const mpreal& v, mp_rnd_t rnd_mode);

    friend const mpreal cos(const mpreal& v, mp_rnd_t rnd_mode);
    friend const mpreal sin(const mpreal& v, mp_rnd_t rnd_mode);
    friend const mpreal tan(const mpreal& v, mp_rnd_t rnd_mode);
    friend const mpreal sec(const mpreal& v, mp_rnd_t rnd_mode);
    friend const mpreal csc(const mpreal& v, mp_rnd_t rnd_mode);
    friend const mpreal cot(const mpreal& v, mp_rnd_t rnd_mode);
    friend int sin_cos(mpreal& s, mpreal& c, const mpreal& v, mp_rnd_t rnd_mode);

    friend const mpreal acos  (const mpreal& v, mp_rnd_t rnd_mode);
    friend const mpreal asin  (const mpreal& v, mp_rnd_t rnd_mode);
    friend const mpreal atan  (const mpreal& v, mp_rnd_t rnd_mode);
    friend const mpreal atan2 (const mpreal& y, const mpreal& x, mp_rnd_t rnd_mode);
    friend const mpreal acot  (const mpreal& v, mp_rnd_t rnd_mode);
    friend const mpreal asec  (const mpreal& v, mp_rnd_t rnd_mode);
    friend const mpreal acsc  (const mpreal& v, mp_rnd_t rnd_mode);

    friend const mpreal cosh  (const mpreal& v, mp_rnd_t rnd_mode);
    friend const mpreal sinh  (const mpreal& v, mp_rnd_t rnd_mode);
    friend const mpreal tanh  (const mpreal& v, mp_rnd_t rnd_mode);
    friend const mpreal sech  (const mpreal& v, mp_rnd_t rnd_mode);
    friend const mpreal csch  (const mpreal& v, mp_rnd_t rnd_mode);
    friend const mpreal coth  (const mpreal& v, mp_rnd_t rnd_mode);
    friend const mpreal acosh (const mpreal& v, mp_rnd_t rnd_mode);
    friend const mpreal asinh (const mpreal& v, mp_rnd_t rnd_mode);
    friend const mpreal atanh (const mpreal& v, mp_rnd_t rnd_mode);
    friend const mpreal acoth (const mpreal& v, mp_rnd_t rnd_mode);
    friend const mpreal asech (const mpreal& v, mp_rnd_t rnd_mode);
    friend const mpreal acsch (const mpreal& v, mp_rnd_t rnd_mode);

    friend const mpreal hypot (const mpreal& x, const mpreal& y, mp_rnd_t rnd_mode);

    friend const mpreal fac_ui (unsigned long int v,  mp_prec_t prec, mp_rnd_t rnd_mode);
    friend const mpreal eint   (const mpreal& v, mp_rnd_t rnd_mode);

    friend const mpreal gamma    (const mpreal& v, mp_rnd_t rnd_mode);
    friend const mpreal lngamma  (const mpreal& v, mp_rnd_t rnd_mode);
    friend const mpreal lgamma   (const mpreal& v, int *signp, mp_rnd_t rnd_mode);
    friend const mpreal zeta     (const mpreal& v, mp_rnd_t rnd_mode);
    friend const mpreal erf      (const mpreal& v, mp_rnd_t rnd_mode);
    friend const mpreal erfc     (const mpreal& v, mp_rnd_t rnd_mode);
    friend const mpreal besselj0 (const mpreal& v, mp_rnd_t rnd_mode); 
    friend const mpreal besselj1 (const mpreal& v, mp_rnd_t rnd_mode); 
    friend const mpreal besseljn (long n, const mpreal& v, mp_rnd_t rnd_mode);
    friend const mpreal bessely0 (const mpreal& v, mp_rnd_t rnd_mode);
    friend const mpreal bessely1 (const mpreal& v, mp_rnd_t rnd_mode);
    friend const mpreal besselyn (long n, const mpreal& v, mp_rnd_t rnd_mode); 
    friend const mpreal fma      (const mpreal& v1, const mpreal& v2, const mpreal& v3, mp_rnd_t rnd_mode);
    friend const mpreal fms      (const mpreal& v1, const mpreal& v2, const mpreal& v3, mp_rnd_t rnd_mode);
    friend const mpreal agm      (const mpreal& v1, const mpreal& v2, mp_rnd_t rnd_mode);
    friend const mpreal sum      (const mpreal tab[], unsigned long int n, mp_rnd_t rnd_mode);
    friend int sgn(const mpreal& v); // returns -1 or +1

// MPFR 2.4.0 Specifics
#if (MPFR_VERSION >= MPFR_VERSION_NUM(2,4,0))
    friend int          sinh_cosh   (mpreal& s, mpreal& c, const mpreal& v, mp_rnd_t rnd_mode);
    friend const mpreal li2         (const mpreal& v,                       mp_rnd_t rnd_mode);
    friend const mpreal fmod        (const mpreal& x, const mpreal& y,      mp_rnd_t rnd_mode);
    friend const mpreal rec_sqrt    (const mpreal& v,                       mp_rnd_t rnd_mode);

    // MATLAB's semantic equivalents
    friend const mpreal rem (const mpreal& x, const mpreal& y, mp_rnd_t rnd_mode); // Remainder after division
    friend const mpreal mod (const mpreal& x, const mpreal& y, mp_rnd_t rnd_mode); // Modulus after division
#endif

// MPFR 3.0.0 Specifics
#if (MPFR_VERSION >= MPFR_VERSION_NUM(3,0,0))
    friend const mpreal digamma (const mpreal& v,        mp_rnd_t rnd_mode);
    friend const mpreal ai      (const mpreal& v,        mp_rnd_t rnd_mode);
    friend const mpreal urandom (gmp_randstate_t& state, mp_rnd_t rnd_mode);     // use gmp_randinit_default() to init state, gmp_randclear() to clear
    friend const mpreal grandom (gmp_randstate_t& state, mp_rnd_t rnd_mode);     // use gmp_randinit_default() to init state, gmp_randclear() to clear
    friend const mpreal grandom (unsigned int seed);
#endif
    
    // Uniformly distributed random number generation in [0,1] using
    // Mersenne-Twister algorithm by default.
    // Use parameter to setup seed, e.g.: random((unsigned)time(NULL))
    // Check urandom() for more precise control.
    friend const mpreal random(unsigned int seed);

    // Exponent and mantissa manipulation
    friend const mpreal frexp(const mpreal& v, mp_exp_t* exp);    
    friend const mpreal ldexp(const mpreal& v, mp_exp_t exp);

    // Splits mpreal value into fractional and integer parts.
    // Returns fractional part and stores integer part in n.
    friend const mpreal modf(const mpreal& v, mpreal& n);    

    // Constants
    // don't forget to call mpfr_free_cache() for every thread where you are using const-functions
    friend const mpreal const_log2      (mp_prec_t prec, mp_rnd_t rnd_mode);
    friend const mpreal const_pi        (mp_prec_t prec, mp_rnd_t rnd_mode);
    friend const mpreal const_euler     (mp_prec_t prec, mp_rnd_t rnd_mode);
    friend const mpreal const_catalan   (mp_prec_t prec, mp_rnd_t rnd_mode);

    // returns +inf iff sign>=0 otherwise -inf
    friend const mpreal const_infinity(int sign, mp_prec_t prec);

    // Output/ Input
    friend std::ostream& operator<<(std::ostream& os, const mpreal& v);
    friend std::istream& operator>>(std::istream& is, mpreal& v);

    // Integer Related Functions
    friend const mpreal rint (const mpreal& v, mp_rnd_t rnd_mode);
    friend const mpreal ceil (const mpreal& v);
    friend const mpreal floor(const mpreal& v);
    friend const mpreal round(const mpreal& v);
    friend const mpreal trunc(const mpreal& v);
    friend const mpreal rint_ceil   (const mpreal& v, mp_rnd_t rnd_mode);
    friend const mpreal rint_floor  (const mpreal& v, mp_rnd_t rnd_mode);
    friend const mpreal rint_round  (const mpreal& v, mp_rnd_t rnd_mode);
    friend const mpreal rint_trunc  (const mpreal& v, mp_rnd_t rnd_mode);
    friend const mpreal frac        (const mpreal& v, mp_rnd_t rnd_mode);
    friend const mpreal remainder   (         const mpreal& x, const mpreal& y, mp_rnd_t rnd_mode);
    friend const mpreal remquo      (long* q, const mpreal& x, const mpreal& y, mp_rnd_t rnd_mode);
    
    // Miscellaneous Functions
    friend const mpreal nexttoward (const mpreal& x, const mpreal& y);
    friend const mpreal nextabove  (const mpreal& x);
    friend const mpreal nextbelow  (const mpreal& x);

    // use gmp_randinit_default() to init state, gmp_randclear() to clear
    friend const mpreal urandomb (gmp_randstate_t& state); 

// MPFR < 2.4.2 Specifics
#if (MPFR_VERSION <= MPFR_VERSION_NUM(2,4,2))
    friend const mpreal random2 (mp_size_t size, mp_exp_t exp);
#endif

    // Instance Checkers
    friend bool (isnan)    (const mpreal& v);
    friend bool (isinf)    (const mpreal& v);
    friend bool (isfinite) (const mpreal& v);

    friend bool isnum    (const mpreal& v);
    friend bool iszero   (const mpreal& v);
    friend bool isint    (const mpreal& v);

#if (MPFR_VERSION >= MPFR_VERSION_NUM(3,0,0))
    friend bool isregular(const mpreal& v);
#endif

    // Set/Get instance properties
    inline mp_prec_t    get_prec() const;
    inline void         set_prec(mp_prec_t prec, mp_rnd_t rnd_mode = get_default_rnd());    // Change precision with rounding mode

    // Aliases for get_prec(), set_prec() - needed for compatibility with std::complex<mpreal> interface
    inline mpreal&      setPrecision(int Precision, mp_rnd_t RoundingMode = get_default_rnd());
    inline int          getPrecision() const;
    
    // Set mpreal to +/- inf, NaN, +/-0
    mpreal&        setInf  (int Sign = +1);    
    mpreal&        setNan  ();
    mpreal&        setZero (int Sign = +1);
    mpreal&        setSign (int Sign, mp_rnd_t RoundingMode = get_default_rnd());

    //Exponent
    mp_exp_t get_exp();
    int set_exp(mp_exp_t e);
    int check_range  (int t, mp_rnd_t rnd_mode = get_default_rnd());
    int subnormalize (int t,mp_rnd_t rnd_mode = get_default_rnd());

    // Inexact conversion from float
    inline bool fits_in_bits(double x, int n);

    // Set/Get global properties
    static void            set_default_prec(mp_prec_t prec);
    static void            set_default_rnd(mp_rnd_t rnd_mode);

    static mp_exp_t  get_emin (void);
    static mp_exp_t  get_emax (void);
    static mp_exp_t  get_emin_min (void);
    static mp_exp_t  get_emin_max (void);
    static mp_exp_t  get_emax_min (void);
    static mp_exp_t  get_emax_max (void);
    static int       set_emin (mp_exp_t exp);
    static int       set_emax (mp_exp_t exp);

    // Efficient swapping of two mpreal values - needed for std algorithms
    friend void swap(mpreal& x, mpreal& y);
    
    friend const mpreal fmax(const mpreal& x, const mpreal& y, mp_rnd_t rnd_mode);
    friend const mpreal fmin(const mpreal& x, const mpreal& y, mp_rnd_t rnd_mode);

private:
    // Human friendly Debug Preview in Visual Studio.
    // Put one of these lines:
    //
    // mpfr::mpreal=<DebugView>                              ; Show value only
    // mpfr::mpreal=<DebugView>, <mp[0]._mpfr_prec,u>bits    ; Show value & precision
    // 
    // at the beginning of
    // [Visual Studio Installation Folder]\Common7\Packages\Debugger\autoexp.dat
    MPREAL_MSVC_DEBUGVIEW_DATA

    // "Smart" resources deallocation. Checks if instance initialized before deletion.
    void clear(::mpfr_ptr);
};

//////////////////////////////////////////////////////////////////////////
// Exceptions
class conversion_overflow : public std::exception {
public:
    std::string why() { return "inexact conversion from floating point"; }
};

//////////////////////////////////////////////////////////////////////////
// Constructors & converters
// Default constructor: creates mp number and initializes it to 0.
inline mpreal::mpreal() 
{ 
    mpfr_init2 (mpfr_ptr(), mpreal::get_default_prec()); 
    mpfr_set_ui(mpfr_ptr(), 0, mpreal::get_default_rnd());

    MPREAL_MSVC_DEBUGVIEW_CODE;
}

inline mpreal::mpreal(const mpreal& u) 
{
    mpfr_init2(mpfr_ptr(),mpfr_get_prec(u.mpfr_srcptr()));
    mpfr_set  (mpfr_ptr(),u.mpfr_srcptr(),mpreal::get_default_rnd());

    MPREAL_MSVC_DEBUGVIEW_CODE;
}

#ifdef MPREAL_HAVE_MOVE_SUPPORT
inline mpreal::mpreal(mpreal&& other)
{
    mpfr_set_uninitialized(mpfr_ptr());     // make sure "other" holds no pinter to actual data 
    mpfr_swap(mpfr_ptr(), other.mpfr_ptr());

    MPREAL_MSVC_DEBUGVIEW_CODE;
}

inline mpreal& mpreal::operator=(mpreal&& other)
{
    mpfr_swap(mpfr_ptr(), other.mpfr_ptr());

    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}
#endif

inline mpreal::mpreal(const mpfr_t  u, bool shared)
{
    if(shared)
    {
        std::memcpy(mpfr_ptr(), u, sizeof(mpfr_t));
    }
    else
    {
        mpfr_init2(mpfr_ptr(), mpfr_get_prec(u));
        mpfr_set  (mpfr_ptr(), u, mpreal::get_default_rnd());
    }

    MPREAL_MSVC_DEBUGVIEW_CODE;
}

inline mpreal::mpreal(const mpf_t u)
{
    mpfr_init2(mpfr_ptr(),(mp_prec_t) mpf_get_prec(u)); // (gmp: mp_bitcnt_t) unsigned long -> long (mpfr: mp_prec_t)
    mpfr_set_f(mpfr_ptr(),u,mpreal::get_default_rnd());

    MPREAL_MSVC_DEBUGVIEW_CODE;
}

inline mpreal::mpreal(const mpz_t u, mp_prec_t prec, mp_rnd_t mode)
{
    mpfr_init2(mpfr_ptr(), prec);
    mpfr_set_z(mpfr_ptr(), u, mode);

    MPREAL_MSVC_DEBUGVIEW_CODE;
}

inline mpreal::mpreal(const mpq_t u, mp_prec_t prec, mp_rnd_t mode)
{
    mpfr_init2(mpfr_ptr(), prec);
    mpfr_set_q(mpfr_ptr(), u, mode);

    MPREAL_MSVC_DEBUGVIEW_CODE;
}

inline mpreal::mpreal(const double u, mp_prec_t prec, mp_rnd_t mode)
{
     mpfr_init2(mpfr_ptr(), prec);

#if (MPREAL_DOUBLE_BITS_OVERFLOW > -1)
  if(fits_in_bits(u, MPREAL_DOUBLE_BITS_OVERFLOW))
  {
    mpfr_set_d(mpfr_ptr(), u, mode);
  }else
    throw conversion_overflow();
#else
  mpfr_set_d(mpfr_ptr(), u, mode);
#endif

    MPREAL_MSVC_DEBUGVIEW_CODE;
}

inline mpreal::mpreal(const long double u, mp_prec_t prec, mp_rnd_t mode)
{ 
    mpfr_init2 (mpfr_ptr(), prec);
    mpfr_set_ld(mpfr_ptr(), u, mode);

    MPREAL_MSVC_DEBUGVIEW_CODE;
}

inline mpreal::mpreal(const unsigned long int u, mp_prec_t prec, mp_rnd_t mode)
{ 
    mpfr_init2 (mpfr_ptr(), prec);
    mpfr_set_ui(mpfr_ptr(), u, mode);

    MPREAL_MSVC_DEBUGVIEW_CODE;
}

inline mpreal::mpreal(const unsigned int u, mp_prec_t prec, mp_rnd_t mode)
{ 
    mpfr_init2 (mpfr_ptr(), prec);
    mpfr_set_ui(mpfr_ptr(), u, mode);

    MPREAL_MSVC_DEBUGVIEW_CODE;
}

inline mpreal::mpreal(const long int u, mp_prec_t prec, mp_rnd_t mode)
{ 
    mpfr_init2 (mpfr_ptr(), prec);
    mpfr_set_si(mpfr_ptr(), u, mode);

    MPREAL_MSVC_DEBUGVIEW_CODE;
}

inline mpreal::mpreal(const int u, mp_prec_t prec, mp_rnd_t mode)
{ 
    mpfr_init2 (mpfr_ptr(), prec);
    mpfr_set_si(mpfr_ptr(), u, mode);

    MPREAL_MSVC_DEBUGVIEW_CODE;
}

#if defined (MPREAL_HAVE_INT64_SUPPORT)
inline mpreal::mpreal(const uint64_t u, mp_prec_t prec, mp_rnd_t mode)
{
    mpfr_init2 (mpfr_ptr(), prec);
    mpfr_set_uj(mpfr_ptr(), u, mode); 

    MPREAL_MSVC_DEBUGVIEW_CODE;
}

inline mpreal::mpreal(const int64_t u, mp_prec_t prec, mp_rnd_t mode)
{
    mpfr_init2 (mpfr_ptr(), prec);
    mpfr_set_sj(mpfr_ptr(), u, mode); 

    MPREAL_MSVC_DEBUGVIEW_CODE;
}
#endif

inline mpreal::mpreal(const char* s, mp_prec_t prec, int base, mp_rnd_t mode)
{
    mpfr_init2  (mpfr_ptr(), prec);
    mpfr_set_str(mpfr_ptr(), s, base, mode); 

    MPREAL_MSVC_DEBUGVIEW_CODE;
}

inline mpreal::mpreal(const std::string& s, mp_prec_t prec, int base, mp_rnd_t mode)
{
    mpfr_init2  (mpfr_ptr(), prec);
    mpfr_set_str(mpfr_ptr(), s.c_str(), base, mode); 

    MPREAL_MSVC_DEBUGVIEW_CODE;
}

inline void mpreal::clear(::mpfr_ptr x)
{
#ifdef MPREAL_HAVE_MOVE_SUPPORT
    if(mpfr_is_initialized(x)) 
#endif
    mpfr_clear(x);
}

inline mpreal::~mpreal() 
{ 
    clear(mpfr_ptr());
}                           

// internal namespace needed for template magic
namespace internal{

    // Use SFINAE to restrict arithmetic operations instantiation only for numeric types
    // This is needed for smooth integration with libraries based on expression templates, like Eigen.
    // TODO: Do the same for boolean operators.
    template <typename ArgumentType> struct result_type {};    
    
    template <> struct result_type<mpreal>              {typedef mpreal type;};    
    template <> struct result_type<mpz_t>               {typedef mpreal type;};    
    template <> struct result_type<mpq_t>               {typedef mpreal type;};    
    template <> struct result_type<long double>         {typedef mpreal type;};    
    template <> struct result_type<double>              {typedef mpreal type;};    
    template <> struct result_type<unsigned long int>   {typedef mpreal type;};    
    template <> struct result_type<unsigned int>        {typedef mpreal type;};    
    template <> struct result_type<long int>            {typedef mpreal type;};    
    template <> struct result_type<int>                 {typedef mpreal type;};    

#if defined (MPREAL_HAVE_INT64_SUPPORT)
    template <> struct result_type<int64_t  >           {typedef mpreal type;};    
    template <> struct result_type<uint64_t >           {typedef mpreal type;};    
#endif
}

// + Addition
template <typename Rhs> 
inline const typename internal::result_type<Rhs>::type 
    operator+(const mpreal& lhs, const Rhs& rhs){ return mpreal(lhs) += rhs;    }

template <typename Lhs> 
inline const typename internal::result_type<Lhs>::type 
    operator+(const Lhs& lhs, const mpreal& rhs){ return mpreal(rhs) += lhs;    } 

// - Subtraction
template <typename Rhs> 
inline const typename internal::result_type<Rhs>::type 
    operator-(const mpreal& lhs, const Rhs& rhs){ return mpreal(lhs) -= rhs;    }

template <typename Lhs> 
inline const typename internal::result_type<Lhs>::type 
    operator-(const Lhs& lhs, const mpreal& rhs){ return mpreal(lhs) -= rhs;    }

// * Multiplication
template <typename Rhs> 
inline const typename internal::result_type<Rhs>::type 
    operator*(const mpreal& lhs, const Rhs& rhs){ return mpreal(lhs) *= rhs;    }

template <typename Lhs> 
inline const typename internal::result_type<Lhs>::type 
    operator*(const Lhs& lhs, const mpreal& rhs){ return mpreal(rhs) *= lhs;    } 

// / Division
template <typename Rhs> 
inline const typename internal::result_type<Rhs>::type 
    operator/(const mpreal& lhs, const Rhs& rhs){ return mpreal(lhs) /= rhs;    }

template <typename Lhs> 
inline const typename internal::result_type<Lhs>::type 
    operator/(const Lhs& lhs, const mpreal& rhs){ return mpreal(lhs) /= rhs;    }

//////////////////////////////////////////////////////////////////////////
// sqrt
const mpreal sqrt(const unsigned int v, mp_rnd_t rnd_mode = mpreal::get_default_rnd());
const mpreal sqrt(const long int v, mp_rnd_t rnd_mode = mpreal::get_default_rnd());
const mpreal sqrt(const int v, mp_rnd_t rnd_mode = mpreal::get_default_rnd());
const mpreal sqrt(const long double v, mp_rnd_t rnd_mode = mpreal::get_default_rnd());
const mpreal sqrt(const double v, mp_rnd_t rnd_mode = mpreal::get_default_rnd());

// abs
inline const mpreal abs(const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd());

//////////////////////////////////////////////////////////////////////////
// pow
const mpreal pow(const mpreal& a, const unsigned int b, mp_rnd_t rnd_mode = mpreal::get_default_rnd());
const mpreal pow(const mpreal& a, const int b, mp_rnd_t rnd_mode = mpreal::get_default_rnd());
const mpreal pow(const mpreal& a, const long double b, mp_rnd_t rnd_mode = mpreal::get_default_rnd());
const mpreal pow(const mpreal& a, const double b, mp_rnd_t rnd_mode = mpreal::get_default_rnd());

const mpreal pow(const unsigned int a, const mpreal& b, mp_rnd_t rnd_mode = mpreal::get_default_rnd());
const mpreal pow(const long int a, const mpreal& b, mp_rnd_t rnd_mode = mpreal::get_default_rnd());
const mpreal pow(const int a, const mpreal& b, mp_rnd_t rnd_mode = mpreal::get_default_rnd());
const mpreal pow(const long double a, const mpreal& b, mp_rnd_t rnd_mode = mpreal::get_default_rnd());
const mpreal pow(const double a, const mpreal& b, mp_rnd_t rnd_mode = mpreal::get_default_rnd());

const mpreal pow(const unsigned long int a, const unsigned int b, mp_rnd_t rnd_mode = mpreal::get_default_rnd());
const mpreal pow(const unsigned long int a, const long int b, mp_rnd_t rnd_mode = mpreal::get_default_rnd());
const mpreal pow(const unsigned long int a, const int b, mp_rnd_t rnd_mode = mpreal::get_default_rnd());
const mpreal pow(const unsigned long int a, const long double b, mp_rnd_t rnd_mode = mpreal::get_default_rnd());
const mpreal pow(const unsigned long int a, const double b, mp_rnd_t rnd_mode = mpreal::get_default_rnd());

const mpreal pow(const unsigned int a, const unsigned long int b, mp_rnd_t rnd_mode = mpreal::get_default_rnd());
const mpreal pow(const unsigned int a, const unsigned int b, mp_rnd_t rnd_mode = mpreal::get_default_rnd());
const mpreal pow(const unsigned int a, const long int b, mp_rnd_t rnd_mode = mpreal::get_default_rnd());
const mpreal pow(const unsigned int a, const int b, mp_rnd_t rnd_mode = mpreal::get_default_rnd());
const mpreal pow(const unsigned int a, const long double b, mp_rnd_t rnd_mode = mpreal::get_default_rnd());
const mpreal pow(const unsigned int a, const double b, mp_rnd_t rnd_mode = mpreal::get_default_rnd());

const mpreal pow(const long int a, const unsigned long int b, mp_rnd_t rnd_mode = mpreal::get_default_rnd());
const mpreal pow(const long int a, const unsigned int b, mp_rnd_t rnd_mode = mpreal::get_default_rnd());
const mpreal pow(const long int a, const long int b, mp_rnd_t rnd_mode = mpreal::get_default_rnd());
const mpreal pow(const long int a, const int b, mp_rnd_t rnd_mode = mpreal::get_default_rnd());
const mpreal pow(const long int a, const long double b, mp_rnd_t rnd_mode = mpreal::get_default_rnd());
const mpreal pow(const long int a, const double b, mp_rnd_t rnd_mode = mpreal::get_default_rnd());

const mpreal pow(const int a, const unsigned long int b, mp_rnd_t rnd_mode = mpreal::get_default_rnd());
const mpreal pow(const int a, const unsigned int b, mp_rnd_t rnd_mode = mpreal::get_default_rnd());
const mpreal pow(const int a, const long int b, mp_rnd_t rnd_mode = mpreal::get_default_rnd());
const mpreal pow(const int a, const int b, mp_rnd_t rnd_mode = mpreal::get_default_rnd()); 
const mpreal pow(const int a, const long double b, mp_rnd_t rnd_mode = mpreal::get_default_rnd());
const mpreal pow(const int a, const double b, mp_rnd_t rnd_mode = mpreal::get_default_rnd()); 

const mpreal pow(const long double a, const long double b, mp_rnd_t rnd_mode = mpreal::get_default_rnd());    
const mpreal pow(const long double a, const unsigned long int b, mp_rnd_t rnd_mode = mpreal::get_default_rnd());
const mpreal pow(const long double a, const unsigned int b, mp_rnd_t rnd_mode = mpreal::get_default_rnd());
const mpreal pow(const long double a, const long int b, mp_rnd_t rnd_mode = mpreal::get_default_rnd());
const mpreal pow(const long double a, const int b, mp_rnd_t rnd_mode = mpreal::get_default_rnd());

const mpreal pow(const double a, const double b, mp_rnd_t rnd_mode = mpreal::get_default_rnd());    
const mpreal pow(const double a, const unsigned long int b, mp_rnd_t rnd_mode = mpreal::get_default_rnd());
const mpreal pow(const double a, const unsigned int b, mp_rnd_t rnd_mode = mpreal::get_default_rnd());
const mpreal pow(const double a, const long int b, mp_rnd_t rnd_mode = mpreal::get_default_rnd());
const mpreal pow(const double a, const int b, mp_rnd_t rnd_mode = mpreal::get_default_rnd());

inline const mpreal mul_2ui(const mpreal& v, unsigned long int k, mp_rnd_t rnd_mode = mpreal::get_default_rnd());
inline const mpreal mul_2si(const mpreal& v, long int k, mp_rnd_t rnd_mode = mpreal::get_default_rnd());
inline const mpreal div_2ui(const mpreal& v, unsigned long int k, mp_rnd_t rnd_mode = mpreal::get_default_rnd());
inline const mpreal div_2si(const mpreal& v, long int k, mp_rnd_t rnd_mode = mpreal::get_default_rnd());

//////////////////////////////////////////////////////////////////////////
// Estimate machine epsilon for the given precision
// Returns smallest eps such that 1.0 + eps != 1.0
inline mpreal machine_epsilon(mp_prec_t prec = mpreal::get_default_prec());

// Returns smallest eps such that x + eps != x (relative machine epsilon)
inline mpreal machine_epsilon(const mpreal& x);        

// Gives max & min values for the required precision, 
// minval is 'safe' meaning 1 / minval does not overflow
// maxval is 'safe' meaning 1 / maxval does not underflow
inline mpreal minval(mp_prec_t prec = mpreal::get_default_prec());
inline mpreal maxval(mp_prec_t prec = mpreal::get_default_prec());

// 'Dirty' equality check 1: |a-b| < min{|a|,|b|} * eps
inline bool isEqualFuzzy(const mpreal& a, const mpreal& b, const mpreal& eps);

// 'Dirty' equality check 2: |a-b| < min{|a|,|b|} * eps( min{|a|,|b|} )
inline bool isEqualFuzzy(const mpreal& a, const mpreal& b);

// 'Bitwise' equality check
//  maxUlps - a and b can be apart by maxUlps binary numbers. 
inline bool isEqualUlps(const mpreal& a, const mpreal& b, int maxUlps);

//////////////////////////////////////////////////////////////////////////
//     Convert precision in 'bits' to decimal digits and vice versa.
//        bits   = ceil(digits*log[2](10))
//        digits = floor(bits*log[10](2))

inline mp_prec_t digits2bits(int d);
inline int       bits2digits(mp_prec_t b);

//////////////////////////////////////////////////////////////////////////
// min, max
const mpreal (max)(const mpreal& x, const mpreal& y);
const mpreal (min)(const mpreal& x, const mpreal& y);

//////////////////////////////////////////////////////////////////////////
// Implementation
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
// Operators - Assignment
inline mpreal& mpreal::operator=(const mpreal& v)
{
    if (this != &v)
    {
    mp_prec_t tp = mpfr_get_prec(  mpfr_srcptr());
    mp_prec_t vp = mpfr_get_prec(v.mpfr_srcptr());

    if(tp != vp){
      clear(mpfr_ptr());
      mpfr_init2(mpfr_ptr(), vp);
    }

        mpfr_set(mpfr_ptr(), v.mpfr_srcptr(), mpreal::get_default_rnd());

        MPREAL_MSVC_DEBUGVIEW_CODE;
    }
    return *this;
}

inline mpreal& mpreal::operator=(const mpf_t v)
{
    mpfr_set_f(mpfr_ptr(), v, mpreal::get_default_rnd());
    
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal& mpreal::operator=(const mpz_t v)
{
    mpfr_set_z(mpfr_ptr(), v, mpreal::get_default_rnd());
    
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal& mpreal::operator=(const mpq_t v)
{
    mpfr_set_q(mpfr_ptr(), v, mpreal::get_default_rnd());

    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal& mpreal::operator=(const long double v)        
{    
    mpfr_set_ld(mpfr_ptr(), v, mpreal::get_default_rnd());

    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal& mpreal::operator=(const double v)                
{   
#if (MPREAL_DOUBLE_BITS_OVERFLOW > -1)
  if(fits_in_bits(v, MPREAL_DOUBLE_BITS_OVERFLOW))
  {
    mpfr_set_d(mpfr_ptr(),v,mpreal::get_default_rnd());
  }else
    throw conversion_overflow();
#else
  mpfr_set_d(mpfr_ptr(),v,mpreal::get_default_rnd());
#endif

  MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal& mpreal::operator=(const unsigned long int v)    
{    
    mpfr_set_ui(mpfr_ptr(), v, mpreal::get_default_rnd());    

    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal& mpreal::operator=(const unsigned int v)        
{    
    mpfr_set_ui(mpfr_ptr(), v, mpreal::get_default_rnd());    

    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal& mpreal::operator=(const long int v)            
{    
    mpfr_set_si(mpfr_ptr(), v, mpreal::get_default_rnd());    

    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal& mpreal::operator=(const int v)
{    
    mpfr_set_si(mpfr_ptr(), v, mpreal::get_default_rnd());    

    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal& mpreal::operator=(const char* s)
{
    // Use other converters for more precise control on base & precision & rounding:
    //
    //        mpreal(const char* s,        mp_prec_t prec, int base, mp_rnd_t mode)
    //        mpreal(const std::string& s,mp_prec_t prec, int base, mp_rnd_t mode)
    //
    // Here we assume base = 10 and we use precision of target variable.

    mpfr_t t;

    mpfr_init2(t, mpfr_get_prec(mpfr_srcptr()));

    if(0 == mpfr_set_str(t, s, 10, mpreal::get_default_rnd()))
    {
        mpfr_set(mpfr_ptr(), t, mpreal::get_default_rnd()); 
        MPREAL_MSVC_DEBUGVIEW_CODE;
    }

    clear(t);
    return *this;
}

inline mpreal& mpreal::operator=(const std::string& s)
{
    // Use other converters for more precise control on base & precision & rounding:
    //
    //        mpreal(const char* s,        mp_prec_t prec, int base, mp_rnd_t mode)
    //        mpreal(const std::string& s,mp_prec_t prec, int base, mp_rnd_t mode)
    //
    // Here we assume base = 10 and we use precision of target variable.

    mpfr_t t;

    mpfr_init2(t, mpfr_get_prec(mpfr_srcptr()));

    if(0 == mpfr_set_str(t, s.c_str(), 10, mpreal::get_default_rnd()))
    {
        mpfr_set(mpfr_ptr(), t, mpreal::get_default_rnd()); 
        MPREAL_MSVC_DEBUGVIEW_CODE;
    }

    clear(t);
    return *this;
}


//////////////////////////////////////////////////////////////////////////
// + Addition
inline mpreal& mpreal::operator+=(const mpreal& v)
{
    mpfr_add(mpfr_ptr(), mpfr_srcptr(), v.mpfr_srcptr(), mpreal::get_default_rnd());
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal& mpreal::operator+=(const mpf_t u)
{
    *this += mpreal(u);
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal& mpreal::operator+=(const mpz_t u)
{
    mpfr_add_z(mpfr_ptr(),mpfr_srcptr(),u,mpreal::get_default_rnd());
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal& mpreal::operator+=(const mpq_t u)
{
    mpfr_add_q(mpfr_ptr(),mpfr_srcptr(),u,mpreal::get_default_rnd());
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal& mpreal::operator+= (const long double u)
{
    *this += mpreal(u);    
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;    
}

inline mpreal& mpreal::operator+= (const double u)
{
#if (MPFR_VERSION >= MPFR_VERSION_NUM(2,4,0))
    mpfr_add_d(mpfr_ptr(),mpfr_srcptr(),u,mpreal::get_default_rnd());
#else
    *this += mpreal(u);
#endif

    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal& mpreal::operator+=(const unsigned long int u)
{
    mpfr_add_ui(mpfr_ptr(),mpfr_srcptr(),u,mpreal::get_default_rnd());
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal& mpreal::operator+=(const unsigned int u)
{
    mpfr_add_ui(mpfr_ptr(),mpfr_srcptr(),u,mpreal::get_default_rnd());
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal& mpreal::operator+=(const long int u)
{
    mpfr_add_si(mpfr_ptr(),mpfr_srcptr(),u,mpreal::get_default_rnd());
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal& mpreal::operator+=(const int u)
{
    mpfr_add_si(mpfr_ptr(),mpfr_srcptr(),u,mpreal::get_default_rnd());
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

#if defined (MPREAL_HAVE_INT64_SUPPORT)
inline mpreal& mpreal::operator+=(const int64_t  u){    *this += mpreal(u); MPREAL_MSVC_DEBUGVIEW_CODE; return *this;    }
inline mpreal& mpreal::operator+=(const uint64_t u){    *this += mpreal(u); MPREAL_MSVC_DEBUGVIEW_CODE; return *this;    }
inline mpreal& mpreal::operator-=(const int64_t  u){    *this -= mpreal(u); MPREAL_MSVC_DEBUGVIEW_CODE; return *this;    }
inline mpreal& mpreal::operator-=(const uint64_t u){    *this -= mpreal(u); MPREAL_MSVC_DEBUGVIEW_CODE; return *this;    }
inline mpreal& mpreal::operator*=(const int64_t  u){    *this *= mpreal(u); MPREAL_MSVC_DEBUGVIEW_CODE; return *this;    }
inline mpreal& mpreal::operator*=(const uint64_t u){    *this *= mpreal(u); MPREAL_MSVC_DEBUGVIEW_CODE; return *this;    }
inline mpreal& mpreal::operator/=(const int64_t  u){    *this /= mpreal(u); MPREAL_MSVC_DEBUGVIEW_CODE; return *this;    }
inline mpreal& mpreal::operator/=(const uint64_t u){    *this /= mpreal(u); MPREAL_MSVC_DEBUGVIEW_CODE; return *this;    }
#endif 

inline const mpreal mpreal::operator+()const    {    return mpreal(*this); }

inline const mpreal operator+(const mpreal& a, const mpreal& b)
{
  mpreal c(0, (std::max)(mpfr_get_prec(a.mpfr_ptr()), mpfr_get_prec(b.mpfr_ptr())));
  mpfr_add(c.mpfr_ptr(), a.mpfr_srcptr(), b.mpfr_srcptr(), mpreal::get_default_rnd());
  return c;
}

inline mpreal& mpreal::operator++() 
{
    return *this += 1;
}

inline const mpreal mpreal::operator++ (int)
{
    mpreal x(*this);
    *this += 1;
    return x;
}

inline mpreal& mpreal::operator--() 
{
    return *this -= 1;
}

inline const mpreal mpreal::operator-- (int)
{
    mpreal x(*this);
    *this -= 1;
    return x;
}

//////////////////////////////////////////////////////////////////////////
// - Subtraction
inline mpreal& mpreal::operator-=(const mpreal& v)
{
    mpfr_sub(mpfr_ptr(),mpfr_srcptr(),v.mpfr_srcptr(),mpreal::get_default_rnd());
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal& mpreal::operator-=(const mpz_t v)
{
    mpfr_sub_z(mpfr_ptr(),mpfr_srcptr(),v,mpreal::get_default_rnd());
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal& mpreal::operator-=(const mpq_t v)
{
    mpfr_sub_q(mpfr_ptr(),mpfr_srcptr(),v,mpreal::get_default_rnd());
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal& mpreal::operator-=(const long double v)
{
    *this -= mpreal(v);    
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;    
}

inline mpreal& mpreal::operator-=(const double v)
{
#if (MPFR_VERSION >= MPFR_VERSION_NUM(2,4,0))
    mpfr_sub_d(mpfr_ptr(),mpfr_srcptr(),v,mpreal::get_default_rnd());
#else
    *this -= mpreal(v);    
#endif

    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal& mpreal::operator-=(const unsigned long int v)
{
    mpfr_sub_ui(mpfr_ptr(),mpfr_srcptr(),v,mpreal::get_default_rnd());
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal& mpreal::operator-=(const unsigned int v)
{
    mpfr_sub_ui(mpfr_ptr(),mpfr_srcptr(),v,mpreal::get_default_rnd());
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal& mpreal::operator-=(const long int v)
{
    mpfr_sub_si(mpfr_ptr(),mpfr_srcptr(),v,mpreal::get_default_rnd());
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal& mpreal::operator-=(const int v)
{
    mpfr_sub_si(mpfr_ptr(),mpfr_srcptr(),v,mpreal::get_default_rnd());
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline const mpreal mpreal::operator-()const
{
    mpreal u(*this);
    mpfr_neg(u.mpfr_ptr(),u.mpfr_srcptr(),mpreal::get_default_rnd());
    return u;
}

inline const mpreal operator-(const mpreal& a, const mpreal& b)
{
  mpreal c(0, (std::max)(mpfr_get_prec(a.mpfr_ptr()), mpfr_get_prec(b.mpfr_ptr())));
  mpfr_sub(c.mpfr_ptr(), a.mpfr_srcptr(), b.mpfr_srcptr(), mpreal::get_default_rnd());
  return c;
}

inline const mpreal operator-(const double  b, const mpreal& a)
{
#if (MPFR_VERSION >= MPFR_VERSION_NUM(2,4,0))
    mpreal x(0, mpfr_get_prec(a.mpfr_ptr()));
    mpfr_d_sub(x.mpfr_ptr(), b, a.mpfr_srcptr(), mpreal::get_default_rnd());
    return x;
#else
    mpreal x(b, mpfr_get_prec(a.mpfr_ptr()));
    x -= a;
    return x;
#endif
}

inline const mpreal operator-(const unsigned long int b, const mpreal& a)
{
    mpreal x(0, mpfr_get_prec(a.mpfr_ptr()));
    mpfr_ui_sub(x.mpfr_ptr(), b, a.mpfr_srcptr(), mpreal::get_default_rnd());
    return x;
}

inline const mpreal operator-(const unsigned int b, const mpreal& a)
{
    mpreal x(0, mpfr_get_prec(a.mpfr_ptr()));
    mpfr_ui_sub(x.mpfr_ptr(), b, a.mpfr_srcptr(), mpreal::get_default_rnd());
    return x;
}

inline const mpreal operator-(const long int b, const mpreal& a)
{
    mpreal x(0, mpfr_get_prec(a.mpfr_ptr()));
    mpfr_si_sub(x.mpfr_ptr(), b, a.mpfr_srcptr(), mpreal::get_default_rnd());
    return x;
}

inline const mpreal operator-(const int b, const mpreal& a)
{
    mpreal x(0, mpfr_get_prec(a.mpfr_ptr()));
    mpfr_si_sub(x.mpfr_ptr(), b, a.mpfr_srcptr(), mpreal::get_default_rnd());
    return x;
}

//////////////////////////////////////////////////////////////////////////
// * Multiplication
inline mpreal& mpreal::operator*= (const mpreal& v)
{
    mpfr_mul(mpfr_ptr(),mpfr_srcptr(),v.mpfr_srcptr(),mpreal::get_default_rnd());
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal& mpreal::operator*=(const mpz_t v)
{
    mpfr_mul_z(mpfr_ptr(),mpfr_srcptr(),v,mpreal::get_default_rnd());
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal& mpreal::operator*=(const mpq_t v)
{
    mpfr_mul_q(mpfr_ptr(),mpfr_srcptr(),v,mpreal::get_default_rnd());
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal& mpreal::operator*=(const long double v)
{
    *this *= mpreal(v);    
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;    
}

inline mpreal& mpreal::operator*=(const double v)
{
#if (MPFR_VERSION >= MPFR_VERSION_NUM(2,4,0))
    mpfr_mul_d(mpfr_ptr(),mpfr_srcptr(),v,mpreal::get_default_rnd());
#else
    *this *= mpreal(v);    
#endif
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal& mpreal::operator*=(const unsigned long int v)
{
    mpfr_mul_ui(mpfr_ptr(),mpfr_srcptr(),v,mpreal::get_default_rnd());
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal& mpreal::operator*=(const unsigned int v)
{
    mpfr_mul_ui(mpfr_ptr(),mpfr_srcptr(),v,mpreal::get_default_rnd());
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal& mpreal::operator*=(const long int v)
{
    mpfr_mul_si(mpfr_ptr(),mpfr_srcptr(),v,mpreal::get_default_rnd());
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal& mpreal::operator*=(const int v)
{
    mpfr_mul_si(mpfr_ptr(),mpfr_srcptr(),v,mpreal::get_default_rnd());
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline const mpreal operator*(const mpreal& a, const mpreal& b)
{
  mpreal c(0, (std::max)(mpfr_get_prec(a.mpfr_ptr()), mpfr_get_prec(b.mpfr_ptr())));
  mpfr_mul(c.mpfr_ptr(), a.mpfr_srcptr(), b.mpfr_srcptr(), mpreal::get_default_rnd());
  return c;
}

//////////////////////////////////////////////////////////////////////////
// / Division
inline mpreal& mpreal::operator/=(const mpreal& v)
{
    mpfr_div(mpfr_ptr(),mpfr_srcptr(),v.mpfr_srcptr(),mpreal::get_default_rnd());
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal& mpreal::operator/=(const mpz_t v)
{
    mpfr_div_z(mpfr_ptr(),mpfr_srcptr(),v,mpreal::get_default_rnd());
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal& mpreal::operator/=(const mpq_t v)
{
    mpfr_div_q(mpfr_ptr(),mpfr_srcptr(),v,mpreal::get_default_rnd());
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal& mpreal::operator/=(const long double v)
{
    *this /= mpreal(v);
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;    
}

inline mpreal& mpreal::operator/=(const double v)
{
#if (MPFR_VERSION >= MPFR_VERSION_NUM(2,4,0))
    mpfr_div_d(mpfr_ptr(),mpfr_srcptr(),v,mpreal::get_default_rnd());
#else
    *this /= mpreal(v);    
#endif
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal& mpreal::operator/=(const unsigned long int v)
{
    mpfr_div_ui(mpfr_ptr(),mpfr_srcptr(),v,mpreal::get_default_rnd());
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal& mpreal::operator/=(const unsigned int v)
{
    mpfr_div_ui(mpfr_ptr(),mpfr_srcptr(),v,mpreal::get_default_rnd());
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal& mpreal::operator/=(const long int v)
{
    mpfr_div_si(mpfr_ptr(),mpfr_srcptr(),v,mpreal::get_default_rnd());
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal& mpreal::operator/=(const int v)
{
    mpfr_div_si(mpfr_ptr(),mpfr_srcptr(),v,mpreal::get_default_rnd());
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline const mpreal operator/(const mpreal& a, const mpreal& b)
{
  mpreal c(0, (std::max)(mpfr_get_prec(a.mpfr_srcptr()), mpfr_get_prec(b.mpfr_srcptr())));
  mpfr_div(c.mpfr_ptr(), a.mpfr_srcptr(), b.mpfr_srcptr(), mpreal::get_default_rnd());
  return c;
}

inline const mpreal operator/(const unsigned long int b, const mpreal& a)
{
    mpreal x(0, mpfr_get_prec(a.mpfr_srcptr()));
    mpfr_ui_div(x.mpfr_ptr(), b, a.mpfr_srcptr(), mpreal::get_default_rnd());
    return x;
}

inline const mpreal operator/(const unsigned int b, const mpreal& a)
{
    mpreal x(0, mpfr_get_prec(a.mpfr_srcptr()));
    mpfr_ui_div(x.mpfr_ptr(), b, a.mpfr_srcptr(), mpreal::get_default_rnd());
    return x;
}

inline const mpreal operator/(const long int b, const mpreal& a)
{
    mpreal x(0, mpfr_get_prec(a.mpfr_srcptr()));
    mpfr_si_div(x.mpfr_ptr(), b, a.mpfr_srcptr(), mpreal::get_default_rnd());
    return x;
}

inline const mpreal operator/(const int b, const mpreal& a)
{
    mpreal x(0, mpfr_get_prec(a.mpfr_srcptr()));
    mpfr_si_div(x.mpfr_ptr(), b, a.mpfr_srcptr(), mpreal::get_default_rnd());
    return x;
}

inline const mpreal operator/(const double  b, const mpreal& a)
{
#if (MPFR_VERSION >= MPFR_VERSION_NUM(2,4,0))
    mpreal x(0, mpfr_get_prec(a.mpfr_srcptr()));
    mpfr_d_div(x.mpfr_ptr(), b, a.mpfr_srcptr(), mpreal::get_default_rnd());
    return x;
#else
    mpreal x(0, mpfr_get_prec(a.mpfr_ptr()));
    x /= a;
    return x;
#endif
}

//////////////////////////////////////////////////////////////////////////
// Shifts operators - Multiplication/Division by power of 2
inline mpreal& mpreal::operator<<=(const unsigned long int u)
{
    mpfr_mul_2ui(mpfr_ptr(),mpfr_srcptr(),u,mpreal::get_default_rnd());
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal& mpreal::operator<<=(const unsigned int u)
{
    mpfr_mul_2ui(mpfr_ptr(),mpfr_srcptr(),static_cast<unsigned long int>(u),mpreal::get_default_rnd());
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal& mpreal::operator<<=(const long int u)
{
    mpfr_mul_2si(mpfr_ptr(),mpfr_srcptr(),u,mpreal::get_default_rnd());
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal& mpreal::operator<<=(const int u)
{
    mpfr_mul_2si(mpfr_ptr(),mpfr_srcptr(),static_cast<long int>(u),mpreal::get_default_rnd());
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal& mpreal::operator>>=(const unsigned long int u)
{
    mpfr_div_2ui(mpfr_ptr(),mpfr_srcptr(),u,mpreal::get_default_rnd());
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal& mpreal::operator>>=(const unsigned int u)
{
    mpfr_div_2ui(mpfr_ptr(),mpfr_srcptr(),static_cast<unsigned long int>(u),mpreal::get_default_rnd());
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal& mpreal::operator>>=(const long int u)
{
    mpfr_div_2si(mpfr_ptr(),mpfr_srcptr(),u,mpreal::get_default_rnd());
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal& mpreal::operator>>=(const int u)
{
    mpfr_div_2si(mpfr_ptr(),mpfr_srcptr(),static_cast<long int>(u),mpreal::get_default_rnd());
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline const mpreal operator<<(const mpreal& v, const unsigned long int k)
{
    return mul_2ui(v,k);
}

inline const mpreal operator<<(const mpreal& v, const unsigned int k)
{
    return mul_2ui(v,static_cast<unsigned long int>(k));
}

inline const mpreal operator<<(const mpreal& v, const long int k)
{
    return mul_2si(v,k);
}

inline const mpreal operator<<(const mpreal& v, const int k)
{
    return mul_2si(v,static_cast<long int>(k));
}

inline const mpreal operator>>(const mpreal& v, const unsigned long int k)
{
    return div_2ui(v,k);
}

inline const mpreal operator>>(const mpreal& v, const long int k)
{
    return div_2si(v,k);
}

inline const mpreal operator>>(const mpreal& v, const unsigned int k)
{
    return div_2ui(v,static_cast<unsigned long int>(k));
}

inline const mpreal operator>>(const mpreal& v, const int k)
{
    return div_2si(v,static_cast<long int>(k));
}

// mul_2ui
inline const mpreal mul_2ui(const mpreal& v, unsigned long int k, mp_rnd_t rnd_mode)
{
    mpreal x(v);
    mpfr_mul_2ui(x.mpfr_ptr(),v.mpfr_srcptr(),k,rnd_mode);
    return x;
}

// mul_2si
inline const mpreal mul_2si(const mpreal& v, long int k, mp_rnd_t rnd_mode)
{
    mpreal x(v);
    mpfr_mul_2si(x.mpfr_ptr(),v.mpfr_srcptr(),k,rnd_mode);
    return x;
}

inline const mpreal div_2ui(const mpreal& v, unsigned long int k, mp_rnd_t rnd_mode)
{
    mpreal x(v);
    mpfr_div_2ui(x.mpfr_ptr(),v.mpfr_srcptr(),k,rnd_mode);
    return x;
}

inline const mpreal div_2si(const mpreal& v, long int k, mp_rnd_t rnd_mode)
{
    mpreal x(v);
    mpfr_div_2si(x.mpfr_ptr(),v.mpfr_srcptr(),k,rnd_mode);
    return x;
}

//////////////////////////////////////////////////////////////////////////
//Boolean operators
inline bool operator >  (const mpreal& a, const mpreal& b){    return (mpfr_greater_p       (a.mpfr_srcptr(),b.mpfr_srcptr()) !=0 );    }
inline bool operator >= (const mpreal& a, const mpreal& b){    return (mpfr_greaterequal_p  (a.mpfr_srcptr(),b.mpfr_srcptr()) !=0 );    }
inline bool operator <  (const mpreal& a, const mpreal& b){    return (mpfr_less_p          (a.mpfr_srcptr(),b.mpfr_srcptr()) !=0 );    }
inline bool operator <= (const mpreal& a, const mpreal& b){    return (mpfr_lessequal_p     (a.mpfr_srcptr(),b.mpfr_srcptr()) !=0 );    }
inline bool operator == (const mpreal& a, const mpreal& b){    return (mpfr_equal_p         (a.mpfr_srcptr(),b.mpfr_srcptr()) !=0 );    }
inline bool operator != (const mpreal& a, const mpreal& b){    return (mpfr_lessgreater_p   (a.mpfr_srcptr(),b.mpfr_srcptr()) !=0 );    }

inline bool operator == (const mpreal& a, const unsigned long int b ){    return (mpfr_cmp_ui(a.mpfr_srcptr(),b) == 0 );    }
inline bool operator == (const mpreal& a, const unsigned int b      ){    return (mpfr_cmp_ui(a.mpfr_srcptr(),b) == 0 );    }
inline bool operator == (const mpreal& a, const long int b          ){    return (mpfr_cmp_si(a.mpfr_srcptr(),b) == 0 );    }
inline bool operator == (const mpreal& a, const int b               ){    return (mpfr_cmp_si(a.mpfr_srcptr(),b) == 0 );    }
inline bool operator == (const mpreal& a, const long double b       ){    return (mpfr_cmp_ld(a.mpfr_srcptr(),b) == 0 );    }
inline bool operator == (const mpreal& a, const double b            ){    return (mpfr_cmp_d (a.mpfr_srcptr(),b) == 0 );    }


inline bool (isnan)    (const mpreal& op){    return (mpfr_nan_p    (op.mpfr_srcptr()) != 0 );    }
inline bool (isinf)    (const mpreal& op){    return (mpfr_inf_p    (op.mpfr_srcptr()) != 0 );    }
inline bool (isfinite) (const mpreal& op){    return (mpfr_number_p (op.mpfr_srcptr()) != 0 );    }
inline bool iszero   (const mpreal& op){    return (mpfr_zero_p   (op.mpfr_srcptr()) != 0 );    }
inline bool isint    (const mpreal& op){    return (mpfr_integer_p(op.mpfr_srcptr()) != 0 );    }

#if (MPFR_VERSION >= MPFR_VERSION_NUM(3,0,0))
inline bool isregular(const mpreal& op){    return (mpfr_regular_p(op.mpfr_srcptr()));}
#endif 

//////////////////////////////////////////////////////////////////////////
// Type Converters
inline bool             mpreal::toBool (mp_rnd_t /*mode*/) const   {    return  mpfr_zero_p (mpfr_srcptr()) == 0;     }
inline long             mpreal::toLong   (mp_rnd_t mode)  const    {    return  mpfr_get_si (mpfr_srcptr(), mode);    }
inline unsigned long    mpreal::toULong  (mp_rnd_t mode)  const    {    return  mpfr_get_ui (mpfr_srcptr(), mode);    }
inline float            mpreal::toFloat  (mp_rnd_t mode)  const    {    return  mpfr_get_flt(mpfr_srcptr(), mode);    }
inline double           mpreal::toDouble (mp_rnd_t mode)  const    {    return  mpfr_get_d  (mpfr_srcptr(), mode);    }
inline long double      mpreal::toLDouble(mp_rnd_t mode)  const    {    return  mpfr_get_ld (mpfr_srcptr(), mode);    }

#if defined (MPREAL_HAVE_INT64_SUPPORT)
inline int64_t      mpreal::toInt64 (mp_rnd_t mode)    const{    return mpfr_get_sj(mpfr_srcptr(), mode);    }
inline uint64_t     mpreal::toUInt64(mp_rnd_t mode)    const{    return mpfr_get_uj(mpfr_srcptr(), mode);    }
#endif

inline ::mpfr_ptr     mpreal::mpfr_ptr()             { return mp; }
inline ::mpfr_srcptr  mpreal::mpfr_ptr()    const    { return mp; }
inline ::mpfr_srcptr  mpreal::mpfr_srcptr() const    { return mp; }

template <class T>
inline std::string toString(T t, std::ios_base & (*f)(std::ios_base&))
{
    std::ostringstream oss;
    oss << f << t;
    return oss.str();
}

#if (MPFR_VERSION >= MPFR_VERSION_NUM(2,4,0))

inline std::string mpreal::toString(const std::string& format) const
{
    char *s = NULL;
    std::string out;

    if( !format.empty() )
    {
        if(!(mpfr_asprintf(&s, format.c_str(), mpfr_srcptr()) < 0))
        {
            out = std::string(s);

            mpfr_free_str(s);
        }
    }

    return out;
}

#endif

inline std::string mpreal::toString(int n, int b, mp_rnd_t mode) const
{
    // TODO: Add extended format specification (f, e, rounding mode) as it done in output operator
    (void)b;
    (void)mode;

#if (MPFR_VERSION >= MPFR_VERSION_NUM(2,4,0))

    std::ostringstream format;

    int digits = (n >= 0) ? n : bits2digits(mpfr_get_prec(mpfr_srcptr()));
    
    format << "%." << digits << "RNg";

    return toString(format.str());

#else

    char *s, *ns = NULL; 
    size_t slen, nslen;
    mp_exp_t exp;
    std::string out;

    if(mpfr_inf_p(mp))
    { 
        if(mpfr_sgn(mp)>0) return "+Inf";
        else               return "-Inf";
    }

    if(mpfr_zero_p(mp)) return "0";
    if(mpfr_nan_p(mp))  return "NaN";

    s  = mpfr_get_str(NULL, &exp, b, 0, mp, mode);
    ns = mpfr_get_str(NULL, &exp, b, (std::max)(0,n), mp, mode);

    if(s!=NULL && ns!=NULL)
    {
        slen  = strlen(s);
        nslen = strlen(ns);
        if(nslen<=slen) 
        {
            mpfr_free_str(s);
            s = ns;
            slen = nslen;
        }
        else {
            mpfr_free_str(ns);
        }

        // Make human eye-friendly formatting if possible
        if (exp>0 && static_cast<size_t>(exp)<slen)
        {
            if(s[0]=='-')
            {
                // Remove zeros starting from right end
                char* ptr = s+slen-1;
                while (*ptr=='0' && ptr>s+exp) ptr--; 

                if(ptr==s+exp) out = std::string(s,exp+1);
                else           out = std::string(s,exp+1)+'.'+std::string(s+exp+1,ptr-(s+exp+1)+1);

                //out = string(s,exp+1)+'.'+string(s+exp+1);
            }
            else
            {
                // Remove zeros starting from right end
                char* ptr = s+slen-1;
                while (*ptr=='0' && ptr>s+exp-1) ptr--; 

                if(ptr==s+exp-1) out = std::string(s,exp);
                else             out = std::string(s,exp)+'.'+std::string(s+exp,ptr-(s+exp)+1);

                //out = string(s,exp)+'.'+string(s+exp);
            }

        }else{ // exp<0 || exp>slen
            if(s[0]=='-')
            {
                // Remove zeros starting from right end
                char* ptr = s+slen-1;
                while (*ptr=='0' && ptr>s+1) ptr--; 

                if(ptr==s+1) out = std::string(s,2);
                else         out = std::string(s,2)+'.'+std::string(s+2,ptr-(s+2)+1);

                //out = string(s,2)+'.'+string(s+2);
            }
            else
            {
                // Remove zeros starting from right end
                char* ptr = s+slen-1;
                while (*ptr=='0' && ptr>s) ptr--; 

                if(ptr==s) out = std::string(s,1);
                else       out = std::string(s,1)+'.'+std::string(s+1,ptr-(s+1)+1);

                //out = string(s,1)+'.'+string(s+1);
            }

            // Make final string
            if(--exp)
            {
                if(exp>0) out += "e+"+mpfr::toString<mp_exp_t>(exp,std::dec);
                else       out += "e"+mpfr::toString<mp_exp_t>(exp,std::dec);
            }
        }

        mpfr_free_str(s);
        return out;
    }else{
        return "conversion error!";
    }
#endif
}


//////////////////////////////////////////////////////////////////////////
// I/O
inline std::ostream& mpreal::output(std::ostream& os) const 
{
    std::ostringstream format;
    const std::ios::fmtflags flags = os.flags();

    format << ((flags & std::ios::showpos) ? "%+" : "%");
    if (os.precision() >= 0)
        format << '.' << os.precision() << "R*"
               << ((flags & std::ios::floatfield) == std::ios::fixed ? 'f' :
                   (flags & std::ios::floatfield) == std::ios::scientific ? 'e' :
                   'g');
    else
        format << "R*e";

    char *s = NULL;
    if(!(mpfr_asprintf(&s, format.str().c_str(),
                        mpfr::mpreal::get_default_rnd(),
                        mpfr_srcptr())
        < 0))
    {
        os << std::string(s);
        mpfr_free_str(s);
    }
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const mpreal& v)
{
    return v.output(os);
}

inline std::istream& operator>>(std::istream &is, mpreal& v)
{
    // TODO: use cout::hexfloat and other flags to setup base
    std::string tmp;
    is >> tmp;
    mpfr_set_str(v.mpfr_ptr(), tmp.c_str(), 10, mpreal::get_default_rnd());
    return is;
}

//////////////////////////////////////////////////////////////////////////
//     Bits - decimal digits relation
//        bits   = ceil(digits*log[2](10))
//        digits = floor(bits*log[10](2))

inline mp_prec_t digits2bits(int d)
{
    const double LOG2_10 = 3.3219280948873624;

    return mp_prec_t(std::ceil( d * LOG2_10 ));
}

inline int bits2digits(mp_prec_t b)
{
    const double LOG10_2 = 0.30102999566398119;

    return int(std::floor( b * LOG10_2 ));
}

//////////////////////////////////////////////////////////////////////////
// Set/Get number properties
inline int sgn(const mpreal& op)
{
    int r = mpfr_signbit(op.mpfr_srcptr());
    return (r > 0? -1 : 1);
}

inline mpreal& mpreal::setSign(int sign, mp_rnd_t RoundingMode)
{
    mpfr_setsign(mpfr_ptr(), mpfr_srcptr(), (sign < 0 ? 1 : 0), RoundingMode);
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline int mpreal::getPrecision() const
{
    return int(mpfr_get_prec(mpfr_srcptr()));
}

inline mpreal& mpreal::setPrecision(int Precision, mp_rnd_t RoundingMode)
{
    mpfr_prec_round(mpfr_ptr(), Precision, RoundingMode);
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal& mpreal::setInf(int sign) 
{ 
    mpfr_set_inf(mpfr_ptr(), sign);
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}    

inline mpreal& mpreal::setNan() 
{
    mpfr_set_nan(mpfr_ptr());
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mpreal&    mpreal::setZero(int sign)
{

#if (MPFR_VERSION >= MPFR_VERSION_NUM(3,0,0))
    mpfr_set_zero(mpfr_ptr(), sign);
#else
    mpfr_set_si(mpfr_ptr(), 0, (mpfr_get_default_rounding_mode)());
    setSign(sign);
#endif 

    MPREAL_MSVC_DEBUGVIEW_CODE;
    return *this;
}

inline mp_prec_t mpreal::get_prec() const
{
    return mpfr_get_prec(mpfr_srcptr());
}

inline void mpreal::set_prec(mp_prec_t prec, mp_rnd_t rnd_mode)
{
    mpfr_prec_round(mpfr_ptr(),prec,rnd_mode);
    MPREAL_MSVC_DEBUGVIEW_CODE;
}

inline mp_exp_t mpreal::get_exp ()
{
    return mpfr_get_exp(mpfr_srcptr());
}

inline int mpreal::set_exp (mp_exp_t e)
{
    int x = mpfr_set_exp(mpfr_ptr(), e);
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return x;
}

inline const mpreal frexp(const mpreal& v, mp_exp_t* exp)
{
    mpreal x(v);
    *exp = x.get_exp();
    x.set_exp(0);
    return x;
}

inline const mpreal ldexp(const mpreal& v, mp_exp_t exp)
{
    mpreal x(v);

    // rounding is not important since we just increasing the exponent
    mpfr_mul_2si(x.mpfr_ptr(), x.mpfr_srcptr(), exp, mpreal::get_default_rnd()); 
    return x;
}

inline mpreal machine_epsilon(mp_prec_t prec)
{
    /* the smallest eps such that 1 + eps != 1 */
    return machine_epsilon(mpreal(1, prec));
}

inline mpreal machine_epsilon(const mpreal& x)
{    
    /* the smallest eps such that x + eps != x */
    if( x < 0)
    {
        return nextabove(-x) + x;
    }else{
        return nextabove( x) - x;
    }
}

// minval is 'safe' meaning 1 / minval does not overflow
inline mpreal minval(mp_prec_t prec)
{
    /* min = 1/2 * 2^emin = 2^(emin - 1) */
    return mpreal(1, prec) << mpreal::get_emin()-1;
}

// maxval is 'safe' meaning 1 / maxval does not underflow
inline mpreal maxval(mp_prec_t prec)
{
    /* max = (1 - eps) * 2^emax, eps is machine epsilon */
    return (mpreal(1, prec) - machine_epsilon(prec)) << mpreal::get_emax(); 
}

inline bool isEqualUlps(const mpreal& a, const mpreal& b, int maxUlps)
{
    return abs(a - b) <= machine_epsilon((max)(abs(a), abs(b))) * maxUlps;
}

inline bool isEqualFuzzy(const mpreal& a, const mpreal& b, const mpreal& eps)
{
    return abs(a - b) <= eps;
}

inline bool isEqualFuzzy(const mpreal& a, const mpreal& b)
{
    return isEqualFuzzy(a, b, machine_epsilon((max)(1, (min)(abs(a), abs(b)))));
}

inline const mpreal modf(const mpreal& v, mpreal& n)
{
    mpreal f(v);

    // rounding is not important since we are using the same number
    mpfr_frac (f.mpfr_ptr(),f.mpfr_srcptr(),mpreal::get_default_rnd());    
    mpfr_trunc(n.mpfr_ptr(),v.mpfr_srcptr());
    return f;
}

inline int mpreal::check_range (int t, mp_rnd_t rnd_mode)
{
    return mpfr_check_range(mpfr_ptr(),t,rnd_mode);
}

inline int mpreal::subnormalize (int t,mp_rnd_t rnd_mode)
{
    int r = mpfr_subnormalize(mpfr_ptr(),t,rnd_mode);
    MPREAL_MSVC_DEBUGVIEW_CODE;
    return r;
}

inline mp_exp_t mpreal::get_emin (void)
{
    return mpfr_get_emin();
}

inline int mpreal::set_emin (mp_exp_t exp)
{
    return mpfr_set_emin(exp);
}

inline mp_exp_t mpreal::get_emax (void)
{
    return mpfr_get_emax();
}

inline int mpreal::set_emax (mp_exp_t exp)
{
    return mpfr_set_emax(exp);
}

inline mp_exp_t mpreal::get_emin_min (void)
{
    return mpfr_get_emin_min();
}

inline mp_exp_t mpreal::get_emin_max (void)
{
    return mpfr_get_emin_max();
}

inline mp_exp_t mpreal::get_emax_min (void)
{
    return mpfr_get_emax_min();
}

inline mp_exp_t mpreal::get_emax_max (void)
{
    return mpfr_get_emax_max();
}

//////////////////////////////////////////////////////////////////////////
// Mathematical Functions
//////////////////////////////////////////////////////////////////////////
#define MPREAL_UNARY_MATH_FUNCTION_BODY(f)                    \
        mpreal y(0, mpfr_get_prec(x.mpfr_srcptr()));          \
        mpfr_##f(y.mpfr_ptr(), x.mpfr_srcptr(), r);           \
        return y; 

inline const mpreal sqr  (const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd())
{   MPREAL_UNARY_MATH_FUNCTION_BODY(sqr );    }

inline const mpreal sqrt (const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd())
{   MPREAL_UNARY_MATH_FUNCTION_BODY(sqrt);    }

inline const mpreal sqrt(const unsigned long int x, mp_rnd_t r)
{
    mpreal y;
    mpfr_sqrt_ui(y.mpfr_ptr(), x, r);
    return y;
}

inline const mpreal sqrt(const unsigned int v, mp_rnd_t rnd_mode)
{
    return sqrt(static_cast<unsigned long int>(v),rnd_mode);
}

inline const mpreal sqrt(const long int v, mp_rnd_t rnd_mode)
{
    if (v>=0)   return sqrt(static_cast<unsigned long int>(v),rnd_mode);
    else        return mpreal().setNan(); // NaN  
}

inline const mpreal sqrt(const int v, mp_rnd_t rnd_mode)
{
    if (v>=0)   return sqrt(static_cast<unsigned long int>(v),rnd_mode);
    else        return mpreal().setNan(); // NaN
}

inline const mpreal root(const mpreal& x, unsigned long int k, mp_rnd_t r = mpreal::get_default_rnd())
{
    mpreal y(0, mpfr_get_prec(x.mpfr_srcptr())); 
    mpfr_root(y.mpfr_ptr(), x.mpfr_srcptr(), k, r);  
    return y; 
}

inline const mpreal dim(const mpreal& a, const mpreal& b, mp_rnd_t r = mpreal::get_default_rnd())
{
    mpreal y(0, mpfr_get_prec(a.mpfr_srcptr()));
    mpfr_dim(y.mpfr_ptr(), a.mpfr_srcptr(), b.mpfr_srcptr(), r);
    return y;
}

inline int cmpabs(const mpreal& a,const mpreal& b)
{
    return mpfr_cmpabs(a.mpfr_ptr(), b.mpfr_srcptr());
}

inline int sin_cos(mpreal& s, mpreal& c, const mpreal& v, mp_rnd_t rnd_mode = mpreal::get_default_rnd())
{
    return mpfr_sin_cos(s.mpfr_ptr(), c.mpfr_ptr(), v.mpfr_srcptr(), rnd_mode);
}

inline const mpreal sqrt  (const long double v, mp_rnd_t rnd_mode)    {   return sqrt(mpreal(v),rnd_mode);    }
inline const mpreal sqrt  (const double v, mp_rnd_t rnd_mode)         {   return sqrt(mpreal(v),rnd_mode);    }

inline const mpreal cbrt  (const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPREAL_UNARY_MATH_FUNCTION_BODY(cbrt );    }
inline const mpreal fabs  (const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPREAL_UNARY_MATH_FUNCTION_BODY(abs  );    }
inline const mpreal abs   (const mpreal& x, mp_rnd_t r)                             {   MPREAL_UNARY_MATH_FUNCTION_BODY(abs  );    }
inline const mpreal log   (const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPREAL_UNARY_MATH_FUNCTION_BODY(log  );    }
inline const mpreal log2  (const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPREAL_UNARY_MATH_FUNCTION_BODY(log2 );    }
inline const mpreal log10 (const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPREAL_UNARY_MATH_FUNCTION_BODY(log10);    }
inline const mpreal exp   (const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPREAL_UNARY_MATH_FUNCTION_BODY(exp  );    }
inline const mpreal exp2  (const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPREAL_UNARY_MATH_FUNCTION_BODY(exp2 );    }
inline const mpreal exp10 (const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPREAL_UNARY_MATH_FUNCTION_BODY(exp10);    }
inline const mpreal cos   (const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPREAL_UNARY_MATH_FUNCTION_BODY(cos  );    }
inline const mpreal sin   (const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPREAL_UNARY_MATH_FUNCTION_BODY(sin  );    }
inline const mpreal tan   (const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPREAL_UNARY_MATH_FUNCTION_BODY(tan  );    }
inline const mpreal sec   (const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPREAL_UNARY_MATH_FUNCTION_BODY(sec  );    }
inline const mpreal csc   (const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPREAL_UNARY_MATH_FUNCTION_BODY(csc  );    }
inline const mpreal cot   (const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPREAL_UNARY_MATH_FUNCTION_BODY(cot  );    }
inline const mpreal acos  (const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPREAL_UNARY_MATH_FUNCTION_BODY(acos );    }
inline const mpreal asin  (const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPREAL_UNARY_MATH_FUNCTION_BODY(asin );    }
inline const mpreal atan  (const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPREAL_UNARY_MATH_FUNCTION_BODY(atan );    }

inline const mpreal acot  (const mpreal& v, mp_rnd_t r = mpreal::get_default_rnd()) {   return atan (1/v, r);                      }
inline const mpreal asec  (const mpreal& v, mp_rnd_t r = mpreal::get_default_rnd()) {   return acos (1/v, r);                      }
inline const mpreal acsc  (const mpreal& v, mp_rnd_t r = mpreal::get_default_rnd()) {   return asin (1/v, r);                      }
inline const mpreal acoth (const mpreal& v, mp_rnd_t r = mpreal::get_default_rnd()) {   return atanh(1/v, r);                      }
inline const mpreal asech (const mpreal& v, mp_rnd_t r = mpreal::get_default_rnd()) {   return acosh(1/v, r);                      }
inline const mpreal acsch (const mpreal& v, mp_rnd_t r = mpreal::get_default_rnd()) {   return asinh(1/v, r);                      }

inline const mpreal cosh  (const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPREAL_UNARY_MATH_FUNCTION_BODY(cosh );    }
inline const mpreal sinh  (const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPREAL_UNARY_MATH_FUNCTION_BODY(sinh );    }
inline const mpreal tanh  (const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPREAL_UNARY_MATH_FUNCTION_BODY(tanh );    }
inline const mpreal sech  (const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPREAL_UNARY_MATH_FUNCTION_BODY(sech );    }
inline const mpreal csch  (const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPREAL_UNARY_MATH_FUNCTION_BODY(csch );    }
inline const mpreal coth  (const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPREAL_UNARY_MATH_FUNCTION_BODY(coth );    }
inline const mpreal acosh (const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPREAL_UNARY_MATH_FUNCTION_BODY(acosh);    }
inline const mpreal asinh (const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPREAL_UNARY_MATH_FUNCTION_BODY(asinh);    }
inline const mpreal atanh (const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPREAL_UNARY_MATH_FUNCTION_BODY(atanh);    }

inline const mpreal log1p   (const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPREAL_UNARY_MATH_FUNCTION_BODY(log1p  );    }
inline const mpreal expm1   (const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPREAL_UNARY_MATH_FUNCTION_BODY(expm1  );    }
inline const mpreal eint    (const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPREAL_UNARY_MATH_FUNCTION_BODY(eint   );    }
inline const mpreal gamma   (const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPREAL_UNARY_MATH_FUNCTION_BODY(gamma  );    }
inline const mpreal lngamma (const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPREAL_UNARY_MATH_FUNCTION_BODY(lngamma);    }
inline const mpreal zeta    (const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPREAL_UNARY_MATH_FUNCTION_BODY(zeta   );    }
inline const mpreal erf     (const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPREAL_UNARY_MATH_FUNCTION_BODY(erf    );    }
inline const mpreal erfc    (const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPREAL_UNARY_MATH_FUNCTION_BODY(erfc   );    }
inline const mpreal besselj0(const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPREAL_UNARY_MATH_FUNCTION_BODY(j0     );    }
inline const mpreal besselj1(const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPREAL_UNARY_MATH_FUNCTION_BODY(j1     );    }
inline const mpreal bessely0(const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPREAL_UNARY_MATH_FUNCTION_BODY(y0     );    }
inline const mpreal bessely1(const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPREAL_UNARY_MATH_FUNCTION_BODY(y1     );    }

inline const mpreal atan2 (const mpreal& y, const mpreal& x, mp_rnd_t rnd_mode = mpreal::get_default_rnd())
{
    mpreal a(0,(std::max)(y.getPrecision(), x.getPrecision()));
    mpfr_atan2(a.mpfr_ptr(), y.mpfr_srcptr(), x.mpfr_srcptr(), rnd_mode);
    return a;
}

inline const mpreal hypot (const mpreal& x, const mpreal& y, mp_rnd_t rnd_mode = mpreal::get_default_rnd())
{
    mpreal a(0,(std::max)(y.getPrecision(), x.getPrecision()));
    mpfr_hypot(a.mpfr_ptr(), x.mpfr_srcptr(), y.mpfr_srcptr(), rnd_mode);
    return a;
}

inline const mpreal remainder (const mpreal& x, const mpreal& y, mp_rnd_t rnd_mode = mpreal::get_default_rnd())
{    
    mpreal a(0,(std::max)(y.getPrecision(), x.getPrecision()));
    mpfr_remainder(a.mpfr_ptr(), x.mpfr_srcptr(), y.mpfr_srcptr(), rnd_mode);
    return a;
}

inline const mpreal remquo (long* q, const mpreal& x, const mpreal& y, mp_rnd_t rnd_mode = mpreal::get_default_rnd())
{
    mpreal a(0,(std::max)(y.getPrecision(), x.getPrecision()));
    mpfr_remquo(a.mpfr_ptr(),q, x.mpfr_srcptr(), y.mpfr_srcptr(), rnd_mode);
    return a;
}

inline const mpreal fac_ui (unsigned long int v, mp_prec_t prec     = mpreal::get_default_prec(),
                                           mp_rnd_t  rnd_mode = mpreal::get_default_rnd())
{
    mpreal x(0, prec);
    mpfr_fac_ui(x.mpfr_ptr(),v,rnd_mode);
    return x;
}


inline const mpreal lgamma (const mpreal& v, int *signp = 0, mp_rnd_t rnd_mode = mpreal::get_default_rnd())
{
    mpreal x(v);
    int tsignp;

    if(signp)   mpfr_lgamma(x.mpfr_ptr(),  signp,v.mpfr_srcptr(),rnd_mode);
    else        mpfr_lgamma(x.mpfr_ptr(),&tsignp,v.mpfr_srcptr(),rnd_mode);

    return x;
}


inline const mpreal besseljn (long n, const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd())
{
    mpreal  y(0, x.getPrecision());
    mpfr_jn(y.mpfr_ptr(), n, x.mpfr_srcptr(), r);
    return y;
}

inline const mpreal besselyn (long n, const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd())
{
    mpreal  y(0, x.getPrecision());
    mpfr_yn(y.mpfr_ptr(), n, x.mpfr_srcptr(), r);
    return y;
}

inline const mpreal fma (const mpreal& v1, const mpreal& v2, const mpreal& v3, mp_rnd_t rnd_mode = mpreal::get_default_rnd())
{
    mpreal a;
    mp_prec_t p1, p2, p3;

    p1 = v1.get_prec(); 
    p2 = v2.get_prec(); 
    p3 = v3.get_prec(); 

    a.set_prec(p3>p2?(p3>p1?p3:p1):(p2>p1?p2:p1));

    mpfr_fma(a.mp,v1.mp,v2.mp,v3.mp,rnd_mode);
    return a;
}

inline const mpreal fms (const mpreal& v1, const mpreal& v2, const mpreal& v3, mp_rnd_t rnd_mode = mpreal::get_default_rnd())
{
    mpreal a;
    mp_prec_t p1, p2, p3;

    p1 = v1.get_prec(); 
    p2 = v2.get_prec(); 
    p3 = v3.get_prec(); 

    a.set_prec(p3>p2?(p3>p1?p3:p1):(p2>p1?p2:p1));

    mpfr_fms(a.mp,v1.mp,v2.mp,v3.mp,rnd_mode);
    return a;
}

inline const mpreal agm (const mpreal& v1, const mpreal& v2, mp_rnd_t rnd_mode = mpreal::get_default_rnd())
{
    mpreal a;
    mp_prec_t p1, p2;

    p1 = v1.get_prec(); 
    p2 = v2.get_prec(); 

    a.set_prec(p1>p2?p1:p2);

    mpfr_agm(a.mp, v1.mp, v2.mp, rnd_mode);

    return a;
}

inline const mpreal sum (const mpreal tab[], unsigned long int n, mp_rnd_t rnd_mode = mpreal::get_default_rnd())
{
    mpreal x;
    mpfr_ptr* t;
    unsigned long int i;

    t = new mpfr_ptr[n];
    for (i=0;i<n;i++) t[i] = (mpfr_ptr)tab[i].mp;
    mpfr_sum(x.mp,t,n,rnd_mode);
    delete[] t;
    return x;
}

//////////////////////////////////////////////////////////////////////////
// MPFR 2.4.0 Specifics
#if (MPFR_VERSION >= MPFR_VERSION_NUM(2,4,0))

inline int sinh_cosh(mpreal& s, mpreal& c, const mpreal& v, mp_rnd_t rnd_mode = mpreal::get_default_rnd())
{
    return mpfr_sinh_cosh(s.mp,c.mp,v.mp,rnd_mode);
}

inline const mpreal li2 (const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) 
{   
    MPREAL_UNARY_MATH_FUNCTION_BODY(li2);    
}

inline const mpreal rem (const mpreal& x, const mpreal& y, mp_rnd_t rnd_mode = mpreal::get_default_rnd())
{
    /*  R = rem(X,Y) if Y != 0, returns X - n * Y where n = trunc(X/Y). */
    return fmod(x, y, rnd_mode);
}

inline const mpreal mod (const mpreal& x, const mpreal& y, mp_rnd_t rnd_mode = mpreal::get_default_rnd())
{
    (void)rnd_mode;
    
    /*  

    m = mod(x,y) if y != 0, returns x - n*y where n = floor(x/y)

    The following are true by convention:
    - mod(x,0) is x
    - mod(x,x) is 0
    - mod(x,y) for x != y and y != 0 has the same sign as y.    
    
    */

    if(iszero(y)) return x;
    if(x == y) return 0;

    mpreal m = x - floor(x / y) * y;
    
    m.setSign(sgn(y)); // make sure result has the same sign as Y

    return m;
}

inline const mpreal fmod (const mpreal& x, const mpreal& y, mp_rnd_t rnd_mode = mpreal::get_default_rnd())
{
    mpreal a;
    mp_prec_t yp, xp;

    yp = y.get_prec(); 
    xp = x.get_prec(); 

    a.set_prec(yp>xp?yp:xp);

    mpfr_fmod(a.mp, x.mp, y.mp, rnd_mode);

    return a;
}

inline const mpreal rec_sqrt(const mpreal& v, mp_rnd_t rnd_mode = mpreal::get_default_rnd())
{
    mpreal x(v);
    mpfr_rec_sqrt(x.mp,v.mp,rnd_mode);
    return x;
}
#endif //  MPFR 2.4.0 Specifics

//////////////////////////////////////////////////////////////////////////
// MPFR 3.0.0 Specifics
#if (MPFR_VERSION >= MPFR_VERSION_NUM(3,0,0))
inline const mpreal digamma (const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPREAL_UNARY_MATH_FUNCTION_BODY(digamma);     }
inline const mpreal ai      (const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPREAL_UNARY_MATH_FUNCTION_BODY(ai);          }
#endif // MPFR 3.0.0 Specifics

//////////////////////////////////////////////////////////////////////////
// Constants
inline const mpreal const_log2 (mp_prec_t p = mpreal::get_default_prec(), mp_rnd_t r = mpreal::get_default_rnd())
{
    mpreal x(0, p);
    mpfr_const_log2(x.mpfr_ptr(), r);
    return x;
}

inline const mpreal const_pi (mp_prec_t p = mpreal::get_default_prec(), mp_rnd_t r = mpreal::get_default_rnd())
{
    mpreal x(0, p);
    mpfr_const_pi(x.mpfr_ptr(), r);
    return x;
}

inline const mpreal const_euler (mp_prec_t p = mpreal::get_default_prec(), mp_rnd_t r = mpreal::get_default_rnd())
{
    mpreal x(0, p);
    mpfr_const_euler(x.mpfr_ptr(), r);
    return x;
}

inline const mpreal const_catalan (mp_prec_t p = mpreal::get_default_prec(), mp_rnd_t r = mpreal::get_default_rnd())
{
    mpreal x(0, p);
    mpfr_const_catalan(x.mpfr_ptr(), r);
    return x;
}

inline const mpreal const_infinity (int sign = 1, mp_prec_t p = mpreal::get_default_prec())
{
    mpreal x(0, p);
    mpfr_set_inf(x.mpfr_ptr(), sign);
    return x;
}

//////////////////////////////////////////////////////////////////////////
// Integer Related Functions
inline const mpreal ceil(const mpreal& v)
{
    mpreal x(v);
    mpfr_ceil(x.mp,v.mp);
    return x;
}

inline const mpreal floor(const mpreal& v)
{
    mpreal x(v);
    mpfr_floor(x.mp,v.mp);
    return x;
}

inline const mpreal round(const mpreal& v)
{
    mpreal x(v);
    mpfr_round(x.mp,v.mp);
    return x;
}

inline const mpreal trunc(const mpreal& v)
{
    mpreal x(v);
    mpfr_trunc(x.mp,v.mp);
    return x;
}

inline const mpreal rint       (const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPREAL_UNARY_MATH_FUNCTION_BODY(rint      );     }
inline const mpreal rint_ceil  (const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPREAL_UNARY_MATH_FUNCTION_BODY(rint_ceil );     }
inline const mpreal rint_floor (const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPREAL_UNARY_MATH_FUNCTION_BODY(rint_floor);     }
inline const mpreal rint_round (const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPREAL_UNARY_MATH_FUNCTION_BODY(rint_round);     }
inline const mpreal rint_trunc (const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPREAL_UNARY_MATH_FUNCTION_BODY(rint_trunc);     }
inline const mpreal frac       (const mpreal& x, mp_rnd_t r = mpreal::get_default_rnd()) {   MPREAL_UNARY_MATH_FUNCTION_BODY(frac      );     }

//////////////////////////////////////////////////////////////////////////
// Miscellaneous Functions
inline void         swap (mpreal& a, mpreal& b)            {    mpfr_swap(a.mp,b.mp);   }
inline const mpreal (max)(const mpreal& x, const mpreal& y){    return (x>y?x:y);       }
inline const mpreal (min)(const mpreal& x, const mpreal& y){    return (x<y?x:y);       }

inline const mpreal fmax(const mpreal& x, const mpreal& y, mp_rnd_t rnd_mode = mpreal::get_default_rnd())
{
    mpreal a;
    mpfr_max(a.mp,x.mp,y.mp,rnd_mode);
    return a;
}

inline const mpreal fmin(const mpreal& x, const mpreal& y,  mp_rnd_t rnd_mode = mpreal::get_default_rnd())
{
    mpreal a;
    mpfr_min(a.mp,x.mp,y.mp,rnd_mode);
    return a;
}

inline const mpreal nexttoward (const mpreal& x, const mpreal& y)
{
    mpreal a(x);
    mpfr_nexttoward(a.mp,y.mp);
    return a;
}

inline const mpreal nextabove  (const mpreal& x)
{
    mpreal a(x);
    mpfr_nextabove(a.mp);
    return a;
}

inline const mpreal nextbelow  (const mpreal& x)
{
    mpreal a(x);
    mpfr_nextbelow(a.mp);
    return a;
}

inline const mpreal urandomb (gmp_randstate_t& state)
{
    mpreal x;
    mpfr_urandomb(x.mp,state);
    return x;
}

#if (MPFR_VERSION >= MPFR_VERSION_NUM(3,1,0))
// use gmp_randinit_default() to init state, gmp_randclear() to clear
inline const mpreal urandom (gmp_randstate_t& state, mp_rnd_t rnd_mode = mpreal::get_default_rnd())
{
    mpreal x;
    mpfr_urandom(x.mp,state,rnd_mode);
    return x;
}

inline const mpreal grandom (gmp_randstate_t& state, mp_rnd_t rnd_mode = mpreal::get_default_rnd())
{
    mpreal x;
    mpfr_grandom(x.mp, NULL, state, rnd_mode);
    return x;
}

#endif 

#if (MPFR_VERSION <= MPFR_VERSION_NUM(2,4,2))
inline const mpreal random2 (mp_size_t size, mp_exp_t exp)
{
    mpreal x;
    mpfr_random2(x.mp,size,exp);
    return x;
}
#endif

// Uniformly distributed random number generation
// a = random(seed); <- initialization & first random number generation
// a = random();     <- next random numbers generation
// seed != 0
inline const mpreal random(unsigned int seed = 0)
{

#if (MPFR_VERSION >= MPFR_VERSION_NUM(3,0,0))
    static gmp_randstate_t state;
    static bool isFirstTime = true;

    if(isFirstTime)
    {
        gmp_randinit_default(state);
        gmp_randseed_ui(state,0);
        isFirstTime = false;
    }

    if(seed != 0)    gmp_randseed_ui(state,seed);

    return mpfr::urandom(state);
#else
    if(seed != 0)    std::srand(seed);
    return mpfr::mpreal(std::rand()/(double)RAND_MAX);
#endif

}

#if (MPFR_VERSION >= MPFR_VERSION_NUM(3,0,0))
inline const mpreal grandom(unsigned int seed = 0)
{
    static gmp_randstate_t state;
    static bool isFirstTime = true;

    if(isFirstTime)
    {
        gmp_randinit_default(state);
        gmp_randseed_ui(state,0);
        isFirstTime = false;
    }

    if(seed != 0) gmp_randseed_ui(state,seed);

    return mpfr::grandom(state);
}
#endif

//////////////////////////////////////////////////////////////////////////
// Set/Get global properties
inline void mpreal::set_default_prec(mp_prec_t prec)
{ 
    mpfr_set_default_prec(prec); 
}

inline void mpreal::set_default_rnd(mp_rnd_t rnd_mode)
{ 
    mpfr_set_default_rounding_mode(rnd_mode); 
}

inline bool mpreal::fits_in_bits(double x, int n)
{   
    int i;
    double t;
    return IsInf(x) || (std::modf ( std::ldexp ( std::frexp ( x, &i ), n ), &t ) == 0.0);
}

inline const mpreal pow(const mpreal& a, const mpreal& b, mp_rnd_t rnd_mode = mpreal::get_default_rnd())
{
    mpreal x(a);
    mpfr_pow(x.mp,x.mp,b.mp,rnd_mode);
    return x;
}

inline const mpreal pow(const mpreal& a, const mpz_t b, mp_rnd_t rnd_mode = mpreal::get_default_rnd())
{
    mpreal x(a);
    mpfr_pow_z(x.mp,x.mp,b,rnd_mode);
    return x;
}

inline const mpreal pow(const mpreal& a, const unsigned long int b, mp_rnd_t rnd_mode = mpreal::get_default_rnd())
{
    mpreal x(a);
    mpfr_pow_ui(x.mp,x.mp,b,rnd_mode);
    return x;
}

inline const mpreal pow(const mpreal& a, const unsigned int b, mp_rnd_t rnd_mode)
{
    return pow(a,static_cast<unsigned long int>(b),rnd_mode);
}

inline const mpreal pow(const mpreal& a, const long int b, mp_rnd_t rnd_mode = mpreal::get_default_rnd())
{
    mpreal x(a);
    mpfr_pow_si(x.mp,x.mp,b,rnd_mode);
    return x;
}

inline const mpreal pow(const mpreal& a, const int b, mp_rnd_t rnd_mode)
{
    return pow(a,static_cast<long int>(b),rnd_mode);
}

inline const mpreal pow(const mpreal& a, const long double b, mp_rnd_t rnd_mode)
{
    return pow(a,mpreal(b),rnd_mode);
}

inline const mpreal pow(const mpreal& a, const double b, mp_rnd_t rnd_mode)
{
    return pow(a,mpreal(b),rnd_mode);
}

inline const mpreal pow(const unsigned long int a, const mpreal& b, mp_rnd_t rnd_mode = mpreal::get_default_rnd())
{
    mpreal x(a);
    mpfr_ui_pow(x.mp,a,b.mp,rnd_mode);
    return x;
}

inline const mpreal pow(const unsigned int a, const mpreal& b, mp_rnd_t rnd_mode)
{
    return pow(static_cast<unsigned long int>(a),b,rnd_mode);
}

inline const mpreal pow(const long int a, const mpreal& b, mp_rnd_t rnd_mode)
{
    if (a>=0)     return pow(static_cast<unsigned long int>(a),b,rnd_mode);
    else          return pow(mpreal(a),b,rnd_mode);
}

inline const mpreal pow(const int a, const mpreal& b, mp_rnd_t rnd_mode)
{
    if (a>=0)     return pow(static_cast<unsigned long int>(a),b,rnd_mode);
    else          return pow(mpreal(a),b,rnd_mode);
}

inline const mpreal pow(const long double a, const mpreal& b, mp_rnd_t rnd_mode)
{
    return pow(mpreal(a),b,rnd_mode);
}

inline const mpreal pow(const double a, const mpreal& b, mp_rnd_t rnd_mode)
{
    return pow(mpreal(a),b,rnd_mode);
}

// pow unsigned long int
inline const mpreal pow(const unsigned long int a, const unsigned long int b, mp_rnd_t rnd_mode)
{
    mpreal x(a);
    mpfr_ui_pow_ui(x.mp,a,b,rnd_mode);
    return x;
}

inline const mpreal pow(const unsigned long int a, const unsigned int b, mp_rnd_t rnd_mode)
{
    return pow(a,static_cast<unsigned long int>(b),rnd_mode); //mpfr_ui_pow_ui
}

inline const mpreal pow(const unsigned long int a, const long int b, mp_rnd_t rnd_mode)
{
    if(b>0)    return pow(a,static_cast<unsigned long int>(b),rnd_mode); //mpfr_ui_pow_ui
    else       return pow(a,mpreal(b),rnd_mode); //mpfr_ui_pow
}

inline const mpreal pow(const unsigned long int a, const int b, mp_rnd_t rnd_mode)
{
    if(b>0)    return pow(a,static_cast<unsigned long int>(b),rnd_mode); //mpfr_ui_pow_ui
    else       return pow(a,mpreal(b),rnd_mode); //mpfr_ui_pow
}

inline const mpreal pow(const unsigned long int a, const long double b, mp_rnd_t rnd_mode)
{
    return pow(a,mpreal(b),rnd_mode); //mpfr_ui_pow
}

inline const mpreal pow(const unsigned long int a, const double b, mp_rnd_t rnd_mode)
{
    return pow(a,mpreal(b),rnd_mode); //mpfr_ui_pow
}

// pow unsigned int
inline const mpreal pow(const unsigned int a, const unsigned long int b, mp_rnd_t rnd_mode)
{
    return pow(static_cast<unsigned long int>(a),b,rnd_mode); //mpfr_ui_pow_ui
}

inline const mpreal pow(const unsigned int a, const unsigned int b, mp_rnd_t rnd_mode)
{
    return pow(static_cast<unsigned long int>(a),static_cast<unsigned long int>(b),rnd_mode); //mpfr_ui_pow_ui
}

inline const mpreal pow(const unsigned int a, const long int b, mp_rnd_t rnd_mode)
{
    if(b>0) return pow(static_cast<unsigned long int>(a),static_cast<unsigned long int>(b),rnd_mode); //mpfr_ui_pow_ui
    else    return pow(static_cast<unsigned long int>(a),mpreal(b),rnd_mode); //mpfr_ui_pow
}

inline const mpreal pow(const unsigned int a, const int b, mp_rnd_t rnd_mode)
{
    if(b>0) return pow(static_cast<unsigned long int>(a),static_cast<unsigned long int>(b),rnd_mode); //mpfr_ui_pow_ui
    else    return pow(static_cast<unsigned long int>(a),mpreal(b),rnd_mode); //mpfr_ui_pow
}

inline const mpreal pow(const unsigned int a, const long double b, mp_rnd_t rnd_mode)
{
    return pow(static_cast<unsigned long int>(a),mpreal(b),rnd_mode); //mpfr_ui_pow
}

inline const mpreal pow(const unsigned int a, const double b, mp_rnd_t rnd_mode)
{
    return pow(static_cast<unsigned long int>(a),mpreal(b),rnd_mode); //mpfr_ui_pow
}

// pow long int
inline const mpreal pow(const long int a, const unsigned long int b, mp_rnd_t rnd_mode)
{
    if (a>0) return pow(static_cast<unsigned long int>(a),b,rnd_mode); //mpfr_ui_pow_ui
    else     return pow(mpreal(a),b,rnd_mode); //mpfr_pow_ui
}

inline const mpreal pow(const long int a, const unsigned int b, mp_rnd_t rnd_mode)
{
    if (a>0) return pow(static_cast<unsigned long int>(a),static_cast<unsigned long int>(b),rnd_mode);  //mpfr_ui_pow_ui
    else     return pow(mpreal(a),static_cast<unsigned long int>(b),rnd_mode); //mpfr_pow_ui
}

inline const mpreal pow(const long int a, const long int b, mp_rnd_t rnd_mode)
{
    if (a>0)
    {
        if(b>0) return pow(static_cast<unsigned long int>(a),static_cast<unsigned long int>(b),rnd_mode); //mpfr_ui_pow_ui
        else    return pow(static_cast<unsigned long int>(a),mpreal(b),rnd_mode); //mpfr_ui_pow
    }else{
        return pow(mpreal(a),b,rnd_mode); // mpfr_pow_si
    }
}

inline const mpreal pow(const long int a, const int b, mp_rnd_t rnd_mode)
{
    if (a>0)
    {
        if(b>0) return pow(static_cast<unsigned long int>(a),static_cast<unsigned long int>(b),rnd_mode); //mpfr_ui_pow_ui
        else    return pow(static_cast<unsigned long int>(a),mpreal(b),rnd_mode); //mpfr_ui_pow
    }else{
        return pow(mpreal(a),static_cast<long int>(b),rnd_mode); // mpfr_pow_si
    }
}

inline const mpreal pow(const long int a, const long double b, mp_rnd_t rnd_mode)
{
    if (a>=0)   return pow(static_cast<unsigned long int>(a),mpreal(b),rnd_mode); //mpfr_ui_pow
    else        return pow(mpreal(a),mpreal(b),rnd_mode); //mpfr_pow
}

inline const mpreal pow(const long int a, const double b, mp_rnd_t rnd_mode)
{
    if (a>=0)   return pow(static_cast<unsigned long int>(a),mpreal(b),rnd_mode); //mpfr_ui_pow
    else        return pow(mpreal(a),mpreal(b),rnd_mode); //mpfr_pow
}

// pow int
inline const mpreal pow(const int a, const unsigned long int b, mp_rnd_t rnd_mode)
{
    if (a>0) return pow(static_cast<unsigned long int>(a),b,rnd_mode); //mpfr_ui_pow_ui
    else     return pow(mpreal(a),b,rnd_mode); //mpfr_pow_ui
}

inline const mpreal pow(const int a, const unsigned int b, mp_rnd_t rnd_mode)
{
    if (a>0) return pow(static_cast<unsigned long int>(a),static_cast<unsigned long int>(b),rnd_mode);  //mpfr_ui_pow_ui
    else     return pow(mpreal(a),static_cast<unsigned long int>(b),rnd_mode); //mpfr_pow_ui
}

inline const mpreal pow(const int a, const long int b, mp_rnd_t rnd_mode)
{
    if (a>0)
    {
        if(b>0) return pow(static_cast<unsigned long int>(a),static_cast<unsigned long int>(b),rnd_mode); //mpfr_ui_pow_ui
        else    return pow(static_cast<unsigned long int>(a),mpreal(b),rnd_mode); //mpfr_ui_pow
    }else{
        return pow(mpreal(a),b,rnd_mode); // mpfr_pow_si
    }
}

inline const mpreal pow(const int a, const int b, mp_rnd_t rnd_mode)
{
    if (a>0)
    {
        if(b>0) return pow(static_cast<unsigned long int>(a),static_cast<unsigned long int>(b),rnd_mode); //mpfr_ui_pow_ui
        else    return pow(static_cast<unsigned long int>(a),mpreal(b),rnd_mode); //mpfr_ui_pow
    }else{
        return pow(mpreal(a),static_cast<long int>(b),rnd_mode); // mpfr_pow_si
    }
}

inline const mpreal pow(const int a, const long double b, mp_rnd_t rnd_mode)
{
    if (a>=0)   return pow(static_cast<unsigned long int>(a),mpreal(b),rnd_mode); //mpfr_ui_pow
    else        return pow(mpreal(a),mpreal(b),rnd_mode); //mpfr_pow
}

inline const mpreal pow(const int a, const double b, mp_rnd_t rnd_mode)
{
    if (a>=0)   return pow(static_cast<unsigned long int>(a),mpreal(b),rnd_mode); //mpfr_ui_pow
    else        return pow(mpreal(a),mpreal(b),rnd_mode); //mpfr_pow
}

// pow long double 
inline const mpreal pow(const long double a, const long double b, mp_rnd_t rnd_mode)
{
    return pow(mpreal(a),mpreal(b),rnd_mode);
}

inline const mpreal pow(const long double a, const unsigned long int b, mp_rnd_t rnd_mode)
{
    return pow(mpreal(a),b,rnd_mode); //mpfr_pow_ui
}

inline const mpreal pow(const long double a, const unsigned int b, mp_rnd_t rnd_mode)
{
    return pow(mpreal(a),static_cast<unsigned long int>(b),rnd_mode); //mpfr_pow_ui
}

inline const mpreal pow(const long double a, const long int b, mp_rnd_t rnd_mode)
{
    return pow(mpreal(a),b,rnd_mode); // mpfr_pow_si
}

inline const mpreal pow(const long double a, const int b, mp_rnd_t rnd_mode)
{
    return pow(mpreal(a),static_cast<long int>(b),rnd_mode); // mpfr_pow_si
}

inline const mpreal pow(const double a, const double b, mp_rnd_t rnd_mode)
{
    return pow(mpreal(a),mpreal(b),rnd_mode);
}

inline const mpreal pow(const double a, const unsigned long int b, mp_rnd_t rnd_mode)
{
    return pow(mpreal(a),b,rnd_mode); // mpfr_pow_ui
}

inline const mpreal pow(const double a, const unsigned int b, mp_rnd_t rnd_mode)
{
    return pow(mpreal(a),static_cast<unsigned long int>(b),rnd_mode); // mpfr_pow_ui
}

inline const mpreal pow(const double a, const long int b, mp_rnd_t rnd_mode)
{
    return pow(mpreal(a),b,rnd_mode); // mpfr_pow_si
}

inline const mpreal pow(const double a, const int b, mp_rnd_t rnd_mode)
{
    return pow(mpreal(a),static_cast<long int>(b),rnd_mode); // mpfr_pow_si
}
} // End of mpfr namespace

// Explicit specialization of std::swap for mpreal numbers
// Thus standard algorithms will use efficient version of swap (due to Koenig lookup)
// Non-throwing swap C++ idiom: http://en.wikibooks.org/wiki/More_C%2B%2B_Idioms/Non-throwing_swap
namespace std
{
  // we are allowed to extend namespace std with specializations only
    template <>
    inline void swap(mpfr::mpreal& x, mpfr::mpreal& y) 
    { 
        return mpfr::swap(x, y); 
    }

    template<>
    class numeric_limits<mpfr::mpreal>
    {
    public:
        static const bool is_specialized    = true;
        static const bool is_signed         = true;
        static const bool is_integer        = false;
        static const bool is_exact          = false;
        static const int  radix             = 2;    

        static const bool has_infinity      = true;
        static const bool has_quiet_NaN     = true;
        static const bool has_signaling_NaN = true;

        static const bool is_iec559         = true;        // = IEEE 754
        static const bool is_bounded        = true;
        static const bool is_modulo         = false;
        static const bool traps             = true;
        static const bool tinyness_before   = true;

        static const float_denorm_style has_denorm  = denorm_absent;

        inline static mpfr::mpreal (min)    (mp_prec_t precision = mpfr::mpreal::get_default_prec()) {  return  mpfr::minval(precision);  }
        inline static mpfr::mpreal (max)    (mp_prec_t precision = mpfr::mpreal::get_default_prec()) {  return  mpfr::maxval(precision);  }
        inline static mpfr::mpreal lowest   (mp_prec_t precision = mpfr::mpreal::get_default_prec()) {  return -mpfr::maxval(precision);  }

        // Returns smallest eps such that 1 + eps != 1 (classic machine epsilon)
        inline static mpfr::mpreal epsilon(mp_prec_t precision = mpfr::mpreal::get_default_prec()) {  return  mpfr::machine_epsilon(precision); }
    
        // Returns smallest eps such that x + eps != x (relative machine epsilon)
        inline static mpfr::mpreal epsilon(const mpfr::mpreal& x) {  return mpfr::machine_epsilon(x);  }

        inline static mpfr::mpreal round_error(mp_prec_t precision = mpfr::mpreal::get_default_prec())
        {
            mp_rnd_t r = mpfr::mpreal::get_default_rnd();

            if(r == GMP_RNDN)  return mpfr::mpreal(0.5, precision); 
            else               return mpfr::mpreal(1.0, precision);    
        }

        inline static const mpfr::mpreal infinity()         { return mpfr::const_infinity();     }
        inline static const mpfr::mpreal quiet_NaN()        { return mpfr::mpreal().setNan();    }
        inline static const mpfr::mpreal signaling_NaN()    { return mpfr::mpreal().setNan();    }
        inline static const mpfr::mpreal denorm_min()       { return (min)();                    }

        // Please note, exponent range is not fixed in MPFR
        static const int min_exponent = MPFR_EMIN_DEFAULT;
        static const int max_exponent = MPFR_EMAX_DEFAULT;
        MPREAL_PERMISSIVE_EXPR static const int min_exponent10 = (int) (MPFR_EMIN_DEFAULT * 0.3010299956639811); 
        MPREAL_PERMISSIVE_EXPR static const int max_exponent10 = (int) (MPFR_EMAX_DEFAULT * 0.3010299956639811); 

#ifdef MPREAL_HAVE_DYNAMIC_STD_NUMERIC_LIMITS

        // Following members should be constant according to standard, but they can be variable in MPFR
        // So we define them as functions here. 
        //
        // This is preferable way for std::numeric_limits<mpfr::mpreal> specialization.
        // But it is incompatible with standard std::numeric_limits and might not work with other libraries, e.g. boost. 
        // See below for compatible implementation. 
        inline static float_round_style round_style()
        {
            mp_rnd_t r = mpfr::mpreal::get_default_rnd();

            switch (r)
            {
            case GMP_RNDN: return round_to_nearest;
            case GMP_RNDZ: return round_toward_zero; 
            case GMP_RNDU: return round_toward_infinity; 
            case GMP_RNDD: return round_toward_neg_infinity; 
            default: return round_indeterminate;
            }
        }

        inline static int digits()                        {    return int(mpfr::mpreal::get_default_prec());    }
        inline static int digits(const mpfr::mpreal& x)   {    return x.getPrecision();                         }

        inline static int digits10(mp_prec_t precision = mpfr::mpreal::get_default_prec())
        {
            return mpfr::bits2digits(precision);
        }

        inline static int digits10(const mpfr::mpreal& x)
        {
            return mpfr::bits2digits(x.getPrecision());
        }

        inline static int max_digits10(mp_prec_t precision = mpfr::mpreal::get_default_prec())
        {
            return digits10(precision);
        }
#else
        // Digits and round_style are NOT constants when it comes to mpreal.
        // If possible, please use functions digits() and round_style() defined above.
        //
        // These (default) values are preserved for compatibility with existing libraries, e.g. boost.
        // Change them accordingly to your application. 
        //
        // For example, if you use 256 bits of precision uniformly in your program, then:
        // digits       = 256
        // digits10     = 77 
        // max_digits10 = 78
        // 
        // Approximate formula for decimal digits is: digits10 = floor(log10(2) * digits). See bits2digits() for more details.

        static const std::float_round_style round_style = round_to_nearest;
        static const int digits       = 53;
        static const int digits10     = 15;
        static const int max_digits10 = 16;
#endif
    };

}

#endif /* __MPREAL_H__ */
