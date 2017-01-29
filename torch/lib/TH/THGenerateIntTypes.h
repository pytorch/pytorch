#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateIntTypes.h"
#endif

#define real unsigned char
#define accreal long
#define Real Byte
#define TH_CONVERT_REAL_TO_ACCREAL(_val) (accreal)(_val)
#define TH_CONVERT_ACCREAL_TO_REAL(_val) (real)(_val)
#define THInf UCHAR_MAX
#define TH_REAL_IS_BYTE
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef THInf
#undef TH_REAL_IS_BYTE
#undef TH_CONVERT_REAL_TO_ACCREAL
#undef TH_CONVERT_ACCREAL_TO_REAL


#define real char
#define accreal long
#define Real Char
#define THInf CHAR_MAX
#define TH_CONVERT_REAL_TO_ACCREAL(_val) (accreal)(_val)
#define TH_CONVERT_ACCREAL_TO_REAL(_val) (real)(_val)
#define TH_REAL_IS_CHAR
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef THInf
#undef TH_REAL_IS_CHAR
#undef TH_CONVERT_REAL_TO_ACCREAL
#undef TH_CONVERT_ACCREAL_TO_REAL

#define real short
#define accreal long
#define TH_CONVERT_REAL_TO_ACCREAL(_val) (accreal)(_val)
#define TH_CONVERT_ACCREAL_TO_REAL(_val) (real)(_val)
#define Real Short
#define THInf SHRT_MAX
#define TH_REAL_IS_SHORT
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef THInf
#undef TH_REAL_IS_SHORT
#undef TH_CONVERT_REAL_TO_ACCREAL
#undef TH_CONVERT_ACCREAL_TO_REAL

#define real int
#define accreal long
#define TH_CONVERT_REAL_TO_ACCREAL(_val) (accreal)(_val)
#define TH_CONVERT_ACCREAL_TO_REAL(_val) (real)(_val)
#define Real Int
#define THInf INT_MAX
#define TH_REAL_IS_INT
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef THInf
#undef TH_REAL_IS_INT
#undef TH_CONVERT_REAL_TO_ACCREAL
#undef TH_CONVERT_ACCREAL_TO_REAL

#define real long
#define accreal long
#define TH_CONVERT_REAL_TO_ACCREAL(_val) (accreal)(_val)
#define TH_CONVERT_ACCREAL_TO_REAL(_val) (real)(_val)
#define Real Long
#define THInf LONG_MAX
#define TH_REAL_IS_LONG
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef THInf
#undef TH_REAL_IS_LONG
#undef TH_CONVERT_REAL_TO_ACCREAL
#undef TH_CONVERT_ACCREAL_TO_REAL

#ifndef THGenerateAllTypes
#undef TH_GENERIC_FILE
#endif
