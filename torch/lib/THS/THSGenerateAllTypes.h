#ifndef THS_GENERIC_FILE
#error "You must define THS_GENERIC_FILE before including THSGenerateAllTypes.h"
#endif

#define real unsigned char
#define accreal long
#define Real Byte
#define THSInf UCHAR_MAX
#define THS_REAL_IS_BYTE
#line 1 THS_GENERIC_FILE
/*#line 1 "THSByteStorage.h"*/
#include THS_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef THSInf
#undef THS_REAL_IS_BYTE

#define real char
#define accreal long
#define Real Char
#define THSInf CHAR_MAX
#define THS_REAL_IS_CHAR
#line 1 THS_GENERIC_FILE
#include THS_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef THSInf
#undef THS_REAL_IS_CHAR

#define real short
#define accreal long
#define Real Short
#define THSInf SHRT_MAX
#define THS_REAL_IS_SHORT
#line 1 THS_GENERIC_FILE
#include THS_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef THSInf
#undef THS_REAL_IS_SHORT

#define real int
#define accreal long
#define Real Int
#define THSInf INT_MAX
#define THS_REAL_IS_INT
#line 1 THS_GENERIC_FILE
#include THS_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef THSInf
#undef THS_REAL_IS_INT

#define real long
#define accreal long
#define Real Long
#define THSInf LONG_MAX
#define THS_REAL_IS_LONG
#line 1 THS_GENERIC_FILE
#include THS_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef THSInf
#undef THS_REAL_IS_LONG

#define real float
#define accreal double
#define Real Float
#define THSInf FLT_MAX
#define THS_REAL_IS_FLOAT
#line 1 THS_GENERIC_FILE
#include THS_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef THSInf
#undef THS_REAL_IS_FLOAT

#define real double
#define accreal double
#define Real Double
#define THSInf DBL_MAX
#define THS_REAL_IS_DOUBLE
#line 1 THS_GENERIC_FILE
#include THS_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef THSInf
#undef THS_REAL_IS_DOUBLE

#undef THS_GENERIC_FILE
