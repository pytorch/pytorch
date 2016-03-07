#ifndef THC_GENERIC_FILE
#error "You must define THC_GENERIC_FILE before including THGenerateAllTypes.h"
#endif

#define THCTypeIdxByte   1
#define THCTypeIdxChar   2
#define THCTypeIdxShort  3
#define THCTypeIdxInt    4
#define THCTypeIdxLong   5
#define THCTypeIdxFloat  6
#define THCTypeIdxDouble 7
#define THCTypeIdxHalf   8
#define THCTypeIdx_(T) TH_CONCAT_2(THCTypeIdx,T)

#define hostreal real
#define hostrealToReal(x) (x)
#define realToHostreal(x) (x)

#define real unsigned char
#define accreal long
#define Real Byte
#define CReal CudaByte
#define THC_REAL_IS_BYTE
#line 1 THC_GENERIC_FILE
/*#line 1 "THByteStorage.h"*/
#include THC_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef CReal
#undef THC_REAL_IS_BYTE

#define real char
#define accreal long
#define Real Char
#define CReal CudaChar
#define THC_REAL_IS_CHAR
#line 1 THC_GENERIC_FILE
#include THC_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef CReal
#undef THC_REAL_IS_CHAR

#define real short
#define accreal long
#define Real Short
#define CReal CudaShort
#define THC_REAL_IS_SHORT
#line 1 THC_GENERIC_FILE
#include THC_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef CReal
#undef THC_REAL_IS_SHORT

#define real int
#define accreal long
#define Real Int
#define CReal CudaInt
#define THC_REAL_IS_INT
#line 1 THC_GENERIC_FILE
#include THC_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef CReal
#undef THC_REAL_IS_INT

#define real long
#define accreal long
#define Real Long
#define CReal CudaLong
#define THC_REAL_IS_LONG
#line 1 THC_GENERIC_FILE
#include THC_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef CReal
#undef THC_REAL_IS_LONG

#define real float
#define accreal double
#define Real Float
#define CReal Cuda
#define THC_REAL_IS_FLOAT
#line 1 THC_GENERIC_FILE
#include THC_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef CReal
#undef THC_REAL_IS_FLOAT

#define real double
#define accreal double
#define Real Double
#define CReal CudaDouble
#define THC_REAL_IS_DOUBLE
#line 1 THC_GENERIC_FILE
#include THC_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef CReal
#undef THC_REAL_IS_DOUBLE

#if CUDA_VERSION >= 7050

#undef hostreal
#undef hostrealToReal
#undef realToHostreal
#define hostreal float
#define hostrealToReal(x) THC_float2half(x);
#define realToHostreal(x) THC_half2float(x);

#define real half
#define accreal half
#define Real Half
#define CReal CudaHalf
#define THC_REAL_IS_HALF
#line 1 THC_GENERIC_FILE
#include THC_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef CReal
#undef THC_REAL_IS_HALF

#endif // CUDA_VERSION >= 7050

#undef hostreal
#undef hostrealToReal
#undef realToHostreal

#undef THCTypeIdxByte
#undef THCTypeIdxChar
#undef THCTypeIdxShort
#undef THCTypeIdxInt
#undef THCTypeIdxLong
#undef THCTypeIdxFloat
#undef THCTypeIdxDouble
#undef THCTypeIdxHalf

#undef THC_GENERIC_FILE
