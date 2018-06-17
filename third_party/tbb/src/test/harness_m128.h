/*
    Copyright (c) 2005-2018 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.




*/

// Header that sets HAVE_m128/HAVE_m256 if vector types (__m128/__m256) are available

//! Class for testing safety of using vector types.
/** Uses circuitous logic forces compiler to put __m128/__m256 objects on stack while
    executing various methods, and thus tempt it to use aligned loads and stores
    on the stack. */
//  Do not create file-scope objects of the class, because MinGW (as of May 2010)
//  did not always provide proper stack alignment in destructors of such objects.

#if (_MSC_VER>=1600)
//TODO: handle /arch:AVX in the right way.
#pragma warning (push)
#pragma warning (disable: 4752)
#endif

template<typename __Mvec>
class ClassWithVectorType {
    static const int n = 16;
    static const int F = sizeof(__Mvec)/sizeof(float);
    __Mvec field[n];
    void init( int start );
public:
    ClassWithVectorType() {init(-n);}
    ClassWithVectorType( int i ) {init(i);}
    void operator=( const ClassWithVectorType& src ) {
        __Mvec stack[n];
        for( int i=0; i<n; ++i )
            stack[i^5] = src.field[i];
        for( int i=0; i<n; ++i )
            field[i^5] = stack[i];
    }
    ~ClassWithVectorType() {init(-2*n);}
    friend bool operator==( const ClassWithVectorType& x, const ClassWithVectorType& y ) {
        for( int i=0; i<F*n; ++i )
            if( ((const float*)x.field)[i]!=((const float*)y.field)[i] )
                return false;
        return true;
    }
    friend bool operator!=( const ClassWithVectorType& x, const ClassWithVectorType& y ) {
        return !(x==y);
    }
};

template<typename __Mvec>
void ClassWithVectorType<__Mvec>::init( int start ) {
    __Mvec stack[n];
    for( int i=0; i<n; ++i ) {
        // Declaring value as a one-element array instead of a scalar quites
        // gratuitous warnings about possible use of "value" before it was set.
        __Mvec value[1];
        for( int j=0; j<F; ++j )
            ((float*)value)[j] = float(n*start+F*i+j);
        stack[i^5] = value[0];
    }
    for( int i=0; i<n; ++i )
        field[i^5] = stack[i];
}

#if (__AVX__ || (_MSC_VER>=1600 && _M_X64)) && !defined(__sun)
#include <immintrin.h>
#define HAVE_m256 1
typedef ClassWithVectorType<__m256> ClassWithAVX;
#if _MSC_VER
#include <intrin.h> // for __cpuid
#endif
bool have_AVX() {
    bool result = false;
    const int avx_mask = 1<<28;
#if _MSC_VER || __INTEL_COMPILER
    int info[4] = {0,0,0,0};
    const int ECX = 2;
    __cpuid(info, 1);
    result = (info[ECX] & avx_mask)!=0;
#elif __GNUC__
    int ECX;
    __asm__( "cpuid"
             : "=c"(ECX)
             : "a" (1)
             : "ebx", "edx" );
    result = (ECX & avx_mask);
#endif
    return result;
}
#endif /* __AVX__ etc */

#if (__SSE__ || _M_IX86_FP || _M_X64) && !defined(__sun)
#include <xmmintrin.h>
#define HAVE_m128 1
typedef ClassWithVectorType<__m128> ClassWithSSE;
#endif

#if (_MSC_VER>=1600)
#pragma warning (pop)
#endif
