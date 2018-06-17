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

#ifndef FAST_RANDOM_H_
#define FAST_RANDOM_H_
namespace utility{
//------------------------------------------------------------------------
// FastRandom
//------------------------------------------------------------------------

namespace internal{
    size_t GetPrime ( size_t seed );
}

//! A fast random number generator.
/** Uses linear congruential method. */
class FastRandom {
    size_t x, a;
public:
    //! Get a random number.
    unsigned short get() {
        return get(x);
    }
        //! Get a random number for the given seed; update the seed for next use.
    unsigned short get( size_t& seed ) {
        unsigned short r = (unsigned short)(seed>>16);
        seed = seed*a+1;
        return r;
    }
    //! Construct a random number generator.
    FastRandom( size_t seed ) {
        x = seed*internal::GetPrime(seed);
        a = internal::GetPrime(x);
    }
};
}

namespace utility {
namespace internal{
//! Table of primes used by fast random-number generator (FastRandom).
    static const unsigned Primes[] = {
        0x9e3779b1, 0xffe6cc59, 0x2109f6dd, 0x43977ab5,
        0xba5703f5, 0xb495a877, 0xe1626741, 0x79695e6b,
        0xbc98c09f, 0xd5bee2b3, 0x287488f9, 0x3af18231,
        0x9677cd4d, 0xbe3a6929, 0xadc6a877, 0xdcf0674b,
        0xbe4d6fe9, 0x5f15e201, 0x99afc3fd, 0xf3f16801,
        0xe222cfff, 0x24ba5fdb, 0x0620452d, 0x79f149e3,
        0xc8b93f49, 0x972702cd, 0xb07dd827, 0x6c97d5ed,
        0x085a3d61, 0x46eb5ea7, 0x3d9910ed, 0x2e687b5b,
        0x29609227, 0x6eb081f1, 0x0954c4e1, 0x9d114db9,
        0x542acfa9, 0xb3e6bd7b, 0x0742d917, 0xe9f3ffa7,
        0x54581edb, 0xf2480f45, 0x0bb9288f, 0xef1affc7,
        0x85fa0ca7, 0x3ccc14db, 0xe6baf34b, 0x343377f7,
        0x5ca19031, 0xe6d9293b, 0xf0a9f391, 0x5d2e980b,
        0xfc411073, 0xc3749363, 0xb892d829, 0x3549366b,
        0x629750ad, 0xb98294e5, 0x892d9483, 0xc235baf3,
        0x3d2402a3, 0x6bdef3c9, 0xbec333cd, 0x40c9520f
    };
    size_t GetPrime ( size_t seed ) {
        return Primes[seed%(sizeof(Primes)/sizeof(Primes[0]))];
    }
}
}

#endif /* FAST_RANDOM_H_ */
