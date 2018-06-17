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

// Header that includes Intel(R) Transactional Synchronization Extensions (Intel(R) TSX) specific test functions

#if __TBB_TSX_AVAILABLE
#define __TBB_TSX_TESTING_ENABLED_FOR_THIS_COMPILER (__INTEL_COMPILER || __GNUC__ || _MSC_VER || __SUNPRO_CC)
#if __TBB_TSX_TESTING_ENABLED_FOR_THIS_COMPILER

#include "harness_defs.h"

inline static bool IsInsideTx()
{
    return __TBB_machine_is_in_transaction() != 0;
}

#if _MSC_VER
#include <intrin.h> // for __cpuid
#endif
// TODO: consider reusing tbb_misc.cpp:cpu_has_speculation() instead of code duplication.
bool have_TSX() {
    bool result = false;
    const int hle_ebx_mask = 1<<4;
    const int rtm_ebx_mask = 1<<11;
#if _MSC_VER
    int info[4] = {0,0,0,0};
    const int reg_ebx = 1;
    int old_ecx = 0;
    __cpuidex(info, 7, old_ecx);
    result = (info[reg_ebx] & rtm_ebx_mask)!=0;
    if( result ) ASSERT( (info[reg_ebx] & hle_ebx_mask)!=0, NULL );
#elif __GNUC__ || __SUNPRO_CC
    int32_t reg_ebx = 0;
    int32_t reg_eax = 7;
    int32_t reg_ecx = 0;
    __asm__ __volatile__ ( "movl %%ebx, %%esi\n"
                           "cpuid\n"
                           "movl %%ebx, %0\n"
                           "movl %%esi, %%ebx\n"
                           : "=a"(reg_ebx) : "0" (reg_eax), "c" (reg_ecx) : "esi",
#if __TBB_x86_64
                           "ebx",
#endif
                           "edx"
                           );
    result = (reg_ebx & rtm_ebx_mask)!=0 ;
    if( result ) ASSERT( (reg_ebx & hle_ebx_mask)!=0, NULL );
#endif
    return result;
}

#endif /* __TBB_TSX_TESTING_ENABLED_FOR_THIS_COMPILER */
#endif /* __TBB_TSX_AVAILABLE */
