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

#include <stdint.h>
#include <sys/atomic_op.h>

/* This file must be compiled with gcc.  The IBM compiler doesn't seem to
   support inline assembly statements (October 2007). */

#ifdef __GNUC__

int32_t __TBB_machine_cas_32 (volatile void* ptr, int32_t value, int32_t comparand) { 
    __asm__ __volatile__ ("sync\n");  /* memory release operation */
    compare_and_swap ((atomic_p) ptr, &comparand, value);
    __asm__ __volatile__ ("isync\n");  /* memory acquire operation */
    return comparand;
}

int64_t __TBB_machine_cas_64 (volatile void* ptr, int64_t value, int64_t comparand) { 
    __asm__ __volatile__ ("sync\n");  /* memory release operation */
    compare_and_swaplp ((atomic_l) ptr, &comparand, value);
    __asm__ __volatile__ ("isync\n");  /* memory acquire operation */
    return comparand;
}

void __TBB_machine_flush () { 
    __asm__ __volatile__ ("sync\n");
}

void __TBB_machine_lwsync () { 
    __asm__ __volatile__ ("lwsync\n");
}

void __TBB_machine_isync () { 
    __asm__ __volatile__ ("isync\n");
}

#endif /* __GNUC__ */
