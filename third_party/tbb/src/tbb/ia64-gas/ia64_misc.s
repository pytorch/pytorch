// Copyright (c) 2005-2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//
//
//

	// RSE backing store pointer retrieval
    .section .text
    .align 16
    .proc __TBB_get_bsp#
    .global __TBB_get_bsp#
__TBB_get_bsp:
        mov r8=ar.bsp
        br.ret.sptk.many b0
    .endp __TBB_get_bsp#

    .section .text
    .align 16
    .proc __TBB_machine_load8_relaxed#
    .global __TBB_machine_load8_relaxed#
__TBB_machine_load8_relaxed:
        ld8 r8=[r32]
        br.ret.sptk.many b0
    .endp __TBB_machine_load8_relaxed#

    .section .text
    .align 16
    .proc __TBB_machine_store8_relaxed#
    .global __TBB_machine_store8_relaxed#
__TBB_machine_store8_relaxed:
        st8 [r32]=r33
        br.ret.sptk.many b0
    .endp __TBB_machine_store8_relaxed#

    .section .text
    .align 16
    .proc __TBB_machine_load4_relaxed#
    .global __TBB_machine_load4_relaxed#
__TBB_machine_load4_relaxed:
        ld4 r8=[r32]
        br.ret.sptk.many b0
    .endp __TBB_machine_load4_relaxed#

    .section .text
    .align 16
    .proc __TBB_machine_store4_relaxed#
    .global __TBB_machine_store4_relaxed#
__TBB_machine_store4_relaxed:
        st4 [r32]=r33
        br.ret.sptk.many b0
    .endp __TBB_machine_store4_relaxed#

    .section .text
    .align 16
    .proc __TBB_machine_load2_relaxed#
    .global __TBB_machine_load2_relaxed#
__TBB_machine_load2_relaxed:
        ld2 r8=[r32]
        br.ret.sptk.many b0
    .endp __TBB_machine_load2_relaxed#

    .section .text
    .align 16
    .proc __TBB_machine_store2_relaxed#
    .global __TBB_machine_store2_relaxed#
__TBB_machine_store2_relaxed:
        st2 [r32]=r33
        br.ret.sptk.many b0
    .endp __TBB_machine_store2_relaxed#

    .section .text
    .align 16
    .proc __TBB_machine_load1_relaxed#
    .global __TBB_machine_load1_relaxed#
__TBB_machine_load1_relaxed:
        ld1 r8=[r32]
        br.ret.sptk.many b0
    .endp __TBB_machine_load1_relaxed#

    .section .text
    .align 16
    .proc __TBB_machine_store1_relaxed#
    .global __TBB_machine_store1_relaxed#
__TBB_machine_store1_relaxed:
        st1 [r32]=r33
        br.ret.sptk.many b0
    .endp __TBB_machine_store1_relaxed#
