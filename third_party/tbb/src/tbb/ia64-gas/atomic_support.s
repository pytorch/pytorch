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

// DO NOT EDIT - AUTOMATICALLY GENERATED FROM tools/generate_atomic/ipf_generate.sh
# 1 "<stdin>"
# 1 "<built-in>"
# 1 "<command line>"
# 1 "<stdin>"





        .section .text
        .align 16


        .proc __TBB_machine_fetchadd1__TBB_full_fence#
        .global __TBB_machine_fetchadd1__TBB_full_fence#
__TBB_machine_fetchadd1__TBB_full_fence:
{
        mf
        br __TBB_machine_fetchadd1acquire
}
        .endp __TBB_machine_fetchadd1__TBB_full_fence#

        .proc __TBB_machine_fetchadd1acquire#
        .global __TBB_machine_fetchadd1acquire#
__TBB_machine_fetchadd1acquire:







        ld1 r9=[r32]
;;
Retry_1acquire:
        mov ar.ccv=r9
        mov r8=r9;
        add r10=r9,r33
;;
        cmpxchg1.acq r9=[r32],r10,ar.ccv
;;
        cmp.ne p7,p0=r8,r9
  (p7) br.cond.dpnt Retry_1acquire
        br.ret.sptk.many b0
# 49 "<stdin>"
        .endp __TBB_machine_fetchadd1acquire#
# 62 "<stdin>"
        .section .text
        .align 16
        .proc __TBB_machine_fetchstore1__TBB_full_fence#
        .global __TBB_machine_fetchstore1__TBB_full_fence#
__TBB_machine_fetchstore1__TBB_full_fence:
        mf
;;
        xchg1 r8=[r32],r33
        br.ret.sptk.many b0
        .endp __TBB_machine_fetchstore1__TBB_full_fence#


        .proc __TBB_machine_fetchstore1acquire#
        .global __TBB_machine_fetchstore1acquire#
__TBB_machine_fetchstore1acquire:
        xchg1 r8=[r32],r33
        br.ret.sptk.many b0
        .endp __TBB_machine_fetchstore1acquire#
# 88 "<stdin>"
        .section .text
        .align 16


        .proc __TBB_machine_cmpswp1__TBB_full_fence#
        .global __TBB_machine_cmpswp1__TBB_full_fence#
__TBB_machine_cmpswp1__TBB_full_fence:
{
        mf
        br __TBB_machine_cmpswp1acquire
}
        .endp __TBB_machine_cmpswp1__TBB_full_fence#

        .proc __TBB_machine_cmpswp1acquire#
        .global __TBB_machine_cmpswp1acquire#
__TBB_machine_cmpswp1acquire:

        zxt1 r34=r34
;;

        mov ar.ccv=r34
;;
        cmpxchg1.acq r8=[r32],r33,ar.ccv
        br.ret.sptk.many b0
        .endp __TBB_machine_cmpswp1acquire#
// DO NOT EDIT - AUTOMATICALLY GENERATED FROM tools/generate_atomic/ipf_generate.sh
# 1 "<stdin>"
# 1 "<built-in>"
# 1 "<command line>"
# 1 "<stdin>"





        .section .text
        .align 16


        .proc __TBB_machine_fetchadd2__TBB_full_fence#
        .global __TBB_machine_fetchadd2__TBB_full_fence#
__TBB_machine_fetchadd2__TBB_full_fence:
{
        mf
        br __TBB_machine_fetchadd2acquire
}
        .endp __TBB_machine_fetchadd2__TBB_full_fence#

        .proc __TBB_machine_fetchadd2acquire#
        .global __TBB_machine_fetchadd2acquire#
__TBB_machine_fetchadd2acquire:







        ld2 r9=[r32]
;;
Retry_2acquire:
        mov ar.ccv=r9
        mov r8=r9;
        add r10=r9,r33
;;
        cmpxchg2.acq r9=[r32],r10,ar.ccv
;;
        cmp.ne p7,p0=r8,r9
  (p7) br.cond.dpnt Retry_2acquire
        br.ret.sptk.many b0
# 49 "<stdin>"
        .endp __TBB_machine_fetchadd2acquire#
# 62 "<stdin>"
        .section .text
        .align 16
        .proc __TBB_machine_fetchstore2__TBB_full_fence#
        .global __TBB_machine_fetchstore2__TBB_full_fence#
__TBB_machine_fetchstore2__TBB_full_fence:
        mf
;;
        xchg2 r8=[r32],r33
        br.ret.sptk.many b0
        .endp __TBB_machine_fetchstore2__TBB_full_fence#


        .proc __TBB_machine_fetchstore2acquire#
        .global __TBB_machine_fetchstore2acquire#
__TBB_machine_fetchstore2acquire:
        xchg2 r8=[r32],r33
        br.ret.sptk.many b0
        .endp __TBB_machine_fetchstore2acquire#
# 88 "<stdin>"
        .section .text
        .align 16


        .proc __TBB_machine_cmpswp2__TBB_full_fence#
        .global __TBB_machine_cmpswp2__TBB_full_fence#
__TBB_machine_cmpswp2__TBB_full_fence:
{
        mf
        br __TBB_machine_cmpswp2acquire
}
        .endp __TBB_machine_cmpswp2__TBB_full_fence#

        .proc __TBB_machine_cmpswp2acquire#
        .global __TBB_machine_cmpswp2acquire#
__TBB_machine_cmpswp2acquire:

        zxt2 r34=r34
;;

        mov ar.ccv=r34
;;
        cmpxchg2.acq r8=[r32],r33,ar.ccv
        br.ret.sptk.many b0
        .endp __TBB_machine_cmpswp2acquire#
// DO NOT EDIT - AUTOMATICALLY GENERATED FROM tools/generate_atomic/ipf_generate.sh
# 1 "<stdin>"
# 1 "<built-in>"
# 1 "<command line>"
# 1 "<stdin>"





        .section .text
        .align 16


        .proc __TBB_machine_fetchadd4__TBB_full_fence#
        .global __TBB_machine_fetchadd4__TBB_full_fence#
__TBB_machine_fetchadd4__TBB_full_fence:
{
        mf
        br __TBB_machine_fetchadd4acquire
}
        .endp __TBB_machine_fetchadd4__TBB_full_fence#

        .proc __TBB_machine_fetchadd4acquire#
        .global __TBB_machine_fetchadd4acquire#
__TBB_machine_fetchadd4acquire:

        cmp.eq p6,p0=1,r33
        cmp.eq p8,p0=-1,r33
  (p6) br.cond.dptk Inc_4acquire
  (p8) br.cond.dpnt Dec_4acquire
;;

        ld4 r9=[r32]
;;
Retry_4acquire:
        mov ar.ccv=r9
        mov r8=r9;
        add r10=r9,r33
;;
        cmpxchg4.acq r9=[r32],r10,ar.ccv
;;
        cmp.ne p7,p0=r8,r9
  (p7) br.cond.dpnt Retry_4acquire
        br.ret.sptk.many b0

Inc_4acquire:
        fetchadd4.acq r8=[r32],1
        br.ret.sptk.many b0
Dec_4acquire:
        fetchadd4.acq r8=[r32],-1
        br.ret.sptk.many b0

        .endp __TBB_machine_fetchadd4acquire#
# 62 "<stdin>"
        .section .text
        .align 16
        .proc __TBB_machine_fetchstore4__TBB_full_fence#
        .global __TBB_machine_fetchstore4__TBB_full_fence#
__TBB_machine_fetchstore4__TBB_full_fence:
        mf
;;
        xchg4 r8=[r32],r33
        br.ret.sptk.many b0
        .endp __TBB_machine_fetchstore4__TBB_full_fence#


        .proc __TBB_machine_fetchstore4acquire#
        .global __TBB_machine_fetchstore4acquire#
__TBB_machine_fetchstore4acquire:
        xchg4 r8=[r32],r33
        br.ret.sptk.many b0
        .endp __TBB_machine_fetchstore4acquire#
# 88 "<stdin>"
        .section .text
        .align 16


        .proc __TBB_machine_cmpswp4__TBB_full_fence#
        .global __TBB_machine_cmpswp4__TBB_full_fence#
__TBB_machine_cmpswp4__TBB_full_fence:
{
        mf
        br __TBB_machine_cmpswp4acquire
}
        .endp __TBB_machine_cmpswp4__TBB_full_fence#

        .proc __TBB_machine_cmpswp4acquire#
        .global __TBB_machine_cmpswp4acquire#
__TBB_machine_cmpswp4acquire:

        zxt4 r34=r34
;;

        mov ar.ccv=r34
;;
        cmpxchg4.acq r8=[r32],r33,ar.ccv
        br.ret.sptk.many b0
        .endp __TBB_machine_cmpswp4acquire#
// DO NOT EDIT - AUTOMATICALLY GENERATED FROM tools/generate_atomic/ipf_generate.sh
# 1 "<stdin>"
# 1 "<built-in>"
# 1 "<command line>"
# 1 "<stdin>"





        .section .text
        .align 16


        .proc __TBB_machine_fetchadd8__TBB_full_fence#
        .global __TBB_machine_fetchadd8__TBB_full_fence#
__TBB_machine_fetchadd8__TBB_full_fence:
{
        mf
        br __TBB_machine_fetchadd8acquire
}
        .endp __TBB_machine_fetchadd8__TBB_full_fence#

        .proc __TBB_machine_fetchadd8acquire#
        .global __TBB_machine_fetchadd8acquire#
__TBB_machine_fetchadd8acquire:

        cmp.eq p6,p0=1,r33
        cmp.eq p8,p0=-1,r33
  (p6) br.cond.dptk Inc_8acquire
  (p8) br.cond.dpnt Dec_8acquire
;;

        ld8 r9=[r32]
;;
Retry_8acquire:
        mov ar.ccv=r9
        mov r8=r9;
        add r10=r9,r33
;;
        cmpxchg8.acq r9=[r32],r10,ar.ccv
;;
        cmp.ne p7,p0=r8,r9
  (p7) br.cond.dpnt Retry_8acquire
        br.ret.sptk.many b0

Inc_8acquire:
        fetchadd8.acq r8=[r32],1
        br.ret.sptk.many b0
Dec_8acquire:
        fetchadd8.acq r8=[r32],-1
        br.ret.sptk.many b0

        .endp __TBB_machine_fetchadd8acquire#
# 62 "<stdin>"
        .section .text
        .align 16
        .proc __TBB_machine_fetchstore8__TBB_full_fence#
        .global __TBB_machine_fetchstore8__TBB_full_fence#
__TBB_machine_fetchstore8__TBB_full_fence:
        mf
;;
        xchg8 r8=[r32],r33
        br.ret.sptk.many b0
        .endp __TBB_machine_fetchstore8__TBB_full_fence#


        .proc __TBB_machine_fetchstore8acquire#
        .global __TBB_machine_fetchstore8acquire#
__TBB_machine_fetchstore8acquire:
        xchg8 r8=[r32],r33
        br.ret.sptk.many b0
        .endp __TBB_machine_fetchstore8acquire#
# 88 "<stdin>"
        .section .text
        .align 16


        .proc __TBB_machine_cmpswp8__TBB_full_fence#
        .global __TBB_machine_cmpswp8__TBB_full_fence#
__TBB_machine_cmpswp8__TBB_full_fence:
{
        mf
        br __TBB_machine_cmpswp8acquire
}
        .endp __TBB_machine_cmpswp8__TBB_full_fence#

        .proc __TBB_machine_cmpswp8acquire#
        .global __TBB_machine_cmpswp8acquire#
__TBB_machine_cmpswp8acquire:




        mov ar.ccv=r34
;;
        cmpxchg8.acq r8=[r32],r33,ar.ccv
        br.ret.sptk.many b0
        .endp __TBB_machine_cmpswp8acquire#
// DO NOT EDIT - AUTOMATICALLY GENERATED FROM tools/generate_atomic/ipf_generate.sh
# 1 "<stdin>"
# 1 "<built-in>"
# 1 "<command line>"
# 1 "<stdin>"





        .section .text
        .align 16
# 19 "<stdin>"
        .proc __TBB_machine_fetchadd1release#
        .global __TBB_machine_fetchadd1release#
__TBB_machine_fetchadd1release:







        ld1 r9=[r32]
;;
Retry_1release:
        mov ar.ccv=r9
        mov r8=r9;
        add r10=r9,r33
;;
        cmpxchg1.rel r9=[r32],r10,ar.ccv
;;
        cmp.ne p7,p0=r8,r9
  (p7) br.cond.dpnt Retry_1release
        br.ret.sptk.many b0
# 49 "<stdin>"
        .endp __TBB_machine_fetchadd1release#
# 62 "<stdin>"
        .section .text
        .align 16
        .proc __TBB_machine_fetchstore1release#
        .global __TBB_machine_fetchstore1release#
__TBB_machine_fetchstore1release:
        mf
;;
        xchg1 r8=[r32],r33
        br.ret.sptk.many b0
        .endp __TBB_machine_fetchstore1release#
# 88 "<stdin>"
        .section .text
        .align 16
# 101 "<stdin>"
        .proc __TBB_machine_cmpswp1release#
        .global __TBB_machine_cmpswp1release#
__TBB_machine_cmpswp1release:

        zxt1 r34=r34
;;

        mov ar.ccv=r34
;;
        cmpxchg1.rel r8=[r32],r33,ar.ccv
        br.ret.sptk.many b0
        .endp __TBB_machine_cmpswp1release#
// DO NOT EDIT - AUTOMATICALLY GENERATED FROM tools/generate_atomic/ipf_generate.sh
# 1 "<stdin>"
# 1 "<built-in>"
# 1 "<command line>"
# 1 "<stdin>"





        .section .text
        .align 16
# 19 "<stdin>"
        .proc __TBB_machine_fetchadd2release#
        .global __TBB_machine_fetchadd2release#
__TBB_machine_fetchadd2release:







        ld2 r9=[r32]
;;
Retry_2release:
        mov ar.ccv=r9
        mov r8=r9;
        add r10=r9,r33
;;
        cmpxchg2.rel r9=[r32],r10,ar.ccv
;;
        cmp.ne p7,p0=r8,r9
  (p7) br.cond.dpnt Retry_2release
        br.ret.sptk.many b0
# 49 "<stdin>"
        .endp __TBB_machine_fetchadd2release#
# 62 "<stdin>"
        .section .text
        .align 16
        .proc __TBB_machine_fetchstore2release#
        .global __TBB_machine_fetchstore2release#
__TBB_machine_fetchstore2release:
        mf
;;
        xchg2 r8=[r32],r33
        br.ret.sptk.many b0
        .endp __TBB_machine_fetchstore2release#
# 88 "<stdin>"
        .section .text
        .align 16
# 101 "<stdin>"
        .proc __TBB_machine_cmpswp2release#
        .global __TBB_machine_cmpswp2release#
__TBB_machine_cmpswp2release:

        zxt2 r34=r34
;;

        mov ar.ccv=r34
;;
        cmpxchg2.rel r8=[r32],r33,ar.ccv
        br.ret.sptk.many b0
        .endp __TBB_machine_cmpswp2release#
// DO NOT EDIT - AUTOMATICALLY GENERATED FROM tools/generate_atomic/ipf_generate.sh
# 1 "<stdin>"
# 1 "<built-in>"
# 1 "<command line>"
# 1 "<stdin>"





        .section .text
        .align 16
# 19 "<stdin>"
        .proc __TBB_machine_fetchadd4release#
        .global __TBB_machine_fetchadd4release#
__TBB_machine_fetchadd4release:

        cmp.eq p6,p0=1,r33
        cmp.eq p8,p0=-1,r33
  (p6) br.cond.dptk Inc_4release
  (p8) br.cond.dpnt Dec_4release
;;

        ld4 r9=[r32]
;;
Retry_4release:
        mov ar.ccv=r9
        mov r8=r9;
        add r10=r9,r33
;;
        cmpxchg4.rel r9=[r32],r10,ar.ccv
;;
        cmp.ne p7,p0=r8,r9
  (p7) br.cond.dpnt Retry_4release
        br.ret.sptk.many b0

Inc_4release:
        fetchadd4.rel r8=[r32],1
        br.ret.sptk.many b0
Dec_4release:
        fetchadd4.rel r8=[r32],-1
        br.ret.sptk.many b0

        .endp __TBB_machine_fetchadd4release#
# 62 "<stdin>"
        .section .text
        .align 16
        .proc __TBB_machine_fetchstore4release#
        .global __TBB_machine_fetchstore4release#
__TBB_machine_fetchstore4release:
        mf
;;
        xchg4 r8=[r32],r33
        br.ret.sptk.many b0
        .endp __TBB_machine_fetchstore4release#
# 88 "<stdin>"
        .section .text
        .align 16
# 101 "<stdin>"
        .proc __TBB_machine_cmpswp4release#
        .global __TBB_machine_cmpswp4release#
__TBB_machine_cmpswp4release:

        zxt4 r34=r34
;;

        mov ar.ccv=r34
;;
        cmpxchg4.rel r8=[r32],r33,ar.ccv
        br.ret.sptk.many b0
        .endp __TBB_machine_cmpswp4release#
// DO NOT EDIT - AUTOMATICALLY GENERATED FROM tools/generate_atomic/ipf_generate.sh
# 1 "<stdin>"
# 1 "<built-in>"
# 1 "<command line>"
# 1 "<stdin>"





        .section .text
        .align 16
# 19 "<stdin>"
        .proc __TBB_machine_fetchadd8release#
        .global __TBB_machine_fetchadd8release#
__TBB_machine_fetchadd8release:

        cmp.eq p6,p0=1,r33
        cmp.eq p8,p0=-1,r33
  (p6) br.cond.dptk Inc_8release
  (p8) br.cond.dpnt Dec_8release
;;

        ld8 r9=[r32]
;;
Retry_8release:
        mov ar.ccv=r9
        mov r8=r9;
        add r10=r9,r33
;;
        cmpxchg8.rel r9=[r32],r10,ar.ccv
;;
        cmp.ne p7,p0=r8,r9
  (p7) br.cond.dpnt Retry_8release
        br.ret.sptk.many b0

Inc_8release:
        fetchadd8.rel r8=[r32],1
        br.ret.sptk.many b0
Dec_8release:
        fetchadd8.rel r8=[r32],-1
        br.ret.sptk.many b0

        .endp __TBB_machine_fetchadd8release#
# 62 "<stdin>"
        .section .text
        .align 16
        .proc __TBB_machine_fetchstore8release#
        .global __TBB_machine_fetchstore8release#
__TBB_machine_fetchstore8release:
        mf
;;
        xchg8 r8=[r32],r33
        br.ret.sptk.many b0
        .endp __TBB_machine_fetchstore8release#
# 88 "<stdin>"
        .section .text
        .align 16
# 101 "<stdin>"
        .proc __TBB_machine_cmpswp8release#
        .global __TBB_machine_cmpswp8release#
__TBB_machine_cmpswp8release:




        mov ar.ccv=r34
;;
        cmpxchg8.rel r8=[r32],r33,ar.ccv
        br.ret.sptk.many b0
        .endp __TBB_machine_cmpswp8release#
