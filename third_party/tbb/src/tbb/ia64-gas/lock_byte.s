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

	// Support for class TinyLock
	.section .text
	.align 16
	// unsigned int __TBB_machine_trylockbyte( byte& flag );
	// r32 = address of flag 
	.proc  __TBB_machine_trylockbyte#
	.global __TBB_machine_trylockbyte#
ADDRESS_OF_FLAG=r32
RETCODE=r8
FLAG=r9
BUSY=r10
SCRATCH=r11
__TBB_machine_trylockbyte:
        ld1.acq FLAG=[ADDRESS_OF_FLAG]
        mov BUSY=1
        mov RETCODE=0
;;
        cmp.ne p6,p0=0,FLAG
        mov ar.ccv=r0
(p6)    br.ret.sptk.many b0
;;
        cmpxchg1.acq SCRATCH=[ADDRESS_OF_FLAG],BUSY,ar.ccv  // Try to acquire lock
;;
        cmp.eq p6,p0=0,SCRATCH
;;
(p6)    mov RETCODE=1
   	br.ret.sptk.many b0	
	.endp __TBB_machine_trylockbyte#
