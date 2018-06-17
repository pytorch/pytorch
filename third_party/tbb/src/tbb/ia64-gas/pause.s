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

	.section .text
	.align 16
	// void __TBB_machine_pause( long count );
	// r32 = count
	.proc  __TBB_machine_pause#
	.global __TBB_machine_pause#
count = r32
__TBB_machine_pause:
        hint.m 0
	add count=-1,count
;;
	cmp.eq p6,p7=0,count
(p7)	br.cond.dpnt __TBB_machine_pause
(p6)   	br.ret.sptk.many b0	
	.endp __TBB_machine_pause#
