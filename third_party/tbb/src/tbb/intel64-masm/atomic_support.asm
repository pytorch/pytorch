; Copyright (c) 2005-2018 Intel Corporation
;
; Licensed under the Apache License, Version 2.0 (the "License");
; you may not use this file except in compliance with the License.
; You may obtain a copy of the License at
;
;     http://www.apache.org/licenses/LICENSE-2.0
;
; Unless required by applicable law or agreed to in writing, software
; distributed under the License is distributed on an "AS IS" BASIS,
; WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
; See the License for the specific language governing permissions and
; limitations under the License.
;
;
;
;

; DO NOT EDIT - AUTOMATICALLY GENERATED FROM .s FILE
.code 
	ALIGN 8
	PUBLIC __TBB_machine_fetchadd1
__TBB_machine_fetchadd1:
	mov rax,rdx
	lock xadd [rcx],al
	ret
.code 
	ALIGN 8
	PUBLIC __TBB_machine_fetchstore1
__TBB_machine_fetchstore1:
	mov rax,rdx
	lock xchg [rcx],al
	ret
.code 
	ALIGN 8
	PUBLIC __TBB_machine_cmpswp1
__TBB_machine_cmpswp1:
	mov rax,r8
	lock cmpxchg [rcx],dl
	ret
.code 
	ALIGN 8
	PUBLIC __TBB_machine_fetchadd2
__TBB_machine_fetchadd2:
	mov rax,rdx
	lock xadd [rcx],ax
	ret
.code 
	ALIGN 8
	PUBLIC __TBB_machine_fetchstore2
__TBB_machine_fetchstore2:
	mov rax,rdx
	lock xchg [rcx],ax
	ret
.code 
	ALIGN 8
	PUBLIC __TBB_machine_cmpswp2
__TBB_machine_cmpswp2:
	mov rax,r8
	lock cmpxchg [rcx],dx
	ret
.code
        ALIGN 8
        PUBLIC __TBB_machine_pause
__TBB_machine_pause:
L1:
        dw 090f3H; pause
        add ecx,-1
        jne L1
        ret
end

