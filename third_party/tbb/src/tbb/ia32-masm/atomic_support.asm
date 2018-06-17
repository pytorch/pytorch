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

.686
.model flat,c
.code 
	ALIGN 4
	PUBLIC c __TBB_machine_fetchadd1
__TBB_machine_fetchadd1:
	mov edx,4[esp]
	mov eax,8[esp]
	lock xadd [edx],al
	ret
.code 
	ALIGN 4
	PUBLIC c __TBB_machine_fetchstore1
__TBB_machine_fetchstore1:
	mov edx,4[esp]
	mov eax,8[esp]
	lock xchg [edx],al
	ret
.code 
	ALIGN 4
	PUBLIC c __TBB_machine_cmpswp1
__TBB_machine_cmpswp1:
	mov edx,4[esp]
	mov ecx,8[esp]
	mov eax,12[esp]
	lock cmpxchg [edx],cl
	ret
.code 
	ALIGN 4
	PUBLIC c __TBB_machine_fetchadd2
__TBB_machine_fetchadd2:
	mov edx,4[esp]
	mov eax,8[esp]
	lock xadd [edx],ax
	ret
.code 
	ALIGN 4
	PUBLIC c __TBB_machine_fetchstore2
__TBB_machine_fetchstore2:
	mov edx,4[esp]
	mov eax,8[esp]
	lock xchg [edx],ax
	ret
.code 
	ALIGN 4
	PUBLIC c __TBB_machine_cmpswp2
__TBB_machine_cmpswp2:
	mov edx,4[esp]
	mov ecx,8[esp]
	mov eax,12[esp]
	lock cmpxchg [edx],cx
	ret
.code 
	ALIGN 4
	PUBLIC c __TBB_machine_fetchadd4
__TBB_machine_fetchadd4:
	mov edx,4[esp]
	mov eax,8[esp]
	lock xadd [edx],eax
	ret
.code 
	ALIGN 4
	PUBLIC c __TBB_machine_fetchstore4
__TBB_machine_fetchstore4:
	mov edx,4[esp]
	mov eax,8[esp]
	lock xchg [edx],eax
	ret
.code 
	ALIGN 4
	PUBLIC c __TBB_machine_cmpswp4
__TBB_machine_cmpswp4:
	mov edx,4[esp]
	mov ecx,8[esp]
	mov eax,12[esp]
	lock cmpxchg [edx],ecx
	ret
.code 
	ALIGN 4
	PUBLIC c __TBB_machine_fetchadd8
__TBB_machine_fetchadd8:
	push ebx
	push edi
	mov edi,12[esp]
	mov eax,[edi]
	mov edx,4[edi]
__TBB_machine_fetchadd8_loop:
	mov ebx,16[esp]
	mov ecx,20[esp]
	add ebx,eax
	adc ecx,edx
	lock cmpxchg8b qword ptr [edi]
	jnz __TBB_machine_fetchadd8_loop
	pop edi
	pop ebx
	ret
.code 
	ALIGN 4
	PUBLIC c __TBB_machine_fetchstore8
__TBB_machine_fetchstore8:
	push ebx
	push edi
	mov edi,12[esp]
	mov ebx,16[esp]
	mov ecx,20[esp]
	mov eax,[edi]
	mov edx,4[edi]
__TBB_machine_fetchstore8_loop:
	lock cmpxchg8b qword ptr [edi]
	jnz __TBB_machine_fetchstore8_loop
	pop edi
	pop ebx
	ret
.code 
	ALIGN 4
	PUBLIC c __TBB_machine_cmpswp8
__TBB_machine_cmpswp8:
	push ebx
	push edi
	mov edi,12[esp]
	mov ebx,16[esp]
	mov ecx,20[esp]
	mov eax,24[esp]
	mov edx,28[esp]
	lock cmpxchg8b qword ptr [edi]
	pop edi
	pop ebx
	ret
.code 
	ALIGN 4
	PUBLIC c __TBB_machine_load8
__TBB_machine_Load8:
	; If location is on stack, compiler may have failed to align it correctly, so we do dynamic check.
	mov ecx,4[esp]
	test ecx,7
	jne load_slow
	; Load within a cache line
	sub esp,12
	fild qword ptr [ecx]
	fistp qword ptr [esp]
	mov eax,[esp]
	mov edx,4[esp]
	add esp,12
	ret
load_slow:
	; Load is misaligned. Use cmpxchg8b.
	push ebx
	push edi
	mov edi,ecx
	xor eax,eax
	xor ebx,ebx
	xor ecx,ecx
	xor edx,edx
	lock cmpxchg8b qword ptr [edi]
	pop edi
	pop ebx
	ret
EXTRN __TBB_machine_store8_slow:PROC
.code 
	ALIGN 4
	PUBLIC c __TBB_machine_store8
__TBB_machine_Store8:
	; If location is on stack, compiler may have failed to align it correctly, so we do dynamic check.
	mov ecx,4[esp]
	test ecx,7
	jne __TBB_machine_store8_slow ;; tail call to tbb_misc.cpp
	fild qword ptr 8[esp]
	fistp qword ptr [ecx]
	ret
end
