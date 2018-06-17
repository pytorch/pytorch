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
.686
.model flat,c
.code 
	ALIGN 4
	PUBLIC c __TBB_machine_trylockbyte
__TBB_machine_trylockbyte:
	mov edx,4[esp]
	mov al,[edx]
	mov cl,1
	test al,1
	jnz __TBB_machine_trylockbyte_contended
	lock cmpxchg [edx],cl
	jne __TBB_machine_trylockbyte_contended
	mov eax,1
	ret
__TBB_machine_trylockbyte_contended:
	xor eax,eax
	ret
end
