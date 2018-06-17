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

.code
        ALIGN 8
        PUBLIC __TBB_machine_try_lock_elided
__TBB_machine_try_lock_elided:
        xor  rax, rax
        mov  al, 1
        BYTE 0F2H
        xchg al, byte ptr [rcx]
        xor  al, 1
        ret
.code
        ALIGN 8
        PUBLIC __TBB_machine_unlock_elided
__TBB_machine_unlock_elided:
        BYTE 0F3H
        mov  byte ptr [rcx], 0
        ret
.code 
	ALIGN 8
	PUBLIC __TBB_machine_begin_transaction
__TBB_machine_begin_transaction:
        mov  eax, -1
        BYTE 0C7H
        BYTE 0F8H
        BYTE 000H
        BYTE 000H
        BYTE 000H
        BYTE 000H
        ret
.code 
	ALIGN 8
	PUBLIC __TBB_machine_end_transaction
__TBB_machine_end_transaction:
        BYTE 00FH
        BYTE 001H
        BYTE 0D5H
        ret
.code 
	ALIGN 8
	PUBLIC __TBB_machine_transaction_conflict_abort
__TBB_machine_transaction_conflict_abort:
        BYTE 0C6H
        BYTE 0F8H
        BYTE 0FFH  ; 12.4.5 Abort argument: lock not free when tested
        ret
.code 
        ALIGN 8
	PUBLIC __TBB_machine_is_in_transaction
__TBB_machine_is_in_transaction:
        xor eax, eax
        BYTE 00FH  ; _xtest sets or clears ZF
        BYTE 001H
        BYTE 0D6H
        jz   rset
        mov  al,1
rset:
        ret
end
