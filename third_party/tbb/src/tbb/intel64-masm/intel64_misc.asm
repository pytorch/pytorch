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
	PUBLIC __TBB_get_cpu_ctl_env
__TBB_get_cpu_ctl_env:
    stmxcsr [rcx]
    fstcw   [rcx+4]
	ret
.code
	ALIGN 8
	PUBLIC __TBB_set_cpu_ctl_env
__TBB_set_cpu_ctl_env:
    ldmxcsr [rcx]
    fldcw   [rcx+4]
	ret
end
