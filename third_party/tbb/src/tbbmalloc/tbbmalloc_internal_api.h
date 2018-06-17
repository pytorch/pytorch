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

#ifndef __TBB_tbbmalloc_internal_api_H
#define __TBB_tbbmalloc_internal_api_H

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

typedef enum {
    /* Tune usage of source included allocator. Selected value is large enough
       to not intercept with constants from AllocationModeParam. */
    TBBMALLOC_INTERNAL_SOURCE_INCLUDED = 65536
} AllocationModeInternalParam;

void MallocInitializeITT();
void __TBB_mallocProcessShutdownNotification();
#if _WIN32||_WIN64
void __TBB_mallocThreadShutdownNotification();
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#endif /* __TBB_tbbmalloc_internal_api_H */
