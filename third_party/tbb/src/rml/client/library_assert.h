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

#ifndef LIBRARY_ASSERT_H
#define LIBRARY_ASSERT_H

#ifndef  LIBRARY_ASSERT
#ifdef KMP_ASSERT2
#define LIBRARY_ASSERT(x,y) KMP_ASSERT2((x),(y))
#else
#include <assert.h>
#define LIBRARY_ASSERT(x,y)         assert(x)
#define __TBB_DYNAMIC_LOAD_ENABLED  1
#endif
#endif /* LIBRARY_ASSERT */

#endif /* LIBRARY_ASSERT_H */
