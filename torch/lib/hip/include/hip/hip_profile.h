/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef HIP_INCLUDE_HIP_HIP_PROFILE_H
#define HIP_INCLUDE_HIP_HIP_PROFILE_H

#if not defined (ENABLE_HIP_PROFILE)
#define ENABLE_HIP_PROFILE 1
#endif

#if defined(__HIP_PLATFORM_HCC__) and (ENABLE_HIP_PROFILE==1)
#include <CXLActivityLogger.h>
#define HIP_SCOPED_MARKER(markerName, group) amdtScopedMarker __scopedMarker(markerName, group, nullptr);
#define HIP_BEGIN_MARKER(markerName, group) amdtBeginMarker(markerName, group, nullptr);
#define HIP_END_MARKER() amdtEndMarker();
#else
#define HIP_SCOPED_MARKER(markerName, group)
#define HIP_BEGIN_MARKER(markerName, group)
#define HIP_END_MARKER()
#endif

#endif
