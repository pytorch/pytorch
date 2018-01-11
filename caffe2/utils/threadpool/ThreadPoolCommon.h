/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CAFFE2_UTILS_THREADPOOL_COMMON_H_
#define CAFFE2_UTILS_THREADPOOL_COMMON_H_

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

// caffe2 depends upon NNPACK, which depends upon this threadpool, so
// unfortunately we can't reference core/common.h here

// This is copied from core/common.h's definition of CAFFE2_MOBILE
// Define enabled when building for iOS or Android devices
#if defined(__ANDROID__)
#define CAFFE2_ANDROID 1
#elif (defined(__APPLE__) &&                                            \
       (TARGET_IPHONE_SIMULATOR || TARGET_OS_SIMULATOR || TARGET_OS_IPHONE))
#define CAFFE2_IOS 1
#elif (defined(__APPLE__) && TARGET_OS_MAC)
#define CAFFE2_IOS 1
#else
#endif // ANDROID / IOS / MACOS

#endif  // CAFFE2_UTILS_THREADPOOL_COMMON_H_
