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

#ifndef CAFFE2_CORE_LOGGING_IS_GOOGLE_GLOG_H_
#define CAFFE2_CORE_LOGGING_IS_GOOGLE_GLOG_H_

#include <iomanip>  // because some of the caffe2 code uses e.g. std::setw
// Using google glog. For glog 0.3.2 versions, stl_logging.h needs to be before
// logging.h to actually use stl_logging. Because template magic.
// In addition, we do not do stl logging in .cu files because nvcc does not like
// it. Some mobile platforms do not like stl_logging, so we add an
// overload in that case as well.

#if !defined(__CUDACC__) && !defined(CAFFE2_USE_MINIMAL_GOOGLE_GLOG)
#include <glog/stl_logging.h>
#else // !defined(__CUDACC__) && !!defined(CAFFE2_USE_MINIMAL_GOOGLE_GLOG)

// here, we need to register a fake overload for vector/string - here,
// we just ignore the entries in the logs.

#define INSTANTIATE_FOR_CONTAINER(container)                                \
  template <class... Types>                                                 \
  std::ostream& operator<<(std::ostream& out, const container<Types...>&) { \
    return out;                                                             \
  }

INSTANTIATE_FOR_CONTAINER(std::vector)
INSTANTIATE_FOR_CONTAINER(std::map)
INSTANTIATE_FOR_CONTAINER(std::set)
#undef INSTANTIATE_FOR_CONTAINER

#endif

#include <glog/logging.h>


#endif  // CAFFE2_CORE_LOGGING_IS_GOOGLE_GLOG_H_
