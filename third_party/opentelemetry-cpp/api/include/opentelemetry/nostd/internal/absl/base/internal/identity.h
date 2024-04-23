// Copyright 2017 The Abseil Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#ifndef OTABSL_BASE_INTERNAL_IDENTITY_H_
#define OTABSL_BASE_INTERNAL_IDENTITY_H_

#include "../config.h"

namespace absl {
OTABSL_NAMESPACE_BEGIN
namespace internal {

template <typename T>
struct identity {
  typedef T type;
};

template <typename T>
using identity_t = typename identity<T>::type;

}  // namespace internal
OTABSL_NAMESPACE_END
}  // namespace absl

#endif  // OTABSL_BASE_INTERNAL_IDENTITY_H_
