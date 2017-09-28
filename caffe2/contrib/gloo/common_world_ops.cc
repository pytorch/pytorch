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

#include "caffe2/contrib/gloo/common_world_ops.h"

#include <gloo/transport/tcp/device.h>

namespace caffe2 {
namespace gloo {

template <>
void CreateCommonWorld<CPUContext>::initializeForContext() {
  // Nothing to initialize for CPUContext.
}

namespace {

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    CreateCommonWorld,
    GLOO,
    CreateCommonWorld<CPUContext>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    CloneCommonWorld,
    GLOO,
    CloneCommonWorld<CPUContext>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(DestroyCommonWorld, GLOO, DestroyCommonWorld);

} // namespace
} // namespace gloo
} // namespace caffe2
