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


#include "GLImage.h"
#include "arm_neon_support.h"
#include "caffe2/core/typeid.h"

namespace caffe2 {
CAFFE_KNOWN_TYPE(GLImage<float>);
CAFFE_KNOWN_TYPE(GLImage<uint8_t>);
CAFFE_KNOWN_TYPE(GLImageVector<float>);
CAFFE_KNOWN_TYPE(GLImageVector<uint8_t>);
#ifdef __ARM_NEON__
CAFFE_KNOWN_TYPE(GLImage<float16_t>);
CAFFE_KNOWN_TYPE(GLImageVector<float16_t>);
#endif
} // namespace caffe2
