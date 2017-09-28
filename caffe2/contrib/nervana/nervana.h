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

#ifndef CAFFE2_FB_NERVANA_INIT_H_
#define CAFFE2_FB_NERVANA_INIT_H_

#include "caffe2/core/init.h"
#include "caffe2/core/flags.h"

#include "nervana_c_api.h"

/**
 * A flag that specifies the nervana cubin path.
 */
CAFFE2_DECLARE_string(nervana_cubin_path);

namespace caffe2 {

/**
 * An empty class to be used in identifying the engine in the math functions.
 */
class NervanaEngine {};

/**
 * Returns whether the nervana kernels are loaded or not.
 */
bool NervanaKernelLoaded();

/**
 * An initialization function that is run once by caffe2::GlobalInit()
 * that initializes the nervana kernels.
 */
bool Caffe2InitializeNervanaKernels();

}  // namespace caffe2

#endif  // CAFFE2_FB_NERVANA_INIT_H_
