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
#include "caffe2/core/hip/common_miopen.h"
#include "caffe2/core/hip/miopen_wrapper.h"

#include "caffe2/core/init.h"

namespace caffe2 {

MIOPENWrapper::PerGPUMIOPENStates& MIOPENWrapper::miopen_states()
{
    // New it (never delete) to avoid calling the destructors on process
    // exit and racing against the CUDA shutdown sequence.
    static auto* p = new MIOPENWrapper::PerGPUMIOPENStates();
    CHECK_NOTNULL(p);
    return *p;
}

namespace {
bool PrintMIOPENInfo(int*, char***)
{
    VLOG(1) << "Caffe2 is built with MIOPEN version " << MIOPEN_VERSION;
    return true;
}

REGISTER_CAFFE2_INIT_FUNCTION(PrintMIOPENInfo, &PrintMIOPENInfo, "Print MIOPEN Info.");

} // namespace
} // namespace caffe2
