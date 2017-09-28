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

#include "caffe2/core/init.h"
#include "caffe2/core/flags.h"

#include "nervana_c_api.h"


CAFFE2_DEFINE_string(nervana_cubin_path,
                     "/usr/local/fbcode/gcc-4.8.1-glibc-2.17/lib/cubin/",
                     "The cubin path for nervana kernels. Currently defaulted "
                     "to the internal fb deployment path.");

namespace caffe2 {

namespace {
static bool g_nervana_kernel_loaded = false;
}  // namespace

bool NervanaKernelLoaded() { return g_nervana_kernel_loaded; }

bool Caffe2InitializeNervanaKernels(int*, char***) {
  // If we do not specify the nervana cubin path, we will simply return.
  if (FLAGS_nervana_cubin_path.size() == 0) {
    VLOG(1) << "Nervana cubin loading skipped.";
    return true;
  }
  g_nervana_kernel_loaded =
      nervana_loadKernels(FLAGS_nervana_cubin_path.c_str());
  if (g_nervana_kernel_loaded) {
    VLOG(1) << "Loaded nervana kernels from path "
            << FLAGS_nervana_cubin_path;
  } else {
    // Since this is not a critical error we will just vlog it.
    VLOG(1) << "Cannot load nervana gpu kernels from path "
            << FLAGS_nervana_cubin_path
            << ", will disable Caffe2 nervana engines.";
  }
  // We will always return true for this initialization, because the loading
  // result is kept and accessible via NervanaKernelLoaded(). This allows us
  // to register an init function but not forcing the user to have to install
  // nervana kernels, delaying the failure to the first time a nervana kernel
  // is actually called.
  return true;
}

REGISTER_CAFFE2_INIT_FUNCTION(Caffe2InitializeNervanaKernels,
                              &Caffe2InitializeNervanaKernels,
                              "Initialize nervana kernels for caffe2.");
}  // namespace caffe2
