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
#include "caffe2/core/operator.h" // for StaticLinkingProtector

#ifndef CAFFE2_BUILD_STRING
#define CAFFE2_BUILD_STRING "build_version_not_set"
#endif
namespace caffe2 {

namespace internal {
Caffe2InitializeRegistry* Caffe2InitializeRegistry::Registry() {
  static Caffe2InitializeRegistry gRegistry;
  return &gRegistry;
}
}

bool GlobalInit(int* pargc, char*** pargv) {
  static bool global_init_was_already_run = false;
  static StaticLinkingProtector g_protector;
  if (global_init_was_already_run) {
    VLOG(1) << "GlobalInit has already been called: did you double-call?";
    return true;
  }
  global_init_was_already_run = true;
  bool success = true;
  success &= internal::Caffe2InitializeRegistry::Registry()
      ->RunRegisteredEarlyInitFunctions(pargc, pargv);
  CAFFE_ENFORCE(success,
                "Failed to run some early init functions for caffe2.");
  success &= ParseCaffeCommandLineFlags(pargc, pargv);
  success &= InitCaffeLogging(pargc, *pargv);
  // Print out the current build version.
  VLOG(1) << "Caffe2 build version: " << CAFFE2_BUILD_STRING;
  // All other initialization functions.
  success &= internal::Caffe2InitializeRegistry::Registry()
      ->RunRegisteredInitFunctions(pargc, pargv);
  if (!success) {
    global_init_was_already_run = false;
  }
  CAFFE_ENFORCE(success,
                "Failed to run some init functions for caffe2.");
  // TODO: if we fail GlobalInit(), should we continue?
  return success;
}
}  // namespace caffe2
