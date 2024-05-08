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

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dlnnapi.h"

#define DLNNAPI_DEBUG_LOG 0
#if DLNNAPI_DEBUG_LOG
#include <android/log.h>
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "NNAPI", __VA_ARGS__)
#endif

#define TAG_API_27 "\x01"

/* clang-format off */
static const char function_names[] =
    TAG_API_27 "ANeuralNetworksMemory_createFromFd\0"
    TAG_API_27 "ANeuralNetworksMemory_free\0"
    TAG_API_27 "ANeuralNetworksModel_create\0"
    TAG_API_27 "ANeuralNetworksModel_finish\0"
    TAG_API_27 "ANeuralNetworksModel_free\0"
    TAG_API_27 "ANeuralNetworksCompilation_create\0"
    TAG_API_27 "ANeuralNetworksCompilation_free\0"
    TAG_API_27 "ANeuralNetworksCompilation_setPreference\0"
    TAG_API_27 "ANeuralNetworksCompilation_finish\0"
    TAG_API_27 "ANeuralNetworksModel_addOperand\0"
    TAG_API_27 "ANeuralNetworksModel_setOperandValue\0"
    TAG_API_27 "ANeuralNetworksModel_setOperandValueFromMemory\0"
    TAG_API_27 "ANeuralNetworksModel_addOperation\0"
    TAG_API_27 "ANeuralNetworksModel_identifyInputsAndOutputs\0"
    TAG_API_27 "ANeuralNetworksExecution_create\0"
    TAG_API_27 "ANeuralNetworksExecution_free\0"
    TAG_API_27 "ANeuralNetworksExecution_setInput\0"
    TAG_API_27 "ANeuralNetworksExecution_setInputFromMemory\0"
    TAG_API_27 "ANeuralNetworksExecution_setOutput\0"
    TAG_API_27 "ANeuralNetworksExecution_setOutputFromMemory\0"
    TAG_API_27 "ANeuralNetworksExecution_startCompute\0"
    TAG_API_27 "ANeuralNetworksEvent_wait\0"
    TAG_API_27 "ANeuralNetworksEvent_free\0";
/* clang-format on */

bool dlnnapi_load(struct dlnnapi* nnapi, uint32_t flags) {
  if (nnapi == NULL) {
    return false;
  }

  memset(nnapi, 0, sizeof(struct dlnnapi));
  if (!(flags & DLNNAPI_FLAG_VERSION_27)) {
    /* No supported NNAPI version is requested */
    return false;
  }

  /* Clear libdl error state */
  dlerror();

  nnapi->handle = dlopen("libneuralnetworks.so", RTLD_LAZY | RTLD_LOCAL);
  if (nnapi->handle != NULL) {
#if DLNNAPI_DEBUG_LOG
    LOGI("note: loaded libneuralnetworks.so\n");
#endif

    uint8_t version_flags = (uint8_t)(flags & DLNNAPI_FLAG_VERSION_MASK);
    const char* function_name = function_names;
    for (size_t i = 0; i < DLNNAPI_FUNCTION_COUNT; i++) {
      const uint8_t tag = (uint8_t)*function_name++;
      if ((tag & version_flags) != 0) {
        void* function = dlsym(nnapi->handle, function_name);
        if (function == NULL) {
#if DLNNAPI_DEBUG_LOG
          LOGI(
              "note: failed to locate %s in libneuralnetworks.so: %s\n",
              function_name,
              dlerror());
#endif
          version_flags &= ~tag;
          if (version_flags == 0) {
            goto failed;
          }
        }
        nnapi->functions[i] = function;
      }

      function_name += strlen(function_name) + 1;
    }
    nnapi->flags = (uint32_t)version_flags;

    return true;
  }
#if DLNNAPI_DEBUG_LOG
  LOGI("note: failed to load libneuralnetworks.so: %s\n", dlerror());
#endif

failed:
  dlnnapi_free(nnapi);
  return false;
}

void dlnnapi_free(struct dlnnapi* nnapi) {
  if (nnapi != NULL) {
    if (nnapi->handle != NULL) {
      /* Clear libdl error state */
      dlerror();
      if (dlclose(nnapi->handle) != 0) {
#if DLNNAPI_DEBUG_LOG
        LOGI("note: failed to unload libneuralnetworks.so: %s\n", dlerror());
#endif
      }
    }
    memset(nnapi, 0, sizeof(struct dlnnapi));
  }
}
