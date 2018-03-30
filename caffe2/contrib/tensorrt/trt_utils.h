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

#pragma once

#include "caffe2/core/logging.h"
#include <NvInfer.h>
#include <iostream>

namespace caffe2 {

  // Logger for GIE info/warning/errors
class TrtLogger : public nvinfer1::ILogger {
  using nvinfer1::ILogger::Severity;

 public:
  TrtLogger(
      Severity verbosity = Severity::kWARNING,
      std::ostream& ostream = std::cout)
      : _verbosity(verbosity), _ostream(&ostream) {}
  void log(Severity severity, const char* msg) override {
    if (severity <= _verbosity) {
      std::string sevstr =
          (severity == Severity::kINTERNAL_ERROR
               ? "INTERNAL ERROR"
               : severity == Severity::kERROR ? "  ERROR"
                                              : severity == Severity::kWARNING
                       ? "WARNING"
                       : severity == Severity::kINFO ? "   INFO" : "UNKNOWN");
      (*_ostream) << "[" << sevstr << "] " << msg << std::endl;
    }
  }

 private:
  Severity _verbosity;
  std::ostream* _ostream;
};

struct InferDeleter {
  template <typename T>
  void operator()(T* obj) const {
    if (obj) {
      obj->destroy();
    }
  }
};

template <typename T>
inline std::shared_ptr<T> InferObject(T* obj) {
  CAFFE_ENFORCE(obj, "Failed to create TensorRt object");
  return std::shared_ptr<T>(obj, InferDeleter());
}


}

