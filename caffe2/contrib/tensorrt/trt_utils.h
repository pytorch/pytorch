#pragma once

#include "caffe2/core/logging.h"
#include <NvInfer.h>
#include <iostream>

namespace caffe2 { namespace tensorrt {

  // Logger for GIE info/warning/errors
class TrtLogger : public nvinfer1::ILogger {
  using nvinfer1::ILogger::Severity;

 public:
  TrtLogger(Severity verbosity = Severity::kWARNING) : _verbosity(verbosity) {}
  void log(Severity severity, const char* msg) override {
    if (severity <= _verbosity) {
      if (severity == Severity::kINTERNAL_ERROR || severity == Severity::kERROR) {
        LOG(ERROR) << msg;
      } else if (severity == Severity::kWARNING) {
        LOG(WARNING)  << msg;
      } else if (severity == Severity::kINFO) {
        LOG(INFO) << msg;
      }
    }
  }

 private:
  Severity _verbosity;
};

struct TrtDeleter {
  template <typename T>
  void operator()(T* obj) const {
    if (obj) {
      obj->destroy();
    }
  }
};

template <typename T>
inline std::shared_ptr<T> TrtObject(T* obj) {
  CAFFE_ENFORCE(obj, "Failed to create TensorRt object");
  return std::shared_ptr<T>(obj, TrtDeleter());
}

}
}

