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

