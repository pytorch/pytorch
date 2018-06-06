// This file defines classes that hold profiling information for
// NeuralNetOperator and NeuralNetData.
#pragma once

#include "caffe2/contrib/prof/prof_dag_net.h"
#include "caffe2/core/nomnigraph/include/nomnigraph/Representations/NeuralNet.h"

using nom::repr::Annotation;

namespace caffe2 {

// Annotations used when profiling a NeuralNetOperator.
class ProfilingOperatorAnnotation : public Annotation {
 public:
  ProfilingOperatorAnnotation()
      : Annotation(AnnotationKind::ProfilingOperator) {}
  // LLVM-style RTTI implementation.
  static bool classof(const Annotation* annotation) {
    return annotation->getKind() == AnnotationKind::ProfilingOperator;
  }
  // Accessors
  const Stats& execution_time_ms() const {
    return execution_time_ms_;
  }
  Stats* mutable_execution_time_ms() {
    return &execution_time_ms_;
  }

 private:
  // Statistics for how long this op took to execute.
  Stats execution_time_ms_;
};

// Annotations used when profiling a NeuralNetData.
class ProfilingDataAnnotation : public Annotation {
 public:
  ProfilingDataAnnotation() : Annotation(AnnotationKind::ProfilingData) {}
  // LLVM-style RTTI implementation.
  static bool classof(const Annotation* annotation) {
    return annotation->getKind() == AnnotationKind::ProfilingData;
  }
  // Accessors
  const Stats& used_bytes() const {
    return used_bytes_;
  }
  Stats* mutable_used_bytes() {
    return &used_bytes_;
  }

 private:
  // Statistics for how much data this tensor/parameter used (per invocation of
  // the op that generated the data).
  Stats used_bytes_;
};

} // namespace caffe2
