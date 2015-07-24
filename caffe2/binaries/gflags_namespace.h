// This file exists solely for the purpose of addressing legacy gflags namespace
// issues.

#ifndef CAFFE2_BINARIES_GFLAGS_NAMESPACE_H_
#define CAFFE2_BINARIES_GFLAGS_NAMESPACE_H_

#include "gflags/gflags.h"

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

#endif  // CAFFE2_BINARIES_GFLAGS_NAMESPACE_H_