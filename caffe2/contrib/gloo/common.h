#pragma once

#include <exception>

#include "caffe2/core/blob.h"

namespace caffe2 {

void signalFailure(Blob* status_blob, std::exception& exception);
}
