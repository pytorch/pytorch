#pragma once

#include "caffe2/core/blob.h"

#include <gloo/common/error.h>

namespace caffe2 {

void signalFailure(Blob* status_blob, ::gloo::IoException& exception);
}
