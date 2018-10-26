#include <ATen/core/blob.h>

caffe2::Blob::~Blob() {
  Reset();
}
