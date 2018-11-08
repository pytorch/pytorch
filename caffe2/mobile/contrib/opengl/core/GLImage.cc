
#include "GLImage.h"
#include "arm_neon_support.h"
#include "caffe2/core/typeid.h"

namespace caffe2 {
CAFFE_KNOWN_TYPE(GLImage<float>);
CAFFE_KNOWN_TYPE(GLImage<uint8_t>);
CAFFE_KNOWN_TYPE(GLImageVector<float>);
CAFFE_KNOWN_TYPE(GLImageVector<uint8_t>);
#ifdef __ARM_NEON__
CAFFE_KNOWN_TYPE(GLImage<float16_t>);
CAFFE_KNOWN_TYPE(GLImageVector<float16_t>);
#endif
} // namespace caffe2
