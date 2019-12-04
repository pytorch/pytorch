#pragma once

#include <c10/util/Half.h>

namespace caffe2 {

void FloatToFloat16(const float* src, at::Half* dst, int size);
void Float16ToFloat(const at::Half* src, float* dst, int size);

} // namespace caffe2
