#include <fbgemm/FbgemmConvert.h>

namespace caffe2 {

template <typename T>
void fp16_wrap(T* tmp) {
  fbgemm::RoundToFloat16(tmp, tmp, 1, FLAGS_caffe2_fbgemm_fake_fp16_clamp);
}

} // namespace caffe2
