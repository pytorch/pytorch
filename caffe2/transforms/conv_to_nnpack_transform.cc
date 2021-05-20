#include "caffe2/transforms/conv_to_nnpack_transform.h"

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_TRANSFORM(ConvToNNPack, ConvToNNPackTransform);

} // namespace caffe2
