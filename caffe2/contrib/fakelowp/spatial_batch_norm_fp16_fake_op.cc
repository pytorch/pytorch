#include "spatial_batch_norm_fp16_fake_op.h"

#include <array>

#include "caffe2/utils/eigen_utils.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(SpatialBNFakeLoweredFp16NNPI, SpatialBNFakeLoweredFp16Op);
REGISTER_CPU_OPERATOR(SpatialBNFakeFp16NNPI, SpatialBNFakeFp16Op);

} // namespace caffe2
