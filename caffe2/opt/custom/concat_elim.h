#ifndef CAFFE2_OPT_CONCAT_ELIM_H_
#define CAFFE2_OPT_CONCAT_ELIM_H_

#include "caffe2/core/common.h"
#include "caffe2/proto/caffe2_pb.h"
#include "nomnigraph/Representations/NeuralNet.h"

namespace caffe2 {
namespace opt {

void concatElim(nom::repr::NNModule* nn);
void concatAddMulNaNClipElim(nom::repr::NNModule* nn);
void gatherFuse8BitRowwiseQuantFloatMulLengthsSumElim(nom::repr::NNModule* nn);

} // namespace opt
} // namespace caffe2

#endif // CAFFE2_OPT_CONCAT_ELIM_H_
