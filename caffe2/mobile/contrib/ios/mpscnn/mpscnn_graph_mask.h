
#pragma once
#include "caffe2/core/net.h"
#include "mpscnn.h"

namespace caffe2 {
// We currently only try to convert a fixed set of operators that handle a subset of a full
// CNN. We also only run when MPSCNN is available, provides a speedup.
// On failure, returns false. On success, returns true, and sets the MPSCNN net in the output
// parameter.
// The rewrite function now supports insertion of copies in intermediate ops.
bool tryConvertToMPSCNNIntermediateCopies(const NetDef& initNet,
                                          const NetDef& predictNet,
                                          NetDef* mpscnnPredictNet);
NetDef setSpecialArgs(const NetDef& def);
} // namespace caffe2
