#include "caffe2/operators/data_couple.h"

namespace caffe2 {
REGISTER_CPU_OPERATOR(DataCouple, DataCoupleOp<CPUContext>);

OPERATOR_SCHEMA(DataCouple)
    .EnforceOneToOneInplace()
    .SetDoc(R"DOC(

A one to one operator that takes an arbitrary number of input and output blobs
such that each input blob is inplace with it's matching output blob. It then proceeds
to do nothing with each of these operators. This serves two purposes. It can make it
appear as if a blob has been written to, as well as can tie together different blobs
in a data dependency

)DOC");
} // namespace caffe2
