#include "caffe2/core/operator_c10wrapper.h"

C10_DEFINE_REGISTRY(
    C10OperatorRegistry,
    caffe2::OperatorBase,
    const caffe2::OperatorDef&,
    caffe2::Workspace*);
