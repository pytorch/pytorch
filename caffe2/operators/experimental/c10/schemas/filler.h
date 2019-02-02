#pragma once

#include <ATen/core/dispatch/OpSchemaRegistration.h>

namespace caffe2 {
namespace ops {

C10_DECLARE_OP_SCHEMA(GivenTensorFill);
C10_DECLARE_OP_SCHEMA(GivenTensorIntFill);
C10_DECLARE_OP_SCHEMA(GivenTensorInt64Fill);
C10_DECLARE_OP_SCHEMA(ConstantFill);
C10_DECLARE_OP_SCHEMA(UniformFill);

} // namespace ops
} // namespace caffe2
