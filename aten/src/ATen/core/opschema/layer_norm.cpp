#include <ATen/core/opschema/layer_norm.h>
#include <ATen/core/dispatch/OpSchemaRegistration.h>

C10_DEFINE_OP_SCHEMA(c10::core::opschema::LayerNorm);

namespace caffe2 {
CAFFE_KNOWN_TYPE(c10::core::opschema::LayerNorm::Cache);
}
