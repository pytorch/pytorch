#include <gtest/gtest.h>

#if !CAFFE2_MOBILE
#include "caffe2/onnx/torch_ops/constants.h"
#include "onnx/defs/schema.h"

namespace caffe2 {
namespace {

using namespace ONNX_NAMESPACE;

TEST(ONNXSchemaTest, TestGetOpSchema) {
  {
      auto* schema = OpSchemaRegistry::Schema("NON_EXIST",
                                              AI_ONNX_PYTORCH_DOMAIN_MAX_OPSET,
                                              AI_ONNX_PYTORCH_DOMAIN);
      ASSERT_EQ(nullptr, schema);
  }

  {
      auto* schema = OpSchemaRegistry::Schema("DUMMY_TEST_ONLY",
                                              AI_ONNX_PYTORCH_DOMAIN_MAX_OPSET,
                                              AI_ONNX_PYTORCH_DOMAIN);
      ASSERT_NE(nullptr, schema);
      EXPECT_EQ(schema->Name(), "DUMMY_TEST_ONLY");
      EXPECT_EQ(schema->min_input(), 1);
      EXPECT_EQ(schema->max_input(), 1);
      EXPECT_EQ(schema->min_output(), 1);
      EXPECT_EQ(schema->max_output(), 1);
  }
}

} // namespace
} // namespace caffe2
#endif
