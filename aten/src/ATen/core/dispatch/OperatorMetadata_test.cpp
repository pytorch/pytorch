#include <gtest/gtest.h>

#include <ATen/core/op_registration/op_registration.h>
#include <ATen/core/dispatch/OperatorMetadata.h>

using c10::RegisterOperators;
using c10::Dispatcher;
using c10::set_op_metadata;
using c10::get_op_metadata;

namespace {

struct MyMetadata final {
  int value;
};
struct MyMetadata2 final {
  int value;
};
}

TORCH_DEFINE_OPERATOR_METADATA_TYPE(MyMetadata);
TORCH_DEFINE_OPERATOR_METADATA_TYPE(MyMetadata2);

namespace {

TEST(OperatorMetadataTest, givenOp_whenSettingAndGettingMetadata_thenValueIsCorrect) {
  auto registry = RegisterOperators().op("my::op() -> ()");

  auto op = Dispatcher::singleton().findSchema("my::op", "").value();

  set_op_metadata<MyMetadata>(op, MyMetadata{1});

  EXPECT_EQ(1, get_op_metadata<MyMetadata>(op).value()->value);
}

TEST(OperatorMetadataTest, givenOp_whenGettingMetadata_thenValueIsEmpty) {
  auto registry = RegisterOperators().op("my::op() -> ()");

  auto op = Dispatcher::singleton().findSchema("my::op", "").value();

  EXPECT_FALSE(get_op_metadata<MyMetadata>(op).has_value());
}

TEST(OperatorMetadataTest, givenOp_whenGettingSettingAndGettingMetadata_thenValueIsEmpty) {
  auto registry = RegisterOperators().op("my::op() -> ()");

  auto op = Dispatcher::singleton().findSchema("my::op", "").value();

  EXPECT_FALSE(get_op_metadata<MyMetadata>(op).has_value());
  set_op_metadata<MyMetadata>(op, MyMetadata{1});
  EXPECT_EQ(1, get_op_metadata<MyMetadata>(op).value()->value);
}

TEST(OperatorMetadataTest, givenMultipleOps_whenSettingAndGettingMetadata_thenValueIsCorrect) {
  auto registry = RegisterOperators()
    .op("my::op1() -> ()")
    .op("my::op2() -> ()")
    .op("my::op3() -> ()");

  auto op1 = Dispatcher::singleton().findSchema("my::op1", "").value();
  auto op2 = Dispatcher::singleton().findSchema("my::op2", "").value();
  auto op3 = Dispatcher::singleton().findSchema("my::op3", "").value();

  set_op_metadata<MyMetadata>(op1, MyMetadata{1});
  set_op_metadata<MyMetadata>(op2, MyMetadata{2});

  EXPECT_EQ(1, get_op_metadata<MyMetadata>(op1).value()->value);
  EXPECT_EQ(2, get_op_metadata<MyMetadata>(op2).value()->value);
  EXPECT_FALSE(get_op_metadata<MyMetadata>(op3).has_value());
}

TEST(OperatorMetadataTest, whenOpGoesOutOfScope_thenMetadataIsDeleted) {
  auto registry = c10::guts::make_unique<RegisterOperators>(RegisterOperators().op("my::op() -> ()"));
  auto op = Dispatcher::singleton().findSchema("my::op", "").value();
  set_op_metadata<MyMetadata>(op, MyMetadata{1});

  EXPECT_TRUE(get_op_metadata<MyMetadata>(op).has_value());

  registry.reset();
  EXPECT_FALSE(get_op_metadata<MyMetadata>(op).has_value());
}

TEST(OperatorMetadataTest, givenMultipleMetadatas_whenSettingAndGettingMetadata_thenValueIsCorrect) {
  auto registry = RegisterOperators()
    .op("my::op() -> ()");

  auto op = Dispatcher::singleton().findSchema("my::op", "").value();

  set_op_metadata<MyMetadata>(op, MyMetadata{1});
  set_op_metadata<MyMetadata2>(op, MyMetadata2{2});

  EXPECT_EQ(1, get_op_metadata<MyMetadata>(op).value()->value);
  EXPECT_EQ(2, get_op_metadata<MyMetadata2>(op).value()->value);
}

}
