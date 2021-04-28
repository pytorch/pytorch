#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "caffe2/utils/cast.h"

namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(CastTest, GetCastDataType) {
  auto castOp = [](std::string t) {
    // Ensure lowercase.
    std::transform(t.begin(), t.end(), t.begin(), ::tolower);
    auto op = CreateOperatorDef("Cast", "", {}, {});
    AddArgument("to", t, &op);
    return op;
  };

#define X(t)                    \
  EXPECT_EQ(                    \
      TensorProto_DataType_##t, \
      cast::GetCastDataType(ArgumentHelper(castOp(#t)), "to"));

  X(FLOAT);
  X(INT32);
  X(BYTE);
  X(STRING);
  X(BOOL);
  X(UINT8);
  X(INT8);
  X(UINT16);
  X(INT16);
  X(INT64);
  X(FLOAT16);
  X(DOUBLE);
#undef X
}

} // namespace caffe2
