#include "caffe2/core/common.h"
#include "caffe2/core/test_utils.h"
#include "caffe2/opt/converter.h"
#include "caffe2/opt/custom/concat_elim.h"
#include "caffe2/predictor/emulator/data_filler.h"
#include "caffe2/utils/proto_utils.h"

#include <gtest/gtest.h>

using namespace caffe2::testing;
using namespace caffe2::emulator;
using caffe2::OperatorDef;
using std::vector;

TEST(Converter, ClipRangesGatherSigridHashConverter) {
  OperatorDef op;
  op.set_type("ClipRangesGatherSigridHash");
  op.add_arg()->CopyFrom(caffe2::MakeArgument<bool>("hash_into_int32", true));
  auto nnDef = convertToNeuralNetOperator(op);
  auto* pNNDef =
      static_cast<nom::repr::ClipRangesGatherSigridHash*>(nnDef.get());
  EXPECT_TRUE(pNNDef);
  EXPECT_TRUE(pNNDef->getHashIntoInt32());

  OperatorDef op2;
  op2.set_type("ClipRangesGatherSigridHash");
  op2.add_arg()->CopyFrom(caffe2::MakeArgument<bool>("hash_into_int32", false));
  auto nnDef2 = convertToNeuralNetOperator(op2);
  auto* pNNDef2 =
      static_cast<nom::repr::ClipRangesGatherSigridHash*>(nnDef2.get());
  EXPECT_TRUE(pNNDef2);
  EXPECT_FALSE(pNNDef2->getHashIntoInt32());
}
