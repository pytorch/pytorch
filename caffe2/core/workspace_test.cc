#include <iostream>

#include "caffe2/core/operator.h"
#include "gtest/gtest.h"


namespace caffe2 {

class Foo {};

TEST(WorkspaceTest, BlobAccess) {
  Workspace ws;

  EXPECT_FALSE(ws.HasBlob("nonexisting"));
  EXPECT_EQ(ws.GetBlob("nonexisting"), nullptr);

  EXPECT_EQ(ws.GetBlob("newblob"), nullptr);
  EXPECT_NE(nullptr, ws.CreateBlob("newblob"));
  EXPECT_NE(nullptr, ws.GetBlob("newblob"));
  EXPECT_TRUE(ws.HasBlob("newblob"));

  // Different names should still be not created.
  EXPECT_FALSE(ws.HasBlob("nonexisting"));
  EXPECT_EQ(ws.GetBlob("nonexisting"), nullptr);

  // Check if the returned Blob is OK for all operations
  Blob* blob = ws.GetBlob("newblob");
  int* int_unused UNUSED_VARIABLE = blob->GetMutable<int>();
  EXPECT_TRUE(blob->IsType<int>());
  EXPECT_FALSE(blob->IsType<Foo>());
  EXPECT_NE(&blob->Get<int>(), nullptr);

  // Re-creating the blob does not change the content as long as it already
  // exists.
  EXPECT_NE(nullptr, ws.CreateBlob("newblob"));
  EXPECT_TRUE(blob->IsType<int>());
  EXPECT_FALSE(blob->IsType<Foo>());
  // When not null, we should only call with the right type.
  EXPECT_NE(&blob->Get<int>(), nullptr);
}

TEST(WorkspaceTest, RunEmptyPlan) {
  PlanDef plan_def;
  Workspace ws;
  EXPECT_TRUE(ws.RunPlan(plan_def));
}

}  // namespace caffe2


