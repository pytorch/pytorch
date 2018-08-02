#include "caffe2/core/active_workspace.h"
#include <gtest/gtest.h>

namespace caffe2 {

TEST(WorkspaceTest, ActiveWorkspace) {
  Workspace ws;

  {
    auto capture = [&]() { return ActiveWorkspace(&ws); };

    auto aws = capture();
    int calls = 0;
    ActiveWorkspace::ForEach([&](Workspace* _ws) {
      EXPECT_EQ(_ws, &ws);
      calls++;
    });
    EXPECT_EQ(calls, 1);

#if GTEST_HAS_DEATH_TEST
    EXPECT_DEATH(capture(), "Workspace is already borrowed as active!");
#endif
  }

  int calls = 0;
  ActiveWorkspace::ForEach([&](Workspace* /* unused */) { calls++; });
  EXPECT_EQ(calls, 0);
}

}  // namespace caffe2
