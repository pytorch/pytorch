#include "caffe2/mobile/contrib/arm-compute/core/context.h"
#include <gtest/gtest.h>

namespace caffe2 {

TEST(OPENGLContextTest, Initialization) {
  auto gc = new GLContext();
  delete gc;
}

} // namespace caffe2
