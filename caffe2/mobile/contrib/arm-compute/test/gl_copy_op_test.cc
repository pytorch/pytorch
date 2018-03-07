#include "gl_operator_test.h"

namespace caffe2 {

TEST(OPENGLOperatorTest, CopyFromGL) {

  for (auto dims: std::vector<std::vector<int>>{
      {1},
      {1, 2},
      {1, 2, 3},
      {1, 2, 3, 4},
      {4, 3, 2, 1},
      {4, 9, 8, 13},
    }) {
    Workspace ws;
    PopulateCPUBlob(&ws, true, std::string("cpu_X"), dims);

    NetDef gpu_net;
    gpu_net.set_type("opengl");
    {
      OperatorDef* def = AddOp(&gpu_net, "CopyFromGL", {"cpu_X"}, {"cpu_X2"});
      MAKE_OPENGL_OPERATOR(def);
    }
    ws.RunNetOnce(gpu_net);
    Blob *cpu_out = ws.GetBlob("cpu_X");
    Blob *gpu_out = ws.GetBlob("cpu_X2");
    EXPECT_NE(nullptr, cpu_out);
    EXPECT_NE(nullptr, gpu_out);

    auto &t1 = cpu_out->Get<TensorCPU>();
    auto &t2 = gpu_out->Get<TensorCPU>();
    double tol=0.01;
    for (auto i = 0; i < t1.size(); ++i) {
      EXPECT_NEAR(t1.data<float>()[i], t2.data<float>()[i], tol)
        << "at index " << i;
    }
  }
}

} // namespace caffe2