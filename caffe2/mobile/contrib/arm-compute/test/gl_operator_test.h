#include "caffe2/mobile/contrib/arm-compute/core/context.h"
#include <gtest/gtest.h>

#include "caffe2/core/graph.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/workspace.h"

namespace caffe2 {

#define DECLARE_OPENGL_OPERATOR(_name)                                         \
  OperatorDef _name;                                                           \
  _name.mutable_device_option()->set_device_type(OPENGL);

#define MAKE_OPENGL_OPERATOR(_op)                                              \
  _op->mutable_device_option()->set_device_type(OPENGL);

#define ADD_ARG(_op, _name, _type, _val)                                       \
  {                                                                            \
    Argument *arg = _op.add_arg();                                             \
    arg->set_name(_name);                                                      \
    arg->set_##_type(_val);                                                    \
  }

// Use value 1337 to generate a blob that is deterministic
// and unique at each value (for debugging purposes)
template<typename T = float>
void PopulateCPUBlob(Workspace *ws, bool random, std::string name,
                     std::vector<int> dims, int val = 1, int dist_shift = 0, float variance = 1) {
  Blob *blob = ws->CreateBlob(name);
  auto *tensor = blob->GetMutable<TensorCPU>();
  tensor->Resize(dims);
  T *t_data = tensor->mutable_data<T>();
  std::random_device rd;
  std::mt19937 e2(rd());
  std::normal_distribution<> dist(0 + dist_shift, variance + dist_shift);
  for (int i = 0; i < tensor->size(); ++i) {
    t_data[i] = T(random ? dist(e2) : (val == 1337 ? i : val));
  }
}

template<typename T = half>
void compareNetResult(Workspace& ws,
                      NetDef& cpu_net, NetDef& gpu_net,
                      string cpu_blob="ref_Y",
                      string gpu_blob="gpu_Y",
                      double tol=0.01,
                      bool relative=false) {
  ws.RunNetOnce(cpu_net);
  ws.RunNetOnce(gpu_net);

  Blob *cpu_out = ws.GetBlob(cpu_blob);
  Blob *gpu_out = ws.GetBlob(gpu_blob);
  EXPECT_NE(nullptr, cpu_out);
  EXPECT_NE(nullptr, gpu_out);

  TensorCPU g;
  auto& g_ = gpu_out->Get<GLTensor<T>>();
  getTensorCPU(g_, g);

  auto &t = cpu_out->Get<TensorCPU>();
  EXPECT_EQ(g.size(), t.size());

  for (auto i = 0; i < g.size(); ++i) {
    if (relative) {
      EXPECT_NEAR(g.data<float>()[i], t.data<float>()[i], tol + tol * std::abs(t.data<float>()[i])) << "at index " << i;
    } else{
      EXPECT_NEAR(g.data<float>()[i], t.data<float>()[i], tol)
        << "at index " << i;
    }
  }
}

template<typename T = half>
void compareNetResult4D(Workspace& ws,
                        NetDef& cpu_net, NetDef& gpu_net,
                        string cpu_blob="ref_Y",
                        string gpu_blob="gpu_Y",
                        double tol=0.05) {
  ws.RunNetOnce(cpu_net);
  ws.RunNetOnce(gpu_net);

  Blob *cpu_out = ws.GetBlob(cpu_blob);
  Blob *gpu_out = ws.GetBlob(gpu_blob);
  auto &g_ = gpu_out->Get<GLTensor<T>>();

  EXPECT_NE(nullptr, cpu_out);
  EXPECT_NE(nullptr, gpu_out);

  TensorCPU g;
  auto &t = cpu_out->Get<TensorCPU>();
  g.Resize(g_.dims());
  T *buffer = g_.map();
  char *byte_buffer = (char *)buffer;
  auto info = g_.get_underlying()->info();

  CAFFE_ENFORCE(byte_buffer != NULL);
  auto C = t.dim32(1);
  auto H = t.dim32(2);
  auto W = t.dim32(3);
  int diff_num = 0;
#define get_elem(_a, _b, _c)                                            \
  (half *)&byte_buffer[info->offset_element_in_bytes(                   \
      arm_compute::Coordinates(_a, _b, _c))]
  for (auto c = 0; c < C; ++c) {
    for (auto h = 0; h < H; ++h) {
      for (auto w = 0; w < W; ++w) {
        auto t_elem = t.data<float>()[(c * H + h) * W + w];
        auto g_elem = get_elem(w, h, c);

        if (!isnan(t_elem) && (std::abs(t_elem - float(*g_elem)) > tol + tol * std::abs(t_elem))) {
            diff_num++;
        }
        CHECK(diff_num <= 0.03 * C*H*W);
      }
    }
  }
#undef get_elem
  g_.unmap();
}


} // namespace caffe2
