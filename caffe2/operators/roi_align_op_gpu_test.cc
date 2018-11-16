#include "caffe2/utils/eigen_utils.h"
#include "caffe2/operators/roi_align_op.h"

#include "caffe2/core/context_gpu.h"
#include "caffe2/core/flags.h"
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"
#include "gtest/gtest.h"

namespace caffe2 {
namespace {

template <class Context>
void AddConstInput(
    const vector<int64_t>& shape,
    const float value,
    const string& name,
    Context* context,
    Workspace* ws) {
  Blob* blob = ws->CreateBlob(name);
  auto* tensor = BlobGetMutableTensor(blob, Context::GetDeviceType());
  tensor->Resize(shape);
  math::Set<float, Context>(
      tensor->size(), value, tensor->template mutable_data<float>(), context);
  return;
}

template <class Context>
void AddInput(
    const vector<int64_t>& shape,
    const vector<float>& values,
    const string& name,
    Workspace* ws);

template <>
void AddInput<CPUContext>(
    const vector<int64_t>& shape,
    const vector<float>& values,
    const string& name,
    Workspace* ws) {
  Blob* blob = ws->CreateBlob(name);
  auto* tensor = BlobGetMutableTensor(blob, CPU);
  tensor->Resize(shape);
  EigenVectorMap<float> tensor_vec(
      tensor->template mutable_data<float>(), tensor->numel());
  tensor_vec.array() = utils::AsEArrXt(values);
}

template <>
void AddInput<CUDAContext>(
    const vector<int64_t>& shape,
    const vector<float>& values,
    const string& name,
    Workspace* ws) {
  Tensor tmp(shape, CPU);
  EigenVectorMap<float> tmp_vec(tmp.mutable_data<float>(), tmp.numel());
  tmp_vec.array() = utils::AsEArrXt(values);

  Blob* blob = ws->CreateBlob(name);
  auto* tensor = BlobGetMutableTensor(blob, CUDA);
  tensor->CopyFrom(tmp);
}

template <class Context>
DeviceType GetDeviceType() {
  return CPU;
}
template <>
DeviceType GetDeviceType<CUDAContext>() {
  return CUDA;
}

int randInt(int a, int b) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  return std::uniform_int_distribution<int>(a, b)(gen);
}

struct TestParams {
  int N;
  int C;
  int H;
  int W;
  int n_rois;
  vector<float> rois_array;
};

template <class Context>
void CreateAndRun(
    TensorCPU* outResult,
    const string& order,
    const TestParams& test_params,
    bool random_test) {
  Workspace ws;
  Context context;

  if (random_test) {
    const int N = test_params.N;
    const int C = test_params.C;
    const int H = test_params.H;
    const int W = test_params.W;
    vector<float> features(N * C * H * W);
    std::iota(features.begin(), features.end(), 0);
    // utils::AsEArrXt(features) /= features.size();
    AddInput<Context>(vector<int64_t>{N, C, H, W}, features, "X", &ws);
    const int n_rois = test_params.n_rois;
    const vector<float>& rois = test_params.rois_array;
    AddInput<Context>(vector<int64_t>{n_rois, 5}, rois, "R", &ws);
  } else {
    const int N = 2;
    const int C = 3;
    const int H = 100;
    const int W = 110;
    vector<float> features(N * C * H * W);
    std::iota(features.begin(), features.end(), 0);
    // utils::AsEArrXt(features) /= features.size();
    AddInput<Context>(vector<int64_t>{N, C, H, W}, features, "X", &ws);
    vector<float> rois{0, 0,            0,            79,           59,
                       0, 0,            5.0005703f,   52.63237f,    43.69501495f,
                       0, 24.13628387f, 7.51243401f,  79,           46.06628418f,
                       0, 0,            7.50924301f,  68.47792816f, 46.03357315f,
                       0, 0,            23.09477997f, 51.61448669f, 59,
                       0, 0,            39.52141571f, 52.44710541f, 59,
                       0, 23.57396317f, 29.98791885f, 79,           59,
                       0, 0,            41.90219116f, 79,           59,
                       0, 0,            23.30098343f, 79,           59};
    AddInput<Context>(vector<int64_t>{9, 5}, rois, "R", &ws);
  }

  std::vector<unique_ptr<OperatorBase>> ops;
  EXPECT_TRUE(order == "NCHW" || order == "NHWC");
  if (order == "NCHW") {
    OperatorDef def;
    def.set_name("test");
    def.set_type("RoIAlign");
    def.add_input("X");
    def.add_input("R");
    def.add_output("Y");
    def.mutable_device_option()->set_device_type(
        TypeToProto(GetDeviceType<Context>()));
    def.add_arg()->CopyFrom(MakeArgument("spatial_scale", 1.0f / 16.0f));
    def.add_arg()->CopyFrom(MakeArgument("pooled_h", 6));
    def.add_arg()->CopyFrom(MakeArgument("pooled_w", 8));
    def.add_arg()->CopyFrom(MakeArgument("sampling_ratio", 2));

    ops.push_back(CreateOperator(def, &ws));
  } else if (order == "NHWC") {
    OperatorDef def_roialign;
    def_roialign.set_name("test");
    def_roialign.set_type("RoIAlign");
    def_roialign.add_input("X_NHWC");
    def_roialign.add_input("R");
    def_roialign.add_output("Y_NHWC");
    def_roialign.mutable_device_option()->set_device_type(
        TypeToProto(GetDeviceType<Context>()));
    def_roialign.add_arg()->CopyFrom(
        MakeArgument("spatial_scale", 1.0f / 16.0f));
    def_roialign.add_arg()->CopyFrom(MakeArgument("pooled_h", 6));
    def_roialign.add_arg()->CopyFrom(MakeArgument("pooled_w", 8));
    def_roialign.add_arg()->CopyFrom(MakeArgument("sampling_ratio", 2));
    def_roialign.add_arg()->CopyFrom(MakeArgument<string>("order", "NHWC"));

    OperatorDef def_x;
    def_x.set_name("test_x");
    def_x.set_type("NCHW2NHWC");
    def_x.add_input("X");
    def_x.add_output("X_NHWC");
    def_x.mutable_device_option()->set_device_type(
        TypeToProto(GetDeviceType<Context>()));

    OperatorDef def_y;
    def_y.set_name("test_y");
    def_y.set_type("NHWC2NCHW");
    def_y.add_input("Y_NHWC");
    def_y.add_output("Y");
    def_y.mutable_device_option()->set_device_type(
        TypeToProto(GetDeviceType<Context>()));

    ops.push_back(CreateOperator(def_x, &ws));
    ops.push_back(CreateOperator(def_roialign, &ws));
    ops.push_back(CreateOperator(def_y, &ws));
  }

  for (auto const& op : ops) {
    EXPECT_NE(nullptr, op.get());
    EXPECT_TRUE(op->Run());
  }

  Blob* Y_blob = ws.GetBlob("Y");
  EXPECT_NE(nullptr, Y_blob);

  auto& Y = Y_blob->Get<Tensor>();
  outResult->CopyFrom(Y);
}

} // namespace

TEST(RoiAlignTest, CheckCPUGPUEqual) {
  if (!caffe2::HasCudaGPU())
    return;

  Tensor y_cpu(CPU);
  Tensor y_gpu(CPU);
  Tensor y_cpu_nhwc(CPU);

  // tests using FAIR example
  {
    TestParams test_params;
    CreateAndRun<CPUContext>(&y_cpu, "NCHW", test_params, false);
    CreateAndRun<CUDAContext>(&y_gpu, "NCHW", test_params, false);
    CreateAndRun<CPUContext>(&y_cpu_nhwc, "NHWC", test_params, false);

    EXPECT_EQ(y_cpu.sizes(), y_gpu.sizes());
    EXPECT_EQ(y_cpu.sizes(), y_cpu_nhwc.sizes());
    ConstEigenVectorMap<float> y_cpu_vec(y_cpu.data<float>(), y_cpu.numel());
    ConstEigenVectorMap<float> y_gpu_vec(y_gpu.data<float>(), y_gpu.numel());
    ConstEigenVectorMap<float> y_cpu_nhwc_vec(
        y_cpu_nhwc.data<float>(), y_cpu_nhwc.numel());
    int max_diff_idx = -1;
    (y_cpu_vec - y_gpu_vec).cwiseAbs().maxCoeff(&max_diff_idx);
    EXPECT_FLOAT_EQ(y_cpu_vec[max_diff_idx], y_gpu_vec[max_diff_idx]);

    max_diff_idx = -1;
    (y_cpu_vec - y_cpu_nhwc_vec).cwiseAbs().maxCoeff(&max_diff_idx);
    EXPECT_FLOAT_EQ(y_cpu_vec[max_diff_idx], y_cpu_nhwc_vec[max_diff_idx]);
  }

  // random tests
  const int random_test_numbers = 100;
  for (int i = 0; i < random_test_numbers; i++) {
    const int N = randInt(1, 5);
    const int C = randInt(1, 5);
    const int H = randInt(1, 50);
    const int W = randInt(1, 50);
    const int n_rois = randInt(1, 30);
    vector<float> rois_array;
    for (int n = 0; n < n_rois; n++) {
      rois_array.push_back(randInt(0, N - 1));
      int w1 = randInt(-20, W + 20);
      int w2 = randInt(-20, W + 20);
      int h1 = randInt(-20, H + 20);
      int h2 = randInt(-20, H + 20);
      rois_array.push_back(std::min(w1, w2));
      rois_array.push_back(std::max(h1, h2));
      rois_array.push_back(std::min(w1, w2));
      rois_array.push_back(std::max(h1, h2));
    }
    TestParams test_params{N, C, H, W, n_rois, rois_array};

    CreateAndRun<CPUContext>(&y_cpu, "NCHW", test_params, true);
    CreateAndRun<CUDAContext>(&y_gpu, "NCHW", test_params, true);
    CreateAndRun<CPUContext>(&y_cpu_nhwc, "NHWC", test_params, true);

    EXPECT_EQ(y_cpu.sizes(), y_gpu.sizes());
    EXPECT_EQ(y_cpu.sizes(), y_cpu_nhwc.sizes());
    ConstEigenVectorMap<float> y_cpu_vec(y_cpu.data<float>(), y_cpu.numel());
    ConstEigenVectorMap<float> y_gpu_vec(y_gpu.data<float>(), y_gpu.numel());
    ConstEigenVectorMap<float> y_cpu_nhwc_vec(
        y_cpu_nhwc.data<float>(), y_cpu_nhwc.numel());
    int max_diff_idx = -1;
    (y_cpu_vec - y_gpu_vec).cwiseAbs().maxCoeff(&max_diff_idx);
    EXPECT_NEAR(y_cpu_vec[max_diff_idx], y_gpu_vec[max_diff_idx], 1e-1);

    max_diff_idx = -1;
    (y_cpu_vec - y_cpu_nhwc_vec).cwiseAbs().maxCoeff(&max_diff_idx);
    EXPECT_FLOAT_EQ(y_cpu_vec[max_diff_idx], y_cpu_nhwc_vec[max_diff_idx]);
  }
}

} // namespace caffe2
