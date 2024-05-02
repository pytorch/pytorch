#include "caffe2/operators/quantized/int8_test_utils.h"
#include "caffe2/operators/quantized/int8_utils.h"
#include <c10/util/irange.h>

namespace caffe2 {

TEST(Int8RoIAlign, RoIAlign) {
  const int N = 2;
  const int C = 3;
  const int H = 100;
  const int W = 110;
  auto XQ = q({N, H, W, C});
  XQ->scale = 0.01f;
  XQ->zero_point = 127;
  auto X = dq(*XQ);
  const int n_rois = 10;
  Workspace ws;
  vector<float> rois_array;
  for (int n = 0; n < n_rois; n++) {
    rois_array.push_back(randomInt(0, N - 1));
    int w1 = randomInt(0, W);
    int w2 = randomInt(0, W);
    int h1 = randomInt(0, H);
    int h2 = randomInt(0, H);
    rois_array.push_back(std::min(w1, w2));
    rois_array.push_back(std::max(h1, h2));
    rois_array.push_back(std::min(w1, w2));
    rois_array.push_back(std::max(h1, h2));
  }
  add_input({n_rois, 5}, rois_array, "RoIs", &ws);
  auto xop = CreateOperatorDef(
      "RoIAlign",
      "",
      {"X", "RoIs"},
      {"Y"},
      {MakeArgument<float>("spatial_scale", 0.25f),
       MakeArgument<int>("pooled_h", 2),
       MakeArgument<int>("pooled_w", 2),
       MakeArgument<string>("order", "NHWC"),
       MakeArgument<int>("sampling_ratio", 2)});
  auto op = CreateOperatorDef(
      "Int8RoIAlign",
      "",
      {"XQ", "RoIs"},
      {"YQ"},
      {MakeArgument<float>("spatial_scale", 0.25f),
       MakeArgument<int>("pooled_h", 2),
       MakeArgument<int>("pooled_w", 2),
       MakeArgument<int>("sampling_ratio", 2),
       MakeArgument<int>("Y_zero_point", 127),
       MakeArgument<float>("Y_scale", 0.01f)});
  int8Copy(ws.CreateBlob("XQ")->GetMutable<int8::Int8TensorCPU>(), *XQ);
  BlobGetMutableTensor(ws.CreateBlob("X"), CPU)->CopyFrom(*X);
  ws.RunOperatorOnce(op);
  ws.RunOperatorOnce(xop);
  const auto& YQ = ws.GetBlob("YQ")->Get<int8::Int8TensorCPU>();
  auto YA = dq(YQ);
  const auto& YE = ws.GetBlob("Y")->Get<TensorCPU>();
  // we cant make sure delta is within XQ->scale since there is interpolation
  EXPECT_TENSOR_APPROX_EQ(*YA, YE, 4 * XQ->scale);
}

} // namespace caffe2
