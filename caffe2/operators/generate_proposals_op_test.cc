#include "caffe2/operators/generate_proposals_op.h"

#include <gtest/gtest.h>
#include "caffe2/core/flags.h"

namespace caffe2 {

static void AddConstInput(
    const vector<TIndex>& shape,
    const float value,
    const string& name,
    Workspace* ws) {
  DeviceOption option;
  CPUContext context(option);
  Blob* blob = ws->CreateBlob(name);
  auto* tensor = blob->GetMutable<TensorCPU>();
  tensor->Resize(shape);
  math::Set<float, CPUContext>(
      tensor->size(), value, tensor->mutable_data<float>(), &context);
  return;
}

static void AddLinSpacedInput(
    const vector<TIndex>& shape,
    const float min_val,
    const float max_val,
    const string& name,
    Workspace* ws) {
  DeviceOption option;
  CPUContext context(option);
  Blob* blob = ws->CreateBlob(name);
  auto* tensor = blob->GetMutable<TensorCPU>();
  tensor->Resize(shape);
  EigenVectorMap<float> tensor_vec(
      tensor->mutable_data<float>(), tensor->size());
  tensor_vec.setLinSpaced(min_val, max_val);

  return;
}

static void AddInput(
    const vector<TIndex>& shape,
    const vector<float>& values,
    const string& name,
    Workspace* ws) {
  DeviceOption option;
  CPUContext context(option);
  Blob* blob = ws->CreateBlob(name);
  auto* tensor = blob->GetMutable<TensorCPU>();
  tensor->Resize(shape);
  EigenVectorMap<float> tensor_vec(
      tensor->mutable_data<float>(), tensor->size());
  tensor_vec.array() = utils::AsEArrXt(values);

  return;
}

TEST(GenerateProposalsTest, TestComputeAllAnchors) {
  ERMatXf anchors(3, 4);
  anchors << -38, -16, 53, 31, -84, -40, 99, 55, -176, -88, 191, 103;

  int height = 4;
  int width = 3;
  float feat_stride = 16;
  ERMatXf all_anchors_gt(36, 4);
  all_anchors_gt << -38, -16, 53, 31, -84, -40, 99, 55, -176, -88, 191, 103,
      -22, -16, 69, 31, -68, -40, 115, 55, -160, -88, 207, 103, -6, -16, 85, 31,
      -52, -40, 131, 55, -144, -88, 223, 103, -38, 0, 53, 47, -84, -24, 99, 71,
      -176, -72, 191, 119, -22, 0, 69, 47, -68, -24, 115, 71, -160, -72, 207,
      119, -6, 0, 85, 47, -52, -24, 131, 71, -144, -72, 223, 119, -38, 16, 53,
      63, -84, -8, 99, 87, -176, -56, 191, 135, -22, 16, 69, 63, -68, -8, 115,
      87, -160, -56, 207, 135, -6, 16, 85, 63, -52, -8, 131, 87, -144, -56, 223,
      135, -38, 32, 53, 79, -84, 8, 99, 103, -176, -40, 191, 151, -22, 32, 69,
      79, -68, 8, 115, 103, -160, -40, 207, 151, -6, 32, 85, 79, -52, 8, 131,
      103, -144, -40, 223, 151;

  TensorCPU anchors_tensor(vector<TIndex>{anchors.rows(), anchors.cols()});
  Eigen::Map<ERMatXf>(
      anchors_tensor.mutable_data<float>(), anchors.rows(), anchors.cols()) =
      anchors;

  auto result =
      utils::ComputeAllAnchors(anchors_tensor, height, width, feat_stride);
  Eigen::Map<const ERMatXf> all_anchors_result(
      result.data(), height * width * anchors.rows(), 4);

  EXPECT_EQ((all_anchors_result - all_anchors_gt).norm(), 0);
}

TEST(GenerateProposalsTest, TestEmpty) {
  Workspace ws;
  OperatorDef def;
  def.set_name("test");
  def.set_type("GenerateProposals");
  def.add_input("scores");
  def.add_input("bbox_deltas");
  def.add_input("im_info");
  def.add_input("anchors");
  def.add_output("rois");
  def.add_output("rois_probs");
  const int img_count = 3;
  const int A = 4;
  const int H = 10;
  const int W = 8;
  AddConstInput(vector<TIndex>{img_count, A, H, W}, 1., "scores", &ws);
  AddLinSpacedInput(
      vector<TIndex>{img_count, 4 * A, H, W}, 0, 10, "bbox_deltas", &ws);
  AddConstInput(vector<TIndex>{img_count, 3}, 0.1, "im_info", &ws);
  AddConstInput(vector<TIndex>{A, 4}, 1.0, "anchors", &ws);

  def.add_arg()->CopyFrom(MakeArgument("spatial_scale", 2.0f));

  unique_ptr<OperatorBase> op(CreateOperator(def, &ws));
  EXPECT_NE(nullptr, op.get());
  EXPECT_TRUE(op->Run());
  Blob* rois_blob = ws.GetBlob("rois");
  EXPECT_NE(nullptr, rois_blob);
  auto& rois = rois_blob->Get<TensorCPU>();
  EXPECT_EQ(rois.size(), 0);

  Blob* rois_probs_blob = ws.GetBlob("rois_probs");
  EXPECT_NE(nullptr, rois_probs_blob);
  auto& rois_probs = rois_probs_blob->Get<TensorCPU>();
  EXPECT_EQ(rois_probs.size(), 0);
}

TEST(GenerateProposalsTest, TestRealDownSampled) {
  Workspace ws;
  OperatorDef def;
  def.set_name("test");
  def.set_type("GenerateProposals");
  def.add_input("scores");
  def.add_input("bbox_deltas");
  def.add_input("im_info");
  def.add_input("anchors");
  def.add_output("rois");
  def.add_output("rois_probs");
  const int img_count = 1;
  const int A = 2;
  const int H = 4;
  const int W = 5;

  vector<float> scores{
      5.44218998e-03, 1.19207997e-03, 1.12379994e-03, 1.17181998e-03,
      1.20544003e-03, 6.17993006e-04, 1.05261997e-05, 8.91025957e-06,
      9.29536981e-09, 6.09605013e-05, 4.72735002e-04, 1.13482002e-10,
      1.50015003e-05, 4.45032993e-06, 3.21612994e-08, 8.02662980e-04,
      1.40488002e-04, 3.12508007e-07, 3.02616991e-06, 1.97759000e-08,
      2.66913995e-02, 5.26766013e-03, 5.05053019e-03, 5.62100019e-03,
      5.37420018e-03, 5.26280981e-03, 2.48894998e-04, 1.06842002e-04,
      3.92931997e-06, 1.79388002e-03, 4.79440019e-03, 3.41609990e-07,
      5.20430971e-04, 3.34090000e-05, 2.19159006e-07, 2.28786003e-03,
      5.16703985e-05, 4.04523007e-06, 1.79227004e-06, 5.32449000e-08};
  vector<float> bbx{
      -1.65040009e-02, -1.84051003e-02, -1.85930002e-02, -2.08263006e-02,
      -1.83814000e-02, -2.89172009e-02, -3.89706008e-02, -7.52277970e-02,
      -1.54091999e-01, -2.55433004e-02, -1.77490003e-02, -1.10340998e-01,
      -4.20190990e-02, -2.71421000e-02, 6.89801015e-03,  5.71171008e-02,
      -1.75665006e-01, 2.30021998e-02,  3.08554992e-02,  -1.39333997e-02,
      3.40579003e-01,  3.91070992e-01,  3.91624004e-01,  3.92527014e-01,
      3.91445011e-01,  3.79328012e-01,  4.26631987e-01,  3.64892989e-01,
      2.76894987e-01,  5.13985991e-01,  3.79999995e-01,  1.80457994e-01,
      4.37402993e-01,  4.18545991e-01,  2.51549989e-01,  4.48318988e-01,
      1.68564007e-01,  4.65440989e-01,  4.21891987e-01,  4.45928007e-01,
      3.27155995e-03,  3.71480011e-03,  3.60032008e-03,  4.27092984e-03,
      3.74579988e-03,  5.95752988e-03,  -3.14473989e-03, 3.52022005e-03,
      -1.88564006e-02, 1.65188999e-03,  1.73791999e-03,  -3.56074013e-02,
      -1.66615995e-04, 3.14146001e-03,  -1.11830998e-02, -5.35363983e-03,
      6.49790000e-03,  -9.27671045e-03, -2.83346009e-02, -1.61233004e-02,
      -2.15505004e-01, -2.19910994e-01, -2.20872998e-01, -2.12831005e-01,
      -2.19145000e-01, -2.27687001e-01, -3.43973994e-01, -2.75869995e-01,
      -3.19516987e-01, -2.50418007e-01, -2.48537004e-01, -5.08224010e-01,
      -2.28724003e-01, -2.82402009e-01, -3.75815988e-01, -2.86352992e-01,
      -5.28333001e-02, -4.43836004e-01, -4.55134988e-01, -4.34897989e-01,
      -5.65053988e-03, -9.25739005e-04, -1.06790999e-03, -2.37016007e-03,
      -9.71166010e-04, -8.90910998e-03, -1.17592998e-02, -2.08992008e-02,
      -4.94231991e-02, 6.63906988e-03,  3.20469006e-03,  -6.44695014e-02,
      -3.11607006e-03, 2.02738005e-03,  1.48096997e-02,  4.39785011e-02,
      -8.28424022e-02, 3.62076014e-02,  2.71668993e-02,  1.38250999e-02,
      6.76669031e-02,  1.03252999e-01,  1.03255004e-01,  9.89722982e-02,
      1.03646003e-01,  4.79663983e-02,  1.11014001e-01,  9.31736007e-02,
      1.15768999e-01,  1.04014002e-01,  -8.90677981e-03, 1.13103002e-01,
      1.33085996e-01,  1.25405997e-01,  1.50051996e-01,  -1.13038003e-01,
      7.01059997e-02,  1.79651007e-01,  1.41055003e-01,  1.62841007e-01,
      -1.00247003e-02, -8.17587040e-03, -8.32176022e-03, -8.90108012e-03,
      -8.13035015e-03, -1.77263003e-02, -3.69572006e-02, -3.51580009e-02,
      -5.92143014e-02, -1.80795006e-02, -5.46086021e-03, -4.10550982e-02,
      -1.83081999e-02, -2.15411000e-02, -1.17953997e-02, 3.33894007e-02,
      -5.29635996e-02, -6.97528012e-03, -3.15250992e-03, -3.27355005e-02,
      1.29676998e-01,  1.16080999e-01,  1.15947001e-01,  1.21797003e-01,
      1.16089001e-01,  1.44875005e-01,  1.15617000e-01,  1.31586999e-01,
      1.74735002e-02,  1.21973999e-01,  1.31596997e-01,  2.48907991e-02,
      6.18605018e-02,  1.12855002e-01,  -6.99798986e-02, 9.58312973e-02,
      1.53593004e-01,  -8.75087008e-02, -4.92327996e-02, -3.32239009e-02};
  vector<float> im_info{60, 80, 0.166667};
  vector<float> anchors{-38, -16, 53, 31, -120, -120, 135, 135};

  ERMatXf rois_gt(9, 5);
  rois_gt << 0, 0, 0, 79, 59, 0, 0, 5.0005703, 52.63237, 43.69501495, 0,
      24.13628387, 7.51243401, 79, 46.06628418, 0, 0, 7.50924301, 68.47792816,
      46.03357315, 0, 0, 23.09477997, 51.61448669, 59, 0, 0, 39.52141571,
      52.44710541, 59, 0, 23.57396317, 29.98791885, 79, 59, 0, 0, 41.90219116,
      79, 59, 0, 0, 23.30098343, 79, 59;
  vector<float> rois_probs_gt{2.66913995e-02,
                              5.44218998e-03,
                              1.20544003e-03,
                              1.19207997e-03,
                              6.17993006e-04,
                              4.72735002e-04,
                              6.09605013e-05,
                              1.50015003e-05,
                              8.91025957e-06};

  AddInput(vector<TIndex>{img_count, A, H, W}, scores, "scores", &ws);
  AddInput(vector<TIndex>{img_count, 4 * A, H, W}, bbx, "bbox_deltas", &ws);
  AddInput(vector<TIndex>{img_count, 3}, im_info, "im_info", &ws);
  AddInput(vector<TIndex>{A, 4}, anchors, "anchors", &ws);

  def.add_arg()->CopyFrom(MakeArgument("spatial_scale", 1.0f / 16.0f));
  def.add_arg()->CopyFrom(MakeArgument("pre_nms_topN", 6000));
  def.add_arg()->CopyFrom(MakeArgument("post_nms_topN", 300));
  def.add_arg()->CopyFrom(MakeArgument("nms_thresh", 0.7f));
  def.add_arg()->CopyFrom(MakeArgument("min_size", 16.0f));

  unique_ptr<OperatorBase> op(CreateOperator(def, &ws));
  EXPECT_NE(nullptr, op.get());
  EXPECT_TRUE(op->Run());

  // test rois
  Blob* rois_blob = ws.GetBlob("rois");
  EXPECT_NE(nullptr, rois_blob);
  auto& rois = rois_blob->Get<TensorCPU>();
  EXPECT_EQ(rois.dims(), (vector<TIndex>{rois_gt.rows(), rois_gt.cols()}));
  auto rois_data =
      Eigen::Map<const ERMatXf>(rois.data<float>(), rois.dim(0), rois.dim(1));
  EXPECT_NEAR((rois_data.matrix() - rois_gt).cwiseAbs().maxCoeff(), 0, 1e-4);

  // test rois_probs
  Blob* rois_probs_blob = ws.GetBlob("rois_probs");
  EXPECT_NE(nullptr, rois_probs_blob);
  auto& rois_probs = rois_probs_blob->Get<TensorCPU>();
  EXPECT_EQ(rois_probs.dims(), (vector<TIndex>{TIndex(rois_probs_gt.size())}));
  auto rois_probs_data =
      ConstEigenVectorArrayMap<float>(rois_probs.data<float>(), rois.dim(0));
  EXPECT_NEAR(
      (rois_probs_data.matrix() - utils::AsEArrXt(rois_probs_gt).matrix())
          .cwiseAbs()
          .maxCoeff(),
      0,
      1e-4);
}

} // namespace caffe2
