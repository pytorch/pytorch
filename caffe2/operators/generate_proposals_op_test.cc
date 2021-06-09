#include "caffe2/operators/generate_proposals_op.h"

#include <gtest/gtest.h>
#include "caffe2/core/flags.h"
#include "caffe2/core/macros.h"

#include "caffe2/operators/generate_proposals_op_util_boxes.h"

#include <c10/util/irange.h>

namespace caffe2 {

static void AddConstInput(
    const vector<int64_t>& shape,
    const float value,
    const string& name,
    Workspace* ws) {
  DeviceOption option;
  CPUContext context(option);
  Blob* blob = ws->CreateBlob(name);
  auto* tensor = BlobGetMutableTensor(blob, CPU);
  tensor->Resize(shape);
  math::Set<float, CPUContext>(
      tensor->numel(), value, tensor->template mutable_data<float>(), &context);
  return;
}

static void AddLinSpacedInput(
    const vector<int64_t>& shape,
    const float min_val,
    const float max_val,
    const string& name,
    Workspace* ws) {
  DeviceOption option;
  CPUContext context(option);
  Blob* blob = ws->CreateBlob(name);
  auto* tensor = BlobGetMutableTensor(blob, CPU);
  tensor->Resize(shape);
  EigenVectorMap<float> tensor_vec(
      tensor->template mutable_data<float>(), tensor->numel());
  tensor_vec.setLinSpaced(min_val, max_val);

  return;
}

static void AddInput(
    const vector<int64_t>& shape,
    const vector<float>& values,
    const string& name,
    Workspace* ws) {
  DeviceOption option;
  CPUContext context(option);
  Blob* blob = ws->CreateBlob(name);
  auto* tensor = BlobGetMutableTensor(blob, CPU);
  tensor->Resize(shape);
  EigenVectorMap<float> tensor_vec(
      tensor->template mutable_data<float>(), tensor->numel());
  tensor_vec.array() = utils::AsEArrXt(values);

  return;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
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

  Tensor anchors_tensor(vector<int64_t>{anchors.rows(), anchors.cols()}, CPU);
  Eigen::Map<ERMatXf>(
      anchors_tensor.mutable_data<float>(), anchors.rows(), anchors.cols()) =
      anchors;

  auto result =
      utils::ComputeAllAnchors(anchors_tensor, height, width, feat_stride);
  Eigen::Map<const ERMatXf> all_anchors_result(
      result.data(), height * width * anchors.rows(), 4);

  EXPECT_EQ((all_anchors_result - all_anchors_gt).norm(), 0);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(GenerateProposalsTest, TestComputeSortedAnchors) {
  ERMatXf anchors(3, 4);
  anchors << -38, -16, 53, 31, -84, -40, 99, 55, -176, -88, 191, 103;

  int height = 4;
  int width = 3;
  int A = anchors.rows();
  float feat_stride = 16;
  int total = height * width * A;

  // Generate all anchors for ground truth
  Tensor anchors_tensor(vector<int64_t>{anchors.rows(), anchors.cols()}, CPU);
  Eigen::Map<ERMatXf>(
      anchors_tensor.mutable_data<float>(), anchors.rows(), anchors.cols()) =
      anchors;
  auto all_anchors =
      utils::ComputeAllAnchors(anchors_tensor, height, width, feat_stride);
  Eigen::Map<const ERMatXf> all_anchors_result(
      all_anchors.data(), height * width * A, 4);

  Eigen::Map<const ERArrXXf> anchors_map(
      anchors.data(), anchors.rows(), anchors.cols());

  // Test with random subsets and ordering of indices
  vector<int> indices(total);
  std::iota(indices.begin(), indices.end(), 0);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::shuffle(indices.begin(), indices.end(), gen);
  for (int count = 0; count <= total; ++count) {
    vector<int> order(indices.begin(), indices.begin() + count);
    auto result = utils::ComputeSortedAnchors(
        anchors_map, height, width, feat_stride, order);

    // Compare the result of ComputeSortedAnchors with first generating all
    // anchors via ComputeAllAnchors and then applying ordering and filtering.
    // Need to convert order from (A, H, W) to (H, W, A) format before for this.
    const auto& order_AHW = utils::AsEArrXt(order);
    const auto& order_AH = order_AHW / width;
    const auto& order_W = order_AHW - order_AH * width;
    const auto& order_A = order_AH / height;
    const auto& order_H = order_AH - order_A * height;
    const auto& order_HWA = (order_H * width + order_W) * A + order_A;

    ERArrXXf gt;
    utils::GetSubArrayRows(all_anchors_result.array(), order_HWA, &gt);
    EXPECT_EQ((result.matrix() - gt.matrix()).norm(), 0);
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(GenerateProposalsTest, TestComputeAllAnchorsRotated) {
  // Similar to TestComputeAllAnchors but for rotated boxes with angle info.
  ERMatXf anchors_xyxy(3, 4);
  anchors_xyxy << -38, -16, 53, 31, -84, -40, 99, 55, -176, -88, 191, 103;

  // Convert to RRPN format and add angles
  ERMatXf anchors(3, 5);
  anchors.block(0, 0, 3, 4) = utils::bbox_xyxy_to_ctrwh(
      anchors_xyxy.array(), true /* legacy_plus_one */);
  std::vector<float> angles{0.0, 45.0, -120.0};
  for (int i = 0; i < anchors.rows(); ++i) {
    anchors(i, 4) = angles[i % angles.size()];
  }

  int height = 4;
  int width = 3;
  float feat_stride = 16;
  ERMatXf all_anchors_gt_xyxy(36, 4);
  all_anchors_gt_xyxy << -38, -16, 53, 31, -84, -40, 99, 55, -176, -88, 191,
      103, -22, -16, 69, 31, -68, -40, 115, 55, -160, -88, 207, 103, -6, -16,
      85, 31, -52, -40, 131, 55, -144, -88, 223, 103, -38, 0, 53, 47, -84, -24,
      99, 71, -176, -72, 191, 119, -22, 0, 69, 47, -68, -24, 115, 71, -160, -72,
      207, 119, -6, 0, 85, 47, -52, -24, 131, 71, -144, -72, 223, 119, -38, 16,
      53, 63, -84, -8, 99, 87, -176, -56, 191, 135, -22, 16, 69, 63, -68, -8,
      115, 87, -160, -56, 207, 135, -6, 16, 85, 63, -52, -8, 131, 87, -144, -56,
      223, 135, -38, 32, 53, 79, -84, 8, 99, 103, -176, -40, 191, 151, -22, 32,
      69, 79, -68, 8, 115, 103, -160, -40, 207, 151, -6, 32, 85, 79, -52, 8,
      131, 103, -144, -40, 223, 151;

  // Convert gt to RRPN format and add angles
  ERMatXf all_anchors_gt(36, 5);
  all_anchors_gt.block(0, 0, 36, 4) = utils::bbox_xyxy_to_ctrwh(
      all_anchors_gt_xyxy.array(), true /* legacy_plus_one */);
  for (int i = 0; i < all_anchors_gt.rows(); ++i) {
    all_anchors_gt(i, 4) = angles[i % angles.size()];
  }

  Tensor anchors_tensor(vector<int64_t>{anchors.rows(), anchors.cols()}, CPU);
  Eigen::Map<ERMatXf>(
      anchors_tensor.mutable_data<float>(), anchors.rows(), anchors.cols()) =
      anchors;

  auto result =
      utils::ComputeAllAnchors(anchors_tensor, height, width, feat_stride);
  Eigen::Map<const ERMatXf> all_anchors_result(
      result.data(), height * width * anchors.rows(), 5);

  EXPECT_EQ((all_anchors_result - all_anchors_gt).norm(), 0);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(GenerateProposalsTest, TestComputeSortedAnchorsRotated) {
  // Similar to TestComputeSortedAnchors but for rotated boxes with angle info.
  ERMatXf anchors_xyxy(3, 4);
  anchors_xyxy << -38, -16, 53, 31, -84, -40, 99, 55, -176, -88, 191, 103;

  // Convert to RRPN format and add angles
  ERMatXf anchors(3, 5);
  anchors.block(0, 0, 3, 4) = utils::bbox_xyxy_to_ctrwh(
      anchors_xyxy.array(), true /* legacy_plus_one */);
  std::vector<float> angles{0.0, 45.0, -120.0};
  for (int i = 0; i < anchors.rows(); ++i) {
    anchors(i, 4) = angles[i % angles.size()];
  }

  int height = 4;
  int width = 3;
  int A = anchors.rows();
  float feat_stride = 16;
  int total = height * width * A;

  // Generate all anchors for ground truth
  Tensor anchors_tensor(vector<int64_t>{anchors.rows(), anchors.cols()}, CPU);
  Eigen::Map<ERMatXf>(
      anchors_tensor.mutable_data<float>(), anchors.rows(), anchors.cols()) =
      anchors;
  auto all_anchors =
      utils::ComputeAllAnchors(anchors_tensor, height, width, feat_stride);
  Eigen::Map<const ERMatXf> all_anchors_result(
      all_anchors.data(), height * width * A, 5);

  Eigen::Map<const ERArrXXf> anchors_map(
      anchors.data(), anchors.rows(), anchors.cols());

  // Test with random subsets and ordering of indices
  vector<int> indices(total);
  std::iota(indices.begin(), indices.end(), 0);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::shuffle(indices.begin(), indices.end(), gen);
  for (int count = 0; count <= total; ++count) {
    vector<int> order(indices.begin(), indices.begin() + count);
    auto result = utils::ComputeSortedAnchors(
        anchors_map, height, width, feat_stride, order);

    // Compare the result of ComputeSortedAnchors with first generating all
    // anchors via ComputeAllAnchors and then applying ordering and filtering.
    // Need to convert order from (A, H, W) to (H, W, A) format before for this.
    const auto& order_AHW = utils::AsEArrXt(order);
    const auto& order_AH = order_AHW / width;
    const auto& order_W = order_AHW - order_AH * width;
    const auto& order_A = order_AH / height;
    const auto& order_H = order_AH - order_A * height;
    const auto& order_HWA = (order_H * width + order_W) * A + order_A;

    ERArrXXf gt;
    utils::GetSubArrayRows(all_anchors_result.array(), order_HWA, &gt);
    EXPECT_EQ((result.matrix() - gt.matrix()).norm(), 0);
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
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
  AddConstInput(vector<int64_t>{img_count, A, H, W}, 1., "scores", &ws);
  AddLinSpacedInput(
      vector<int64_t>{img_count, 4 * A, H, W}, 0, 10, "bbox_deltas", &ws);
  AddConstInput(vector<int64_t>{img_count, 3}, 0.1, "im_info", &ws);
  AddConstInput(vector<int64_t>{A, 4}, 1.0, "anchors", &ws);

  def.add_arg()->CopyFrom(MakeArgument("spatial_scale", 2.0f));

  unique_ptr<OperatorBase> op(CreateOperator(def, &ws));
  EXPECT_NE(nullptr, op.get());
  EXPECT_TRUE(op->Run());
  Blob* rois_blob = ws.GetBlob("rois");
  EXPECT_NE(nullptr, rois_blob);
  auto& rois = rois_blob->Get<TensorCPU>();
  EXPECT_EQ(rois.numel(), 0);

  Blob* rois_probs_blob = ws.GetBlob("rois_probs");
  EXPECT_NE(nullptr, rois_probs_blob);
  auto& rois_probs = rois_probs_blob->Get<TensorCPU>();
  EXPECT_EQ(rois_probs.numel(), 0);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
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
      5.44218998e-03f, 1.19207997e-03f, 1.12379994e-03f, 1.17181998e-03f,
      1.20544003e-03f, 6.17993006e-04f, 1.05261997e-05f, 8.91025957e-06f,
      9.29536981e-09f, 6.09605013e-05f, 4.72735002e-04f, 1.13482002e-10f,
      1.50015003e-05f, 4.45032993e-06f, 3.21612994e-08f, 8.02662980e-04f,
      1.40488002e-04f, 3.12508007e-07f, 3.02616991e-06f, 1.97759000e-08f,
      2.66913995e-02f, 5.26766013e-03f, 5.05053019e-03f, 5.62100019e-03f,
      5.37420018e-03f, 5.26280981e-03f, 2.48894998e-04f, 1.06842002e-04f,
      3.92931997e-06f, 1.79388002e-03f, 4.79440019e-03f, 3.41609990e-07f,
      5.20430971e-04f, 3.34090000e-05f, 2.19159006e-07f, 2.28786003e-03f,
      5.16703985e-05f, 4.04523007e-06f, 1.79227004e-06f, 5.32449000e-08f};
  vector<float> bbx{
      -1.65040009e-02f, -1.84051003e-02f, -1.85930002e-02f, -2.08263006e-02f,
      -1.83814000e-02f, -2.89172009e-02f, -3.89706008e-02f, -7.52277970e-02f,
      -1.54091999e-01f, -2.55433004e-02f, -1.77490003e-02f, -1.10340998e-01f,
      -4.20190990e-02f, -2.71421000e-02f, 6.89801015e-03f,  5.71171008e-02f,
      -1.75665006e-01f, 2.30021998e-02f,  3.08554992e-02f,  -1.39333997e-02f,
      3.40579003e-01f,  3.91070992e-01f,  3.91624004e-01f,  3.92527014e-01f,
      3.91445011e-01f,  3.79328012e-01f,  4.26631987e-01f,  3.64892989e-01f,
      2.76894987e-01f,  5.13985991e-01f,  3.79999995e-01f,  1.80457994e-01f,
      4.37402993e-01f,  4.18545991e-01f,  2.51549989e-01f,  4.48318988e-01f,
      1.68564007e-01f,  4.65440989e-01f,  4.21891987e-01f,  4.45928007e-01f,
      3.27155995e-03f,  3.71480011e-03f,  3.60032008e-03f,  4.27092984e-03f,
      3.74579988e-03f,  5.95752988e-03f,  -3.14473989e-03f, 3.52022005e-03f,
      -1.88564006e-02f, 1.65188999e-03f,  1.73791999e-03f,  -3.56074013e-02f,
      -1.66615995e-04f, 3.14146001e-03f,  -1.11830998e-02f, -5.35363983e-03f,
      6.49790000e-03f,  -9.27671045e-03f, -2.83346009e-02f, -1.61233004e-02f,
      -2.15505004e-01f, -2.19910994e-01f, -2.20872998e-01f, -2.12831005e-01f,
      -2.19145000e-01f, -2.27687001e-01f, -3.43973994e-01f, -2.75869995e-01f,
      -3.19516987e-01f, -2.50418007e-01f, -2.48537004e-01f, -5.08224010e-01f,
      -2.28724003e-01f, -2.82402009e-01f, -3.75815988e-01f, -2.86352992e-01f,
      -5.28333001e-02f, -4.43836004e-01f, -4.55134988e-01f, -4.34897989e-01f,
      -5.65053988e-03f, -9.25739005e-04f, -1.06790999e-03f, -2.37016007e-03f,
      -9.71166010e-04f, -8.90910998e-03f, -1.17592998e-02f, -2.08992008e-02f,
      -4.94231991e-02f, 6.63906988e-03f,  3.20469006e-03f,  -6.44695014e-02f,
      -3.11607006e-03f, 2.02738005e-03f,  1.48096997e-02f,  4.39785011e-02f,
      -8.28424022e-02f, 3.62076014e-02f,  2.71668993e-02f,  1.38250999e-02f,
      6.76669031e-02f,  1.03252999e-01f,  1.03255004e-01f,  9.89722982e-02f,
      1.03646003e-01f,  4.79663983e-02f,  1.11014001e-01f,  9.31736007e-02f,
      1.15768999e-01f,  1.04014002e-01f,  -8.90677981e-03f, 1.13103002e-01f,
      1.33085996e-01f,  1.25405997e-01f,  1.50051996e-01f,  -1.13038003e-01f,
      7.01059997e-02f,  1.79651007e-01f,  1.41055003e-01f,  1.62841007e-01f,
      -1.00247003e-02f, -8.17587040e-03f, -8.32176022e-03f, -8.90108012e-03f,
      -8.13035015e-03f, -1.77263003e-02f, -3.69572006e-02f, -3.51580009e-02f,
      -5.92143014e-02f, -1.80795006e-02f, -5.46086021e-03f, -4.10550982e-02f,
      -1.83081999e-02f, -2.15411000e-02f, -1.17953997e-02f, 3.33894007e-02f,
      -5.29635996e-02f, -6.97528012e-03f, -3.15250992e-03f, -3.27355005e-02f,
      1.29676998e-01f,  1.16080999e-01f,  1.15947001e-01f,  1.21797003e-01f,
      1.16089001e-01f,  1.44875005e-01f,  1.15617000e-01f,  1.31586999e-01f,
      1.74735002e-02f,  1.21973999e-01f,  1.31596997e-01f,  2.48907991e-02f,
      6.18605018e-02f,  1.12855002e-01f,  -6.99798986e-02f, 9.58312973e-02f,
      1.53593004e-01f,  -8.75087008e-02f, -4.92327996e-02f, -3.32239009e-02f};
  vector<float> im_info{60, 80, 0.166667f};
  vector<float> anchors{-38, -16, 53, 31, -120, -120, 135, 135};

  ERMatXf rois_gt(9, 5);
  rois_gt << 0, 0, 0, 79, 59, 0, 0, 5.0005703f, 51.6324f, 42.6950f, 0,
      24.13628387f, 7.51243401f, 79, 45.0663f, 0, 0, 7.50924301f, 67.4779f,
      45.0336, 0, 0, 23.09477997f, 50.61448669f, 59, 0, 0, 39.52141571f,
      51.44710541f, 59, 0, 23.57396317f, 29.98791885f, 79, 59, 0, 0,
      41.90219116f, 79, 59, 0, 0, 23.30098343f, 78.2413f, 58.7287f;
  vector<float> rois_probs_gt{2.66913995e-02f,
                              5.44218998e-03f,
                              1.20544003e-03f,
                              1.19207997e-03f,
                              6.17993006e-04f,
                              4.72735002e-04f,
                              6.09605013e-05f,
                              1.50015003e-05f,
                              8.91025957e-06f};

  AddInput(vector<int64_t>{img_count, A, H, W}, scores, "scores", &ws);
  AddInput(vector<int64_t>{img_count, 4 * A, H, W}, bbx, "bbox_deltas", &ws);
  AddInput(vector<int64_t>{img_count, 3}, im_info, "im_info", &ws);
  AddInput(vector<int64_t>{A, 4}, anchors, "anchors", &ws);

  def.add_arg()->CopyFrom(MakeArgument("spatial_scale", 1.0f / 16.0f));
  def.add_arg()->CopyFrom(MakeArgument("pre_nms_topN", 6000));
  def.add_arg()->CopyFrom(MakeArgument("post_nms_topN", 300));
  def.add_arg()->CopyFrom(MakeArgument("nms_thresh", 0.7f));
  def.add_arg()->CopyFrom(MakeArgument("min_size", 16.0f));
  def.add_arg()->CopyFrom(MakeArgument("correct_transform_coords", true));

  unique_ptr<OperatorBase> op(CreateOperator(def, &ws));
  EXPECT_NE(nullptr, op.get());
  EXPECT_TRUE(op->Run());

  // test rois
  Blob* rois_blob = ws.GetBlob("rois");
  EXPECT_NE(nullptr, rois_blob);
  auto& rois = rois_blob->Get<TensorCPU>();
  EXPECT_EQ(rois.sizes(), (vector<int64_t>{rois_gt.rows(), rois_gt.cols()}));
  auto rois_data =
      Eigen::Map<const ERMatXf>(rois.data<float>(), rois.size(0), rois.size(1));
  EXPECT_NEAR((rois_data.matrix() - rois_gt).cwiseAbs().maxCoeff(), 0, 1e-4);

  // test rois_probs
  Blob* rois_probs_blob = ws.GetBlob("rois_probs");
  EXPECT_NE(nullptr, rois_probs_blob);
  auto& rois_probs = rois_probs_blob->Get<TensorCPU>();
  EXPECT_EQ(
      rois_probs.sizes(), (vector<int64_t>{int64_t(rois_probs_gt.size())}));
  auto rois_probs_data =
      ConstEigenVectorArrayMap<float>(rois_probs.data<float>(), rois.size(0));
  EXPECT_NEAR(
      (rois_probs_data.matrix() - utils::AsEArrXt(rois_probs_gt).matrix())
          .cwiseAbs()
          .maxCoeff(),
      0,
      1e-4);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(GenerateProposalsTest, TestRealDownSampledRotatedAngle0) {
  // Similar to TestRealDownSampled but for rotated boxes with angle info.
  const float angle = 0;
  const float delta_angle = 0;
  const float clip_angle_thresh = 1.0;
  const int box_dim = 5;

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
      5.44218998e-03f, 1.19207997e-03f, 1.12379994e-03f, 1.17181998e-03f,
      1.20544003e-03f, 6.17993006e-04f, 1.05261997e-05f, 8.91025957e-06f,
      9.29536981e-09f, 6.09605013e-05f, 4.72735002e-04f, 1.13482002e-10f,
      1.50015003e-05f, 4.45032993e-06f, 3.21612994e-08f, 8.02662980e-04f,
      1.40488002e-04f, 3.12508007e-07f, 3.02616991e-06f, 1.97759000e-08f,
      2.66913995e-02f, 5.26766013e-03f, 5.05053019e-03f, 5.62100019e-03f,
      5.37420018e-03f, 5.26280981e-03f, 2.48894998e-04f, 1.06842002e-04f,
      3.92931997e-06f, 1.79388002e-03f, 4.79440019e-03f, 3.41609990e-07f,
      5.20430971e-04f, 3.34090000e-05f, 2.19159006e-07f, 2.28786003e-03f,
      5.16703985e-05f, 4.04523007e-06f, 1.79227004e-06f, 5.32449000e-08f};
  vector<float> bbx{
      -1.65040009e-02f, -1.84051003e-02f, -1.85930002e-02f, -2.08263006e-02f,
      -1.83814000e-02f, -2.89172009e-02f, -3.89706008e-02f, -7.52277970e-02f,
      -1.54091999e-01f, -2.55433004e-02f, -1.77490003e-02f, -1.10340998e-01f,
      -4.20190990e-02f, -2.71421000e-02f, 6.89801015e-03f,  5.71171008e-02f,
      -1.75665006e-01f, 2.30021998e-02f,  3.08554992e-02f,  -1.39333997e-02f,
      3.40579003e-01f,  3.91070992e-01f,  3.91624004e-01f,  3.92527014e-01f,
      3.91445011e-01f,  3.79328012e-01f,  4.26631987e-01f,  3.64892989e-01f,
      2.76894987e-01f,  5.13985991e-01f,  3.79999995e-01f,  1.80457994e-01f,
      4.37402993e-01f,  4.18545991e-01f,  2.51549989e-01f,  4.48318988e-01f,
      1.68564007e-01f,  4.65440989e-01f,  4.21891987e-01f,  4.45928007e-01f,
      3.27155995e-03f,  3.71480011e-03f,  3.60032008e-03f,  4.27092984e-03f,
      3.74579988e-03f,  5.95752988e-03f,  -3.14473989e-03f, 3.52022005e-03f,
      -1.88564006e-02f, 1.65188999e-03f,  1.73791999e-03f,  -3.56074013e-02f,
      -1.66615995e-04f, 3.14146001e-03f,  -1.11830998e-02f, -5.35363983e-03f,
      6.49790000e-03f,  -9.27671045e-03f, -2.83346009e-02f, -1.61233004e-02f,
      -2.15505004e-01f, -2.19910994e-01f, -2.20872998e-01f, -2.12831005e-01f,
      -2.19145000e-01f, -2.27687001e-01f, -3.43973994e-01f, -2.75869995e-01f,
      -3.19516987e-01f, -2.50418007e-01f, -2.48537004e-01f, -5.08224010e-01f,
      -2.28724003e-01f, -2.82402009e-01f, -3.75815988e-01f, -2.86352992e-01f,
      -5.28333001e-02f, -4.43836004e-01f, -4.55134988e-01f, -4.34897989e-01f,
      -5.65053988e-03f, -9.25739005e-04f, -1.06790999e-03f, -2.37016007e-03f,
      -9.71166010e-04f, -8.90910998e-03f, -1.17592998e-02f, -2.08992008e-02f,
      -4.94231991e-02f, 6.63906988e-03f,  3.20469006e-03f,  -6.44695014e-02f,
      -3.11607006e-03f, 2.02738005e-03f,  1.48096997e-02f,  4.39785011e-02f,
      -8.28424022e-02f, 3.62076014e-02f,  2.71668993e-02f,  1.38250999e-02f,
      6.76669031e-02f,  1.03252999e-01f,  1.03255004e-01f,  9.89722982e-02f,
      1.03646003e-01f,  4.79663983e-02f,  1.11014001e-01f,  9.31736007e-02f,
      1.15768999e-01f,  1.04014002e-01f,  -8.90677981e-03f, 1.13103002e-01f,
      1.33085996e-01f,  1.25405997e-01f,  1.50051996e-01f,  -1.13038003e-01f,
      7.01059997e-02f,  1.79651007e-01f,  1.41055003e-01f,  1.62841007e-01f,
      -1.00247003e-02f, -8.17587040e-03f, -8.32176022e-03f, -8.90108012e-03f,
      -8.13035015e-03f, -1.77263003e-02f, -3.69572006e-02f, -3.51580009e-02f,
      -5.92143014e-02f, -1.80795006e-02f, -5.46086021e-03f, -4.10550982e-02f,
      -1.83081999e-02f, -2.15411000e-02f, -1.17953997e-02f, 3.33894007e-02f,
      -5.29635996e-02f, -6.97528012e-03f, -3.15250992e-03f, -3.27355005e-02f,
      1.29676998e-01f,  1.16080999e-01f,  1.15947001e-01f,  1.21797003e-01f,
      1.16089001e-01f,  1.44875005e-01f,  1.15617000e-01f,  1.31586999e-01f,
      1.74735002e-02f,  1.21973999e-01f,  1.31596997e-01f,  2.48907991e-02f,
      6.18605018e-02f,  1.12855002e-01f,  -6.99798986e-02f, 9.58312973e-02f,
      1.53593004e-01f,  -8.75087008e-02f, -4.92327996e-02f, -3.32239009e-02f};

  // Add angle in bbox deltas
  int num_boxes = scores.size();
  CHECK_EQ(bbx.size() / 4, num_boxes);
  vector<float> bbx_with_angle(num_boxes * box_dim);
  // bbx (deltas) is in shape (A * 4, H, W). Insert angle delta
  // at each spatial location for each anchor.
  int i = 0, j = 0;
  for (int a = 0; a < A; ++a) {
    for (int k = 0; k < 4 * H * W; ++k) {
      bbx_with_angle[i++] = bbx[j++];
    }
    for (int k = 0; k < H * W; ++k) {
      bbx_with_angle[i++] = delta_angle;
    }
  }

  vector<float> im_info{60, 80, 0.166667f};
  // vector<float> anchors{-38, -16, 53, 31, -120, -120, 135, 135};
  // Anchors in [x_ctr, y_ctr, w, h, angle] format
  vector<float> anchors{7.5, 7.5, 92, 48, angle, 7.5, 7.5, 256, 256, angle};

  // Results should exactly be the same as TestRealDownSampled since
  // angle = 0 for all boxes and clip_angle_thresh > 0 (which means
  // all horizontal boxes will be clipped to maintain backward compatibility).
  ERMatXf rois_gt_xyxy(9, 5);
  rois_gt_xyxy << 0, 0, 0, 79, 59, 0, 0, 5.0005703f, 51.6324f, 42.6950f, 0,
      24.13628387f, 7.51243401f, 79, 45.0663f, 0, 0, 7.50924301f, 67.4779f,
      45.0336, 0, 0, 23.09477997f, 50.61448669f, 59, 0, 0, 39.52141571f,
      51.44710541f, 59, 0, 23.57396317f, 29.98791885f, 79, 59, 0, 0,
      41.90219116f, 79, 59, 0, 0, 23.30098343f, 78.2413f, 58.7287f;
  ERMatXf rois_gt(rois_gt_xyxy.rows(), 6);
  // Batch ID
  rois_gt.block(0, 0, rois_gt.rows(), 1) =
      rois_gt_xyxy.block(0, 0, rois_gt.rows(), 1);
  // rois_gt in [x_ctr, y_ctr, w, h] format
  rois_gt.block(0, 1, rois_gt.rows(), 4) = utils::bbox_xyxy_to_ctrwh(
      rois_gt_xyxy.block(0, 1, rois_gt.rows(), 4).array(),
      true /* legacy_plus_one */);
  // Angle
  rois_gt.block(0, 5, rois_gt.rows(), 1) =
      ERMatXf::Constant(rois_gt.rows(), 1, angle);
  vector<float> rois_probs_gt{2.66913995e-02f,
                              5.44218998e-03f,
                              1.20544003e-03f,
                              1.19207997e-03f,
                              6.17993006e-04f,
                              4.72735002e-04f,
                              6.09605013e-05f,
                              1.50015003e-05f,
                              8.91025957e-06f};

  AddInput(vector<int64_t>{img_count, A, H, W}, scores, "scores", &ws);
  AddInput(
      vector<int64_t>{img_count, box_dim * A, H, W},
      bbx_with_angle,
      "bbox_deltas",
      &ws);
  AddInput(vector<int64_t>{img_count, 3}, im_info, "im_info", &ws);
  AddInput(vector<int64_t>{A, box_dim}, anchors, "anchors", &ws);

  def.add_arg()->CopyFrom(MakeArgument("spatial_scale", 1.0f / 16.0f));
  def.add_arg()->CopyFrom(MakeArgument("pre_nms_topN", 6000));
  def.add_arg()->CopyFrom(MakeArgument("post_nms_topN", 300));
  def.add_arg()->CopyFrom(MakeArgument("nms_thresh", 0.7f));
  def.add_arg()->CopyFrom(MakeArgument("min_size", 16.0f));
  def.add_arg()->CopyFrom(MakeArgument("correct_transform_coords", true));
  def.add_arg()->CopyFrom(MakeArgument("clip_angle_thresh", clip_angle_thresh));

  unique_ptr<OperatorBase> op(CreateOperator(def, &ws));
  EXPECT_NE(nullptr, op.get());
  EXPECT_TRUE(op->Run());

  // test rois
  Blob* rois_blob = ws.GetBlob("rois");
  EXPECT_NE(nullptr, rois_blob);
  auto& rois = rois_blob->Get<TensorCPU>();
  EXPECT_EQ(rois.sizes(), (vector<int64_t>{rois_gt.rows(), rois_gt.cols()}));
  auto rois_data =
      Eigen::Map<const ERMatXf>(rois.data<float>(), rois.size(0), rois.size(1));
  EXPECT_NEAR((rois_data.matrix() - rois_gt).cwiseAbs().maxCoeff(), 0, 1e-3);

  // test rois_probs
  Blob* rois_probs_blob = ws.GetBlob("rois_probs");
  EXPECT_NE(nullptr, rois_probs_blob);
  auto& rois_probs = rois_probs_blob->Get<TensorCPU>();
  EXPECT_EQ(
      rois_probs.sizes(), (vector<int64_t>{int64_t(rois_probs_gt.size())}));
  auto rois_probs_data =
      ConstEigenVectorArrayMap<float>(rois_probs.data<float>(), rois.size(0));
  EXPECT_NEAR(
      (rois_probs_data.matrix() - utils::AsEArrXt(rois_probs_gt).matrix())
          .cwiseAbs()
          .maxCoeff(),
      0,
      1e-4);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(GenerateProposalsTest, TestRealDownSampledRotated) {
  // Similar to TestRealDownSampled but for rotated boxes with angle info.
  const float angle = 45.0;
  const float delta_angle = 0.174533; // 0.174533 radians -> 10 degrees
  const float expected_angle = 55.0;
  const float clip_angle_thresh = 1.0;
  const int box_dim = 5;

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
      5.44218998e-03f, 1.19207997e-03f, 1.12379994e-03f, 1.17181998e-03f,
      1.20544003e-03f, 6.17993006e-04f, 1.05261997e-05f, 8.91025957e-06f,
      9.29536981e-09f, 6.09605013e-05f, 4.72735002e-04f, 1.13482002e-10f,
      1.50015003e-05f, 4.45032993e-06f, 3.21612994e-08f, 8.02662980e-04f,
      1.40488002e-04f, 3.12508007e-07f, 3.02616991e-06f, 1.97759000e-08f,
      2.66913995e-02f, 5.26766013e-03f, 5.05053019e-03f, 5.62100019e-03f,
      5.37420018e-03f, 5.26280981e-03f, 2.48894998e-04f, 1.06842002e-04f,
      3.92931997e-06f, 1.79388002e-03f, 4.79440019e-03f, 3.41609990e-07f,
      5.20430971e-04f, 3.34090000e-05f, 2.19159006e-07f, 2.28786003e-03f,
      5.16703985e-05f, 4.04523007e-06f, 1.79227004e-06f, 5.32449000e-08f};
  vector<float> bbx{
      -1.65040009e-02f, -1.84051003e-02f, -1.85930002e-02f, -2.08263006e-02f,
      -1.83814000e-02f, -2.89172009e-02f, -3.89706008e-02f, -7.52277970e-02f,
      -1.54091999e-01f, -2.55433004e-02f, -1.77490003e-02f, -1.10340998e-01f,
      -4.20190990e-02f, -2.71421000e-02f, 6.89801015e-03f,  5.71171008e-02f,
      -1.75665006e-01f, 2.30021998e-02f,  3.08554992e-02f,  -1.39333997e-02f,
      3.40579003e-01f,  3.91070992e-01f,  3.91624004e-01f,  3.92527014e-01f,
      3.91445011e-01f,  3.79328012e-01f,  4.26631987e-01f,  3.64892989e-01f,
      2.76894987e-01f,  5.13985991e-01f,  3.79999995e-01f,  1.80457994e-01f,
      4.37402993e-01f,  4.18545991e-01f,  2.51549989e-01f,  4.48318988e-01f,
      1.68564007e-01f,  4.65440989e-01f,  4.21891987e-01f,  4.45928007e-01f,
      3.27155995e-03f,  3.71480011e-03f,  3.60032008e-03f,  4.27092984e-03f,
      3.74579988e-03f,  5.95752988e-03f,  -3.14473989e-03f, 3.52022005e-03f,
      -1.88564006e-02f, 1.65188999e-03f,  1.73791999e-03f,  -3.56074013e-02f,
      -1.66615995e-04f, 3.14146001e-03f,  -1.11830998e-02f, -5.35363983e-03f,
      6.49790000e-03f,  -9.27671045e-03f, -2.83346009e-02f, -1.61233004e-02f,
      -2.15505004e-01f, -2.19910994e-01f, -2.20872998e-01f, -2.12831005e-01f,
      -2.19145000e-01f, -2.27687001e-01f, -3.43973994e-01f, -2.75869995e-01f,
      -3.19516987e-01f, -2.50418007e-01f, -2.48537004e-01f, -5.08224010e-01f,
      -2.28724003e-01f, -2.82402009e-01f, -3.75815988e-01f, -2.86352992e-01f,
      -5.28333001e-02f, -4.43836004e-01f, -4.55134988e-01f, -4.34897989e-01f,
      -5.65053988e-03f, -9.25739005e-04f, -1.06790999e-03f, -2.37016007e-03f,
      -9.71166010e-04f, -8.90910998e-03f, -1.17592998e-02f, -2.08992008e-02f,
      -4.94231991e-02f, 6.63906988e-03f,  3.20469006e-03f,  -6.44695014e-02f,
      -3.11607006e-03f, 2.02738005e-03f,  1.48096997e-02f,  4.39785011e-02f,
      -8.28424022e-02f, 3.62076014e-02f,  2.71668993e-02f,  1.38250999e-02f,
      6.76669031e-02f,  1.03252999e-01f,  1.03255004e-01f,  9.89722982e-02f,
      1.03646003e-01f,  4.79663983e-02f,  1.11014001e-01f,  9.31736007e-02f,
      1.15768999e-01f,  1.04014002e-01f,  -8.90677981e-03f, 1.13103002e-01f,
      1.33085996e-01f,  1.25405997e-01f,  1.50051996e-01f,  -1.13038003e-01f,
      7.01059997e-02f,  1.79651007e-01f,  1.41055003e-01f,  1.62841007e-01f,
      -1.00247003e-02f, -8.17587040e-03f, -8.32176022e-03f, -8.90108012e-03f,
      -8.13035015e-03f, -1.77263003e-02f, -3.69572006e-02f, -3.51580009e-02f,
      -5.92143014e-02f, -1.80795006e-02f, -5.46086021e-03f, -4.10550982e-02f,
      -1.83081999e-02f, -2.15411000e-02f, -1.17953997e-02f, 3.33894007e-02f,
      -5.29635996e-02f, -6.97528012e-03f, -3.15250992e-03f, -3.27355005e-02f,
      1.29676998e-01f,  1.16080999e-01f,  1.15947001e-01f,  1.21797003e-01f,
      1.16089001e-01f,  1.44875005e-01f,  1.15617000e-01f,  1.31586999e-01f,
      1.74735002e-02f,  1.21973999e-01f,  1.31596997e-01f,  2.48907991e-02f,
      6.18605018e-02f,  1.12855002e-01f,  -6.99798986e-02f, 9.58312973e-02f,
      1.53593004e-01f,  -8.75087008e-02f, -4.92327996e-02f, -3.32239009e-02f};

  // Add angle in bbox deltas
  int num_boxes = scores.size();
  CHECK_EQ(bbx.size() / 4, num_boxes);
  vector<float> bbx_with_angle(num_boxes * box_dim);
  // bbx (deltas) is in shape (A * 4, H, W). Insert angle delta
  // at each spatial location for each anchor.
  {
    int i = 0, j = 0;
    for (int a = 0; a < A; ++a) {
      for (int k = 0; k < 4 * H * W; ++k) {
        bbx_with_angle[i++] = bbx[j++];
      }
      for (int k = 0; k < H * W; ++k) {
        bbx_with_angle[i++] = delta_angle;
      }
    }
  }

  vector<float> im_info{60, 80, 0.166667f};
  // vector<float> anchors{-38, -16, 53, 31, -120, -120, 135, 135};
  vector<float> anchors{8, 8, 92, 48, angle, 8, 8, 256, 256, angle};

  AddInput(vector<int64_t>{img_count, A, H, W}, scores, "scores", &ws);
  AddInput(
      vector<int64_t>{img_count, box_dim * A, H, W},
      bbx_with_angle,
      "bbox_deltas",
      &ws);
  AddInput(vector<int64_t>{img_count, 3}, im_info, "im_info", &ws);
  AddInput(vector<int64_t>{A, box_dim}, anchors, "anchors", &ws);

  def.add_arg()->CopyFrom(MakeArgument("spatial_scale", 1.0f / 16.0f));
  def.add_arg()->CopyFrom(MakeArgument("pre_nms_topN", 6000));
  def.add_arg()->CopyFrom(MakeArgument("post_nms_topN", 300));
  def.add_arg()->CopyFrom(MakeArgument("nms_thresh", 0.7f));
  def.add_arg()->CopyFrom(MakeArgument("min_size", 16.0f));
  def.add_arg()->CopyFrom(MakeArgument("correct_transform_coords", true));
  def.add_arg()->CopyFrom(MakeArgument("clip_angle_thresh", clip_angle_thresh));

  unique_ptr<OperatorBase> op(CreateOperator(def, &ws));
  EXPECT_NE(nullptr, op.get());
  EXPECT_TRUE(op->Run());

  Blob* rois_blob = ws.GetBlob("rois");
  EXPECT_NE(nullptr, rois_blob);
  auto& rois = rois_blob->Get<TensorCPU>();
  EXPECT_EQ(rois.sizes(), (vector<int64_t>{13, 6}));

  Blob* rois_probs_blob = ws.GetBlob("rois_probs");
  EXPECT_NE(nullptr, rois_probs_blob);
  auto& rois_probs = rois_probs_blob->Get<TensorCPU>();
  EXPECT_EQ(rois_probs.sizes(), (vector<int64_t>{13}));

  // Verify that the resulting angles are correct
  auto rois_data =
      Eigen::Map<const ERMatXf>(rois.data<float>(), rois.size(0), rois.size(1));
  for (const auto i : c10::irange(rois.size(0))) {
    EXPECT_LE(std::abs(rois_data(i, 5) - expected_angle), 1e-4);
  }
}

} // namespace caffe2
