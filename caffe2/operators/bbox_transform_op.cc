#include "bbox_transform_op.h"
#include "caffe2/operators/generate_proposals_op_util_boxes.h"

#ifdef CAFFE2_USE_MKL
#include "caffe2/mkl/operators/operator_fallback_mkl.h"
#endif // CAFFE2_USE_MKL

namespace caffe2 {
namespace {

REGISTER_CPU_OPERATOR(BBoxTransform, BBoxTransformOp<float, CPUContext>);

#ifdef CAFFE2_HAS_MKL_DNN
REGISTER_MKL_OPERATOR(
    BBoxTransform,
    mkl::MKLFallbackOp<BBoxTransformOp<float, CPUContext>>);
#endif // CAFFE2_HAS_MKL_DNN

// Input: box, delta Output: box
OPERATOR_SCHEMA(BBoxTransform)
    .NumInputs(3)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Transform proposal bounding boxes to target bounding box using bounding box
    regression deltas.
)DOC")
    .Arg("weights", "vector<float> weights [wx, wy, ww, wh] for the deltas")
    .Input(
        0,
        "rois",
        "Bounding box proposals in pixel coordinates, "
        "Size (M, 4), format [x1, y1, x2, y2], or"
        "Size (M, 5), format [img_index_IGNORED, x1, y1, x2, y2]")
    .Input(
        1,
        "deltas",
        "bounding box translations and scales,"
        "size (M, 4*K), format [dx, dy, dw, dh], K = # classes")
    .Input(
        2,
        "im_info",
        "Image dimensions, size (1, 3), "
        "format [img_height, img_width, img_scale_IGNORED]")
    .Output(
        0,
        "box_out",
        "Pixel coordinates of the transformed bounding boxes,"
        "Size (M, 4*K), format [x1, y1, x2, y2]");

SHOULD_NOT_DO_GRADIENT(BBoxTransform);
} // namespace

template <>
bool BBoxTransformOp<float, CPUContext>::RunOnDevice() {
  const auto& roi_in = Input(0);
  const auto& delta_in = Input(1);
  const auto& iminfo_in = Input(2);
  auto* box_out = Output(0);

  const int N = roi_in.dim32(0);
  CAFFE_ENFORCE_EQ(roi_in.ndim(), 2);
  CAFFE_ENFORCE_GE(roi_in.dim32(1), 4);

  CAFFE_ENFORCE_EQ(roi_in.ndim(), 2);
  CAFFE_ENFORCE_EQ(delta_in.dim32(0), N);
  CAFFE_ENFORCE_EQ(delta_in.dim32(1) % 4, 0);

  DCHECK_EQ(weights_.size(), 4);

  CAFFE_ENFORCE_EQ(iminfo_in.size(), 3);
  ConstEigenVectorArrayMap<float> iminfo(iminfo_in.data<float>(), 3);
  int img_h = iminfo(0);
  int img_w = iminfo(1);

  Eigen::Map<const ERArrXXf> boxes0(
      roi_in.data<float>(), roi_in.dim32(0), roi_in.dim32(1));
  auto boxes = boxes0.rightCols(4);

  Eigen::Map<const ERArrXXf> deltas0(
      delta_in.data<float>(), delta_in.dim32(0), delta_in.dim32(1));

  box_out->ResizeLike(delta_in);
  Eigen::Map<ERArrXXf> new_boxes(
      box_out->mutable_data<float>(), box_out->dim32(0), box_out->dim32(1));

  int num_classes = deltas0.cols() / 4;
  for (int k = 0; k < num_classes; k++) {
    auto deltas = deltas0.block(0, k * 4, N, 4);
    auto trans_boxes = utils::bbox_transform(boxes, deltas, weights_);
    auto clip_boxes = utils::clip_boxes(trans_boxes, img_h, img_w);
    new_boxes.block(0, k * 4, N, 4) = clip_boxes;
  }

  return true;
}

} // namespace caffe2
