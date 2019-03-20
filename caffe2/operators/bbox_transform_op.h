// Copyright 2004-present Facebook. All Rights Reserved.

#ifndef BBOX_TRANSFORM_OP_H_
#define BBOX_TRANSFORM_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

C10_DECLARE_CAFFE2_OPERATOR(BBoxTransform)

namespace caffe2 {

template <typename T, class Context>
class BBoxTransformOp final : public Operator<Context> {
 public:
  template<class... Args>
  explicit BBoxTransformOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        weights_(this->template GetRepeatedArgument<T>(
            "weights",
            vector<T>{1.0f, 1.0f, 1.0f, 1.0f})),
        apply_scale_(
            this->template GetSingleArgument<bool>("apply_scale", true)),
        rotated_(this->template GetSingleArgument<bool>("rotated", false)),
        angle_bound_on_(
            this->template GetSingleArgument<bool>("angle_bound_on", true)),
        angle_bound_lo_(
            this->template GetSingleArgument<int>("angle_bound_lo", -90)),
        angle_bound_hi_(
            this->template GetSingleArgument<int>("angle_bound_hi", 90)),
        clip_angle_thresh_(
            this->template GetSingleArgument<float>("clip_angle_thresh", 1.0)) {
    CAFFE_ENFORCE_EQ(
        weights_.size(),
        4,
        "weights size " + c10::to_string(weights_.size()) + "must be 4.");
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  // weights [wx, wy, ww, wh] to apply to the regression target
  vector<T> weights_;
  // Transform the boxes to the scaled image space after applying the bbox
  //   deltas.
  // Set to false to match the detectron code, set to true for the keypoint
  //   model and for backward compatibility
  bool apply_scale_{true};
  // Set for RRPN case to handle rotated boxes. Inputs should be in format
  // [ctr_x, ctr_y, width, height, angle (in degrees)].
  bool rotated_{false};
  // If set, for rotated boxes in RRPN, output angles are normalized to be
  // within [angle_bound_lo, angle_bound_hi].
  bool angle_bound_on_{true};
  int angle_bound_lo_{-90};
  int angle_bound_hi_{90};
  // For RRPN, clip almost horizontal boxes within this threshold of
  // tolerance for backward compatibility. Set to negative value for
  // no clipping.
  float clip_angle_thresh_{1.0};
};

} // namespace caffe2

#endif // BBOX_TRANSFORM_OP_H_
