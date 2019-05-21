#include "caffe2/operators/crop_and_resize_op.h"

#include <math.h>

namespace caffe2 {

template<>
bool CropAndResizeOp<float, CPUContext>::RunOnDevice() {

    const auto& fm = Input(FM);
    const auto& boxes = Input(BOXES);

    auto* crops = Output(0);

    CAFFE_ENFORCE_EQ(fm.dim(), 4);
    CAFFE_ENFORCE_EQ(boxes.dim(), 2);

    // [batch, x1, y1, x2, y2]
    CAFFE_ENFORCE_EQ(boxes.dim32(1), 5);

    const int batch_size = fm.dim32(0);
    const int channels = fm.dim32(1);
    const int fm_height = fm.dim32(2);
    const int fm_width = fm.dim32(3);

    crops->Resize(boxes.dim32(0), channels, crop_height_, crop_width_);

    const auto* fm_data = fm.template data<float>();
    const auto* boxes_data = boxes.template data<float>();
    auto* crops_data = crops->template mutable_data<float>();

    const int box_dim = boxes.dim32(1);

    for (int b = 0; b < boxes.dim32(0); ++b) {

        const int batch_index = static_cast<int>(boxes_data[b * box_dim]);

        if (batch_index < 0 || batch_index >= batch_size) {
            continue;
        }

        const float x1 = boxes_data[b * box_dim + 1];
        const float y1 = boxes_data[b * box_dim + 2];
        const float x2 = boxes_data[b * box_dim + 3];
        const float y2 = boxes_data[b * box_dim + 4];

        const float height_scale = (crop_height_ > 1) ?
                                    ((y2 - y1) * (fm_height - 1) / (crop_height_ - 1)) : 0.0f;

        const float width_scale = (crop_width_ > 1) ?
                                    ((x2 - x1) * (fm_width - 1) / (crop_width_ - 1)) : 0.0f;

        for (int c = 0; c < channels; ++c) {
            for (int y = 0; y < crop_height_; ++y) {

                const float in_y = (crop_height_ > 1) ?
                    (y1 * (fm_height - 1) + y * height_scale) :
                    (0.5f * (y1 + y2) * (fm_height - 1));


                if (in_y < 0 || in_y > fm_height - 1) {
                    for (int x = 0; x < crop_width_; ++x) {
                        crops_data[b * channels * crop_height_ * crop_width_ +
                            c * crop_height_ * crop_width_ +
                            y * crop_width_ +
                            x] = extrapolation_value_;
                    }
                    continue;
                }


                if (method_ == CropAndResizeMethod::BILINEAR) {

                    const int top_y_index = floorf(in_y);
                    const int bottom_y_index = ceilf(in_y);
                    const float y_lerp = in_y - top_y_index;

                    for (int x = 0; x < crop_width_; ++x) {
                        const float in_x = (crop_width_ > 1) ?
                            (x1 * (fm_width - 1) + x * width_scale) :
                            (0.5f * (x1 + x2) * (fm_width - 1));

                        if (in_x < 0 || in_x > fm_width - 1) {
                            crops_data[b * channels * crop_height_ * crop_width_ +
                                c * crop_height_ * crop_width_ +
                                y * crop_width_ +
                                x] = extrapolation_value_;
                            continue;
                        }

                        const int left_x_index = floorf(in_x);
                        const int right_x_index = ceilf(in_x);
                        const float x_lerp = in_x - left_x_index;

                        const float top_left =
                            fm_data[batch_index * channels * fm_height * fm_width +
                                c * fm_height * fm_width +
                                top_y_index * fm_width +
                                left_x_index];
                        const float top_right =
                            fm_data[batch_index * channels * fm_height * fm_width +
                                c * fm_height * fm_width +
                                top_y_index * fm_width +
                                right_x_index];
                        const float bottom_left =
                            fm_data[batch_index * channels * fm_height * fm_width +
                                c * fm_height * fm_width +
                                bottom_y_index * fm_width +
                                left_x_index];
                        const float bottom_right =
                            fm_data[batch_index * channels * fm_height * fm_width +
                                c * fm_height * fm_width +
                                bottom_y_index * fm_width +
                                right_x_index];

                        const float top = top_left + (top_right - top_left) * x_lerp;
                        const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;

                        crops_data[b * channels * crop_height_ * crop_width_ +
                            c * crop_height_ * crop_width_ +
                            y * crop_width_ +
                            x] = top + (bottom - top) * y_lerp;

                    }

                } else { // method_ == CropAndResizeMethod::NEAREST

                    CAFFE_ENFORCE_EQ(method_, CropAndResizeMethod::NEAREST);

                    for (int x = 0; x < crop_width_; ++x) {
                        const float in_x = (crop_width_ > 1) ?
                            (x1 * (fm_width - 1) + x * width_scale) :
                            (0.5f * (x1 + x2) * (fm_width - 1));

                        if (in_x < 0 || in_x > fm_width - 1) {
                            crops_data[b * channels * crop_height_ * crop_width_ +
                                c * crop_height_ * crop_width_ +
                                y * crop_width_ +
                                x] = extrapolation_value_;
                            continue;
                        }

                        const int nearest_x_index = static_cast<int>(in_x + 0.5f);
                        const int nearest_y_index = static_cast<int>(in_y + 0.5f);

                        crops_data[b * channels * crop_height_ * crop_width_ +
                            c * crop_height_ * crop_width_ +
                            y * crop_width_ +
                            x] = fm_data[batch_index * channels * fm_height * fm_width +
                                    c * fm_height * fm_width +
                                    nearest_y_index * fm_width +
                                    nearest_x_index];
                    }

                }
            }
        }
    }

    return true;
}

template<>
bool CropAndResizeGradientOp<float, CPUContext>::RunOnDevice() {

    const auto& X = Input(0);
    const auto& boxes = Input(1);
    const auto& dY = Input(2);

    auto* dX = Output(0);

    dX->ResizeLike(X);

    CAFFE_ENFORCE_EQ(dX->dim(), 4);
    CAFFE_ENFORCE_EQ(boxes.dim(), 2);

    // [batch, x1, y1, x2, y2]
    CAFFE_ENFORCE_EQ(boxes.dim32(1), 5);

    CAFFE_ENFORCE_EQ(dY.dim32(0), boxes.dim32(0));
    CAFFE_ENFORCE_EQ(dY.dim32(1), dX->dim32(1));
    CAFFE_ENFORCE_EQ(dY.dim32(2), crop_height_);
    CAFFE_ENFORCE_EQ(dY.dim32(3), crop_width_);

    const int batch_size = dX->dim32(0);
    const int channels = dX->dim32(1);
    const int fm_height = dX->dim32(2);
    const int fm_width = dX->dim32(3);

    const auto* X_data = X.template data<float>();
    const auto* boxes_data = boxes.template data<float>();
    const auto* dY_data = dY.template data<float>();

    auto* dX_data = dX->template mutable_data<float>();

    math::Set<float, CPUContext>(
        dX->numel(), 0.0f, dX_data, &context_);

    const int box_dim = boxes.dim32(1);

    for (int b = 0; b < boxes.dim32(0); ++b) {

        const int batch_index = static_cast<int>(boxes_data[b * box_dim]);

        if (batch_index < 0 || batch_index >= batch_size) {
            continue;
        }

        const float x1 = boxes_data[b * box_dim + 1];
        const float y1 = boxes_data[b * box_dim + 2];
        const float x2 = boxes_data[b * box_dim + 3];
        const float y2 = boxes_data[b * box_dim + 4];

        const float height_scale = (crop_height_ > 1) ?
            ((y2 - y1) * (fm_height - 1) / (crop_height_ - 1)) : 0.0f;

        const float width_scale = (crop_width_ > 1) ?
            ((x2 - x1) * (fm_width - 1) / (crop_width_ - 1)) : 0.0f;

        for (int c = 0; c < channels; ++c) {
            for (int y = 0; y < crop_height_; ++y) {

                const float in_y = (crop_height_ > 1) ?
                    (y1 * (fm_height - 1) + y * height_scale) :
                    (0.5f * (y1 + y2) * (fm_height - 1));


                if (in_y < 0 || in_y > fm_height - 1) {
                    continue;
                }

                if (method_ == CropAndResizeMethod::BILINEAR) {

                    const int top_y_index = floorf(in_y);
                    const int bottom_y_index = ceilf(in_y);
                    const float y_lerp = in_y - top_y_index;

                    for (int x = 0; x < crop_width_; ++x) {
                        const float in_x = (crop_width_ > 1) ?
                            (x1 * (fm_width - 1) + x * width_scale) :
                            (0.5f * (x1 + x2) * (fm_width - 1));

                        if (in_x < 0 || in_x > fm_width - 1) {
                            continue;
                        }

                        const int left_x_index = floorf(in_x);
                        const int right_x_index = ceilf(in_x);
                        const float x_lerp = in_x - left_x_index;

                        const float dtop = (1 - y_lerp) * dY_data[b * channels * crop_height_ * crop_width_ +
                                                                c * crop_height_ * crop_width_ +
                                                                y * crop_width_ +
                                                                x];

                        dX_data[batch_index * channels * fm_height * fm_width +
                            c * fm_height * fm_width +
                            top_y_index * fm_width +
                            left_x_index] += ((1 - x_lerp) * dtop);

                        dX_data[batch_index * channels * fm_height * fm_width +
                            c * fm_height * fm_width +
                            top_y_index * fm_width +
                            right_x_index] += x_lerp * dtop;

                        const float dbottom = y_lerp * dY_data[b * channels * crop_height_ * crop_width_ +
                                                            c * crop_height_ * crop_width_ +
                                                            y * crop_width_ +
                                                            x];

                        dX_data[batch_index * channels * fm_height * fm_width +
                            c * fm_height * fm_width +
                            bottom_y_index * fm_width +
                            left_x_index] += (1 - x_lerp) * dbottom;

                        dX_data[batch_index * channels * fm_height * fm_width +
                            c * fm_height * fm_width +
                            bottom_y_index * fm_width +
                            right_x_index] += x_lerp * dbottom;

                    }

                } else { // method_ == CropAndResizeMethod::NEAREST

                    CAFFE_ENFORCE_EQ(method_, CropAndResizeMethod::NEAREST);

                    for (int x = 0; x < crop_width_; ++x) {

                        const float in_x = (crop_width_ > 1) ?
                            (x1 * (fm_width - 1) + x * width_scale) :
                            (0.5f * (x1 + x2) * (fm_width - 1));

                        if (in_x < 0 || in_x > fm_width - 1) {
                            continue;
                        }

                        const int nearest_x_index = static_cast<int>(in_x + 0.5f);
                        const int nearest_y_index = static_cast<int>(in_y + 0.5f);

                        dX_data[batch_index * channels * fm_height * fm_width +
                            c * fm_height * fm_width +
                            nearest_y_index * fm_width +
                            nearest_x_index] += dY_data[b * channels * crop_height_ * crop_width_ +
                                                    c * crop_height_ * crop_width_ +
                                                    y * crop_width_ +
                                                    x];
                    }

                }
            }
        }
    }


    return true;
}


REGISTER_CPU_OPERATOR(CropAndResize, CropAndResizeOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(CropAndResizeGradient, CropAndResizeGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(CropAndResize)
    .NumInputs(2)
    .NumOutputs(1)
    .Arg("crop_width", "Width of the cropped region.")
    .Arg("crop_height", "Height of the cropped region.")
    .Arg("method", "Interpolation method."
        "int (default 0). Supported options are 0 for bilinear and 1 for nearest")
    .Arg("extrapolation_value", "float (deafault 0). Extrapolation value for pixels outside of image region.")
    .SetDoc(R"DOC(
Takes input NCHW tensor and outputs regions, specified by boxes, which come as the second input tensor. The output
regions are being resized to [crop_width x crop_height]. Interpolation method is specified by method argument.
)DOC")
.Input(0, "FM", "Input NCHW tensor.")
.Input(1, "boxes", "Boxes, specifying cropping coordinates.")
.Output(0, "crops", "Output cropped images.");

OPERATOR_SCHEMA(CropAndResizeGradient).NumInputs(3).NumOutputs(1);

class GetCropAndResizeGradient : public GradientMakerBase {
    using GradientMakerBase::GradientMakerBase;
    vector<OperatorDef> GetGradientDefs() override {
        return SingleGradientDef(
            "CropAndResizeGradient", "",
            vector<string>{I(0), I(1), GO(0)},
            vector<string>{GI(0)});
    }
};

REGISTER_GRADIENT(CropAndResize, GetCropAndResizeGradient);

} // namespace caffe2

