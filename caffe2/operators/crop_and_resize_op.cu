#include "caffe2/operators/crop_and_resize_op.h"
#include "caffe2/core/context_gpu.h"

#include <cmath>

namespace caffe2 {

namespace {

template <typename T>
inline __device__ T gpu_atomic_add(const T val, T* address);

template <>
inline __device__ float gpu_atomic_add(const float val, float* address) {
    return atomicAdd(address, val);
}

template<typename T>
__global__ void CropAndResizeForward(
    const int nthreads,
    const T* fm_data,
    const T* boxes_data,
    int batch_size,
    int fm_height,
    int fm_width,
    int crop_height,
    int crop_width,
    int channels,
    int num_boxes,
    int box_dim,
    int method,
    float extrapolation_value,
    T* crops_data
) {

    CUDA_1D_KERNEL_LOOP(out_idx, nthreads) {

        const int index = out_idx;

        const int x = index % crop_width;
        const int y = (index / crop_width) % crop_height;
        const int c = (index / crop_width / crop_height) % channels;
        const int b = index / crop_width / crop_height / channels;

        const int batch_index = static_cast<int>(boxes_data[b * box_dim]);

        if (batch_index < 0 || batch_index >= batch_size) {
            continue;
        }

        const float x1 = static_cast<float>(boxes_data[b * box_dim + 1]);
        const float y1 = static_cast<float>(boxes_data[b * box_dim + 2]);
        const float x2 = static_cast<float>(boxes_data[b * box_dim + 3]);
        const float y2 = static_cast<float>(boxes_data[b * box_dim + 4]);

        const float height_scale = (crop_height > 1) ?
            ((y2 - y1) * (fm_height - 1) / (crop_height - 1)) : 0.0f;

        const float width_scale = (crop_width > 1) ?
            ((x2 - x1) * (fm_width - 1) / (crop_width - 1)) : 0.0f;

        const float in_y = (crop_height > 1) ?
            (y1 * (fm_height - 1) + y * height_scale) :
            (0.5f * (y1 + y2) * (fm_height - 1));


        if (in_y < 0 || in_y > fm_height - 1) {
            crops_data[out_idx] = extrapolation_value;
            continue;
        }

        const float in_x = (crop_width > 1) ?
            (x1 * (fm_width - 1) + x * width_scale) :
            (0.5f * (x1 + x2) * (fm_width - 1));

        if (in_x < 0 || in_x > fm_width - 1) {
            crops_data[out_idx] = extrapolation_value;
            continue;
        }

        if (method == BILINEAR) {

            const int top_y_index = floorf(in_y);
            const int bottom_y_index = ceilf(in_y);
            const float y_lerp = in_y - top_y_index;

            const int left_x_index = floorf(in_x);
            const int right_x_index = ceilf(in_x);
            const float x_lerp = in_x - left_x_index;

            const float top_left =
                static_cast<float>(fm_data[batch_index * channels * fm_height * fm_width +
                c * fm_height * fm_width +
                top_y_index * fm_width +
                left_x_index]);
            const float top_right =
                static_cast<float>(fm_data[batch_index * channels * fm_height * fm_width +
                c * fm_height * fm_width +
                top_y_index * fm_width +
                right_x_index]);
            const float bottom_left =
                static_cast<float>(fm_data[batch_index * channels * fm_height * fm_width +
                c * fm_height * fm_width +
                bottom_y_index * fm_width +
                left_x_index]);
            const float bottom_right =
                static_cast<float>(fm_data[batch_index * channels * fm_height * fm_width +
                c * fm_height * fm_width +
                bottom_y_index * fm_width +
                right_x_index]);

            const float top = top_left + (top_right - top_left) * x_lerp;
            const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;

            crops_data[out_idx] = top + (bottom - top) * y_lerp;

        } else {

            const int nearest_x_index = static_cast<int>(in_x + 0.5f);
            const int nearest_y_index = static_cast<int>(in_y + 0.5f);

            crops_data[out_idx] = static_cast<float>(fm_data[batch_index * channels * fm_height * fm_width +
                c * fm_height * fm_width +
                nearest_y_index * fm_width +
                nearest_x_index]);

        }

    }

}

template<typename T>
__global__ void CropAndResizeBackward(
    const int nthreads,
    const T* dY_data,
    const T* boxes_data,
    int batch_size,
    int fm_height,
    int fm_width,
    int crop_height,
    int crop_width,
    int channels,
    int num_boxes,
    int box_dim,
    int method,
    T* dX_data)
{

    CUDA_1D_KERNEL_LOOP(out_idx, nthreads) {

        const int index = out_idx;

        const int x = index % crop_width;
        const int y = (index / crop_width) % crop_height;
        const int c = (index / crop_width / crop_height) % channels;
        const int b = index / crop_width / crop_height / channels;

        const int batch_index = static_cast<int>(boxes_data[b * box_dim]);

        if (batch_index < 0 || batch_index >= batch_size) {
            continue;
        }

        const float x1 = static_cast<float>(boxes_data[b * box_dim + 1]);
        const float y1 = static_cast<float>(boxes_data[b * box_dim + 2]);
        const float x2 = static_cast<float>(boxes_data[b * box_dim + 3]);
        const float y2 = static_cast<float>(boxes_data[b * box_dim + 4]);

        const float height_scale = (crop_height > 1) ?
            ((y2 - y1) * (fm_height - 1) / (crop_height - 1)) : 0.0f;

        const float width_scale = (crop_width > 1) ?
            ((x2 - x1) * (fm_width - 1) / (crop_width - 1)) : 0.0f;

        const float in_y = (crop_height > 1) ?
            (y1 * (fm_height - 1) + y * height_scale) :
            (0.5f * (y1 + y2) * (fm_height - 1));


        if (in_y < 0 || in_y > fm_height - 1) {
            continue;
        }

        const float in_x = (crop_width > 1) ?
            (x1 * (fm_width - 1) + x * width_scale) :
            (0.5f * (x1 + x2) * (fm_width - 1));

        if (in_x < 0 || in_x > fm_width - 1) {
            continue;
        }

        if (method == BILINEAR) {

            const int top_y_index = floorf(in_y);
            const int bottom_y_index = ceilf(in_y);
            const float y_lerp = in_y - top_y_index;

            const int left_x_index = floorf(in_x);
            const int right_x_index = ceilf(in_x);
            const float x_lerp = in_x - left_x_index;

            const float dtop = (1 - y_lerp) * dY_data[out_idx];

            gpu_atomic_add(static_cast<T>((1 - x_lerp) * dtop),
                dX_data + batch_index * channels * fm_height * fm_width +
                        c * fm_height * fm_width +
                        top_y_index * fm_width +
                        left_x_index);
            gpu_atomic_add(static_cast<T>(x_lerp * dtop),
                dX_data + batch_index * channels * fm_height * fm_width +
                        c * fm_height * fm_width +
                        top_y_index * fm_width +
                        right_x_index);

            const float dbottom = y_lerp * dY_data[out_idx];

            gpu_atomic_add(static_cast<T>((1 - x_lerp) * dbottom),
                dX_data + batch_index * channels * fm_height * fm_width +
                    c * fm_height * fm_width +
                    bottom_y_index * fm_width +
                    left_x_index);

            gpu_atomic_add(static_cast<T>(x_lerp * dbottom),
                dX_data + batch_index * channels * fm_height * fm_width +
                        c * fm_height * fm_width +
                        bottom_y_index * fm_width +
                        right_x_index);

        } else {

            const int nearest_x_index = static_cast<int>(in_x + 0.5f);
            const int nearest_y_index = static_cast<int>(in_y + 0.5f);

            gpu_atomic_add(static_cast<T>(dY_data[out_idx]),
                dX_data + batch_index * channels * fm_height * fm_width +
                c * fm_height * fm_width +
                nearest_y_index * fm_width +
                nearest_x_index);

        }

    }

}

} // namespace

template<>
bool CropAndResizeOp<float, CUDAContext>::RunOnDevice() {

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

    const auto output_size = crops->numel();
    const int num_boxes = boxes.dim32(0);
    const int box_dim = boxes.dim32(1);


    CAFFE_ENFORCE_EQ(output_size, num_boxes * channels * crop_height_ * crop_width_);

    if (output_size > 0) { // handle case of no rois

        CropAndResizeForward<<<CAFFE_GET_BLOCKS(output_size),
            CAFFE_CUDA_NUM_THREADS,
            0, context_.cuda_stream()>>>(
                output_size,
                fm_data,
                boxes_data,
                batch_size,
                fm_height,
                fm_width,
                crop_height_,
                crop_width_,
                channels,
                num_boxes,
                box_dim,
                static_cast<int>(method_),
                extrapolation_value_,
                crops_data);

    }

    return true;
}

template<>
bool CropAndResizeGradientOp<float, CUDAContext>::RunOnDevice() {

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

    math::Set<float, CUDAContext>(dX->numel(), 0.0f, dX_data, &context_);

    const auto output_size = dY.numel();
    const int num_boxes = boxes.dim32(0);
    const int box_dim = boxes.dim32(1);

    CAFFE_ENFORCE_EQ(output_size, num_boxes * channels * crop_height_ * crop_width_);

    if (output_size > 0) { // handle case of no rois

        CropAndResizeBackward<<<CAFFE_GET_BLOCKS(output_size),
            CAFFE_CUDA_NUM_THREADS,
            0, context_.cuda_stream()>>>(
                output_size,
                dY_data,
                boxes_data,
                batch_size,
                fm_height,
                fm_width,
                crop_height_,
                crop_width_,
                channels,
                num_boxes,
                box_dim,
                static_cast<int>(method_),
                dX_data);

    }

    return true;
}

REGISTER_CUDA_OPERATOR(CropAndResize, CropAndResizeOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(CropAndResizeGradient, CropAndResizeGradientOp<float, CUDAContext>);

} // namespace caffe2

