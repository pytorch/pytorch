#include "caffe2/image/image_input_op.h"

#ifdef USE_MKLDNN
#include <caffe2/ideep/operators/operator_fallback_ideep.h>
#include <caffe2/ideep/utils/ideep_operator.h>
#endif

namespace caffe2 {

template <>
bool ImageInputOp<CPUContext>::ApplyTransformOnGPU(
    const std::vector<std::int64_t>&,
    const c10::Device&) {
  return false;
}

REGISTER_CPU_OPERATOR(ImageInput, ImageInputOp<CPUContext>);

OPERATOR_SCHEMA(ImageInput)
    .NumInputs(0, 1)
    .NumOutputs(2, INT_MAX)
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& /* unused */) {
      vector<TensorShape> out(2);
      ArgumentHelper helper(def);
      int batch_size = helper.GetSingleArgument<int>("batch_size", 0);
      int crop = helper.GetSingleArgument<int>("crop", -1);
      int color = helper.GetSingleArgument<int>("color", 1);
      TORCH_CHECK_GT(crop, 0);
      out[0] = CreateTensorShape(
          vector<int>{batch_size, crop, crop, color ? 3 : 1},
          TensorProto::FLOAT);
      out[1] =
          CreateTensorShape(vector<int>{1, batch_size}, TensorProto::INT32);
      return out;
    })
    .SetDoc(R"DOC(
Imports and processes images from a database. For each run of the operator,
batch_size images will be processed. GPUs can optionally be used for
part of the processing.

The following transformations are applied to the image
  - A bounding box is applied to the initial image (optional)
  - The image is rescaled either up or down (with the scale argument) or
    just up (with the minsize argument)
  - The image is randomly cropped (crop size is passed as an argument but
    the location of the crop is random except if is_test is passed in which case
    the image in cropped at the center)
  - The image is normalized. Each of its color channels can have separate
    normalization values

The dimension of the output image will always be cropxcrop
)DOC")
    .Arg(
        "batch_size",
        "Number of images to output for each run of the operator"
        ". Must be 1 or greater")
    .Arg("color", "Number of color channels (1 or 3). Defaults to 1")
    .Arg("color_jitter", "Whether or not to do color jitter. Defaults to 0")
    .Arg(
        "img_saturation",
        "Image saturation scale used in color jittering. "
        "Defaults to 0.4")
    .Arg(
        "img_brightness",
        "Image brightness scale used in color jittering. "
        "Defaults to 0.4")
    .Arg(
        "img_contrast",
        "Image contrast scale used in color jittering. "
        "Defaults to 0.4")
    .Arg(
        "color_lighting",
        "Whether or not to do color lighting."
        " Defaults to 0")
    .Arg(
        "color_lighting_std",
        "Std of normal distribution where color lighting"
        " scaling factor is sampled. Defaults to 0.1")
    .Arg(
        "scale_jitter_type",
        "Type 0: No scale jittering "
        "Type 1: Inception-style scale jittering")
    .Arg(
        "label_type",
        "Type 0: single integer label for multi-class "
        "classification. Type 1: sparse active label indices for multi-label "
        "classification. Type 2: dense label embedding vector for label "
        "embedding regression")
    .Arg(
        "scale",
        "Scale the size of the smallest dimension of the image to"
        " this. Scale and minsize are mutually exclusive."
        " Must be larger than crop")
    .Arg(
        "minsize",
        "Scale the size of the smallest dimension of the image to"
        " this only if the size is initially smaller. Scale and minsize are"
        " mutually exclusive. Must be larger than crop.")
    .Arg(
        "warp",
        "If 1, both dimensions of the image will be set to minsize or"
        " scale; otherwise, the other dimension is proportionally scaled."
        " Defaults to 0")
    .Arg("crop", "Size to crop the image to. Must be provided")
    .Arg("mirror", "Whether or not to mirror the image. Defaults to 0")
    .Arg(
        "mean",
        "Mean by which to normalize color channels."
        " Defaults to 0.")
    .Arg(
        "mean_per_channel",
        "Vector of means per color channel "
        " (1 or 3 elements). Defaults to mean argument. Channel order BGR")
    .Arg(
        "std",
        "Standard deviation by which to normalize color channels."
        " Defaults to 1.")
    .Arg(
        "std_per_channel",
        "Vector of standard dev. per color channel "
        " (1 or 3 elements). Defaults to std argument. Channel order is BGR")
    .Arg("bounding_ymin", "Bounding box coordinate. Defaults to -1 (none)")
    .Arg("bounding_xmin", "Bounding box coordinate. Defaults to -1 (none)")
    .Arg("bounding_height", "Bounding box coordinate. Defaults to -1 (none)")
    .Arg("bounding_width", "Bounding box coordinate. Defaults to -1 (none)")
    .ArgIsTest("Set to 1 to do deterministic cropping. Defaults to 0")
    .Arg("use_caffe_datum", "1 if the input is in Caffe format. Defaults to 0")
    .Arg(
        "use_gpu_transform",
        "1 if GPU acceleration should be used."
        " Defaults to 0. Can only be 1 in a CUDAContext")
    .Arg(
        "decode_threads",
        "Number of CPU decode/transform threads."
        " Defaults to 4")
    .Arg("output_type", "If gpu_transform, can set to FLOAT or FLOAT16.")
    .Arg("db", "Name of the database (if not passed as input)")
    .Arg(
        "db_type",
        "Type of database (if not passed as input)."
        " Defaults to leveldb")
    .Arg(
        "output_sizes",
        "The sizes of any outputs besides the data and label "
        "(should have a number of elements equal to the number of additional "
        "outputs)")
    .Arg(
        "random_scale",
        "[min, max] shortest-side desired for image resize. "
        "Defaults to [-1, -1] or no random resize desired.")
    .Input(0, "reader", "The input reader (a db::DBReader)")
    .Output(0, "data", "Tensor containing the images")
    .Output(1, "label", "Tensor containing the labels")
    .Output(
        2,
        "additional outputs",
        "Any outputs after the first 2 will be "
        "Tensors read from the input TensorProtos");

NO_GRADIENT(ImageInput);

#ifdef USE_MKLDNN
REGISTER_IDEEP_OPERATOR(ImageInput, IDEEPFallbackOp<ImageInputOp<CPUContext>>);
#endif

} // namespace caffe2
