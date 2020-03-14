#include <caffe2/video/video_input_op.h>

namespace caffe2 {

REGISTER_CPU_OPERATOR(VideoInput, VideoInputOp<CPUContext>);

OPERATOR_SCHEMA(VideoInput)
    .NumInputs(0, 1)
    .NumOutputs(2, 5)
    .TensorInferenceFunction(
        [](const OperatorDef& def,
           const vector<TensorShape>& /* unused */ /*in*/) {
          ArgumentHelper helper(def);
          int batch_size = helper.GetSingleArgument<int>("batch_size", 0);
          int clip_per_video =
              helper.GetSingleArgument<int>("clip_per_video", 1);
          int crop_size = helper.GetSingleArgument<int>("crop_size", -1);
          int length_rgb = helper.GetSingleArgument<int>("length_rgb", 0);
          int channels_rgb = helper.GetSingleArgument<int>("channels_rgb", 3);
          int length_of = helper.GetSingleArgument<int>("length_of", 0);
          int channels_of = helper.GetSingleArgument<int>("channels_of", 2);

          // get the flags
          bool get_rgb = helper.GetSingleArgument<bool>("get_rgb", true);
          bool get_optical_flow =
              helper.GetSingleArgument<bool>("get_optical_flow", false);
          bool do_multi_label =
              helper.GetSingleArgument<bool>("do_multi_label", false);
          bool get_video_id =
              helper.GetSingleArgument<bool>("get_video_id", false);
          bool get_start_frame =
              helper.GetSingleArgument<bool>("get_start_frame", false);
          // get starting positions if available
          vector<int> clip_start_positions =
              helper.GetRepeatedArgument<int>("clip_start_positions", {});
          // In case clip_start_positions are given, set the clip_per_video arg
          if (clip_start_positions.size() > 0) {
            clip_per_video = clip_start_positions.size();
          }

          int output_size = 1;
          if (get_rgb) {
            output_size++;
          }
          if (get_optical_flow) {
            output_size++;
          }
          if (get_video_id) {
            output_size++;
          }
          if (get_start_frame) {
            output_size++;
          }

          int index = 0;
          vector<TensorShape> out(output_size);
          CHECK_GT(crop_size, 0);
          batch_size *= clip_per_video;
          if (get_rgb) {
            out[index++] = CreateTensorShape(
                vector<int>{
                    batch_size, channels_rgb, length_rgb, crop_size, crop_size},
                TensorProto::FLOAT);
          }
          if (get_optical_flow) {
            out[index++] = CreateTensorShape(
                vector<int>{
                    batch_size, channels_of, length_of, crop_size, crop_size},
                TensorProto::FLOAT);
          }
          if (!do_multi_label) {
            out[index++] = CreateTensorShape(
                vector<int>{1, batch_size}, TensorProto::INT32);
          } else {
            int num_of_class = helper.GetSingleArgument<int>("num_of_class", 0);
            out[index++] = CreateTensorShape(
                vector<int>{batch_size, num_of_class}, TensorProto::INT32);
          }
          if (get_video_id) {
            out[index++] = CreateTensorShape(
                vector<int64_t>{1, batch_size}, TensorProto::INT64);
          }
          if (get_start_frame) {
            out[index] = CreateTensorShape(
                vector<int>{1, batch_size}, TensorProto::INT32);
          }

          return out;
        });

NO_GRADIENT(VideoInput);

} // namespace caffe2
