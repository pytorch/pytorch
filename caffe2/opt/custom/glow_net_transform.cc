#include "glow_net_transform.h"

#include <caffe2/opt/onnxifi_transformer.h>
#include <caffe2/opt/shape_info.h>
#include <caffe2/utils/string_utils.h>

#include <unordered_set>

C10_DEFINE_bool(onnxifi_debug_mode, false, "Enable onnxifi debug mode.");

C10_DEFINE_bool(
    onnxifi_adjust_batch,
    true,
    "Attach AdjustBatch ops at input/outputs of the Onnxifi ops");

C10_DEFINE_bool(
    onnxifi_loop_test_mode,
    false,
    "For test purpose only. Build a dummy net just to test the functionality");

C10_DEFINE_bool(
    merge_fp32_inputs_into_fp16,
    false,
    "Merge all the fp32 input tensors into one, convert it to fp16 and split it back");

C10_DEFINE_int32(
    onnxifi_min_ops,
    1,
    "Minimum number of ops for a subgraph to be lowered to backend");

C10_DEFINE_string(
    onnxifi_shape_hints,
    "",
    "Shape hints in the form of Name:d0,d1:d2;");

C10_DEFINE_string(
    onnxifi_blacklist,
    "",
    "A list of net positions whose corresponding op will be ignored "
    "to onnxifi. Example 0-50,61,62-70");

C10_DEFINE_string(
    onnxifi_blacklist_ops,
    "",
    "A list of operator types that will be ignored "
    "to onnxifi. Example Tanh,Mul");

C10_DEFINE_string(
    onnxifi_input_output_observe_list,
    "",
    "A list of net positions whose corresponding op's inputs and outputs will be"
    " observed. ");

namespace caffe2 {
namespace glow {

// The list in in the form of "0-3,5,6-7" which means, we will black list ops
// with net positions in [0,1,2,3,5,6,7]
std::unordered_set<int> ParseNetPositionList(const std::string& str) {
  std::unordered_set<int> net_position_list;
  if (str.empty()) {
    return net_position_list;
  }
  auto tokens = caffe2::split(',', str);
  for (const auto& token : tokens) {
    auto range = caffe2::split('-', token);
    if (range.size() == 1) {
      net_position_list.emplace(std::stoi(range[0]));
    } else if (range.size() == 2) {
      int from = std::stoi(range[0]);
      int to = std::stoi(range[1]);
      for (int i = from; i <= to; ++i) {
        net_position_list.emplace(i);
      }
    } else if (range.size() > 2) {
      LOG(WARNING) << "Ignoring illegal range: " << token;
    }
  }
  return net_position_list;
}

std::unordered_set<std::string> ParseBlackListOps(const std::string& str) {
  std::unordered_set<std::string> ops;
  if (str.empty()) {
    return ops;
  }
  auto tokens = caffe2::split(',', str);
  for (const auto& token : tokens) {
    ops.emplace(token);
  }
  return ops;
}

// Carrying out the ONNXIFI transform
void onnxifi(
    NetDef* net,
    Workspace* ws,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::vector<std::string>& weight_names,
    const std::unordered_set<int>& blacklist,
    const ShapeInfoMap& shape_hints,
    bool use_onnx,
    size_t max_batch_size,
    size_t max_seq_size,
    bool load_model_by_blob) {
  // Clean up the external input/output of the net
  net->mutable_external_input()->Clear();
  net->mutable_external_output()->Clear();
  for (const auto& i : input_names) {
    net->add_external_input(i);
  }
  for (const auto& w : weight_names) {
    net->add_external_input(w);
  }
  for (const auto& o : output_names) {
    net->add_external_output(o);
  }

  // ONNXIFI transform
  OnnxifiTransformerOptions opts;
  opts.use_onnx = use_onnx;
  opts.bound_shape_spec.max_batch_size = max_batch_size;
  opts.bound_shape_spec.max_seq_size = max_seq_size;
  opts.debug = FLAGS_onnxifi_debug_mode;
  opts.adjust_batch = FLAGS_onnxifi_adjust_batch;
  opts.min_ops = FLAGS_onnxifi_min_ops;
  opts.load_model_by_blob = load_model_by_blob;
  opts.merge_fp32_inputs_into_fp16 = FLAGS_merge_fp32_inputs_into_fp16;
  opts.loop_test = FLAGS_onnxifi_loop_test_mode;

  ShapeInfoMap more_shape_hints = shape_hints;
  if (!FLAGS_onnxifi_shape_hints.empty()) {
    parseShapeInfoMapFromString(FLAGS_onnxifi_shape_hints, more_shape_hints);
  }

  // Before applying backlist, make sure the ops in the net all have an net_pos;
  caffe2::BackendTransformerBase::annotateOpIndex(net);

  // Parse the blacklist
  auto more_blacklist = ParseNetPositionList(FLAGS_onnxifi_blacklist);
  for (const auto& b : blacklist) {
    more_blacklist.emplace(b);
  }

  // ONNX mode will change the op order so it doesn't apply here
  if (!opts.use_onnx) {
    auto blacklisted_ops = ParseBlackListOps(FLAGS_onnxifi_blacklist_ops);
    for (const auto& op : net->op()) {
      if (blacklisted_ops.count(op.type())) {
        ArgumentHelper helper(op);
        more_blacklist.emplace(helper.GetSingleArgument(op, kNetPos, -1));
      }
    }
  }

  // Attach observation nodes
  //
  // When we want to observe intermediate tensors value out of the onnxifi op,
  // we use the following trick:
  //
  // 1. for specified op, we find its input and outputs.
  // 2. for each input and output, we create a new copy op and attach it as an
  // input to the copy.
  // 3. we blacklist these new copy operators from onnxification. This forces
  // these intermediate tensors to also become outputs of the onnxifi op.
  // 4. we put the right arguments on the copy ops so TensorObserver can print
  // out the values.
  auto ops_to_observe =
      ParseNetPositionList(FLAGS_onnxifi_input_output_observe_list);
  std::unordered_set<std::string> tensors_to_observe;
  for (const auto& op : ops_to_observe) {
    if (op >= net->op().size()) {
      CAFFE_THROW(
          "Cannot observe operator at position ", op, " (out of range)");
    }
    const auto& op_to_observe = net->op(op);
    tensors_to_observe.insert(
        op_to_observe.input().begin(), op_to_observe.input().end());

    if ((op_to_observe.type() == "Concat" ||
         op_to_observe.type() == "Reshape") &&
        op_to_observe.output().size() == 2) {
      tensors_to_observe.insert(op_to_observe.output(0));
    } else {
      tensors_to_observe.insert(
          op_to_observe.output().begin(), op_to_observe.output().end());
    }
  }
  for (const auto& tensor : tensors_to_observe) {
    OperatorDef copy_op;
    copy_op.set_type("Copy");
    copy_op.add_input(tensor);
    copy_op.add_output(tensor + "_copy_output_ignore");
    auto pos = net->op().size();
    AddArgument(kNetPos, pos, &copy_op);
    AddArgument("observe_input_tensors", 1, &copy_op);
    net->add_op()->CopyFrom(copy_op);
    more_blacklist.emplace(pos);
  }

  OnnxifiTransformer ts(opts);
  ts.transform(ws, net, weight_names, more_shape_hints, more_blacklist);
  if (FLAGS_onnxifi_debug_mode) {
    WriteProtoToTextFile(*net, "debug_transformed_net.pb_txt");
  }

  // Cleanup the input from the workspace
  for (const auto& i : input_names) {
    ws->RemoveBlob(i);
  }
}

} // namespace glow
} // namespace caffe2
