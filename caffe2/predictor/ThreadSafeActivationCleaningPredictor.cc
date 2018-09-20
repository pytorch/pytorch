#include "caffe2/predictor/ThreadSafeActivationCleaningPredictor.h"

#include "caffe2/core/db.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/memonger.h"
#include "caffe2/proto/predictor_consts.pb.h"
#include "caffe2/utils/proto_utils.h"

CAFFE2_DEFINE_bool(
    caffe2_predictor_cleanup_activations,
    true,
    "Clean up activations after each prediction");

namespace caffe2 {
namespace {

size_t TensorAdjustedCapacity(const TensorCPU& tensor) {
  // Don't change the size if it is already aligned, otherwise increase the size
  // to make it aligned
  // Note: everything below is size_t
  return (tensor.capacity_nbytes() + gCaffe2Alignment - 1) &
      (~(gCaffe2Alignment - 1));
}

void ApplyMemonger(const InferenceGraph& graph) {
  // Run memonger so it doesn't affect outputs
  // Other blobs are okay to touch - nothing writes to parameters
  // as this is guaranteed by ThreadSafeActivationCleaningPredictor and inputs
  // are owned by ThreadSafeActivationCleaningPredictor
  // TODO: make inputs participate in memory reuse. Currently memonger
  // Doesn't reuse blobs you don't write into

  // Memonger currently doesn't make inputs to participate in the blob reuse.
  // But we explicitly define them static. If memonger starts to make them
  // participate, we don't want that for a case where caller provided inputs
  // which were filled via ShareExternalPointer.

  std::set<string> static_blobs{graph.input_names.begin(),
                                graph.input_names.end()};
  static_blobs.insert(graph.output_names.begin(), graph.output_names.end());

  *graph.predict_net_def = caffe2::memonger::optimize_inference_net(
      *graph.predict_net_def, static_blobs);
  VLOG(1) << "memonger optimized net: "
          << ProtoDebugString(*graph.predict_net_def);
}

void EnforceWorkspaceIsConstant(
    const Workspace* parameter_workspace,
    const NetDef& net,
    const std::string& net_type) {
  for (const auto& op : net.op()) {
    for (const auto& out : op.output()) {
      CAFFE_ENFORCE(
          !parameter_workspace->HasBlob(out),
          "Net ",
          net.name(),
          " of type ",
          net_type,
          " writes to blob ",
          out,
          " which exists in the parameter workspace");
    }
  }
}

void EnforceAllOutputsArePresent(
    const NetDef& predict_net,
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs) {
  std::set<std::string> all_possible_outputs;
  for (const auto& op : predict_net.op()) {
    for (const auto& out : op.output()) {
      all_possible_outputs.insert(out);
    }
  }
  all_possible_outputs.insert(inputs.begin(), inputs.end());
  for (const auto& output : outputs) {
    CAFFE_ENFORCE(
        all_possible_outputs.count(output),
        "output ",
        output,
        " is not part of the prediction net and a list of inputs.",
        " All inputs: ",
        caffe2::Join(", ", inputs),
        " All possible outputs: ",
        caffe2::Join(", ", all_possible_outputs),
        " Prediction net proto: ",
        ProtoDebugString(predict_net));
  }
}

void EnforcePredictorInitializationInvariants(
    const Workspace* parameter_workspace,
    const InferenceGraph& graph) {
  VLOG(1) << "predict_init_net: "
          << ProtoDebugString(*graph.predict_init_net_def);
  VLOG(1) << "predict_net: " << ProtoDebugString(*graph.predict_net_def);
  for (const auto& param : graph.parameter_names) {
    CAFFE_ENFORCE(
        parameter_workspace->HasBlob(param),
        "Parameter blob ",
        param,
        " is not in the parameter workspace");
  }

  // Now lets make sure that prediction nets don't change parameter master
  // workspace. This way we guarantee that model parameters don't get changed
  // by accident. For example, if some of learning ops got saved
  EnforceWorkspaceIsConstant(
      parameter_workspace, *graph.predict_init_net_def, "predict_init_net");
  EnforceWorkspaceIsConstant(
      parameter_workspace, *graph.predict_net_def, "predict_net");

  // A few more checks - this time make sure that inputs and outputs don't
  // belong to the parameter workspace workspace
  for (const auto& input : graph.input_names) {
    CAFFE_ENFORCE(
        !parameter_workspace->HasBlob(input),
        "Input blob ",
        input,
        " found in the parameter workspace");
  }

  for (const auto& output : graph.output_names) {
    CAFFE_ENFORCE(
        !parameter_workspace->HasBlob(output),
        "Output blob ",
        output,
        " found in the parameter workspace");
  }

  EnforceAllOutputsArePresent(
      *graph.predict_net_def, graph.input_names, graph.output_names);
}

std::vector<std::string> DeduceInternalBlobs(
    const NetDef& predict_net,
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& final_outputs) {
  std::set<std::string> internal_blobs;

  std::set<std::string> final_outputs_set(
      final_outputs.begin(), final_outputs.end());
  std::set<std::string> inputs_set(inputs.begin(), inputs.end());

  for (const auto& op : predict_net.op()) {
    for (const auto& blob : op.output()) {
      if (final_outputs_set.count(blob) == 0 && inputs_set.count(blob) == 0) {
        internal_blobs.insert(blob);
      }
    }
  }

  return {internal_blobs.begin(), internal_blobs.end()};
}

} // namespace

ThreadSafeActivationCleaningPredictor::ThreadSafeActivationCleaningPredictor(
    Workspace* parameter_workspace,
    const NetDef& predict_init_net,
    std::shared_ptr<NetDef> predict_net,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::vector<std::string>& parameter_names)
    : ThreadSafeActivationCleaningPredictor(
          std::shared_ptr<Workspace>(parameter_workspace, [](Workspace*) {}),
          predict_init_net,
          predict_net,
          input_names,
          output_names,
          parameter_names) {}

ThreadSafeActivationCleaningPredictor::ThreadSafeActivationCleaningPredictor(
    std::shared_ptr<Workspace> parameter_workspace,
    const NetDef& predict_init_net,
    std::shared_ptr<NetDef> predict_net,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::vector<std::string>& parameter_names)
    : parameter_workspace_(parameter_workspace) {
  predict_net->clear_external_input(); // We don't care about those, predictor
                                       // has its own metadata
  predict_net->clear_external_output(); // We don't care about those, predictor
                                        // has its own metadata
  graph_.predict_init_net_def = caffe2::make_unique<NetDef>(predict_init_net);
  graph_.predict_net_def = predict_net;
  graph_.input_names = input_names;
  graph_.output_names = output_names;
  graph_.parameter_names = parameter_names;

  for (const auto& name : input_names) {
    CAFFE_ENFORCE(!input_idx_.count(name), "Duplicate name ", name);
    input_idx_[name] = input_idx_.size();
  }

  EnforcePredictorInitializationInvariants(parameter_workspace.get(), graph_);

  OptimizeNetwork();
  // internal blobs could be deduced only after all the transformations
  internal_blob_names_ = DeduceInternalBlobs(
      *graph_.predict_net_def, graph_.input_names, output_names);

  // Double check the invariants after all transformation have been applied
  EnforcePredictorInitializationInvariants(parameter_workspace.get(), graph_);
  // Here we make sure the prediction net could be created. If not, model
  // loading should fail. Otherwise we will fail individual requests and keep
  // serving the model. One of the reasons for model to fail at creation time is
  // a not linked operator. Or it could be an ENFORCE from some of the operators
  // constructors (in case exported model is inconsistent).
  PrepareThreadLocalState();
  // Discarding ThreadLocalPtr state for the main thread as predicions probably
  // won't happen here in any way.
  thread_local_state_.reset();
}

void ThreadSafeActivationCleaningPredictor::PrepareThreadLocalState() {
  if (!thread_local_state_.get()) {
    auto state =
        caffe2::make_unique<PredictorState>(parameter_workspace_.get());
    // We are going to perform each prediction in this threadlocal workspace
    CAFFE_ENFORCE(
        state->workspace.RunNetOnce(*graph_.predict_init_net_def),
        "Failed running the prendict_init_net: ",
        ProtoDebugString(*graph_.predict_init_net_def));

    for (const auto& input_name : graph_.input_names) {
      // pre-create all input blobs so predict_net can be created
      state->workspace.CreateBlob(input_name);
    }

    // predict_net is owned by workspace, we just store a pointer to avoid
    // further lookups
    state->predict_net = state->workspace.CreateNet(
        std::const_pointer_cast<const NetDef>(graph_.predict_net_def));
    CAFFE_ENFORCE(state->predict_net != nullptr);

    for (const auto& input_name : graph_.input_names) {
      Blob* blob = state->workspace.GetBlob(input_name);
      CAFFE_ENFORCE(blob, "Blob ", input_name, " was not initialized");
      state->input_blobs.push_back(blob);
    }
    for (const auto& output_name : graph_.output_names) {
      Blob* blob = state->workspace.GetBlob(output_name);
      CAFFE_ENFORCE(blob, "Blob ", output_name, " was not initialized");
      state->output_blobs.push_back(blob);
    }

    // And again we do all the workspace lookups here and avoid them at
    // serving time
    state->internal_blobs.reserve(internal_blob_names_.size());
    for (const auto& output_name : internal_blob_names_) {
      Blob* blob = state->workspace.GetBlob(output_name);
      CAFFE_ENFORCE(
          blob, "Internal blob ", output_name, " was not initialized");
      state->internal_blobs.push_back(blob);
    }

    // So far all blobs have 0 size
    state->internal_blob_max_sizes.resize(internal_blob_names_.size());

    // Start from an empty buffer
    state->buffer.Resize(0);
    state->buffer.template mutable_data<char>();

    thread_local_state_.reset(std::move(state));
  }
}

const ThreadSafeActivationCleaningPredictor::TensorList&
ThreadSafeActivationCleaningPredictor::operator()(
    const TensorList& inputs,
    TensorList* outputs) {
  CHECK(outputs);

  PrepareThreadLocalState();

  auto* state = thread_local_state_.get();
  if (FLAGS_caffe2_predictor_cleanup_activations) {
    AllocateMemory(state);
  }

  // We do sharing for outputs first. So if an input ends up being also an
  // output, we don't overwrite it with an empty output tensor. Instead output
  // will end up sharing data from the input
  TensorVectorResize(*outputs, graph_.output_names.size(), CPU);

  for (int i = 0; i < graph_.output_names.size(); ++i) {
    state->output_blobs[i]->ShareExternal<TensorCPU>(&(*outputs)[i]);
  }

  CAFFE_ENFORCE_EQ(inputs.size(), state->input_blobs.size());
  for (int i = 0; i < inputs.size(); ++i) {
    auto* tensor = state->input_blobs[i]->GetMutableTensor(CPU);
    const auto& input = inputs[i];
    tensor->Resize(input.dims());
    tensor->ShareData(inputs[i]);
  }

  state->predict_net->Run();

  if (FLAGS_caffe2_predictor_cleanup_activations) {
    CleanUpMemory(state);
  }
  return *outputs;
}

ThreadSafeActivationCleaningPredictor::TensorList
ThreadSafeActivationCleaningPredictor::operator()(const TensorList& args) {
  TensorList output;
  (*this)(args, &output);
  return output;
}

ThreadSafeActivationCleaningPredictor::TensorList
ThreadSafeActivationCleaningPredictor::operator()(const TensorMap& kwargs) {
  TensorList output;
  (*this)(kwargs, &output);
  return output;
}

ThreadSafeActivationCleaningPredictor::TensorList
ThreadSafeActivationCleaningPredictor::operator()(
    const TensorList& args,
    const TensorMap& kwargs) {
  TensorList output;
  (*this)(args, kwargs, &output);
  return output;
}

const ThreadSafeActivationCleaningPredictor::TensorList&
ThreadSafeActivationCleaningPredictor::operator()(
    const TensorMap& kwargs,
    TensorList* outputs) {
  return (*this)({}, kwargs, outputs);
}

const ThreadSafeActivationCleaningPredictor::TensorList&
ThreadSafeActivationCleaningPredictor::operator()(
    const TensorList& args,
    const TensorMap& kwargs,
    TensorList* outputs) {
  CAFFE_ENFORCE_EQ(graph_.input_names.size(), args.size() + kwargs.size());

  TensorList inputs_vector;
  TensorVectorResize(inputs_vector, args.size() + kwargs.size(), CPU);
  auto shareTensor = [](Tensor& tensor, const Tensor& input) {
    tensor.Resize(input.dims());
    tensor.ShareData(input);
  };

  for (int i = 0; i < args.size(); ++i) {
    shareTensor(inputs_vector[i], args[i]);
  }

  // The input before args should be covered by args
  for (int i = args.size(); i < graph_.input_names.size(); ++i) {
    const auto& name = graph_.input_names[i];
    auto it = kwargs.find(name);
    CAFFE_ENFORCE(it != kwargs.end(), "Input ", name, " is not provided");
    shareTensor(inputs_vector[i], it->second);
  }
  return (*this)(inputs_vector, outputs);
}

void ThreadSafeActivationCleaningPredictor::CleanUpMemory(
    PredictorState* state) {
  state->internal_blob_max_sizes_sum = 0;
  for (int i = 0; i < state->internal_blobs.size(); ++i) {
    Blob* blob = state->internal_blobs[i];
    TensorCPU* tensor = blob->GetMutableTensor(CPU);
    CAFFE_ENFORCE_GE(
        tensor->size(),
        0,
        "All intermidiate blobs have to be"
        " valid initialized TensorCPU objects");
    auto& tensor_size = state->internal_blob_max_sizes[i];
    // We keep maximum of an ever created tensor. This way we minimize
    // possibility of reallocations during the inference time. Also we assume
    // that data type change is unlikely. In case things don't match a
    // reallocation will happen
    tensor_size = std::max(tensor_size, TensorAdjustedCapacity(*tensor));
    state->internal_blob_max_sizes_sum += tensor_size;

    // tensor->data_ will perform cleanup in a case tensor owns its own memory
    // Othwerwise, in a case memory is shared, dummy deleter will do nothing and
    // memory will be cleaned as a part of the buffer_. The latter is the most
    // common case after first net iteration happened and the size of the blob
    // didn't increase
    tensor->FreeMemory();
  }

  for (Blob* input_blob : state->input_blobs) {
    // No preallocation for inputs for now, we just clean them
    // Potentially we can allow Predictor caller to fill inputs within a
    // buffer allocated once per prediction. This would give a speedup in a case
    // there are a lot of inputs and their allocation takes significant wrt to
    // total inference time
    auto* tensor = input_blob->GetMutableTensor(CPU);
    tensor->FreeMemory();
  }

  state->buffer.FreeMemory();
}

void ThreadSafeActivationCleaningPredictor::AllocateMemory(
    PredictorState* state) {
  if (state->internal_blob_max_sizes_sum == 0) {
    // This is probably the first iteration, we don't have any tensor shape info
    // yet. Also this can be a case if model produced all 0 blobs or if previous
    // run failed.
    return;
  }
  size_t offset = 0;
  state->buffer.Resize(state->internal_blob_max_sizes_sum);
  char* start = state->buffer.template mutable_data<char>();
  for (int i = 0; i < state->internal_blobs.size(); ++i) {
    auto tensor_size = state->internal_blob_max_sizes[i];
    if (tensor_size == 0) {
      // There is no need to pre-allocate empty tensors
      continue;
    }
    CAFFE_ENFORCE_LE(
        tensor_size + offset,
        state->buffer.size(),
        "Inconsistent state, tensors from the previous iteration"
        " are not fitting into the buffer");

    Blob* blob = state->internal_blobs[i];
    TensorCPU* tensor = blob->GetMutableTensor(CPU);
    CAFFE_ENFORCE_LE(
        tensor->nbytes(),
        tensor_size,
        "Somehow maximum tensor size is smaller than current size...");

    tensor->ShareExternalPointer(
        static_cast<void*>(
            start + offset) /* starting position in the buffer */,
        tensor->meta() /* we use latest tensor's meta */,
        tensor_size /* capacity */);

    offset += tensor_size;
  }
  CAFFE_ENFORCE_EQ(offset, state->internal_blob_max_sizes_sum);
}

const std::vector<std::string>&
ThreadSafeActivationCleaningPredictor::InputNames() {
  return graph_.input_names;
}

const std::vector<std::string>&
ThreadSafeActivationCleaningPredictor::OutputNames() {
  return graph_.output_names;
}

int ThreadSafeActivationCleaningPredictor::InputIndex(
    const std::string& name) const {
  auto iter = input_idx_.find(name);
  CAFFE_ENFORCE(iter != input_idx_.end(), "Invalid input name: ", name);
  return iter->second;
}

namespace {
// The function is for logging only that does not alter the predictor logic
void ValidateOrAddParam(InferenceGraph* graph) {
  // Add parameters to arguments to make predict net self-contained
  Argument* param_arg = nullptr;
  for (size_t i = 0; i < graph->predict_net_def->arg_size(); ++i) {
    auto* arg = graph->predict_net_def->mutable_arg(i);
    if (arg->name() ==
        ThreadSafeActivationCleaningPredictor::kPredictorParamName) {
      param_arg = arg;
      break;
    }
  }
  if (param_arg != nullptr) {
    // Validate the params containment
    // param_arg usually contains parameters that won't be used by a net
    CAFFE_ENFORCE_GE(param_arg->strings_size(), graph->parameter_names.size());
    std::unordered_set<std::string> params(
        graph->parameter_names.begin(), graph->parameter_names.end());

    for (size_t i = 0; i < param_arg->strings_size(); ++i) {
      params.erase(param_arg->strings(i));
    }
    CAFFE_ENFORCE(params.empty());
    return;
  }
  param_arg = graph->predict_net_def->add_arg();
  param_arg->set_name(
      ThreadSafeActivationCleaningPredictor::kPredictorParamName);
  for (const auto& name : graph->parameter_names) {
    param_arg->add_strings(name);
  }
}
} // namespace

std::string ThreadSafeActivationCleaningPredictor::kPredictorParamName =
    "PredictorParameters";

void ThreadSafeActivationCleaningPredictor::OptimizeNetwork() {
  ValidateOrAddParam(&graph_);
  ApplyMemonger(graph_);
}

const InferenceGraph&
ThreadSafeActivationCleaningPredictor::GetInferenceGraph() {
  return graph_;
}

} // namespace caffe2
