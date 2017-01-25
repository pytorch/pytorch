#ifndef CAFFE2_OPERATORS_RECURRENT_NETWORK_OP_H_
#define CAFFE2_OPERATORS_RECURRENT_NETWORK_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"
#include "google/protobuf/text_format.h"

namespace caffe2 {
namespace detail {

struct Param {
  std::string param;
  std::string grad;
  std::string accGrad;
};

struct RecurrentInput {
  std::string state;
  std::string input;
};

struct RecurrentGradient {
  std::string param;
  std::string grad;
  std::string externalGrad;
  int32_t offset;
};

struct OffsetAlias {
  std::string src;
  std::string dst;
  int32_t offset{0};
};

struct Scratch {
  std::string name;
  int32_t sizePerStep;
};

struct Link {
  std::string internal;
  std::string external;
  int32_t offset{0};
};

template <typename T, typename Context>
void applyOffsetAlias(const OffsetAlias& oc, Workspace* ws, Context* context) {
  VLOG(1) << "Aliasing: " << oc.src << " to: " << oc.dst
          << " at offset: " << oc.offset;
  auto srcBlob = ws->GetBlob(oc.src);
  CAFFE_ENFORCE(srcBlob);
  auto* src = srcBlob->template GetMutable<Tensor<Context>>();
  auto* dst = ws->CreateBlob(oc.dst)->template GetMutable<Tensor<Context>>();
  auto timestep = src->size() / src->dim(0);
  auto dims = src->dims();
  const int32_t startDstTimestep =
      oc.offset >= 0 ? oc.offset : src->dim(0) + oc.offset;
  const int32_t numDstTimesteps = src->dim(0) - startDstTimestep;
  CAFFE_ENFORCE(
      numDstTimesteps >= 1, "Invalid number of timesteps: ", numDstTimesteps);
  dims[0] = numDstTimesteps;
  dst->Resize(dims);
  CAFFE_ENFORCE(timestep == dst->size() / numDstTimesteps, "Invalid offset");
  dst->ShareExternalPointer(
      src->template mutable_data<T>() + startDstTimestep * timestep,
      dst->size());
}

/**
 * Copy external input to the step net into the first item of
 * (T + 1) X batch_size X input_size tensor
 */
template <typename T, typename Context>
void initializeRecurrentInput(
    const RecurrentInput& rc,
    int32_t seqLen,
    int32_t batchSize,
    Workspace* ws,
    Context* context) {
  auto stateBlob = ws->GetBlob(rc.state);
  CAFFE_ENFORCE(stateBlob);
  auto* state = stateBlob->template GetMutable<Tensor<Context>>();

  auto inputBlob = ws->GetBlob(rc.input);
  CAFFE_ENFORCE(inputBlob);
  const auto& input = inputBlob->template Get<Tensor<Context>>();
  CAFFE_ENFORCE_EQ(input.ndim(), 3, rc.input);
  CAFFE_ENFORCE_EQ(input.dim(0), 1, rc.input);
  CAFFE_ENFORCE_EQ(input.dim(1), batchSize, rc.input);

  // States at [0, ..., T] (inclusive)
  state->Resize(seqLen + 1, batchSize, input.dim(2));
  context->template Copy<T, Context, Context>(
      batchSize * input.dim(2),
      input.template data<T>(),
      state->template mutable_data<T>());
}

template <typename Context>
void initializeScratch(
    const Scratch& scratch,
    int32_t seqLength,
    int32_t batchSize,
    Workspace* ws) {
  CAFFE_ENFORCE(ws);
  VLOG(1) << "Initializing scratch: " << scratch.name;
  ws->GetBlob(scratch.name)
      ->template GetMutable<Tensor<Context>>()
      ->Resize(std::vector<TIndex>{seqLength, batchSize, scratch.sizePerStep});
}

template <typename T, typename Context>
void applyLink(const Link& link, size_t t, Workspace* ws) {
  VLOG(1) << "Linking: " << link.internal << " to: " << link.external
          << " at offset: " << link.offset;
  auto internalTensorBlob = ws->CreateBlob(link.internal);
  CAFFE_ENFORCE(internalTensorBlob);
  auto* internalTensor =
      internalTensorBlob->template GetMutable<Tensor<Context>>();

  auto externalTensorBlob = ws->GetBlob(link.external);
  CAFFE_ENFORCE(externalTensorBlob);
  auto* externalTensor =
      externalTensorBlob->template GetMutable<Tensor<Context>>();
  CAFFE_ENFORCE_GT(externalTensor->size(), 0);
  const TIndex externalTimestepSize =
      externalTensor->size() / externalTensor->dim(0);
  auto* externalData = externalTensor->template mutable_data<T>() +
      (t + link.offset) * externalTimestepSize;
  auto internalDims = externalTensor->dims();
  // Single timestep
  internalDims[0] = 1;
  internalTensor->Resize(internalDims);
  internalTensor->ShareExternalPointer(externalData, externalTimestepSize);
}

void extractLinks(
    OperatorBase* op,
    const std::string& internalArg,
    const std::string& externalArg,
    const std::string offsetArg,
    std::vector<detail::Link>* links) {
  const auto& internal = op->GetRepeatedArgument<std::string>(internalArg);
  const auto& external = op->GetRepeatedArgument<std::string>(externalArg);
  const auto& offset = op->GetRepeatedArgument<int32_t>(offsetArg);
  CAFFE_ENFORCE(
      internal.size() == offset.size(),
      "internal/offset mismatch: ",
      internalArg,
      externalArg);
  CAFFE_ENFORCE(
      external.size() == offset.size(),
      "external/offset mismatch",
      externalArg,
      offsetArg);
  for (auto i = 0; i < internal.size(); ++i) {
    detail::Link l;
    l.internal = internal[i];
    l.external = external[i];
    l.offset = offset[i];
    links->push_back(l);
  }
}
} // namespace detail

template <typename T, class Context>
class RecurrentNetworkOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  RecurrentNetworkOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        ws_(ws),
        timestep_(OperatorBase::template GetSingleArgument<std::string>(
            "timestep",
            "timestep")) {
    CAFFE_ENFORCE(ws);
    const auto stepNet =
        OperatorBase::GetSingleArgument<string>("step_net", "");
    NetDef stepNetDef;
    CAFFE_ENFORCE(
        google::protobuf::TextFormat::ParseFromString(stepNet, &stepNetDef),
        "Invalid netdef");
    CAFFE_ENFORCE(
        stepNetDef.type() == "simple", "Step Net must be `simple`", stepNet);

    ws_.CreateBlob(timestep_)->template GetMutable<TensorCPU>()->Resize(1);

    for (const auto& blob : stepNetDef.external_input()) {
      ws_.CreateBlob(blob);
    }

    stepNet_ = ws_.CreateNet(stepNetDef);
    CAFFE_ENFORCE(stepNet_, "Step Net construction failure");

    scratches_ = constructScratches(ws);
    recurrentInputs_ = constructRecurrentInputs(ws);
    links_ = constructLinks();
    aliases_ = constructAliases();
  }

  std::vector<detail::Scratch> constructScratches(Workspace* sharedWs) {
    const auto& names =
        OperatorBase::GetRepeatedArgument<std::string>("scratch");
    const auto& sizes =
        OperatorBase::GetRepeatedArgument<int32_t>("scratch_sizes");
    CAFFE_ENFORCE(
        sizes.size() == names.size(), "scratch and scratch_sizes mismatch");
    std::vector<detail::Scratch> scratch;
    for (auto i = 0; i < names.size(); ++i) {
      // Scratches need to be "global" (since they are shared between
      // forward and backward).
      sharedWs->CreateBlob(names[i]);
      detail::Scratch s;
      s.name = names[i];
      s.sizePerStep = sizes[i];
      scratch.push_back(s);
    }
    return scratch;
  }

  std::vector<detail::RecurrentInput> constructRecurrentInputs(
      Workspace* sharedWs) {
    const auto states =
        OperatorBase::GetRepeatedArgument<std::string>("recurrent_states");
    const auto inputs =
        OperatorBase::GetRepeatedArgument<std::string>("recurrent_inputs");
    CAFFE_ENFORCE_EQ(states.size(), inputs.size(), "states/inputs mismatch");
    std::vector<detail::RecurrentInput> ris;
    for (auto i = 0; i < states.size(); ++i) {
      // States need to be "global" (since they are shared between
      // forward and backward).
      sharedWs->CreateBlob(states[i]);

      detail::RecurrentInput ri;
      ri.state = states[i];
      ri.input = inputs[i];
      ris.push_back(ri);
    }
    return ris;
  }

  std::vector<detail::OffsetAlias> constructAliases() {
    const auto& src =
        OperatorBase::GetRepeatedArgument<std::string>("alias_src");
    const auto& dst =
        OperatorBase::GetRepeatedArgument<std::string>("alias_dst");
    const auto& offset =
        OperatorBase::GetRepeatedArgument<int32_t>("alias_offset");
    CAFFE_ENFORCE(
        src.size() == offset.size(), "alias_src/alias_offset mismatch");
    CAFFE_ENFORCE(
        dst.size() == offset.size(), "alias_dst/alias_offset mismatch");
    std::vector<detail::OffsetAlias> aliases;
    for (auto i = 0; i < src.size(); ++i) {
      detail::OffsetAlias oc;
      oc.src = src[i];
      oc.dst = dst[i];
      oc.offset = offset[i];
      aliases.push_back(oc);
    }
    return aliases;
  }

  std::vector<detail::Link> constructLinks() {
    std::vector<detail::Link> links;
    detail::extractLinks(
        this, "link_internal", "link_external", "link_offset", &links);
    return links;
  }

  bool RunOnDevice() {
    const auto seqLen = Input(0).dim32(0);
    const auto batchSize = Input(0).dim32(1);
    for (const auto& ri : recurrentInputs_) {
      detail::initializeRecurrentInput<T, Context>(
          ri, seqLen, batchSize, &ws_, &context_);
    }

    for (auto& scratch : scratches_) {
      detail::initializeScratch<Context>(scratch, seqLen, batchSize, &ws_);
    }

    for (auto t = 0; t < seqLen; ++t) {
      for (const auto& link : links_) {
        detail::applyLink<T, Context>(link, t, &ws_);
      }
      // Since we have a SimpleNet, there are no races here.
      auto timestepBlob = ws_.GetBlob(timestep_);
      CAFFE_ENFORCE(timestepBlob);
      timestepBlob->template GetMutable<TensorCPU>()
          ->template mutable_data<int32_t>()[0] = t;
      stepNet_->RunAsync();
    }

    for (const auto& alias : aliases_) {
      detail::applyOffsetAlias<T, Context>(alias, &ws_, &context_);
    }

    return true;
  }

 protected:
  NetBase* stepNet_{nullptr};
  Workspace ws_;
  std::vector<detail::Scratch> scratches_;
  std::vector<detail::Link> links_;
  std::vector<detail::OffsetAlias> aliases_;
  std::vector<detail::RecurrentInput> recurrentInputs_;
  std::string timestep_;
};

template <typename T, class Context>
class RecurrentNetworkGradientOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  RecurrentNetworkGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        ws_(ws),

        timestep_(OperatorBase::template GetSingleArgument<std::string>(
            "timestep",
            "timestep")) {
    scratches_ = constructScratches(ws);
    links_ = constructLinks();
    params_ = constructParams();
    recurrentGradients_ = constructRecurrentGradients();
    aliases_ = constructAliases();
    recurrentInputIds_ = OperatorBase::template GetRepeatedArgument<int32_t>(
        "recurrent_input_ids");

    CAFFE_ENFORCE(ws);
    const auto stepNet =
        OperatorBase::GetSingleArgument<string>("backward_step_net", "");
    NetDef stepNetDef;
    CAFFE_ENFORCE(
        google::protobuf::TextFormat::ParseFromString(stepNet, &stepNetDef));
    ws_.CreateBlob(timestep_)->template GetMutable<TensorCPU>()->Resize(1);

    for (const auto& blob : stepNetDef.external_input()) {
      ws_.CreateBlob(blob);
    }
    stepNet_ = ws_.CreateNet(stepNetDef);
    CAFFE_ENFORCE(stepNet_);
  }

  std::vector<detail::Scratch> constructScratches(Workspace* sharedWs) {
    const auto& names =
        OperatorBase::GetRepeatedArgument<std::string>("backward_scratch");
    const auto& sizes =
        OperatorBase::GetRepeatedArgument<int32_t>("scratch_sizes");
    CAFFE_ENFORCE(
        sizes.size() == names.size(),
        "backward_scratch and scratch_sizes mismatch");
    std::vector<detail::Scratch> scratch;
    for (auto i = 0; i < names.size(); ++i) {
      // Scratches need to be "global" (since they are shared between
      // forward and backward).
      sharedWs->CreateBlob(names[i]);
      detail::Scratch s;
      s.name = names[i];
      s.sizePerStep = sizes[i];
      scratch.push_back(s);
    }
    return scratch;
  }

  std::vector<detail::OffsetAlias> constructAliases() {
    const auto& src =
        OperatorBase::GetRepeatedArgument<std::string>("backward_alias_src");
    const auto& dst =
        OperatorBase::GetRepeatedArgument<std::string>("backward_alias_dst");
    const auto& offset =
        OperatorBase::GetRepeatedArgument<int32_t>("backward_alias_offset");
    CAFFE_ENFORCE_EQ(src.size(), offset.size(), "src/offset mismatch");
    CAFFE_ENFORCE_EQ(dst.size(), offset.size(), "dst/offset mismatch");
    std::vector<detail::OffsetAlias> aliases;
    for (auto i = 0; i < src.size(); ++i) {
      detail::OffsetAlias oc;
      oc.src = src[i];
      oc.dst = dst[i];
      oc.offset = offset[i];
      aliases.push_back(oc);
    }
    return aliases;
  }

  std::vector<detail::Param> constructParams() {
    std::vector<detail::Param> params;
    const auto& param = OperatorBase::GetRepeatedArgument<int32_t>("param");
    for (int i = 0; i < param.size(); ++i) {
      detail::Param p;
      // Forward inputs come after the main input (sequence)
      p.param = def().input(param[i] + numSequences_);
      // See GetRecurrentNetworkGradient to understand offseting here"
      p.grad = def().output(i + numSequences_);
      p.accGrad = p.grad + "_acc";
      params.push_back(p);
    }
    return params;
  }

  std::vector<detail::RecurrentGradient> constructRecurrentGradients() {
    std::vector<detail::RecurrentGradient> rgs;
    const auto& recurrent =
        OperatorBase::GetRepeatedArgument<std::string>("recurrent_states");
    for (auto i = 0; i < recurrent.size(); ++i) {
      detail::RecurrentGradient rg;
      rg.param = recurrent[i];
      rg.grad = recurrent[i] + "_grad";
      rg.externalGrad = i == 0 ? def().input(0) : "";
      rg.offset = 1;
      rgs.push_back(rg);
    }
    return rgs;
  }

  std::vector<detail::Link> constructLinks() {
    std::vector<detail::Link> links;
    detail::extractLinks(
        this, "link_internal", "link_external", "link_offset", &links);
    detail::extractLinks(
        this,
        "backward_link_internal",
        "backward_link_external",
        "backward_link_offset",
        &links);
    return links;
  }

  bool RunOnDevice() {
    const auto seqLen = Input(0).dim32(0);
    VLOG(1) << "seqLen: " << seqLen;
    const auto batchSize = Input(0).dim32(1);
    // See GetRecurrentNetworkGradient to understand offseting here
    for (int i = 0; i < recurrentInputIds_.size(); ++i) {
      // See GetRecurrentNetworkGradient to understand offseting here
      // Outputs of the gradient are inputs of the forward pass.
      // So we need to offset on all inputs that go before recurrent
      // initial ones
      auto outputIdx = i + params_.size() + numSequences_;
      int inputId = recurrentInputIds_[i];
      VLOG(1) << "Resetting output " << outputIdx << " like input " << inputId;
      Output(outputIdx)->ResizeLike(Input(inputId));
      Output(outputIdx)->template mutable_data<float>();
    }
    for (auto& param : params_) {
      auto pBlob = ws_.GetBlob(param.param);
      CAFFE_ENFORCE(pBlob);
      const auto& p = pBlob->template Get<Tensor<Context>>();

      auto gBlob = ws_.CreateBlob(param.grad);
      CAFFE_ENFORCE(gBlob);
      auto* g = gBlob->template GetMutable<Tensor<Context>>();

      auto agBlob = ws_.CreateBlob(param.accGrad);
      CAFFE_ENFORCE(agBlob);
      auto* ag = agBlob->template GetMutable<Tensor<Context>>();
      g->ResizeLike(p);
      ag->ResizeLike(p);
      math::Set<T, Context>(
          ag->size(), 0.0, ag->template mutable_data<T>(), &context_);
    }

    for (auto& rg : recurrentGradients_) {
      auto pBlob = ws_.GetBlob(rg.param);
      CAFFE_ENFORCE(pBlob);
      const auto& p = pBlob->template Get<Tensor<Context>>();

      auto gBlob = ws_.CreateBlob(rg.grad);
      CAFFE_ENFORCE(gBlob);
      auto* g = gBlob->template GetMutable<Tensor<Context>>();
      g->ResizeLike(p);
      CAFFE_ENFORCE_EQ(g->ndim(), 3);
      const auto timestep = g->size() / g->dim(0);
      // Fill the last timestep with zeros for the gradient
      math::Set<T, Context>(
          timestep,
          0.0,
          g->template mutable_data<T>() + (g->dim(0) - 1) * timestep,
          &context_);
    }

    // This code assumes that there are several input
    // sequences. Actually it is not supported by the rest of the code
    // and here we either have 0 or 1 inputGradientId.
    const auto& inputGradientIds =
        OperatorBase::GetRepeatedArgument<int32_t>("input_gradient_ids");
    for (int id : inputGradientIds) {
      // Offseting as the first input of the op is GO(0). Then all I(0..N) go.
      id = id + 1;
      const auto& inputName = def().input(id);
      auto gradientName = inputName + "_grad";
      VLOG(1) << "Initializing gradient for input " << id << " (" << inputName
              << ") "
              << " as blob " << gradientName << ". Size: " << Input(id).size();
      auto pGradientBlob = ws_.CreateBlob(gradientName);
      CAFFE_ENFORCE(pGradientBlob);
      auto* g = pGradientBlob->template GetMutable<Tensor<Context>>();
      g->ResizeLike(Input(id));
      g->template mutable_data<T>();
    }

    for (auto& scratch : scratches_) {
      detail::initializeScratch<Context>(scratch, seqLen, batchSize, &ws_);
    }

    auto accumulateParameterGradients = [&]() {
      for (const auto& param : params_) {
        auto gBlob = ws_.GetBlob(param.grad);
        CAFFE_ENFORCE(gBlob);
        const auto& g = gBlob->template Get<Tensor<Context>>();

        auto agBlob = ws_.GetBlob(param.accGrad);
        CAFFE_ENFORCE(agBlob);
        auto* ag = agBlob->template GetMutable<Tensor<Context>>();
        CAFFE_ENFORCE(ag->dims() == g.dims());
        math::Add<T, Context>(
            g.size(),
            g.template data<T>(),
            ag->template data<T>(),
            ag->template mutable_data<T>(),
            &context_);
      }
    };

    auto accumulateInputGradients = [&](int t) {
      // Input gradients
      for (const auto& rg : recurrentGradients_) {
        if (rg.externalGrad.empty()) {
          continue;
        }
        VLOG(1) << "Accumulating into: " << rg.grad << " from "
                << rg.externalGrad << " at time: " << t
                << ", offset: " << rg.offset;
        auto gBlob = ws_.GetBlob(rg.grad);
        CAFFE_ENFORCE(gBlob);
        auto* g = gBlob->template GetMutable<Tensor<Context>>();

        auto ogBlob = ws_.GetBlob(rg.externalGrad);
        CAFFE_ENFORCE(ogBlob);
        const auto& og = ogBlob->template Get<Tensor<Context>>();

        // g[T+offset] += og[T]
        CAFFE_ENFORCE_EQ(g->size() / g->dim(0), og.size() / og.dim(0));
        const auto timestep = g->size() / g->dim(0);
        CAFFE_ENFORCE_EQ(timestep, og.size() / og.dim(0));
        math::Add<T, Context>(
            timestep,
            og.template data<T>() + t * timestep,
            g->template data<T>() + (t + rg.offset) * timestep,
            g->template mutable_data<T>() + (t + rg.offset) * timestep,
            &context_);
      }
    };

    for (int32_t t = seqLen - 1; t >= 0; --t) {
      VLOG(1) << "Running step: " << t;
      accumulateInputGradients(t);
      for (const auto& link : links_) {
        detail::applyLink<T, Context>(link, t, &ws_);
      }
      // Since we have a SimpleNet, there are no races here.
      ws_.GetBlob(timestep_)
          ->template GetMutable<TensorCPU>()
          ->template mutable_data<int32_t>()[0] = t;
      stepNet_->RunAsync();
      accumulateParameterGradients();
    }

    for (const auto& alias : aliases_) {
      detail::applyOffsetAlias<T, Context>(alias, &ws_, &context_);
    }

    for (const auto& param : params_) {
      // Swap the accumulated gradients with the actual gradients so
      // the rest of the network sees the accumulated gradients.
      using std::swap;
      auto accGradBlob = ws_.GetBlob(param.accGrad);
      auto gradBlob = ws_.GetBlob(param.grad);
      CAFFE_ENFORCE(accGradBlob);
      CAFFE_ENFORCE(gradBlob);
      swap(*accGradBlob, *gradBlob);
    }
    return true;
  }

 protected:
  NetBase* stepNet_{nullptr};
  Workspace ws_;
  std::vector<detail::Scratch> scratches_;
  std::vector<detail::Link> links_;
  std::vector<detail::Param> params_;
  std::vector<detail::RecurrentGradient> recurrentGradients_;
  std::vector<detail::OffsetAlias> aliases_;
  std::string timestep_;
  // For now we support only one input sequence
  const int numSequences_{1};
  std::vector<int32_t> recurrentInputIds_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_RECURRENT_NETWORK_OP_H_
