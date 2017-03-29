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
  std::string lastExternalGrad;
  int32_t offset;
};

struct OffsetAlias {
  std::string src;
  std::string dst;
  int32_t offset{0};
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
  auto* dst = ws->GetBlob(oc.dst)->template GetMutable<Tensor<Context>>();
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
  CAFFE_ENFORCE_GE(input.ndim(), 1, rc.input);
  CAFFE_ENFORCE_LE(input.ndim(), 3, rc.input);

  const auto stateSize = input.dim(input.ndim() - 1);
  // States at [0, ..., T] (inclusive)
  state->Resize(seqLen + 1, batchSize, stateSize);

  if (input.ndim() == 3) {
    CAFFE_ENFORCE_EQ(input.dim(0), 1, rc.input);
  }
  if (input.ndim() >= 2) {
    CAFFE_ENFORCE_EQ(input.dim(input.ndim() - 2), batchSize, rc.input);
    context->template Copy<T, Context, Context>(
        batchSize * stateSize,
        input.template data<T>(),
        state->template mutable_data<T>());
  } else {
    for (int i = 0; i < batchSize; ++i) {
      // Usually, the initial state is the same for all inputs in the batch.
      // So the op conveniently accepts 1-D input and copies it batchSize times.
      context->template Copy<T, Context, Context>(
          stateSize,
          input.template data<T>(),
          state->template mutable_data<T>() + i * stateSize);
    }
  }
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
    std::vector<detail::Link>* links);
} // namespace detail

template <typename T, class Context>
class RecurrentNetworkOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  RecurrentNetworkOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        sharedWs_(ws),
        timestep_(OperatorBase::template GetSingleArgument<std::string>(
            "timestep",
            "timestep")) {
    CAFFE_ENFORCE(ws);
    const auto stepNet =
        OperatorBase::GetSingleArgument<string>("step_net", "");
    CAFFE_ENFORCE(
        google::protobuf::TextFormat::ParseFromString(stepNet, &stepNetDef_),
        "Invalid netdef");

    recurrentInputs_ = constructRecurrentInputs(sharedWs_);
    links_ = constructLinks();
    aliases_ = constructAliases();
  }

  std::vector<detail::RecurrentInput> constructRecurrentInputs(
      Workspace* sharedWs) {
    const auto states =
        OperatorBase::GetRepeatedArgument<std::string>("recurrent_states");
    const auto inputs =
        OperatorBase::GetRepeatedArgument<int>("initial_recurrent_state_ids");
    CAFFE_ENFORCE_EQ(states.size(), inputs.size(), "states/inputs mismatch");
    std::vector<detail::RecurrentInput> ris;
    for (auto i = 0; i < states.size(); ++i) {
      // States need to be "global" (since they are shared between
      // forward and backward).
      sharedWs->CreateBlob(states[i]);

      detail::RecurrentInput ri;
      ri.state = states[i];
      ri.input = def().input(inputs[i]);
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
          ri, seqLen, batchSize, sharedWs_, &context_);
    }

    std::vector<std::shared_ptr<Workspace>>& stepWorkspaces =
        *OperatorBase::Output<std::vector<std::shared_ptr<Workspace>>>(
            OutputSize() - 1);

    if (seqLen > stepWorkspaces.size()) {
      stepWorkspaces.resize(seqLen);
    }

    for (auto t = 0; t < seqLen; ++t) {
      auto& currentStepWorkspace = stepWorkspaces[t];
      if (!currentStepWorkspace) {
        currentStepWorkspace = std::make_shared<Workspace>(sharedWs_);
      }

      for (const auto& link : links_) {
        detail::applyLink<T, Context>(link, t, currentStepWorkspace.get());
      }

      currentStepWorkspace->CreateBlob(timestep_)
          ->template GetMutable<TensorCPU>()
          ->Resize(1);
      auto timestepBlob = currentStepWorkspace->GetBlob(timestep_);
      CAFFE_ENFORCE(timestepBlob);
      timestepBlob->template GetMutable<TensorCPU>()
          ->template mutable_data<int32_t>()[0] = t;

      auto* stepNet = currentStepWorkspace->GetNet(stepNetDef_.name());
      if (stepNet == nullptr) {
        stepNet = currentStepWorkspace->CreateNet(stepNetDef_);
      }
      CAFFE_ENFORCE(stepNet, "Step Net construction failure");
      // Since we have a SimpleNet, there are no races here.
      stepNet->RunAsync();
    }

    for (const auto& alias : aliases_) {
      detail::applyOffsetAlias<T, Context>(alias, sharedWs_, &context_);
    }

    return true;
  }

 protected:
  NetDef stepNetDef_;
  Workspace* sharedWs_;
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
        sharedWs_(ws),
        localWs_(ws),
        timestep_(OperatorBase::template GetSingleArgument<std::string>(
            "timestep",
            "timestep")),
        gradInputs_(OperatorBase::template GetRepeatedArgument<int32_t>(
            "outputs_with_grads")) {
    links_ = constructLinks();
    params_ = constructParams();
    recurrentGradients_ = constructRecurrentGradients();
    recurrentInputIds_ = OperatorBase::template GetRepeatedArgument<int32_t>(
        "initial_recurrent_state_ids");

    CAFFE_ENFORCE(ws);
    const auto stepNet =
        OperatorBase::GetSingleArgument<string>("backward_step_net", "");
    CAFFE_ENFORCE(
        google::protobuf::TextFormat::ParseFromString(stepNet, &stepNetDef_));
  }

  std::vector<detail::Param> constructParams() {
    std::vector<detail::Param> params;
    const auto& param = OperatorBase::GetRepeatedArgument<int32_t>("param");
    for (int i = 0; i < param.size(); ++i) {
      detail::Param p;
      // Forward inputs come after [outputs_with_grads] gradient inputs
      p.param = def().input(param[i] + gradInputs_.size());
      // See GetRecurrentNetworkGradient to understand offseting here
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
    const auto& alias_src =
        OperatorBase::GetRepeatedArgument<std::string>("alias_src");
    const auto& offset =
        OperatorBase::GetRepeatedArgument<int32_t>("alias_offset");

    for (auto i = 0; i < recurrent.size(); ++i) {
      detail::RecurrentGradient rg;
      rg.param = recurrent[i];
      rg.grad = recurrent[i] + "_grad";

      for (int j = 0; j < alias_src.size(); ++j) {
        if (alias_src[j] != recurrent[i]) {
          continue;
        }
        int idx = -1;
        for (int k = 0; k < gradInputs_.size(); ++k) {
          if (gradInputs_[k] == j) {
            idx = k;
          }
        }
        if (idx == -1) {
          continue;
        }

        CAFFE_ENFORCE(offset[j] == 1 || offset[j] == -1);
        if (offset[j] == 1) {
          rg.externalGrad = def().input(idx);
        } else if (offset[j] == -1) {
          rg.lastExternalGrad = def().input(idx);
        }
      }
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
    const auto seqLen = Input(gradInputs_.size()).dim32(0);
    VLOG(1) << "seqLen: " << seqLen;

    const auto batchSize = Input(0).dim32(1);
    for (auto& param : params_) {
      auto pBlob = sharedWs_->GetBlob(param.param);
      CAFFE_ENFORCE(pBlob);
      const auto& p = pBlob->template Get<Tensor<Context>>();

      auto gBlob = sharedWs_->GetBlob(param.grad);
      CAFFE_ENFORCE(gBlob);
      auto* g = gBlob->template GetMutable<Tensor<Context>>();

      auto agBlob = localWs_.CreateBlob(param.accGrad);
      CAFFE_ENFORCE(agBlob);
      auto* ag = agBlob->template GetMutable<Tensor<Context>>();
      g->ResizeLike(p);
      ag->ResizeLike(p);
      math::Set<T, Context>(
          ag->size(), 0.0, ag->template mutable_data<T>(), &context_);
    }

    for (auto& rg : recurrentGradients_) {
      auto pBlob = sharedWs_->GetBlob(rg.param);
      CAFFE_ENFORCE(pBlob);
      const auto& p = pBlob->template Get<Tensor<Context>>();

      auto gBlob = sharedWs_->CreateBlob(rg.grad);
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
    // sequences. Actually it is not supported by the rest of the code,
    // and numSequences_ is a constant, equal to 1.
    for (int i = 0; i < numSequences_; ++i) {
      // Offseting as the first gradInputs_.size() inputs of the op
      // are from GO. Then all I(0..N).
      const int gradientInputIndex = i + gradInputs_.size();
      const auto& inputName = def().input(gradientInputIndex);
      auto gradientName = inputName + "_grad";
      VLOG(1) << "Initializing gradient for input " << gradientInputIndex
              << " (" << inputName << ") "
              << " as blob " << gradientName
              << ". Size: " << Input(gradientInputIndex).size();
      auto pGradientBlob = sharedWs_->GetBlob(gradientName);
      CAFFE_ENFORCE(pGradientBlob);
      auto* g = pGradientBlob->template GetMutable<Tensor<Context>>();
      g->ResizeLike(Input(gradientInputIndex));
      g->template mutable_data<T>();
    }

    auto accumulateParameterGradients = [&]() {
      for (const auto& param : params_) {
        auto gBlob = sharedWs_->GetBlob(param.grad);
        CAFFE_ENFORCE(gBlob);
        const auto& g = gBlob->template Get<Tensor<Context>>();

        auto agBlob = localWs_.GetBlob(param.accGrad);
        CAFFE_ENFORCE(agBlob);
        auto* ag = agBlob->template GetMutable<Tensor<Context>>();
        CAFFE_ENFORCE(ag->dims() == g.dims());
        T* ag_data = ag->template mutable_data<T>();
        math::Add<T, Context>(
            g.size(),
            g.template data<T>(),
            ag_data,
            ag_data,
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
        auto gBlob = sharedWs_->GetBlob(rg.grad);
        CAFFE_ENFORCE(gBlob);
        auto* g = gBlob->template GetMutable<Tensor<Context>>();

        auto ogBlob = sharedWs_->GetBlob(rg.externalGrad);
        CAFFE_ENFORCE(ogBlob);
        const auto& og = ogBlob->template Get<Tensor<Context>>();

        // g[T+offset] += og[T]
        CAFFE_ENFORCE_EQ(g->size() / g->dim(0), og.size() / og.dim(0));
        const auto timestep_size = g->size() / g->dim(0);
        CAFFE_ENFORCE_EQ(timestep_size, og.size() / og.dim(0));
        T* g_data = g->template mutable_data<T>();
        math::Add<T, Context>(
            timestep_size,
            og.template data<T>() + t * timestep_size,
            g_data + (t + rg.offset) * timestep_size,
            g_data + (t + rg.offset) * timestep_size,
            &context_);
      }
    };

    auto accumulateFinalInputGradients = [&]() {
      for (const auto& rg : recurrentGradients_) {
        if (rg.lastExternalGrad.empty()) {
          continue;
        }
        VLOG(1) << "Accumulating into: " << rg.grad << " from "
                << rg.lastExternalGrad << " for final time step (sep. blob)";
        auto gBlob = sharedWs_->GetBlob(rg.grad);
        CAFFE_ENFORCE(gBlob);
        auto* g = gBlob->template GetMutable<Tensor<Context>>();

        auto oglastBlob = sharedWs_->GetBlob(rg.lastExternalGrad);
        CAFFE_ENFORCE(oglastBlob);
        const auto& oglast = oglastBlob->template Get<Tensor<Context>>();
        CAFFE_ENFORCE_EQ(g->dim(1), oglast.dim(1));
        CAFFE_ENFORCE_EQ(g->dim(2), oglast.dim(2));

        const auto t = g->dim(0) - 1;
        const auto timestep_size = g->size() / g->dim(0);
        CAFFE_ENFORCE_EQ(timestep_size, oglast.size());
        T* g_data_with_offset =
            g->template mutable_data<T>() + t * timestep_size;
        math::Add<T, Context>(
            timestep_size,
            oglast.template data<T>(),
            g_data_with_offset,
            g_data_with_offset,
            &context_);
      }
    };

    const std::vector<std::shared_ptr<Workspace>>& stepWorkspaces =
        OperatorBase::Input<std::vector<std::shared_ptr<Workspace>>>(
            InputSize() - 1);
    CAFFE_ENFORCE_GE(stepWorkspaces.size(), seqLen);

    accumulateFinalInputGradients();
    for (int32_t t = seqLen - 1; t >= 0; --t) {
      // We use local workspace for all the blobs which are not a part
      // of backward links. This way we reuse memory for all the internal
      // gradient blobs of the backward step net across all the timesteps
      localWs_.AddParentWorkspace(stepWorkspaces[t].get());
      accumulateInputGradients(t);
      for (const auto& link : links_) {
        detail::applyLink<T, Context>(link, t, &localWs_);
      }
      // We create different nets in the localWs_.
      // There is no name clash here as localWs_ is a private
      // workspace of this operator. The reason for this is that
      // otherwise if we use the same net at each timestep,
      // its inputs / outputs won't peak up new blobs after
      // attaching a new parent using AddParentWorkspace
      auto old_net_name = stepNetDef_.name();
      auto net_name = MakeString(old_net_name, "_", t);
      auto* stepNet = localWs_.GetNet(net_name);
      if (stepNet == nullptr) {
        stepNetDef_.set_name(net_name);
        stepNet = localWs_.CreateNet(stepNetDef_);
        stepNetDef_.set_name(old_net_name);
      }
      CAFFE_ENFORCE(stepNet);
      stepNet->RunAsync();
      accumulateParameterGradients();
    }

    for (const auto& param : params_) {
      // Swap the accumulated gradients with the actual gradients so
      // the rest of the network sees the accumulated gradients.
      using std::swap;
      auto accGradBlob = localWs_.GetBlob(param.accGrad);
      auto gradBlob = sharedWs_->GetBlob(param.grad);
      CAFFE_ENFORCE(accGradBlob);
      CAFFE_ENFORCE(gradBlob);
      swap(*accGradBlob, *gradBlob);
    }

    CAFFE_ENFORCE_EQ(recurrentInputIds_.size(), recurrentGradients_.size());
    for (int i = 0; i < recurrentInputIds_.size(); ++i) {
      // See GetRecurrentNetworkGradient to understand offseting here
      // Outputs of the gradient are inputs of the forward pass.
      // So we need to offset on all inputs that go before recurrent
      // initial ones
      auto outputIdx = i + params_.size() + numSequences_;
      // because first gradInputs_.size() inputs are from GO
      int inputId = recurrentInputIds_[i] + gradInputs_.size();
      VLOG(1) << "Resetting output " << def().output(outputIdx)
              << " like input " << def().input(inputId);
      Output(outputIdx)->ResizeLike(Input(inputId));
      T* output_data = Output(outputIdx)->template mutable_data<T>();
      auto pBlob = sharedWs_->GetBlob(recurrentGradients_[i].grad);
      CAFFE_ENFORCE(pBlob);
      auto* p = pBlob->template GetMutable<Tensor<Context>>();

      if (Input(inputId).ndim() >= 2) {
        // Gradient states blob should live. And if it gets changed by the
        // backward pass, then output should be changed as well. Thus it should
        // be okay to share data here
        Output(outputIdx)->template ShareExternalPointer<T>(
            p->template mutable_data<T>());
      } else {
        // We need to do a bunch of Adds any way. So lets not worry about
        // copy / share data here. One way to speed this up could be a kernel
        // which sums up several tensors together instead of going 1 by 1
        const auto recurrentStateSize = Input(inputId).dim32(0);
        context_.template Copy<T, Context, Context>(
            recurrentStateSize,
            p->template data<T>(),
            output_data);
        for (int j = 1; j < batchSize; ++j) {
          math::Add<T, Context>(
              recurrentStateSize,
              p->template data<T>() + j * recurrentStateSize,
              output_data,
              output_data,
              &context_);
        }
      }
    }

    return true;
  }

 protected:
  NetDef stepNetDef_;
  Workspace* sharedWs_;
  Workspace localWs_;
  std::vector<detail::Link> links_;
  std::vector<detail::Param> params_;
  std::vector<detail::RecurrentGradient> recurrentGradients_;
  std::string timestep_;
  // For now we support only one input sequence
  const int numSequences_{1};
  std::vector<int32_t> recurrentInputIds_;
  std::vector<int32_t> gradInputs_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_RECURRENT_NETWORK_OP_H_
