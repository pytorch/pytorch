#pragma once

#include "caffe2/core/operator.h"
#include "caffe2/distributed/store_handler.h"

#include "fbcollective/context.h"
#include "fbcollective/rendezvous/prefix_store.h"

#include "store_handler.h"

namespace caffe2 {
namespace fbcollective {

template <class Context>
class CreateCommonWorld final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  CreateCommonWorld(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        size_(OperatorBase::template GetSingleArgument<int>("size", 0)),
        rank_(OperatorBase::template GetSingleArgument<int>("rank", 0)) {
    CAFFE_ENFORCE(def().has_name(), "CreateCommonWorld operator requires name");
    CAFFE_ENFORCE(rank_ >= 0 && rank_ < size_);
    name_ = def().name();
    device_ = createDevice();
  }

  virtual ~CreateCommonWorld() {}

  bool RunOnDevice() override {
    // Use PrefixStore to isolate different CreateCommonWorld instances
    const auto& handler =
        OperatorBase::Input<std::unique_ptr<StoreHandler>>(STORE_HANDLER);
    auto wrapper = std::unique_ptr<::fbcollective::rendezvous::Store>(
        new StoreHandlerWrapper(*handler));
    ::fbcollective::rendezvous::PrefixStore store(name_, wrapper);

    // Create context and connect everyone to everyone
    auto context = std::make_shared<::fbcollective::Context>(rank_, size_);
    context->connectFullMesh(store, device_);
    *OperatorBase::Output<std::shared_ptr<::fbcollective::Context>>(COMM) =
        std::move(context);
    return true;
  }

 private:
  std::shared_ptr<::fbcollective::transport::Device> createDevice();

  const int size_;
  const int rank_;

  std::string name_;
  std::shared_ptr<::fbcollective::transport::Device> device_;

  INPUT_TAGS(STORE_HANDLER);
  OUTPUT_TAGS(COMM);
};

} // namespace fbcollective
} // namespace caffe2
