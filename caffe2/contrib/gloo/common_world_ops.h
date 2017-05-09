#pragma once

#include "caffe2/core/operator.h"
#include "caffe2/distributed/store_handler.h"

#include <gloo/rendezvous/context.h>
#include <gloo/rendezvous/prefix_store.h>

#include "store_handler.h"

namespace caffe2 {
namespace gloo {

template <class Context>
class CreateCommonWorld final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  CreateCommonWorld(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        size_(OperatorBase::template GetSingleArgument<int>("size", 0)),
        rank_(OperatorBase::template GetSingleArgument<int>("rank", 0)),
        sync_(OperatorBase::template GetSingleArgument<bool>("sync", false)) {
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
    StoreHandlerWrapper wrapper(*handler);
    ::gloo::rendezvous::PrefixStore store(name_, wrapper);

    // Create context and connect everyone to everyone
    auto context = std::make_shared<::gloo::rendezvous::Context>(rank_, size_);
    context->connectFullMesh(store, device_);

    // Switch pairs to synchronous mode if configured to do so
    if (sync_) {
      for (int i = 0; i < context->size; i++) {
        auto& pair = context->getPair(i);
        if (pair) {
          pair->setSync(true, false);
        }
      }
    }

    *OperatorBase::Output<std::shared_ptr<::gloo::Context>>(COMM) =
        std::move(context);
    return true;
  }

 private:
  std::shared_ptr<::gloo::transport::Device> createDevice();

  const int size_;
  const int rank_;
  const bool sync_;

  std::string name_;
  std::shared_ptr<::gloo::transport::Device> device_;

  INPUT_TAGS(STORE_HANDLER);
  OUTPUT_TAGS(COMM);
};

} // namespace gloo
} // namespace caffe2
