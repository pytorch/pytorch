#pragma once

#include "caffe2/contrib/gloo/common.h"
#include "caffe2/contrib/gloo/store_handler.h"
#include "caffe2/core/operator.h"
#include "caffe2/distributed/store_handler.h"

#include <gloo/common/error.h>
#include <gloo/rendezvous/context.h>
#include <gloo/rendezvous/prefix_store.h>

namespace caffe2 {
namespace gloo {

template <class Context>
class CreateCommonWorld final : public Operator<Context> {
 public:
  using CommonWorld = std::shared_ptr<::gloo::Context>;

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  CreateCommonWorld(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        size_(OperatorBase::template GetSingleArgument<int>("size", 0)),
        rank_(OperatorBase::template GetSingleArgument<int>("rank", 0)),
        sync_(OperatorBase::template GetSingleArgument<bool>("sync", false)),
        status_blob_(
            OperatorBase::GetSingleArgument<std::string>("status_blob", "")),
        timeout_ms_(OperatorBase::GetSingleArgument<int>("timeout_ms", -1)),
        ws_(ws) {
    CAFFE_ENFORCE(
        operator_def.has_name(), "CreateCommonWorld operator requires name");
    CAFFE_ENFORCE(rank_ >= 0 && rank_ < size_);
    name_ = operator_def.name();
    device_ = createDevice();
    if (status_blob_ != "") {
      ws_->CreateBlob(status_blob_);
    }
  }

  virtual ~CreateCommonWorld() {}

  bool RunOnDevice() override {
    // Use PrefixStore to isolate different CreateCommonWorld instances
    const auto& handler =
        OperatorBase::Input<std::unique_ptr<StoreHandler>>(STORE_HANDLER);
    StoreHandlerWrapper wrapper(*handler);
    ::gloo::rendezvous::PrefixStore store(name_, wrapper);

    try {
      // Create context and connect everyone to everyone
      auto context =
          std::make_shared<::gloo::rendezvous::Context>(rank_, size_);
      if (timeout_ms_ != -1) {
        context->setTimeout(std::chrono::milliseconds(timeout_ms_));
      }
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

      *OperatorBase::Output<CommonWorld>(COMM) = std::move(context);
    } catch (::gloo::IoException& ioe) {
      LOG(ERROR) << "Caught gloo IO exception: " << ioe.what();
      return handleException(ioe);
    } catch (::caffe2::StoreHandlerTimeoutException& te) {
      LOG(ERROR) << "Caught store handler timeout exception: " << te.what();
      return handleException(te);
    }
    return true;
  }

 private:
  bool handleException(std::exception& ex) {
    if (status_blob_ != "") {
      signalFailure(ws_->GetBlob(status_blob_), ex);
      return false;
    } else {
      throw ex;
    }
  }

  std::shared_ptr<::gloo::transport::Device> createDevice();

  const int size_;
  const int rank_;
  const bool sync_;
  const std::string status_blob_;
  const int timeout_ms_;
  Workspace* ws_;

  std::string name_;
  std::shared_ptr<::gloo::transport::Device> device_;

  INPUT_TAGS(STORE_HANDLER);
  OUTPUT_TAGS(COMM);
};

template <class Context>
class CloneCommonWorld final : public Operator<Context> {
 public:
  using CommonWorld = std::shared_ptr<::gloo::Context>;

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  CloneCommonWorld(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        sync_(OperatorBase::template GetSingleArgument<bool>("sync", false)),
        ws_(ws),
        status_blob_(
            OperatorBase::GetSingleArgument<std::string>("status_blob", "")) {
    if (status_blob_ != "") {
      ws_->CreateBlob(status_blob_);
    }
  }

  virtual ~CloneCommonWorld() {}

  bool RunOnDevice() override {
    try {
      auto existing = OperatorBase::Input<CommonWorld>(EXISTING_COMM);
      ::gloo::rendezvous::ContextFactory factory(existing);
      auto clone = factory.makeContext(existing->getDevice());

      // Switch pairs to synchronous mode if configured to do so
      if (sync_) {
        for (int i = 0; i < clone->size; i++) {
          auto& pair = clone->getPair(i);
          if (pair) {
            pair->setSync(true, false);
          }
        }
      }

      *OperatorBase::Output<CommonWorld>(CLONED_COMM) = std::move(clone);
    } catch (::gloo::IoException& ioe) {
      LOG(ERROR) << "Caught gloo IO exception: " << ioe.what();
      return handleException(ioe);
    }
    return true;
  }

 private:
  bool handleException(std::exception& ex) {
    if (status_blob_ != "") {
      signalFailure(ws_->GetBlob(status_blob_), ex);
      return false;
    } else {
      throw ex;
    }
  }

  const bool sync_;
  Workspace* ws_;
  std::string status_blob_;

  INPUT_TAGS(EXISTING_COMM);
  OUTPUT_TAGS(CLONED_COMM);
};

class DestroyCommonWorld final : public Operator<CPUContext> {
 public:
  DestroyCommonWorld(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {
    cw_name_ = operator_def.input(0);
  }

  bool RunOnDevice() override {
    if (OperatorBase::InputBlob(0).GetRaw() == nullptr) {
      return true;
    }
    const auto& context =
        OperatorBase::Input<std::shared_ptr<::gloo::Context>>(0);

    if (context) {
      LOG(INFO) << "Closing connections: " << cw_name_;
      context->closeConnections();
    }
    return true;
  }

 private:
  std::string cw_name_;
};

} // namespace gloo
} // namespace caffe2
