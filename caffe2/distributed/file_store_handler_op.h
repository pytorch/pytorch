#pragma once

#include "file_store_handler.h"

#include <caffe2/core/operator.h>

namespace caffe2 {

template <class Context>
class FileStoreHandlerCreateOp final : public Operator<Context> {
 public:
  explicit FileStoreHandlerCreateOp(
      const OperatorDef& operator_def,
      Workspace* ws)
      : Operator<Context>(operator_def, ws),
        basePath_(
            OperatorBase::template GetSingleArgument<std::string>("path", "")),
        prefix_(OperatorBase::template GetSingleArgument<std::string>(
            "prefix",
            "")) {
    CAFFE_ENFORCE_NE(basePath_, "", "path is a required argument");
  }

  bool RunOnDevice() override {
    auto ptr =
        std::unique_ptr<StoreHandler>(new FileStoreHandler(basePath_, prefix_));
    *OperatorBase::Output<std::unique_ptr<StoreHandler>>(HANDLER) =
        std::move(ptr);
    return true;
  }

 private:
  std::string basePath_;
  std::string prefix_;

  OUTPUT_TAGS(HANDLER);
};

} // namespace caffe2
