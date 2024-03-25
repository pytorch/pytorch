#pragma once

#include "redis_store_handler.h"

#include <caffe2/core/operator.h>

#include <string>

namespace caffe2 {

template <class Context>
class RedisStoreHandlerCreateOp final : public Operator<Context> {
 public:
  explicit RedisStoreHandlerCreateOp(
      const OperatorDef& operator_def,
      Workspace* ws)
      : Operator<Context>(operator_def, ws),
        host_(
            OperatorBase::template GetSingleArgument<std::string>("host", "")),
        port_(OperatorBase::template GetSingleArgument<int>("port", 0)),
        prefix_(OperatorBase::template GetSingleArgument<std::string>(
            "prefix",
            "")) {
    CAFFE_ENFORCE_NE(host_, "", "host is a required argument");
    CAFFE_ENFORCE_NE(port_, 0, "port is a required argument");
  }

  bool RunOnDevice() override {
    auto ptr = std::unique_ptr<StoreHandler>(
        new RedisStoreHandler(host_, port_, prefix_));
    *OperatorBase::Output<std::unique_ptr<StoreHandler>>(HANDLER) =
        std::move(ptr);
    return true;
  }

 private:
  std::string host_;
  int port_;
  std::string prefix_;

  OUTPUT_TAGS(HANDLER);
};

} // namespace caffe2
