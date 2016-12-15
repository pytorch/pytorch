#pragma once

#include "redis_store_handler.h"

#include <caffe2/core/operator.h>

#include <string>

namespace caffe2 {

class RedisStoreHandlerCreateOp final : public Operator<CPUContext> {
 public:
  explicit RedisStoreHandlerCreateOp(
      const OperatorDef& operator_def,
      Workspace* ws);

  bool RunOnDevice() override;

 private:
  std::string host_;
  int port_;
  std::string prefix_;

  OUTPUT_TAGS(HANDLER);
};

} // namespace caffe2
