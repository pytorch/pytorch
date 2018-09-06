#pragma once

#include "store_handler.h"

#include <caffe2/core/operator.h>

namespace caffe2 {

class CAFFE2_API StoreSetOp final : public Operator<CPUContext> {
 public:
  StoreSetOp(const OperatorDef& operator_def, Workspace* ws);
  bool RunOnDevice() override;

 private:
  std::string blobName_;

  INPUT_TAGS(HANDLER, DATA);
};

class CAFFE2_API StoreGetOp final : public Operator<CPUContext> {
 public:
  StoreGetOp(const OperatorDef& operator_def, Workspace* ws);
  bool RunOnDevice() override;

 private:
  std::string blobName_;

  INPUT_TAGS(HANDLER);
  OUTPUT_TAGS(DATA);
};

class CAFFE2_API StoreAddOp final : public Operator<CPUContext> {
 public:
  StoreAddOp(const OperatorDef& operator_def, Workspace* ws);
  bool RunOnDevice() override;

 private:
  std::string blobName_;
  int addValue_;

  INPUT_TAGS(HANDLER);
  OUTPUT_TAGS(VALUE);
};

class CAFFE2_API StoreWaitOp final : public Operator<CPUContext> {
 public:
  StoreWaitOp(const OperatorDef& operator_def, Workspace* ws);
  bool RunOnDevice() override;

 private:
  std::vector<std::string> blobNames_;

  INPUT_TAGS(HANDLER);
};
}
