#pragma once

#include "file_store_handler.h"

#include <caffe2/core/operator.h>

namespace caffe2 {

class FileStoreHandlerCreateOp final : public Operator<CPUContext> {
 public:
  FileStoreHandlerCreateOp(const OperatorDef& operator_def, Workspace* ws);

  bool RunOnDevice() override;

 private:
  std::string basePath_;

  OUTPUT_TAGS(HANDLER);
};

} // namespace caffe2
