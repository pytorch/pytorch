#include "caffe2/core/context.h"
#include "caffe2/core/db.h"
#include "caffe2/core/operator.h"


namespace caffe2 {

class CreateDBOp final : public OperatorBase {
 public:
  CreateDBOp(const OperatorDef& operator_def, Workspace* ws)
      : OperatorBase(operator_def, ws),
        db_type_(OperatorBase::template GetSingleArgument<string>(
        "db_type", "leveldb")),
        db_name_(
            OperatorBase::template GetSingleArgument<string>("db", "")) {
    CAFFE_CHECK_GT(db_name_.size(), 0) << "Must specify a db name.";

  }

  bool Run() override {
    OperatorBase::Output<db::DBReader>(0)->Open(db_type_, db_name_);
    return true;
  }

 private:
  string db_type_;
  string db_name_;
  DISABLE_COPY_AND_ASSIGN(CreateDBOp);
};

namespace {
REGISTER_CPU_OPERATOR(CreateDB, CreateDBOp);
REGISTER_CUDA_OPERATOR(CreateDB, CreateDBOp);

OPERATOR_SCHEMA(CreateDB).NumInputs(0).NumOutputs(1);

NO_GRADIENT(CreateDB);
}
}  // namespace caffe2
