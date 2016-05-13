#ifndef CAFFE2_OPERATORS_LOAD_SAVE_OP_H_
#define CAFFE2_OPERATORS_LOAD_SAVE_OP_H_

#include <cstdio>
#include <map>

#include "caffe2/core/context.h"
#include "caffe2/core/db.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/core/logging.h"

namespace caffe2 {

using db::Cursor;
using db::DB;
using db::Transaction;

template <class Context>
class LoadTensorOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  LoadTensorOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        db_name_(OperatorBase::GetSingleArgument<string>("db", "")),
        db_type_(OperatorBase::GetSingleArgument<string>("db_type", "")) {
    CAFFE_CHECK_GT(db_name_.size(), 0) << "Must specify a db name.";
    CAFFE_CHECK_GT(db_type_.size(), 0) << "Must specify a db type.";
    int idx = 0;
    for (const string& output_name : this->def().output()) {
      output_indices_[output_name] = idx;
    }
  }

  // TODO(Yangqing): put the load functionality into a registration pattern
  // as well?
  bool RunOnDevice() override {
    std::unique_ptr<DB> in_db(caffe2::db::CreateDB(
        db_type_, db_name_, caffe2::db::READ));
    std::unique_ptr<Cursor> cursor(in_db->NewCursor());
    for (; cursor->Valid(); cursor->Next()) {
      const string& key = cursor->key();
      if (!output_indices_.count(key)) {
        CAFFE_VLOG(1) << "Key " << key << " not used. Skipping.";
        continue;
      } else {
        TensorProto proto;
        CAFFE_CHECK(proto.ParseFromString(cursor->value()));
        CAFFE_CHECK_GT(proto.dims_size(), 0);
        int idx = output_indices_[key];
        auto* output = Output(idx);
        output->Reshape(
            vector<TIndex>(proto.dims().begin(), proto.dims().end()));
        switch (proto.data_type()) {
        case TensorProto::FLOAT:
        {
          CAFFE_CHECK_EQ(output->size(), proto.float_data_size());
          this->context_.template Copy<float, CPUContext, Context>(
              output->size(), proto.float_data().data(),
              output->template mutable_data<float>());
          CAFFE_VLOG(1) << "Loaded float tensor " << key << ".";
          break;
        }
        case TensorProto::INT32:
        {
          static_assert(sizeof(int) == 4,
                        "int in this compiler does not equal to 4 bytes.");
          CAFFE_CHECK_EQ(output->size(), proto.int32_data_size());
          this->context_.template Copy<int, CPUContext, Context>(
              output->size(), proto.int32_data().data(),
              output->template mutable_data<int>());
          CAFFE_VLOG(1) << "Loaded int32 tensor " << key << ".";
          break;
        }
        default:
          CAFFE_LOG_FATAL << "Tensor proto data type " << proto.data_type()
                     << " not currently supported.";
        }
      }
    }
    return true;
  }

 private:
  string db_name_;
  string db_type_;
  std::map<string, int> output_indices_;
  DISABLE_COPY_AND_ASSIGN(LoadTensorOp);
};


template <class Context>
class SaveOp final : public OperatorBase {
 public:
  SaveOp(const OperatorDef& operator_def, Workspace* ws)
      : OperatorBase(operator_def, ws),
        db_name_(OperatorBase::GetSingleArgument<string>("db", "")),
        db_type_(OperatorBase::GetSingleArgument<string>("db_type", "")) {
    CAFFE_CHECK_GT(db_name_.size(), 0) << "Must specify a db name.";
    CAFFE_CHECK_GT(db_type_.size(), 0) << "Must specify a db type.";
  }

  bool Run() override {
    std::unique_ptr<DB> out_db(caffe2::db::CreateDB(
        db_type_, db_name_, caffe2::db::NEW));
    std::unique_ptr<Transaction> transaction(out_db->NewTransaction());
    const vector<const Blob*>& inputs = Inputs();
    for (int i = 0; i < inputs.size(); ++i) {
      transaction->Put(def().input(i), inputs[i]->Serialize(def().input(i)));
    }
    transaction->Commit();
    return true;
  }

 private:
  string db_name_;
  string db_type_;
  DISABLE_COPY_AND_ASSIGN(SaveOp);
};

template <typename ... Ts>
string FormatString(const string& pattern, Ts... values) {
  // Note(Yangqing): We believe that 1024 is enough, but who are we to assert
  // that?
  // As a result, if things go wrong, we'll just throw the towel and quit loud.
  // Yeah, I know that there is snprintf, but it is not present in *some*
  // platforms unfortunately.
  char buffer[1024];
  int written = sprintf(buffer, pattern.c_str(), values...);
  if (written < 0 || written + 1 > 1024) {
    CAFFE_LOG_FATAL << "FormatString fails: total bytes written " << written;
  }
  return string(buffer);
  /*
   * The following is the snprintf version that is safe; enable it one day?
  unsigned int required =
      std::snprintf(nullptr, 0, pattern.c_str(), values...) + 1;
  char bytes[required];
  std::snprintf(bytes, required, pattern.c_str(), values...);
  return string(bytes);
  */
}

// SnapshotOp is a wrapper over a SaveFloatTensorOp that basically allows
// flexible naming over iterations.
// The file pattern in db_name should be a format string that can be passed into
// sprintf with an int argument specifying the current iteration. An example:
//     "/path/to/my/snapshot/snapshot_at_%d.pb"
template <class Context>
class SnapshotOp final : public Operator<Context> {
 public:
  SnapshotOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        db_pattern_(OperatorBase::GetSingleArgument<string>("db", "")),
        every_(OperatorBase::GetSingleArgument<int>("every", 1)),
        ws_(ws), save_op_def_(operator_def), db_arg_index_(-1) {
    CAFFE_CHECK_GT(db_pattern_.size(), 0)
        << "Must specify a snapshot file pattern.";
    CAFFE_CHECK_GT(every_, 0) << "Snapshot interval should be positive.";
    if (every_ == 1) {
      // Just issue a warning, but it's totally legal so we don't do anything.
      CAFFE_LOG_WARNING << "It seems that we are snapshotting every iteration. "
                   << "Is that intended?";
    }
    save_op_def_.set_type("Save");
    for (int i = 0; i < save_op_def_.arg_size(); ++i) {
      if (save_op_def_.arg(i).name() == "db") {
        db_arg_index_ = i;
        break;
      }
    }
  }

  bool RunOnDevice() override {
    int iter = OperatorBase::Input<TensorCPU>(0).template data<int>()[0];
    if (iter % every_ == 0) {
      save_op_def_.mutable_arg(db_arg_index_)->set_s(
          FormatString(db_pattern_, iter));
      SaveOp<Context> sub_op(save_op_def_, ws_);
      return sub_op.Run();
    } else {
      return true;
    }
  }

 private:
  string db_pattern_;
  int every_;
  Workspace* ws_;
  OperatorDef save_op_def_;
  int db_arg_index_;
};


}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_LOAD_SAVE_OP_H_
