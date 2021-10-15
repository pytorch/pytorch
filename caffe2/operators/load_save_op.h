#ifndef CAFFE2_OPERATORS_LOAD_SAVE_OP_H_
#define CAFFE2_OPERATORS_LOAD_SAVE_OP_H_

#include <cstdio>
#include <map>
#include <unordered_set>

#include <c10/util/string_view.h>
#include "caffe2/core/blob_serialization.h"
#include "caffe2/core/context.h"
#include "caffe2/core/db.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/load_save_op_util.h"
#include "caffe2/utils/math.h"
#include "caffe2/utils/proto_utils.h"

namespace caffe2 {

using db::Cursor;
using db::DB;
using db::Transaction;

template <class Context>
class DBExistsOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  explicit DBExistsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        ws_(ws),
        absolute_path_(
            this->template GetSingleArgument<int>("absolute_path", false)),
        db_name_(this->template GetSingleArgument<string>("db_name", "")),
        db_type_(this->template GetSingleArgument<string>("db_type", "")) {}

  bool RunOnDevice() override {
    string full_db_name =
        absolute_path_ ? db_name_ : (ws_->RootFolder() + "/" + db_name_);
    auto* output = Output(0);
    output->Resize();
    bool* exists = output->template mutable_data<bool>();

    *exists = caffe2::db::DBExists(db_type_, full_db_name);
    return true;
  }

 private:
  Workspace* ws_;
  bool absolute_path_;
  std::string db_name_;
  std::string db_type_;
};

template <class Context>
class LoadOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  explicit LoadOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        ws_(ws),
        absolute_path_(
            this->template GetSingleArgument<int>("absolute_path", false)),
        add_prefix_(this->template GetSingleArgument<string>("add_prefix", "")),
        strip_prefix_(
            this->template GetSingleArgument<string>("strip_prefix", "")),
        db_name_(this->template GetSingleArgument<string>("db", "")),
        db_names_(this->template GetRepeatedArgument<string>("dbs")),
        db_type_(this->template GetSingleArgument<string>("db_type", "")),
        db_options_(this->template GetSingleArgument<string>("db_options", "")),
        keep_device_(this->template GetSingleArgument<int>("keep_device", 0)),
        load_all_(this->template GetSingleArgument<int>("load_all", 0)),
        allow_incomplete_(
            this->template GetSingleArgument<bool>("allow_incomplete", false)),
        blob_names_(
            this->template GetRepeatedArgument<string>("source_blob_names")),
        shape_(this->template GetRepeatedArgument<int64_t>("shape")) {
    if (InputSize() == 0) {
      CAFFE_ENFORCE_GT(db_type_.size(), 0, "Must specify a db type.");
      if (db_names_.empty()) {
        CAFFE_ENFORCE_GT(db_name_.size(), 0, "Must specify a db name.");
        db_names_.push_back(db_name_);
        db_name_ = "";
      } else {
        std::set<std::string> db_name_set;
        for (const string& db_name : db_names_) {
          CAFFE_ENFORCE_GT(db_name.size(), 0, "Db name should not be empty.");
          CAFFE_ENFORCE(
              db_name_set.insert(db_name).second,
              "Duplicated db name: ",
              db_name);
        }
        db_name_ = "";
      }
    }
    CAFFE_ENFORCE(
        // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
        blob_names_.empty() || blob_names_.size() == OutputSize(),
        "Number of output blobs and source_blob_names mismatch.");
    CAFFE_ENFORCE(
        blob_names_.empty() || strip_prefix_.empty(),
        "strip_prefix and source_blob_names are mutually exclusive.");
    CAFFE_ENFORCE(
        blob_names_.empty() || !load_all_,
        "cannot load_all_ while using source_blob_names.");
    if (!load_all_) {
      // blob_names_ will be filled with ''source blob names'' in file/db
      // if argument source_blob_names is not given, then blob_names_ is
      // inferred from operator output
      if (blob_names_.empty()) {
        for (const string& name : operator_def.output()) {
          blob_names_.push_back(name);
        }
      }
      int idx = 0;
      std::set<std::string> name_set;
      for (const string& name : blob_names_) {
        CAFFE_ENFORCE(
            name_set.insert(name).second,
            "Duplicated source blob name: ",
            name);
        output_indices_[name] = idx++;
      }
    }
  }

  void SetCurrentDevice(BlobProto* proto);

  bool RunOnDevice() override {
    int total_loaded_blobs = 0;
    std::unordered_map<string, load_save_op_util::BlobState> blob_states;
    if (InputSize() > 0) {
      for (int i = 0; i < InputSize(); ++i) {
        const db::DBReader& reader = this->template Input<db::DBReader>(i);
        extract(i, reader.cursor(), &blob_states, &total_loaded_blobs);
      }
    } else {
      // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
      for (int i = 0; i < db_names_.size(); ++i) {
        string full_db_name = absolute_path_
            ? db_names_[i]
            : (ws_->RootFolder() + "/" + db_names_[i]);
        std::unique_ptr<DB> in_db(
            caffe2::db::CreateDB(db_type_, full_db_name, caffe2::db::READ));
        if (!db_options_.empty()) {
          in_db->SetOptions(db_options_);
        }
        CAFFE_ENFORCE(
            in_db.get(),
            "Cannot find db implementation of type ",
            db_type_,
            " (while trying to open ",
            full_db_name,
            ")");
        std::unique_ptr<Cursor> cursor(in_db->NewCursor());
        extract(i, cursor.get(), &blob_states, &total_loaded_blobs);
      }
    }

    load_save_op_util::validateBlobStates(blob_states);
    // Loaded all the needed blobs.
    if (!load_all_ && total_loaded_blobs == OutputSize()) {
      VLOG(1) << "Loaded " << total_loaded_blobs << " blobs fully from db(s)";
      return true;
    }

    if (load_all_) {
      for (const string& name : this->debug_def().output()) {
        CAFFE_ENFORCE(
            blob_states.count(name),
            "Output blob name ",
            name,
            " does not exist in the db(s).");
      }
      return true;
    }

    // Only loaded a subset of the blobs.
    if (allow_incomplete_) {
      VLOG(1) << "Loaded " << total_loaded_blobs << " blobs out of "
              << OutputSize() << " blobs from db(s).";
      for (const auto& output_index : output_indices_) {
        if (!blob_states.count(output_index.first)) {
          const auto& blobName = output_index.first;
          const auto* blob = ws_->GetBlob(output_index.first);
          if (blob == nullptr || blob->GetRaw() == nullptr){
            // If blob was not loaded in this op and
            // it did not exist in the workspace before,
            // remove it.
            ws_->RemoveBlob(blobName);
          }
        }
      }
    } else {
      for (const string& output_name : this->debug_def().output()) {
        if (blob_states.count(output_name) == 0) {
          LOG(ERROR) << "Failed to load blob: " << output_name;
        }
      }
      CAFFE_THROW(
          "Expected to load ",
          OutputSize(),
          " blobs, got ",
          total_loaded_blobs,
          " only.\n");
    }

    return true;
  }

 private:
  void extract(
      int db_id,
      Cursor* cursor,
      std::unordered_map<string, load_save_op_util::BlobState>* blob_states,
      int* total_loaded_blobs) {
    if (load_all_) {
      extractAll(db_id, cursor, blob_states, total_loaded_blobs);
    } else {
      extractFrom(
          db_id,
          cursor,
          OperatorBase::Outputs(),
          blob_states,
          total_loaded_blobs);
    }
  }

  void extractAll(
      int db_id,
      Cursor* cursor,
      std::unordered_map<string, load_save_op_util::BlobState>* blob_states,
      int* total_loaded_blobs) {
    CAFFE_ENFORCE(cursor, "cursor is not valid");
    int loaded_blobs = 0;
    for (; cursor->Valid(); cursor->Next()) {
      const auto key = load_save_op_util::buildBlobNameFromDbKey(
          cursor->key(), strip_prefix_, add_prefix_);
      if (key_to_dbid_.count(key) && key_to_dbid_[key] != db_id) {
        CAFFE_THROW("Duplicate Key ", key, " is found!\n");
      } else {
        key_to_dbid_[key] = db_id;
      }

      BlobProto proto;
      CAFFE_ENFORCE(
          proto.ParseFromString(cursor->value()), "Couldn't parse Proto");
      if (!keep_device_) {
        // If we are not keeping the device as the one specified in the
        // proto, we will set the current device.
        SetCurrentDevice(&proto);
      }
      Blob* blob = ws_->CreateBlob(key);
      load_save_op_util::ProcessBlob(
          blob, proto, blob_states, key, &loaded_blobs);
    }
    *total_loaded_blobs += loaded_blobs;
  }

  void extractFrom(
      int db_id,
      Cursor* cursor,
      const vector<Blob*>& outputs,
      std::unordered_map<string, load_save_op_util::BlobState>* blob_states,
      int* total_loaded_blobs) {
    CAFFE_ENFORCE(cursor);
    int loaded_blobs = 0;
    for (; cursor->Valid(); cursor->Next()) {
      const auto key = load_save_op_util::buildBlobNameFromDbKey(
          cursor->key(), strip_prefix_, add_prefix_);
      if (!output_indices_.count(key)) {
        VLOG(1) << "Key " << key << " not used. Skipping.";
      } else {
        if (key_to_dbid_.count(key) && key_to_dbid_[key] != db_id) {
          CAFFE_THROW("Duplicate Key ", key, " is found!\n");
        } else {
          key_to_dbid_[key] = db_id;
        }

        VLOG(2) << "Deserializing blob " << key;
        BlobProto proto;
        CAFFE_ENFORCE(proto.ParseFromString(cursor->value()));
        if (!keep_device_) {
          // If we are not keeping the device as the one specified in the
          // proto, we will set the current device.
          SetCurrentDevice(&proto);
        }
        auto blobIndex = output_indices_[key];
        Blob* blob = outputs.at(blobIndex);
        load_save_op_util::ProcessBlob(
            blob, proto, blob_states, key, &loaded_blobs);

        if (*total_loaded_blobs + loaded_blobs == OutputSize()) {
          break;
        }
      }
    }

    *total_loaded_blobs += loaded_blobs;
  }

 private:
  Workspace* ws_;
  bool absolute_path_;
  string add_prefix_;
  string strip_prefix_;
  string db_name_;
  std::vector<std::string> db_names_;
  string db_type_;
  std::string db_options_;
  bool keep_device_;
  bool load_all_;
  bool allow_incomplete_;
  std::map<string, int> output_indices_;
  std::map<string, int> key_to_dbid_;
  std::vector<std::string> blob_names_;
  std::vector<int64_t> shape_;
};

namespace internal {
class TORCH_API SaveOpImpl {
 public:
  SaveOpImpl(OperatorBase* op, const OperatorDef& operator_def, Workspace* ws);

  bool RunOnDevice();

 private:
  OperatorBase* operator_;
  std::string strip_prefix_;
  std::string full_db_name_;
  std::string db_type_;
  std::string db_options_;
  std::vector<std::string> blob_names_;
  SerializationOptions options_;
};
} // namespace internal

template <class Context>
class SaveOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  explicit SaveOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws), impl_(this, operator_def, ws) {}

  bool RunOnDevice() override {
    return impl_.RunOnDevice();
  }

 private:
  internal::SaveOpImpl impl_;
};

template <typename... Ts>
std::string FormatString(const std::string& pattern, Ts... values) {
  // Start with an initial buffer size that is probably enough most of the time.
  std::string buffer(256, '\0');
  auto bytes_written =
      snprintf(&buffer[0], buffer.size(), pattern.c_str(), values...);
  if (bytes_written < 0) {
    throw std::runtime_error("FormatString failed");
  }
  // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
  if (bytes_written > buffer.size()) {
    // Our initial buffer size wasn't enough, resize and run again.
    buffer.resize(bytes_written + 1);
    bytes_written =
        snprintf(&buffer[0], buffer.size(), pattern.c_str(), values...);
    if (bytes_written < 0) {
      throw std::runtime_error("FormatString failed");
    }
  }
  // Truncate the string to the correct size to trim off the nul terminator.
  buffer.resize(bytes_written);
  return buffer;
}

// CheckpointOp is a wrapper over a SaveFloatTensorOp that basically allows
// flexible naming over iterations.
// The file pattern in db_name should be a format string that can be passed into
// sprintf with an int argument specifying the current iteration. An example:
//     "/path/to/my/checkpoint/checkpoint_at_%d.pb"
template <class Context>
class CheckpointOp final : public Operator<Context> {
 public:
  explicit CheckpointOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        db_pattern_(this->template GetSingleArgument<string>("db", "")),
        every_(this->template GetSingleArgument<int>("every", 1)),
        ws_(ws),
        save_op_def_(operator_def) {
    CAFFE_ENFORCE_GT(
        db_pattern_.size(), 0, "Must specify a checkpoint file pattern.");
    CAFFE_ENFORCE_GT(every_, 0, "Checkpoint interval should be positive.");
    if (every_ == 1) {
      // Just issue a warning, but it's totally legal so we don't do anything.
      LOG(WARNING) << "It seems that we are checkpointting every iteration. "
                   << "Is that intended?";
    }
    save_op_def_.set_type("Save");
  }

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    int64_t iter =
        this->template Input<Tensor>(0, CPU).template data<int64_t>()[0];
    if (iter % every_ == 0) {
      GetMutableArgument("db", true, &save_op_def_)
          ->set_s(FormatString(db_pattern_, iter));
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
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_LOAD_SAVE_OP_H_
