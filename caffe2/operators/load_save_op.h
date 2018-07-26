#ifndef CAFFE2_OPERATORS_LOAD_SAVE_OP_H_
#define CAFFE2_OPERATORS_LOAD_SAVE_OP_H_

#include <cstdio>
#include <map>
#include <unordered_set>

#include "caffe2/core/blob_serialization.h"
#include "caffe2/core/context.h"
#include "caffe2/core/db.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"
#include "caffe2/utils/proto_utils.h"

namespace caffe2 {

namespace {
struct BlobState {
  int64_t total_size;
  int64_t current_size;
  bool is_tensor;
  std::set<int32_t> seen_chunks_ids;

  explicit BlobState(
      int64_t total_size = 0,
      int64_t current_size = 0,
      bool is_tensor = false)
      : total_size(total_size),
        current_size(current_size),
        is_tensor(is_tensor) {}
};
} // namespace

using db::Cursor;
using db::DB;
using db::Transaction;

template <class Context>
class DBExistsOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  DBExistsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        ws_(ws),
        absolute_path_(
            OperatorBase::GetSingleArgument<int>("absolute_path", false)),
        db_name_(OperatorBase::GetSingleArgument<string>("db_name", "")),
        db_type_(OperatorBase::GetSingleArgument<string>("db_type", "")) {}

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
  LoadOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        ws_(ws),
        absolute_path_(
            OperatorBase::GetSingleArgument<int>("absolute_path", false)),
        add_prefix_(OperatorBase::GetSingleArgument<string>("add_prefix", "")),
        strip_prefix_(
            OperatorBase::GetSingleArgument<string>("strip_prefix", "")),
        db_name_(OperatorBase::GetSingleArgument<string>("db", "")),
        db_names_(OperatorBase::GetRepeatedArgument<string>("dbs")),
        db_type_(OperatorBase::GetSingleArgument<string>("db_type", "")),
        keep_device_(OperatorBase::GetSingleArgument<int>("keep_device", 0)),
        load_all_(OperatorBase::GetSingleArgument<int>("load_all", 0)),
        allow_incomplete_(
            OperatorBase::GetSingleArgument<bool>("allow_incomplete", false)),
        blob_names_(
            OperatorBase::GetRepeatedArgument<string>("source_blob_names")) {
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
    CAFFE_ENFORCE(blob_names_.empty() || blob_names_.size() == OutputSize(),
      "Number of output blobs and source_blob_names mismatch.");
    CAFFE_ENFORCE(blob_names_.empty() || strip_prefix_.empty(),
        "strip_prefix and source_blob_names are mutually exclusive.");
    CAFFE_ENFORCE(blob_names_.empty() || !load_all_,
        "cannot load_all_ while using source_blob_names.");
    if (!load_all_) {
      // blob_names_ will be filled with ''source blob names'' in file/db
      // if argument source_blob_names is not given, then blob_names_ is
      // inferred from operator output
      if(blob_names_.empty()) {
        for (const string& name : operator_def.output()) {
          blob_names_.push_back(name);
        }
      }
      int idx = 0;
      std::set<std::string> name_set;
      for (const string& name : blob_names_) {
        CAFFE_ENFORCE(name_set.insert(name).second,
            "Duplicated source blob name: ", name);
        output_indices_[name] = idx++;
      }
    }
  }

  void SetCurrentDevice(BlobProto* proto);

  bool RunOnDevice() override {
    int total_loaded_blobs = 0;
    std::unordered_map<string, BlobState> blob_states;
    if (InputSize() > 0) {
      for (int i = 0; i < InputSize(); ++i) {
        const db::DBReader& reader = OperatorBase::Input<db::DBReader>(i);
        extract(i, reader.cursor(), &blob_states, &total_loaded_blobs);
      }
    } else {
      for (int i = 0; i < db_names_.size(); ++i) {
        string full_db_name = absolute_path_
            ? db_names_[i]
            : (ws_->RootFolder() + "/" + db_names_[i]);
        std::unique_ptr<DB> in_db(
            caffe2::db::CreateDB(db_type_, full_db_name, caffe2::db::READ));
        CAFFE_ENFORCE(in_db.get(), "Cannot open db: ", full_db_name);
        std::unique_ptr<Cursor> cursor(in_db->NewCursor());
        extract(i, cursor.get(), &blob_states, &total_loaded_blobs);
      }
    }

    validateBlobStates(blob_states);
    // Loaded all the needed blobs.
    if (load_all_ || total_loaded_blobs == OutputSize()) {
      VLOG(1) << "Loaded " << total_loaded_blobs << " blobs fully from db(s)";
      return true;
    }

    // Only loaded a subset of the blobs.
    if (allow_incomplete_) {
      VLOG(1) << "Loaded " << total_loaded_blobs << " blobs out of "
              << OutputSize() << " blobs from db(s).";
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
      std::unordered_map<string, BlobState>* blob_states,
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
      std::unordered_map<string, BlobState>* blob_states,
      int* total_loaded_blobs) {
    CAFFE_ENFORCE(cursor, "cursor is not valid");
    int loaded_blobs = 0;
    for (; cursor->Valid(); cursor->Next()) {
      const auto key = buildBlobNameFromDbKey(cursor->key());
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
      ProcessBlob(blob, proto, blob_states, key, &loaded_blobs);
    }
    *total_loaded_blobs += loaded_blobs;
  }

  void extractFrom(
      int db_id,
      Cursor* cursor,
      const vector<Blob*>& outputs,
      std::unordered_map<string, BlobState>* blob_states,
      int* total_loaded_blobs) {
    CAFFE_ENFORCE(cursor);
    int loaded_blobs = 0;
    for (; cursor->Valid(); cursor->Next()) {
      const auto key = buildBlobNameFromDbKey(cursor->key());
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
        ProcessBlob(blob, proto, blob_states, key, &loaded_blobs);

        if (*total_loaded_blobs + loaded_blobs == OutputSize()) {
          break;
        }
      }
    }

    *total_loaded_blobs += loaded_blobs;
  }

  string buildBlobNameFromDbKey(const string& dbKey) {
    string key = dbKey.substr(0, dbKey.find(kChunkIdSeparator));
    if (!strip_prefix_.empty()) {
      auto match_pos = key.find(strip_prefix_);
      if (match_pos != string::npos) {
        key = key.substr(match_pos + strip_prefix_.size());
      }
    }
    key = add_prefix_ + key;
    return key;
  }

 private:
  // We are tracking sizes of already read tensor parts while reading data
  // chunks. This way we can make sure that all chunks were loaded in the end.
  void ProcessBlob(
      Blob* blob,
      const BlobProto& proto,
      std::unordered_map<string, BlobState>* blob_states_ptr,
      const string& key,
      int* loaded_blobs) {
    auto& blob_states = *blob_states_ptr;
    if (blob_states.count(key) == 0) {
      // We reset the blob so that any existing content is destroyed. This
      // is to guaranee correct device placement: if we are deserializing
      // into a TensorCUDA, without explicit Reset we might be loading data
      // into an existing TensorCUDA that has pre-allocated memory on a
      // different GPU.
      blob->Reset();
    }
    blob->Deserialize(proto);
    if (proto.has_content_num_chunks()) {
      if (!blob_states.count(key)) {
        blob_states[key] = BlobState(proto.content_num_chunks());
      }
      CAFFE_ENFORCE(
          blob_states[key]
              .seen_chunks_ids.insert(proto.content_chunk_id())
              .second,
          "Chunk with the same id has occured twice for: ",
          key);
      CAFFE_ENFORCE(
          proto.content_chunk_id() >= 0 &&
              proto.content_chunk_id() < blob_states[key].total_size,
          "Chunk id has to be not less than 0 and "
          "less than content_num_chunks for key: ",
          key);
      blob_states[key].current_size++;
      CAFFE_ENFORCE(
          !blob_states[key].is_tensor,
          "Proto with content_chunks can not store tensor: ",
          key);
      CAFFE_ENFORCE(
          blob_states[key].current_size <= blob_states[key].total_size,
          "Found an extra part for an already filled blob: ",
          key);
      if (blob_states[key].current_size == blob_states[key].total_size) {
        (*loaded_blobs)++;
      }
      return;
    }
    if (!proto.has_tensor()) {
      // If blob is divided into chunks the field content_chunks has to be set,
      // otherwise only tensors can be seen multiple times as chunks.
      CAFFE_ENFORCE(blob_states.count(key) == 0, "Blob duplicated: ", key);
      blob_states[key] = BlobState();
      (*loaded_blobs)++;
      return;
    }
    CAFFE_ENFORCE(proto.has_tensor());
    if (blob_states.count(key)) {
      CAFFE_ENFORCE(blob_states[key].is_tensor, "Must be tensor ", key);
      CAFFE_ENFORCE(
          blob_states[key].current_size < blob_states[key].total_size,
          "Found an extra part for an already filled tensor: ",
          key);
      CAFFE_ENFORCE(
          proto.tensor().has_segment(),
          "Partial tensor must have a segment: ",
          key);
      blob_states[key].current_size +=
          proto.tensor().segment().end() - proto.tensor().segment().begin();
      CAFFE_ENFORCE(
          blob_states[key].current_size <= blob_states[key].total_size,
          "Tensor parts are bigger than target size for tensor: ",
          key);
    } else {
      const auto& dims = proto.tensor().dims();
      int64_t total_size = 1;
      for (const auto& dim : dims) {
        total_size *= dim;
      }
      auto current_size = total_size;
      if (proto.tensor().has_segment()) {
        current_size =
            proto.tensor().segment().end() - proto.tensor().segment().begin();
      }
      blob_states[key] =
          BlobState(total_size, current_size, true /* is_tensor */);
    }

    if (blob_states[key].current_size == blob_states[key].total_size) {
      (*loaded_blobs)++;
    }
  }

  void validateBlobStates(
      const std::unordered_map<string, BlobState>& blob_states) {
    for (const auto& iter : blob_states) {
      const BlobState& blob_state = iter.second;
      CAFFE_ENFORCE(
          blob_state.current_size == blob_state.total_size,
          "Data size mismatch for blob ",
          iter.first,
          ". Expected: ",
          blob_state.total_size,
          " Read: ",
          blob_state.current_size);
    }
  }

  Workspace* ws_;
  bool absolute_path_;
  string add_prefix_;
  string strip_prefix_;
  string db_name_;
  std::vector<std::string> db_names_;
  string db_type_;
  bool keep_device_;
  bool load_all_;
  bool allow_incomplete_;
  std::map<string, int> output_indices_;
  std::map<string, int> key_to_dbid_;
  std::vector<std::string> blob_names_;
};

template <class Context>
class SaveOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  SaveOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        ws_(ws),
        absolute_path_(
            OperatorBase::GetSingleArgument<int>("absolute_path", false)),
        strip_prefix_(
            OperatorBase::GetSingleArgument<string>("strip_prefix", "")),
        db_name_(OperatorBase::GetSingleArgument<string>("db", "")),
        db_type_(OperatorBase::GetSingleArgument<string>("db_type", "")),
        blob_names_(
            OperatorBase::GetRepeatedArgument<string>("blob_name_overrides")) {
    CAFFE_ENFORCE_GT(db_name_.size(), 0, "Must specify a db name.");
    CAFFE_ENFORCE_GT(db_type_.size(), 0, "Must specify a db type.");
    CAFFE_ENFORCE(
        blob_names_.empty() ||
            blob_names_.size() == OperatorBase::Inputs().size(),
        "Number of blobs and blob_name_overrides mismatch.");
    CAFFE_ENFORCE(
        blob_names_.empty() || strip_prefix_.empty(),
        "strip_prefix and blob_name_overrides are mutually exclusive.");

    if (blob_names_.empty()) {
      std::set<std::string> input_names;
      blob_names_.resize(OperatorBase::Inputs().size());
      for (int i = 0; i < blob_names_.size(); ++i) {
        std::string name;
        if (strip_prefix_.empty()) {
          name = operator_def.input(i);
        } else {
          auto match_pos = operator_def.input(i).find(strip_prefix_);
          if (match_pos == string::npos) {
            name = operator_def.input(i);
          } else {
            name = operator_def.input(i).substr(
                match_pos + strip_prefix_.size(), string::npos);
          }
        }
        CAFFE_ENFORCE(
            input_names.insert(name).second, "Duplicated input: ", name);
        blob_names_[i] = name;
      }
    }
  }

  bool RunOnDevice() override {
    string full_db_name =
        absolute_path_ ? db_name_ : (ws_->RootFolder() + "/" + db_name_);
    std::unique_ptr<DB> out_db(
        caffe2::db::CreateDB(db_type_, full_db_name, caffe2::db::NEW));
    CAFFE_ENFORCE(out_db.get(), "Cannot open db for writing: ", full_db_name);

    BlobSerializerBase::SerializationAcceptor acceptor = [&](
        const std::string& blobName, const std::string& data) {
      // transaction should take care of locking
      VLOG(2) << "Sending " << blobName << " blob's data of size "
              << data.size() << " to db";
      auto transaction = out_db->NewTransaction();
      transaction->Put(blobName, data);
      transaction->Commit();
    };

    const vector<const Blob*>& inputs = OperatorBase::Inputs();
    for (int i = 0; i < inputs.size(); ++i) {
      inputs[i]->Serialize(blob_names_[i], acceptor);
    }
    out_db->Close();
    return true;
  }

 private:
  Workspace* ws_;
  bool absolute_path_;
  string strip_prefix_;
  string db_name_;
  string db_type_;
  std::vector<std::string> blob_names_;
};

template <typename... Ts>
string FormatString(const string& pattern, Ts... values) {
  // Note(Yangqing): We believe that 1024 is enough, but who are we to assert
  // that?
  // As a result, if things go wrong, we'll just throw the towel and quit loud.
  // Yeah, I know that there is snprintf, but it is not present in *some*
  // platforms unfortunately.
  char buffer[1024];
  int written = sprintf(buffer, pattern.c_str(), values...);
  if (written < 0 || written + 1 > 1024) {
    LOG(FATAL) << "FormatString fails: total bytes written " << written;
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

// CheckpointOp is a wrapper over a SaveFloatTensorOp that basically allows
// flexible naming over iterations.
// The file pattern in db_name should be a format string that can be passed into
// sprintf with an int argument specifying the current iteration. An example:
//     "/path/to/my/checkpoint/checkpoint_at_%d.pb"
template <class Context>
class CheckpointOp final : public Operator<Context> {
 public:
  CheckpointOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        db_pattern_(OperatorBase::GetSingleArgument<string>("db", "")),
        every_(OperatorBase::GetSingleArgument<int>("every", 1)),
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

  bool RunOnDevice() override {
    int64_t iter =
        OperatorBase::Input<TensorCPU>(0).template data<int64_t>()[0];
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
