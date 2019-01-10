#include "caffe2/core/db.h"

#include <mutex>

#include "caffe2/core/blob_serialization.h"
#include "caffe2/core/logging.h"

namespace caffe2 {

CAFFE_KNOWN_TYPE(db::DBReader);
CAFFE_KNOWN_TYPE(db::Cursor);

namespace db {

CAFFE_DEFINE_REGISTRY(Caffe2DBRegistry, DB, const string&, Mode);

// Below, we provide a bare minimum database "minidb" as a reference
// implementation as well as a portable choice to store data.
// Note that the MiniDB classes are not exposed via a header file - they should
// be created directly via the db interface. See MiniDB for details.

class MiniDBCursor : public Cursor {
 public:
  explicit MiniDBCursor(FILE* f, std::mutex* mutex)
    : file_(f), lock_(*mutex), valid_(true) {
    // We call Next() to read in the first entry.
    Next();
  }
  ~MiniDBCursor() {}

  void Seek(const string& /*key*/) override {
    LOG(FATAL) << "MiniDB does not support seeking to a specific key.";
  }

  void SeekToFirst() override {
    fseek(file_, 0, SEEK_SET);
    CAFFE_ENFORCE(!feof(file_), "Hmm, empty file?");
    // Read the first item.
    valid_ = true;
    Next();
  }

  void Next() override {
    // First, read in the key and value length.
    if (fread(&key_len_, sizeof(int), 1, file_) == 0) {
      // Reaching EOF.
      VLOG(1) << "EOF reached, setting valid to false";
      valid_ = false;
      return;
    }
    CAFFE_ENFORCE_EQ(fread(&value_len_, sizeof(int), 1, file_), 1);
    CAFFE_ENFORCE_GT(key_len_, 0);
    CAFFE_ENFORCE_GT(value_len_, 0);
    // Resize if the key and value len is larger than the current one.
    if (key_len_ > key_.size()) {
      key_.resize(key_len_);
    }
    if (value_len_ > value_.size()) {
      value_.resize(value_len_);
    }
    // Actually read in the contents.
    CAFFE_ENFORCE_EQ(
        fread(key_.data(), sizeof(char), key_len_, file_), key_len_);
    CAFFE_ENFORCE_EQ(
        fread(value_.data(), sizeof(char), value_len_, file_), value_len_);
    // Note(Yangqing): as we read the file, the cursor naturally moves to the
    // beginning of the next entry.
  }

  string key() override {
    CAFFE_ENFORCE(valid_, "Cursor is at invalid location!");
    return string(key_.data(), key_len_);
  }

  string value() override {
    CAFFE_ENFORCE(valid_, "Cursor is at invalid location!");
    return string(value_.data(), value_len_);
  }

  bool Valid() override { return valid_; }

 private:
  FILE* file_;
  std::lock_guard<std::mutex> lock_;
  bool valid_;
  int key_len_;
  vector<char> key_;
  int value_len_;
  vector<char> value_;
};

class MiniDBTransaction : public Transaction {
 public:
  explicit MiniDBTransaction(FILE* f, std::mutex* mutex)
    : file_(f), lock_(*mutex) {}
  ~MiniDBTransaction() {
    Commit();
  }

  void Put(const string& key, const string& value) override {
    int key_len = key.size();
    int value_len = value.size();
    CAFFE_ENFORCE_EQ(fwrite(&key_len, sizeof(int), 1, file_), 1);
    CAFFE_ENFORCE_EQ(fwrite(&value_len, sizeof(int), 1, file_), 1);
    CAFFE_ENFORCE_EQ(
        fwrite(key.c_str(), sizeof(char), key_len, file_), key_len);
    CAFFE_ENFORCE_EQ(
        fwrite(value.c_str(), sizeof(char), value_len, file_), value_len);
  }

  void Commit() override {
    if (file_ != nullptr) {
      CAFFE_ENFORCE_EQ(fflush(file_), 0);
      file_ = nullptr;
    }
  }

 private:
  FILE* file_;
  std::lock_guard<std::mutex> lock_;

  DISABLE_COPY_AND_ASSIGN(MiniDBTransaction);
};

class MiniDB : public DB {
 public:
  MiniDB(const string& source, Mode mode) : DB(source, mode), file_(nullptr) {
    switch (mode) {
      case NEW:
        file_ = fopen(source.c_str(), "wb");
        break;
      case WRITE:
        file_ = fopen(source.c_str(), "ab");
        fseek(file_, 0, SEEK_END);
        break;
      case READ:
        file_ = fopen(source.c_str(), "rb");
        break;
    }
    CAFFE_ENFORCE(file_, "Cannot open file: " + source);
    VLOG(1) << "Opened MiniDB " << source;
  }
  ~MiniDB() { Close(); }

  void Close() override {
    if (file_) {
      fclose(file_);
    }
    file_ = nullptr;
  }

  unique_ptr<Cursor> NewCursor() override {
    CAFFE_ENFORCE_EQ(this->mode_, READ);
    return make_unique<MiniDBCursor>(file_, &file_access_mutex_);
  }

  unique_ptr<Transaction> NewTransaction() override {
    CAFFE_ENFORCE(this->mode_ == NEW || this->mode_ == WRITE);
    return make_unique<MiniDBTransaction>(file_, &file_access_mutex_);
  }

 private:
  FILE* file_;
  // access mutex makes sure we don't have multiple cursors/transactions
  // reading the same file.
  std::mutex file_access_mutex_;
};

REGISTER_CAFFE2_DB(MiniDB, MiniDB);
REGISTER_CAFFE2_DB(minidb, MiniDB);

void DBReaderSerializer::Serialize(
    const Blob& blob,
    const string& name,
    BlobSerializerBase::SerializationAcceptor acceptor) {
  CAFFE_ENFORCE(blob.IsType<DBReader>());
  auto& reader = blob.Get<DBReader>();
  DBReaderProto proto;
  proto.set_name(name);
  proto.set_source(reader.source_);
  proto.set_db_type(reader.db_type_);
  if (reader.cursor() && reader.cursor()->SupportsSeek()) {
    proto.set_key(reader.cursor()->key());
  }
  BlobProto blob_proto;
  blob_proto.set_name(name);
  blob_proto.set_type("DBReader");
  blob_proto.set_content(proto.SerializeAsString());
  acceptor(name, blob_proto.SerializeAsString());
}

void DBReaderDeserializer::Deserialize(const BlobProto& proto, Blob* blob) {
  DBReaderProto reader_proto;
  CAFFE_ENFORCE(
      reader_proto.ParseFromString(proto.content()),
      "Cannot parse content into a DBReaderProto.");
  blob->Reset(new DBReader(reader_proto));
}

namespace {
// Serialize TensorCPU.
REGISTER_BLOB_SERIALIZER((TypeMeta::Id<DBReader>()),
                         DBReaderSerializer);
REGISTER_BLOB_DESERIALIZER(DBReader, DBReaderDeserializer);
}  // namespace

}  // namespace db
}  // namespace caffe2
