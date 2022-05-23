#include "caffe2/core/db.h"
#include "caffe2/core/flags.h"
#include "caffe2/core/logging.h"
#include "leveldb/db.h"
#include "leveldb/write_batch.h"

C10_DEFINE_int(
    caffe2_leveldb_block_size,
    65536,
    "The caffe2 leveldb block size when writing a leveldb.");

namespace caffe2 {
namespace db {

class LevelDBCursor : public Cursor {
 public:
  explicit LevelDBCursor(leveldb::DB* db)
      : iter_(db->NewIterator(leveldb::ReadOptions())) {
    SeekToFirst();
  }
  ~LevelDBCursor() override {}
  void Seek(const string& key) override {
    iter_->Seek(key);
  }
  bool SupportsSeek() override {
    return true;
  }
  void SeekToFirst() override {
    iter_->SeekToFirst();
  }
  void Next() override {
    iter_->Next();
  }
  string key() override {
    return iter_->key().ToString();
  }
  string value() override {
    return iter_->value().ToString();
  }
  bool Valid() override {
    return iter_->Valid();
  }

 private:
  std::unique_ptr<leveldb::Iterator> iter_;
};

class LevelDBTransaction : public Transaction {
 public:
  explicit LevelDBTransaction(leveldb::DB* db) : db_(db) {
    CAFFE_ENFORCE(db_);
    batch_.reset(new leveldb::WriteBatch());
  }
  ~LevelDBTransaction() override {
    Commit();
  }
  void Put(const string& key, string&& value) override {
    batch_->Put(key, value);
  }
  void Commit() override {
    leveldb::Status status = db_->Write(leveldb::WriteOptions(), batch_.get());
    batch_.reset(new leveldb::WriteBatch());
    CAFFE_ENFORCE(
        status.ok(), "Failed to write batch to leveldb. ", status.ToString());
  }

 private:
  leveldb::DB* db_;
  std::unique_ptr<leveldb::WriteBatch> batch_;

  C10_DISABLE_COPY_AND_ASSIGN(LevelDBTransaction);
};

class LevelDB : public DB {
 public:
  LevelDB(const string& source, Mode mode) : DB(source, mode) {
    leveldb::Options options;
    options.block_size = FLAGS_caffe2_leveldb_block_size;
    options.write_buffer_size = 268435456;
    options.max_open_files = 100;
    options.error_if_exists = mode == NEW;
    options.create_if_missing = mode != READ;
    leveldb::DB* db_temp;
    leveldb::Status status = leveldb::DB::Open(options, source, &db_temp);
    CAFFE_ENFORCE(
        status.ok(),
        "Failed to open leveldb ",
        source,
        ". ",
        status.ToString());
    db_.reset(db_temp);
    VLOG(1) << "Opened leveldb " << source;
  }

  void Close() override {
    db_.reset();
  }
  unique_ptr<Cursor> NewCursor() override {
    return make_unique<LevelDBCursor>(db_.get());
  }
  unique_ptr<Transaction> NewTransaction() override {
    return make_unique<LevelDBTransaction>(db_.get());
  }

 private:
  std::unique_ptr<leveldb::DB> db_;
};

REGISTER_CAFFE2_DB(LevelDB, LevelDB);
// For lazy-minded, one can also call with lower-case name.
REGISTER_CAFFE2_DB(leveldb, LevelDB);

} // namespace db
} // namespace caffe2
