#include "caffe2/core/db.h"
#include "caffe2/core/logging.h"
#include "leveldb/db.h"
#include "leveldb/write_batch.h"

namespace caffe2 {
namespace db {

class LevelDBCursor : public Cursor {
 public:
  explicit LevelDBCursor(leveldb::Iterator* iter)
    : iter_(iter) { SeekToFirst(); }
  ~LevelDBCursor() { delete iter_; }
  void SeekToFirst() override { iter_->SeekToFirst(); }
  void Next() override { iter_->Next(); }
  string key() override { return iter_->key().ToString(); }
  string value() override { return iter_->value().ToString(); }
  bool Valid() override { return iter_->Valid(); }

 private:
  leveldb::Iterator* iter_;
};

class LevelDBTransaction : public Transaction {
 public:
  explicit LevelDBTransaction(leveldb::DB* db) : db_(db) {
    CAFFE_CHECK_NOTNULL(db_);
    batch_.reset(new leveldb::WriteBatch());
  }
  ~LevelDBTransaction() { Commit(); }
  void Put(const string& key, const string& value) override {
    batch_->Put(key, value);
  }
  void Commit() override {
    leveldb::Status status = db_->Write(leveldb::WriteOptions(), batch_.get());
    batch_.reset(new leveldb::WriteBatch());
    CAFFE_CHECK(status.ok()) << "Failed to write batch to leveldb "
                       << std::endl << status.ToString();
  }

 private:
  leveldb::DB* db_;
  std::unique_ptr<leveldb::WriteBatch> batch_;

  DISABLE_COPY_AND_ASSIGN(LevelDBTransaction);
};

class LevelDB : public DB {
 public:
  LevelDB(const string& source, Mode mode) : DB(source, mode) {
    leveldb::Options options;
    options.block_size = 65536;
    options.write_buffer_size = 268435456;
    options.max_open_files = 100;
    options.error_if_exists = mode == NEW;
    options.create_if_missing = mode != READ;
    leveldb::DB* db_temp;
    leveldb::Status status = leveldb::DB::Open(options, source, &db_temp);
    CAFFE_CHECK(status.ok()) << "Failed to open leveldb " << source
                       << std::endl << status.ToString();
    db_.reset(db_temp);
    CAFFE_LOG_INFO << "Opened leveldb " << source;
  }

  void Close() override { db_.reset(); }
  Cursor* NewCursor() override {
    return new LevelDBCursor(db_->NewIterator(leveldb::ReadOptions()));
  }
  Transaction* NewTransaction() override {
    return new LevelDBTransaction(db_.get());
  }

 private:
  std::unique_ptr<leveldb::DB> db_;
};

REGISTER_CAFFE2_DB(LevelDB, LevelDB);
// For lazy-minded, one can also call with lower-case name.
REGISTER_CAFFE2_DB(leveldb, LevelDB);

}  // namespace db
}  // namespace caffe2
