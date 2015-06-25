#include <cstdio>
#include <mutex>

#include "caffe2/core/db.h"
#include "glog/logging.h"

namespace caffe2 {
namespace db {

class MiniDBCursor : public Cursor {
 public:
  explicit MiniDBCursor(FILE* f, std::mutex* mutex)
    : file_(f), lock_(*mutex) {}
  ~MiniDBCursor() {}

  void SeekToFirst() override {
    fseek(file_, 0, SEEK_SET);
    CHECK(!feof(file_)) << "Hmm, empty file?";
    // Read the first item.
    valid_ = true;
    Next();
  }

  void Next() override {
    if (fread(&key_len_, sizeof(int), 1, file_) == 0) {
      // Reaching EOF.
      valid_ = false;
      return;
    }
    CHECK_EQ(fread(&value_len_, sizeof(int), 1, file_), 1);
    CHECK_GT(key_len_, 0);
    CHECK_GT(value_len_, 0);
    if (key_len_ > key_.size()) {
      key_.resize(key_len_);
    }
    if (value_len_ > value_.size()) {
      value_.resize(value_len_);
    }
    CHECK_EQ(fread(key_.data(), sizeof(char), key_len_, file_), key_len_);
    CHECK_EQ(fread(value_.data(), sizeof(char), value_len_, file_), value_len_);
  }

  string key() override {
    CHECK(valid_) << "Invalid position!";
    return string(key_.data(), key_len_);
  }

  string value() override {
    CHECK(valid_) << "Invalid position!";
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
  ~MiniDBTransaction() { Commit(); }

  void Put(const string& key, const string& value) override {
    int key_len = key.size();
    int value_len = value.size();
    CHECK_EQ(fwrite(&key_len, sizeof(int), 1, file_), 1);
    CHECK_EQ(fwrite(&value_len, sizeof(int), 1, file_), 1);
    CHECK_EQ(fwrite(key.c_str(), sizeof(char), key_len, file_), key_len);
    CHECK_EQ(fwrite(value.c_str(), sizeof(char), value_len, file_), value_len);
  }

  void Commit() override {
    CHECK_EQ(fflush(file_), 0);
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
    CHECK(file_) << "Cannot open file: " << source;
    LOG(INFO) << "Opened MiniDB " << source;
  }
  ~MiniDB() { Close(); }

  void Close() override { fclose(file_); }

  Cursor* NewCursor() override {
    CHECK_EQ(this->mode_, READ);
    return new MiniDBCursor(file_, &file_access_mutex_);
  }

  Transaction* NewTransaction() override {
    CHECK(this->mode_ == NEW || this->mode_ == WRITE);
    return new MiniDBTransaction(file_, &file_access_mutex_);
  }

 private:
  FILE* file_;
  // access mutex makes sure we don't have multiple cursors/transactions
  // reading the same file.
  std::mutex file_access_mutex_;
};

REGISTER_CAFFE2_DB(MiniDB, MiniDB);
REGISTER_CAFFE2_DB(minidb, MiniDB);

}  // namespace db
}  // namespace caffe2
