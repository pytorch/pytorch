#ifndef CAFFE2_CORE_TEST_UTILS_H_
#define CAFFE2_CORE_TEST_UTILS_H_
#include <mutex>
#include "caffe2/core/db.h"
#include "caffe2/core/logging.h"

namespace caffe2 {
using StringMap = std::vector<std::pair<string, string>>;

class VectorCursor : public db::Cursor {
 public:
  explicit VectorCursor(StringMap* data) : data_(data) {
    pos_ = 0;
  }
  ~VectorCursor() {}
  void Seek(const string& /* unused */) override {}
  void SeekToFirst() override {}
  void Next() override {
    ++pos_;
  }
  string key() override {
    return (*data_)[pos_].first;
  }
  string value() override {
    return (*data_)[pos_].second;
  }
  bool Valid() override {
    return pos_ < data_->size();
  }

 private:
  StringMap* data_ = nullptr;
  size_t pos_ = 0;
};

class VectorDB : public db::DB {
 public:
  VectorDB(const string& source, db::Mode mode)
      : DB(source, mode), name_(source) {}
  ~VectorDB() {
    Data().erase(name_);
  }
  void Close() override {}
  std::unique_ptr<db::Cursor> NewCursor() override {
    return make_unique<VectorCursor>(getData());
  }
  std::unique_ptr<db::Transaction> NewTransaction() override {
    CAFFE_THROW("Not implemented");
  }
  static void registerData(const string& name, StringMap&& data) {
    std::lock_guard<std::mutex> guard(DataRegistryMutex());
    Data()[name] = std::move(data);
  }

 private:
  StringMap* getData() {
    auto it = Data().find(name_);
    CAFFE_ENFORCE(it != Data().end(), "Can't find ", name_);
    return &(it->second);
  }

 private:
  string name_;
  static std::mutex& DataRegistryMutex();
  static std::map<string, StringMap>& Data();
};
}
#endif // CAFFE2_CORE_TEST_UTILS_H_
