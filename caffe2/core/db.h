#ifndef CAFFE2_CORE_DB_H_
#define CAFFE2_CORE_DB_H_

#include "caffe2/core/registry.h"

namespace caffe2 {
namespace db {

enum Mode { READ, WRITE, NEW };

class Cursor {
 public:
  Cursor() { }
  virtual ~Cursor() { }
  virtual void SeekToFirst() = 0;
  virtual void Next() = 0;
  virtual string key() = 0;
  virtual string value() = 0;
  virtual bool Valid() = 0;

  DISABLE_COPY_AND_ASSIGN(Cursor);
};

class Transaction {
 public:
  Transaction() { }
  virtual ~Transaction() { }
  virtual void Put(const string& key, const string& value) = 0;
  virtual void Commit() = 0;

  DISABLE_COPY_AND_ASSIGN(Transaction);
};

class DB {
 public:
  DB(const string& source, Mode mode) : mode_(mode) {
    // This constructor does nothing. The actual opening should be done in the
    // derived constructors.
  }
  virtual ~DB() { }
  virtual void Close() = 0;
  virtual Cursor* NewCursor() = 0;
  virtual Transaction* NewTransaction() = 0;

 protected:
  Mode mode_;

  DISABLE_COPY_AND_ASSIGN(DB);
};

DECLARE_REGISTRY(Caffe2DBRegistry, DB, const string&, Mode);
#define REGISTER_CAFFE2_DB(name, ...) \
  REGISTER_CLASS(Caffe2DBRegistry, name, __VA_ARGS__)

inline DB* CreateDB(const string& db_type, const string& source, Mode mode) {
  return Caffe2DBRegistry()->Create(db_type, source, mode);
}

}  // namespace db
}  // namespace caffe2

#endif  // CAFFE2_CORE_DB_H_
