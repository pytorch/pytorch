#ifndef CAFFE2_CORE_DB_H_
#define CAFFE2_CORE_DB_H_

#include "caffe2/core/registry.h"

namespace caffe2 {
namespace db {

/**
 * The mode of the database, whether we are doing a read, write, or creating
 * a new database.
 */
enum Mode { READ, WRITE, NEW };

/**
 * An abstract class for the cursor of the database while reading.
 */
class Cursor {
 public:
  Cursor() { }
  virtual ~Cursor() { }
  /**
   * Seek to the first key in the database.
   */
  virtual void SeekToFirst() = 0;
  /**
   * Go to the next location in the database.
   */
  virtual void Next() = 0;
  /**
   * Returns the current key.
   */
  virtual string key() = 0;
  /**
   * Returns the current value.
   */
  virtual string value() = 0;
  /**
   * Returns whether the current location is valid - for example, if we have
   * reached the end of the database, return false.
   */
  virtual bool Valid() = 0;

  DISABLE_COPY_AND_ASSIGN(Cursor);
};

/**
 * An abstract class for the current database transaction while writing.
 */
class Transaction {
 public:
  Transaction() { }
  virtual ~Transaction() { }
  /**
   * Puts the key value pair to the database.
   */
  virtual void Put(const string& key, const string& value) = 0;
  /**
   * Commits the current writes.
   */
  virtual void Commit() = 0;

  DISABLE_COPY_AND_ASSIGN(Transaction);
};

/**
 * An abstract class for accessing a database of key-value pairs.
 */
class DB {
 public:
  DB(const string& source, Mode mode) : mode_(mode) {}
  virtual ~DB() { }
  /**
   * Closes the database.
   */
  virtual void Close() = 0;
  /**
   * Returns a cursor to read the database. The caller takes the ownership of
   * the pointer.
   */
  virtual Cursor* NewCursor() = 0;
  /**
   * Returns a transaction to write data to the database. The caller takes the
   * ownership of the pointer.
   */
  virtual Transaction* NewTransaction() = 0;

 protected:
  Mode mode_;

  DISABLE_COPY_AND_ASSIGN(DB);
};

// Database classes are registered by their names so we can do optional
// dependencies.
DECLARE_REGISTRY(Caffe2DBRegistry, DB, const string&, Mode);
#define REGISTER_CAFFE2_DB(name, ...) \
  REGISTER_CLASS(Caffe2DBRegistry, name, __VA_ARGS__)

/**
 * Returns a database object of the given database type, source and mode. The
 * caller takes the ownership of the pointer. If the database type is not
 * supported, a nullptr is returned. The caller is responsible for examining the
 * validity of the pointer.
 */
inline DB* CreateDB(const string& db_type, const string& source, Mode mode) {
  return Caffe2DBRegistry()->Create(db_type, source, mode);
}

}  // namespace db
}  // namespace caffe2

#endif  // CAFFE2_CORE_DB_H_
