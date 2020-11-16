#ifndef CAFFE2_CORE_DB_H_
#define CAFFE2_CORE_DB_H_

#include <mutex>

#include "c10/util/Registry.h"
#include "caffe2/core/blob_serialization.h"
#include "caffe2/proto/caffe2_pb.h"

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
class CAFFE2_API Cursor {
 public:
  Cursor() {}
  virtual ~Cursor() {}
  /**
   * Seek to a specific key (or if the key does not exist, seek to the
   * immediate next). This is optional for dbs, and in default, SupportsSeek()
   * returns false meaning that the db cursor does not support it.
   */
  virtual void Seek(const string& key) = 0;
  virtual bool SupportsSeek() {
    return false;
  }
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

  C10_DISABLE_COPY_AND_ASSIGN(Cursor);
};

/**
 * An abstract class for the current database transaction while writing.
 */
class CAFFE2_API Transaction {
 public:
  Transaction() {}
  virtual ~Transaction() {}
  /**
   * Puts the key value pair to the database.
   */
  virtual void Put(const string& key, const string& value) = 0;
  /**
   * Commits the current writes.
   */
  virtual void Commit() = 0;

  C10_DISABLE_COPY_AND_ASSIGN(Transaction);
};

/**
 * An abstract class for accessing a database of key-value pairs.
 */
class CAFFE2_API DB {
 public:
  DB(const string& /*source*/, Mode mode) : mode_(mode) {}
  virtual ~DB() {}
  /**
   * Closes the database.
   */
  virtual void Close() = 0;
  /**
   * Returns a cursor to read the database. The caller takes the ownership of
   * the pointer.
   */
  virtual std::unique_ptr<Cursor> NewCursor() = 0;
  /**
   * Returns a transaction to write data to the database. The caller takes the
   * ownership of the pointer.
   */
  virtual std::unique_ptr<Transaction> NewTransaction() = 0;

 protected:
  Mode mode_;

  C10_DISABLE_COPY_AND_ASSIGN(DB);
};

// Database classes are registered by their names so we can do optional
// dependencies.
C10_DECLARE_REGISTRY(Caffe2DBRegistry, DB, const string&, Mode);
#define REGISTER_CAFFE2_DB(name, ...) \
  C10_REGISTER_CLASS(Caffe2DBRegistry, name, __VA_ARGS__)

/**
 * Returns a database object of the given database type, source and mode. The
 * caller takes the ownership of the pointer. If the database type is not
 * supported, a nullptr is returned. The caller is responsible for examining the
 * validity of the pointer.
 */
inline unique_ptr<DB>
CreateDB(const string& db_type, const string& source, Mode mode) {
  auto result = Caffe2DBRegistry()->Create(db_type, source, mode);
  VLOG(1) << ((!result) ? "not found db " : "found db ") << db_type;
  return result;
}

/**
 * Returns whether or not a database exists given the database type and path.
 */
inline bool DBExists(const string& db_type, const string& full_db_name) {
  // Warning! We assume that creating a DB throws an exception if the DB
  // does not exist. If the DB constructor does not follow this design
  // pattern,
  // the returned output (the existence tensor) can be wrong.
  try {
    std::unique_ptr<DB> db(
        caffe2::db::CreateDB(db_type, full_db_name, caffe2::db::READ));
    return true;
  } catch (...) {
    return false;
  }
}

/**
 * A reader wrapper for DB that also allows us to serialize it.
 */
class CAFFE2_API DBReader {
 public:
  friend class DBReaderSerializer;
  DBReader() {}

  DBReader(
      const string& db_type,
      const string& source,
      const int32_t num_shards = 1,
      const int32_t shard_id = 0) {
    Open(db_type, source, num_shards, shard_id);
  }

  explicit DBReader(const DBReaderProto& proto) {
    Open(proto.db_type(), proto.source());
    if (proto.has_key()) {
      CAFFE_ENFORCE(
          cursor_->SupportsSeek(),
          "Encountering a proto that needs seeking but the db type "
          "does not support it.");
      cursor_->Seek(proto.key());
    }
    num_shards_ = 1;
    shard_id_ = 0;
  }

  explicit DBReader(std::unique_ptr<DB> db)
      : db_type_("<memory-type>"),
        source_("<memory-source>"),
        db_(std::move(db)) {
    CAFFE_ENFORCE(db_.get(), "Passed null db");
    cursor_ = db_->NewCursor();
  }

  void Open(
      const string& db_type,
      const string& source,
      const int32_t num_shards = 1,
      const int32_t shard_id = 0) {
    // Note(jiayq): resetting is needed when we re-open e.g. leveldb where no
    // concurrent access is allowed.
    cursor_.reset();
    db_.reset();
    db_type_ = db_type;
    source_ = source;
    db_ = CreateDB(db_type_, source_, READ);
    CAFFE_ENFORCE(
        db_,
        "Cannot find db implementation of type ",
        db_type,
        " (while trying to open ",
        source_,
        ")");
    InitializeCursor(num_shards, shard_id);
  }

  void Open(
      unique_ptr<DB>&& db,
      const int32_t num_shards = 1,
      const int32_t shard_id = 0) {
    cursor_.reset();
    db_.reset();
    db_ = std::move(db);
    CAFFE_ENFORCE(db_.get(), "Passed null db");
    InitializeCursor(num_shards, shard_id);
  }

 public:
  /**
   * Read a set of key and value from the db and move to next. Thread safe.
   *
   * The string objects key and value must be created by the caller and
   * explicitly passed in to this function. This saves one additional object
   * copy.
   *
   * If the cursor reaches its end, the reader will go back to the head of
   * the db. This function can be used to enable multiple input ops to read
   * the same db.
   *
   * Note(jiayq): we loosen the definition of a const function here a little
   * bit: the state of the cursor is actually changed. However, this allows
   * us to pass in a DBReader to an Operator without the need of a duplicated
   * output blob.
   */
  void Read(string* key, string* value) const {
    CAFFE_ENFORCE(cursor_ != nullptr, "Reader not initialized.");
    std::unique_lock<std::mutex> mutex_lock(reader_mutex_);
    *key = cursor_->key();
    *value = cursor_->value();

    // In sharded mode, each read skips num_shards_ records
    for (uint32_t s = 0; s < num_shards_; s++) {
      cursor_->Next();
      if (!cursor_->Valid()) {
        MoveToBeginning();
        break;
      }
    }
  }

  /**
   * @brief Seeks to the first key. Thread safe.
   */
  void SeekToFirst() const {
    CAFFE_ENFORCE(cursor_ != nullptr, "Reader not initialized.");
    std::unique_lock<std::mutex> mutex_lock(reader_mutex_);
    MoveToBeginning();
  }

  /**
   * Returns the underlying cursor of the db reader.
   *
   * Note that if you directly use the cursor, the read will not be thread
   * safe, because there is no mechanism to stop multiple threads from
   * accessing the same cursor. You should consider using Read() explicitly.
   */
  inline Cursor* cursor() const {
    VLOG(1) << "Usually for a DBReader you should use Read() to be "
               "thread safe. Consider refactoring your code.";
    return cursor_.get();
  }

 private:
  void InitializeCursor(const int32_t num_shards, const int32_t shard_id) {
    CAFFE_ENFORCE(num_shards >= 1);
    CAFFE_ENFORCE(shard_id >= 0);
    CAFFE_ENFORCE(shard_id < num_shards);
    num_shards_ = num_shards;
    shard_id_ = shard_id;
    cursor_ = db_->NewCursor();
    SeekToFirst();
  }

  void MoveToBeginning() const {
    cursor_->SeekToFirst();
    for (uint32_t s = 0; s < shard_id_; s++) {
      cursor_->Next();
      CAFFE_ENFORCE(
          cursor_->Valid(), "Db has fewer rows than shard id: ", s, shard_id_);
    }
  }

  string db_type_;
  string source_;
  unique_ptr<DB> db_;
  unique_ptr<Cursor> cursor_;
  mutable std::mutex reader_mutex_;
  uint32_t num_shards_{};
  uint32_t shard_id_{};

  C10_DISABLE_COPY_AND_ASSIGN(DBReader);
};

class CAFFE2_API DBReaderSerializer : public BlobSerializerBase {
 public:
  /**
   * Serializes a DBReader. Note that this blob has to contain DBReader,
   * otherwise this function produces a fatal error.
   */
  void Serialize(
      const void* pointer,
      TypeMeta typeMeta,
      const string& name,
      BlobSerializerBase::SerializationAcceptor acceptor) override;
};

class CAFFE2_API DBReaderDeserializer : public BlobDeserializerBase {
 public:
  void Deserialize(const BlobProto& proto, Blob* blob) override;
};

} // namespace db
} // namespace caffe2

#endif // CAFFE2_CORE_DB_H_
