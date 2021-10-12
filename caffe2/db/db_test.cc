#include <cstdio>
#include <iomanip>
#include <sstream>
#include <thread>

#include <gtest/gtest.h>
#include "caffe2/core/blob_serialization.h"
#include "caffe2/core/db.h"
#include "caffe2/core/logging.h"
#include "caffe2/proto/caffe2_pb.h"
#include "common/gtest/gtest_extensions.h"

namespace caffe2 {
namespace db {

constexpr int kMaxItems = 10;

static bool CreateAndFill(const string& db_type, const string& name) {
  VLOG(1) << "Creating db: " << name;
  std::unique_ptr<DB> db(CreateDB(db_type, name, NEW));
  if (!db.get()) {
    LOG(ERROR) << "Cannot create db of type " << db_type;
    return false;
  }
  std::unique_ptr<Transaction> trans(db->NewTransaction());
  for (int i = 0; i < kMaxItems; ++i) {
    std::stringstream ss;
    ss << std::setw(2) << std::setfill('0') << i;
    trans->Put(ss.str(), std::string{ss.str()});
  }
  trans->Commit();
  trans.reset();
  db.reset();
  return true;
}

static void TestCursor(Cursor* cursor) {
  // Test the first key.
  cursor->SeekToFirst();
  EXPECT_EQ(cursor->key(), "00");
  EXPECT_EQ(cursor->value(), "00");
  // Test if Next() works.
  cursor->Next();
  EXPECT_EQ(cursor->key(), "01");
  cursor->Next();
  EXPECT_EQ(cursor->key(), "02");
  // Test if we can return to the first key.
  cursor->SeekToFirst();
  EXPECT_EQ(cursor->key(), "00");
  // Test seeking to a key that exists.
  cursor->Seek("05");
  EXPECT_EQ(cursor->key(), "05");
  // Test seeking to a key that does not exist - that should give us the
  // immediate next key.
  cursor->Seek("07.5");
  EXPECT_EQ(cursor->key(), "08");
  // Test seeking over the end of the db - that should make the current
  // cursor invalid.
  cursor->Seek("11");
  EXPECT_FALSE(cursor->Valid());
  // Test seeking to empty string, aka the beginning
  cursor->Seek("");
  EXPECT_EQ(cursor->key(), "00");
}

static void DBSeekTestWrapper(const string& db_type) {
  std::string name = std::tmpnam(nullptr);
  if (!CreateAndFill(db_type, name)) {
    // Manually fail the test, and not do anything onwards.
    EXPECT_TRUE(0);
  } else {
    std::unique_ptr<DB> db(CreateDB(db_type, name, READ));
    std::unique_ptr<Cursor> cursor(db->NewCursor());
    TestCursor(cursor.get());
  }
}

TEST(DBSeekTest, RocksDB) {
  SKIP() << "The test is broken. So skip.";
  DBSeekTestWrapper("rocksdb");
}

TEST(DBSeekTest, LevelDB) {
  DBSeekTestWrapper("leveldb");
}

TEST(DBSeekTest, LMDB) {
  SKIP() << "The test is broken. So skip.";
  DBSeekTestWrapper("lmdb");
}

TEST(DBReaderTest, Reader) {
  std::string name = std::tmpnam(nullptr);
  CreateAndFill("leveldb", name);
  std::unique_ptr<DBReader> reader(new DBReader("leveldb", name));
  EXPECT_TRUE(reader->cursor() != nullptr);
  // DBReader should have a full-fledged cursor.
  TestCursor(reader->cursor());
  // Test the Read() functionality.
  reader->cursor()->Seek("05");
  EXPECT_EQ(reader->cursor()->key(), "05");
  string key;
  string value;
  reader->Read(&key, &value);
  EXPECT_EQ(key, "05");
  EXPECT_EQ(value, "05");
  reader->Read(&key, &value);
  EXPECT_EQ(key, "06");
  EXPECT_EQ(value, "06");

  // Test if we are able to serialize it using the blob serialization
  // interface.
  reader->cursor()->Seek("05");
  EXPECT_EQ(reader->cursor()->key(), "05");
  Blob reader_blob;
  reader_blob.Reset(reader.release());
  std::string str = SerializeBlob(reader_blob, "saved_reader");
  // Release to close the old reader.
  reader_blob.Reset();
  BlobProto blob_proto;
  CHECK(blob_proto.ParseFromString(str));
  EXPECT_EQ(blob_proto.name(), "saved_reader");
  EXPECT_EQ(blob_proto.type(), "DBReader");
  DBReaderProto proto;
  CHECK(proto.ParseFromString(blob_proto.content()));
  EXPECT_EQ(proto.source(), name);
  EXPECT_EQ(proto.db_type(), "leveldb");
  EXPECT_EQ(proto.key(), "05");
  // Test restoring the reader from the serialized proto.
  EXPECT_NO_THROW(DeserializeBlob(str, &reader_blob));
  EXPECT_TRUE(reader_blob.IsType<DBReader>());
  const DBReader& new_reader = reader_blob.Get<DBReader>();
  EXPECT_TRUE(new_reader.cursor() != nullptr);
  EXPECT_EQ(new_reader.cursor()->key(), "05");

  // Test Reader's multi-threading capability.
  vector<unique_ptr<std::thread>> threads(kMaxItems);
  vector<string> keys(kMaxItems);
  vector<string> values(kMaxItems);
  for (int i = 0; i < kMaxItems; ++i) {
    threads[i].reset(new std::thread(
        [&new_reader](string* key, string* value) {
          new_reader.Read(key, value);
        },
        &keys[i],
        &values[i]));
  }
  for (int i = 0; i < kMaxItems; ++i) {
    threads[i]->join();
    EXPECT_TRUE(keys[i].size() > 0);
  }
  // Check if the names are all unique by putting them into a set and
  // checking the size.
  std::set<string> keys_set(keys.begin(), keys.end());
  EXPECT_EQ(keys_set.size(), kMaxItems);
}

TEST(DBReaderShardedTest, Reader) {
  std::string name = std::tmpnam(nullptr);
  CreateAndFill("leveldb", name);

  std::unique_ptr<DBReader> reader0(new DBReader("leveldb", name, 3, 0));
  string key;
  string value;
  reader0->Read(&key, &value);
  EXPECT_EQ(key, "00");
  EXPECT_EQ(value, "00");
  reader0->Read(&key, &value);
  EXPECT_EQ(key, "03");
  EXPECT_EQ(value, "03");
  reader0->Read(&key, &value);
  EXPECT_EQ(key, "06");
  EXPECT_EQ(value, "06");
  reader0->Read(&key, &value);
  EXPECT_EQ(key, "09");
  EXPECT_EQ(value, "09");
  reader0->Read(&key, &value);
  EXPECT_EQ(key, "00");
  EXPECT_EQ(value, "00");
  reader0->Read(&key, &value);
  EXPECT_EQ(key, "03");
  EXPECT_EQ(value, "03");

  CreateAndFill("leveldb", name + "1");
  std::unique_ptr<DBReader> reader1(new DBReader("leveldb", name + "1", 3, 1));
  reader1->Read(&key, &value);
  EXPECT_EQ(key, "01");
  EXPECT_EQ(value, "01");
  reader1->Read(&key, &value);
  EXPECT_EQ(key, "04");
  EXPECT_EQ(value, "04");

  CreateAndFill("leveldb", name + "2");
  std::unique_ptr<DBReader> reader2(new DBReader("leveldb", name + "2", 3, 2));
  reader2->Read(&key, &value);
  EXPECT_EQ(key, "02");
  EXPECT_EQ(value, "02");
  reader2->Read(&key, &value);
  EXPECT_EQ(key, "05");
  EXPECT_EQ(value, "05");
}

} // namespace db
} // namespace caffe2
