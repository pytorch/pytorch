#include "caffe2/core/utils_test.h"
using namespace ::caffe2::db;
namespace caffe2 {
REGISTER_CAFFE2_DB(vector_db, VectorDB);
std::mutex& VectorDB::DataRegistryMutex() {
  static std::mutex dataRegistryMutexVar;
  return dataRegistryMutexVar;
}
std::map<string, StringMap>& VectorDB::Data() {
  static std::map<string, StringMap> dataVar;
  return dataVar;
}
}
