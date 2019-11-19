#ifndef CAFFE2_OPERATORS_LOAD_SAVE_OP_UTIL_H_
#define CAFFE2_OPERATORS_LOAD_SAVE_OP_UTIL_H_

#include <set>
#include <string>
#include <unordered_map>

#include "caffe2/core/blob.h"
#include "caffe2/core/blob_serialization.h"

namespace caffe2 {
namespace load_save_op_util {

struct BlobState {
  int64_t total_size;
  int64_t current_size;
  bool is_tensor;
  std::set<int32_t> seen_chunks_ids;

  explicit BlobState(
      int64_t total_size = 0,
      int64_t current_size = 0,
      bool is_tensor = false)
      : total_size(total_size),
        current_size(current_size),
        is_tensor(is_tensor) {}
};

std::string buildBlobNameFromDbKey(
    const std::string& dbKey,
    const std::string& strip_prefix = "",
    const std::string& add_prefix = "");

// We are tracking sizes of already read tensor parts while reading data
// chunks. This way we can make sure that all chunks were loaded in the end.
void ProcessBlob(
    Blob* blob,
    const BlobProto& proto,
    std::unordered_map<std::string, BlobState>* blob_states_ptr,
    const std::string& key,
    int* loaded_blobs);

void validateBlobStates(
    const std::unordered_map<std::string, BlobState>& blob_states);

} // namespace load_save_op_util
} // namespace caffe2

#endif // CAFFE2_OPERATORS_LOAD_SAVE_OP_UTIL_H_
