#include "fbgemm_pack_matrix_cache.h"

#include <map>
#include <memory>
#include <mutex>

using namespace std;

namespace caffe2 {

template <typename ACC_T>
shared_ptr<fbgemm::PackBMatrix<int8_t, ACC_T>> GetOrCreateFbgemmPackBMatrix(
    fbgemm::matrix_op_t trans,
    int32_t m,
    int32_t n,
    const void* orig_data,
    const int8_t* quantized_data,
    int32_t ld) {
  static std::map<
      std::tuple<int, int, const void*>,
      weak_ptr<fbgemm::PackBMatrix<int8_t, ACC_T>>>
      cache;
  static mutex cache_mutex;

  // Create a new packed matrix and compare with cached one if there's any.
  // Note that a cache miss is as expensive as a cache hit here, the purpose of
  // this cache is only to deduplicate the quantized tensors for improved
  // memory bandwidth if different nets share copies of the same operator.
  // TODO: make this cheaper by computing hash of fdata.
  auto new_packed = make_shared<fbgemm::PackBMatrix<int8_t, ACC_T>>(
      trans,
      m,
      n,
      quantized_data,
      ld,
      nullptr, // pmat
      1); // groups

  std::tuple<int, int, const void*> key(m, n, orig_data);
  std::shared_ptr<fbgemm::PackBMatrix<int8_t, ACC_T>> cache_entry;
  {
    lock_guard<mutex> lock(cache_mutex);
    auto itr = cache.find(key);
    if (itr != cache.end()) {
      cache_entry = itr->second.lock();
    }
  } // release lock here during expensive equals()

  if (!cache_entry || !cache_entry->metaEquals(*new_packed) ||
      !cache_entry->equals(*new_packed)) {
    // cache miss
    lock_guard<mutex> lock(cache_mutex);
    cache[key] = new_packed;
    return new_packed;
  } else {
    return cache_entry;
  }
}

template shared_ptr<fbgemm::PackBMatrix<int8_t, int16_t>>
GetOrCreateFbgemmPackBMatrix<int16_t>(
    fbgemm::matrix_op_t trans,
    int32_t m,
    int32_t n,
    const void* orig_data,
    const int8_t* quantized_data,
    int32_t ld);

template shared_ptr<fbgemm::PackBMatrix<int8_t, int32_t>>
GetOrCreateFbgemmPackBMatrix<int32_t>(
    fbgemm::matrix_op_t trans,
    int32_t m,
    int32_t n,
    const void* orig_data,
    const int8_t* quantized_data,
    int32_t ld);

} // namespace caffe2
