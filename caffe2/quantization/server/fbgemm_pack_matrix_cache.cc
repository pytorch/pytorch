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
    int32_t ld,
    int32_t zero_point) {
  static std::map<
      std::tuple<int, int, const void*>,
      weak_ptr<fbgemm::PackBMatrix<int8_t, ACC_T>>>
      cache;
  static mutex cache_mutex;

  lock_guard<mutex> lock(cache_mutex);

  // Create a new packed matrix and compare with cached one if there's any.
  // TODO: make this cheaper by computing hash of fdata.
  auto new_packed = make_shared<fbgemm::PackBMatrix<int8_t, ACC_T>>(
      trans,
      m,
      n,
      quantized_data,
      ld,
      nullptr, // pmat
      1, // groups
      zero_point);

  std::tuple<int, int, const void*> key(m, n, orig_data);
  auto itr = cache.find(key);

  if (itr == cache.end() || !itr->second.lock() ||
      !itr->second.lock()->metaEquals(*new_packed)) {
    // cache miss
    cache[key] = new_packed;
    return new_packed;
  } else if (!itr->second.lock()->equals(*new_packed)) {
    // cache hit but content is different then just copy the packed matrix
    memcpy(
        itr->second.lock()->getBuf(),
        new_packed->getBuf(),
        new_packed->blockRows() * new_packed->blockRowSize() *
            new_packed->blockCols() * new_packed->blockColSize() *
            sizeof(uint8_t));
    return itr->second.lock();
  } else {
    return itr->second.lock();
  }
}

template shared_ptr<fbgemm::PackBMatrix<int8_t, int16_t>>
GetOrCreateFbgemmPackBMatrix<int16_t>(
    fbgemm::matrix_op_t trans,
    int32_t m,
    int32_t n,
    const void* orig_data,
    const int8_t* quantized_data,
    int32_t ld,
    int32_t zero_point);

template shared_ptr<fbgemm::PackBMatrix<int8_t, int32_t>>
GetOrCreateFbgemmPackBMatrix<int32_t>(
    fbgemm::matrix_op_t trans,
    int32_t m,
    int32_t n,
    const void* orig_data,
    const int8_t* quantized_data,
    int32_t ld,
    int32_t zero_point);

} // namespace caffe2
