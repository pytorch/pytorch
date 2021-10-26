#pragma once


namespace lazy_tensors {

inline c10::ArrayRef<int64_t> AsInt64Slice(c10::ArrayRef<int64_t> slice) {
  return slice;
}

}  // namespace lazy_tensors
