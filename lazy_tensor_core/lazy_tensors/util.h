#pragma once


namespace lazy_tensors {

inline c10::ArrayRef<int64> AsInt64Slice(c10::ArrayRef<int64> slice) {
  return slice;
}

}  // namespace lazy_tensors
