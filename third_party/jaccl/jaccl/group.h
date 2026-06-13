// Copyright © 2025 Apple Inc.

#pragma once

#include <cstddef>
#include <memory>

namespace jaccl {

/**
 * Abstract base class for a JACCL communication group.
 */
class Group {
 public:
  virtual ~Group() {}

  virtual int rank() = 0;
  virtual int size() = 0;

  virtual void
  all_sum(const void* input, void* output, size_t n_bytes, int dtype) = 0;

  virtual void
  all_max(const void* input, void* output, size_t n_bytes, int dtype) = 0;

  virtual void
  all_min(const void* input, void* output, size_t n_bytes, int dtype) = 0;

  virtual void all_gather(const void* input, void* output, size_t n_bytes) = 0;

  virtual void send(const void* input, size_t n_bytes, int dst) = 0;
  virtual void recv(void* output, size_t n_bytes, int src) = 0;
  virtual void barrier() = 0;
};

/**
 * Type IDs for dispatch in the standalone JACCL library.
 *
 * Users pass one of these to all_sum/all_max/all_min so JACCL knows how to
 * interpret the data for typed reduction operations.
 */
enum Dtype {
  Bool = 0,
  Int8,
  Int16,
  Int32,
  Int64,
  UInt8,
  UInt16,
  UInt32,
  UInt64,
  Float16,
  BFloat16,
  Float32,
  Float64,
  Complex64,
};

} // namespace jaccl
