#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace lazy_tensors {

enum class PrimitiveType {
  PRED,
  S8,
  S16,
  S32,
  S64,
  U8,
  U16,
  U32,
  U64,
  F16,
  F32,
  BF16,
  F64,
  C64,
  C128,
  TUPLE,
  INVALID
};

namespace client {

class ShapeData {
 public:
  ShapeData() : element_type_(PrimitiveType::INVALID) {}

  ShapeData(PrimitiveType element_type, std::vector<int64_t> dimensions)
      : element_type_(element_type), dimensions_(std::move(dimensions)) {}

  ShapeData(std::vector<ShapeData> element_shapes)
      : element_type_(PrimitiveType::TUPLE),
        element_shapes_(std::move(element_shapes)) {}

  ShapeData(PrimitiveType element_type, std::vector<int64_t> dimensions,
            std::vector<ShapeData> element_shapes,
            std::vector<int64_t> minor_to_major)
      : element_type_(element_type),
        dimensions_(std::move(dimensions)),
        element_shapes_(std::move(element_shapes)),
        minor_to_major_(std::move(minor_to_major)) {}

  PrimitiveType element_type() const { return element_type_; }

  const std::vector<int64_t>& dimensions() const { return dimensions_; }

  const std::vector<ShapeData>& element_shapes() const {
    return element_shapes_;
  }

  const std::vector<int64_t>& minor_to_major() const { return minor_to_major_; }

 private:
  PrimitiveType element_type_;
  std::vector<int64_t> dimensions_;
  std::vector<ShapeData> element_shapes_;
  std::vector<int64_t> minor_to_major_;
};

class Data {
 public:
  struct Info {
    virtual ~Info() {}
  };

  using OpaqueHandle = int64_t;

  Data(std::string device, ShapeData shape)
      : device_(std::move(device)), shape_(std::move(shape)) {}

  virtual ~Data() {}

  const std::string& device() const { return device_; }

  const ShapeData& shape() const { return shape_; }

  Info* info() const { return info_.get(); }

  std::shared_ptr<Info> SetInfo(std::shared_ptr<Info> info) {
    std::swap(info, info_);
    return info;
  }

  virtual OpaqueHandle GetOpaqueHandle() = 0;

  virtual void Assign(const Data& data) = 0;

  virtual bool HasValue() const = 0;

 private:
  std::string device_;
  ShapeData shape_;
  std::shared_ptr<Info> info_;
};

using DataPtr = std::shared_ptr<Data>;

// The TensorSource provides a way for a client to populate a buffer allocated
// by the core computation client code.
struct TensorSource {
  // The PopulateFn accepts a dense buffer is standard array layout
  // (dim0-major) and deposits the source tensor data directly over the
  // provided buffer.
  using PopulateFn = std::function<void(const TensorSource&, void*, size_t)>;

  TensorSource() = default;
  TensorSource(ShapeData shape, std::string device, PopulateFn populate_fn)
      : shape(std::move(shape)),
        device(std::move(device)),
        populate_fn(std::move(populate_fn)) {}

  ShapeData shape;
  std::string device;
  PopulateFn populate_fn;
};

}  // namespace client
}  // namespace lazy_tensors
