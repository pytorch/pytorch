#pragma once

#include <c10/util/Exception.h>
#include <torch/csrc/distributed/spmd/event.h>

// This file contains implementations of all specific Event classes.

namespace torch {
namespace distributed {
namespace spmd {

class PrepareModuleEvent : public Event {
 public:
  explicit PrepareModuleEvent(std::vector<at::Tensor> params)
      : Event(EventSchema(EventType::PREPARE_MODULE)),
        params_(std::move(params)) {}

  inline const std::vector<at::Tensor>& parameters() const {
    return params_;
  }

 private:
  std::vector<at::Tensor> params_;
};

class PreForwardEvent : public Event {
 public:
  explicit PreForwardEvent() : Event(EventSchema(EventType::PRE_FORWARD)) {}
};

class LocalGradReadyEvent : public Event {
 public:
  LocalGradReadyEvent(size_t index, const at::Tensor& grad)
      : Event(EventSchema(EventType::LOCAL_GRAD_READY)),
        index_(index),
        grad_(grad) {}

  inline const at::Tensor& grad() {
    return grad_;
  }

  inline size_t index() const {
    return index_;
  }

 private:
  const size_t index_;
  const at::Tensor& grad_;
};

struct Bucket final {
  Bucket(
      size_t index,
      const at::Tensor& tensor,
      std::vector<size_t> paramIndices)
      : index_(index),
        tensor_(tensor),
        paramIndices_(std::move(paramIndices)) {}

  const size_t index_;
  const at::Tensor& tensor_;
  const std::vector<size_t> paramIndices_;
};

class BucketReadyEvent : public Event {
 public:
  explicit BucketReadyEvent(std::shared_ptr<Bucket> bucket)
      : Event(EventSchema(EventType::BUCKET_READY)),
        bucket_(std::move(bucket)) {}

  inline const std::shared_ptr<Bucket>& bucket() const {
    return bucket_;
  }

 private:
  const std::shared_ptr<Bucket> bucket_;
};

class CommDoneEvent : public Event {
 public:
  explicit CommDoneEvent(std::shared_ptr<Bucket> bucket)
      : Event(EventSchema(EventType::COMM_DONE)), bucket_(std::move(bucket)) {}

  inline const std::shared_ptr<Bucket>& bucket() const {
    return bucket_;
  }

 private:
  const std::shared_ptr<Bucket> bucket_;
};

class GlobalGradReadyEvent : public Event {
 public:
  explicit GlobalGradReadyEvent(size_t index)
      : Event(EventSchema(EventType::LOCAL_GRAD_READY)), index_(index) {}

  inline size_t index() const {
    return index_;
  }

 private:
  const size_t index_;
};

} // namespace spmd
} // namespace distributed
} // namespace torch
