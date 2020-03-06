#pragma once

#include <ATen/ATen.h>

namespace at {

// TODO: use a real key
constexpr auto BatchTensorKey = DispatchKey::TESTING_ONLY_GenericWrapperTensorId;

struct BatchTensorImpl : public c10::TensorImpl {
  explicit BatchTensorImpl(
      at::Tensor rep,
      optional<int64_t> batch_dim,
      int64_t level)
    : TensorImpl(
        c10::DispatchKeySet(BatchTensorKey),
        rep.dtype(),
        rep.device()
      )
    , rep_(std::move(rep))
    , batch_dim_(batch_dim)
    , level_(level)
    {
      TORCH_INTERNAL_ASSERT(!batch_dim.has_value() || *batch_dim >= 0);
      TORCH_INTERNAL_ASSERT(level >= 0);

      // TODO: Ugh, are there operations that change the size in-place?
      if (rep_.defined()) {
        owned_sizes_ = rep_.sizes().vec();
        if (batch_dim_) {
          owned_sizes_.erase(owned_sizes_.begin() + *batch_dim_);
        }
      }
    }

  IntArrayRef sizes() const {
    return owned_sizes_;
  }

  // TODO: Do we have to inherit this??
  int64_t dim() const {
    return owned_sizes_.size();
  }

  optional<int64_t> batch_size() const {
    if (!batch_dim_) return nullopt;
    return rep_.sizes()[*batch_dim_];
  }

  at::Tensor rep_;
  optional<int64_t> batch_dim_;
  int64_t level_;

  std::vector<int64_t> owned_sizes_;
  // TODO: Doesn't TensorImpl have 10000 fields that we don't want to inherit here?
};

inline Tensor makeBatched(Tensor tensor, optional<int64_t> batch_dim, int64_t level) {
  return at::detail::make_tensor<BatchTensorImpl>(tensor, batch_dim, level);
}

inline BatchTensorImpl* getBatched(Tensor tensor) {
  return static_cast<BatchTensorImpl*>(tensor.unsafeGetTensorImpl());
}

inline Tensor unwrapBatched(Tensor tensor, int64_t ntimes=1) {
  for (auto time = 0; time < ntimes; time++) {
    tensor = getBatched(tensor)->rep_;
  }
  return tensor;
}

inline bool isBatched(Tensor tensor) {
  return tensor.unsafeGetTensorImpl()->key_set().has(BatchTensorKey);
}

inline std::ostream& operator<<(std::ostream& out, const BatchTensorImpl& batch_tensor) {
  int64_t batch_dim = -1;
  if (batch_tensor.batch_dim_.has_value()) {
    batch_dim = batch_tensor.batch_dim_.value();
  }
  // TODO: this prints out really bizarrely
  out << "BatchTensor[lvl" << batch_tensor.level_ << "/bdim" << batch_dim << "]"
      << batch_tensor.sizes();
  return out;
}

}
