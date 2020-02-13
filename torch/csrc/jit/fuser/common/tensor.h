#pragma once

#include <torch/csrc/jit/fuser/common/ir.h>
#include <torch/csrc/jit/fuser/common/tensor_meta.h>

namespace torch {
namespace jit {
namespace fuser {

struct TORCH_API IterDomain : public Val {
  ~IterDomain() = default;

  IterDomain() = delete;

  IterDomain(
      const Int* _size,
      ParallelType _parallel_method = ParallelType::Serial,
      bool _reduction_domain = false)
      : Val(ValType::IterDomain, DataType::Int),
        size_(_size),
        parallel_method_(_parallel_method),
        reduction_domain_(_reduction_domain) {}

  IterDomain(
      const Val* int_size,
      ParallelType _parallel_method = ParallelType::Serial,
      bool _reduction_domain = false)
      : Val(ValType::IterDomain, DataType::Int),
        size_(static_cast<const Int*>(int_size)),
        parallel_method_(_parallel_method),
        reduction_domain_(_reduction_domain) {
    assert(int_size->isVal());
    assert(int_size->getDataType() == DataType::Int);
  }

  bool isReduction() const noexcept {
    return reduction_domain_;
  }
  ParallelType parallel_method() const noexcept {
    return parallel_method_;
  }
  const Int* size() const noexcept {
    return size_;
  }

  IterDomain(const IterDomain& other) = delete;
  IterDomain& operator=(const IterDomain& other) = delete;

  IterDomain(IterDomain&& other) = delete;
  IterDomain& operator=(IterDomain&& other) = delete;

 private:
  const Int* size_;
  const ParallelType parallel_method_;
  const bool reduction_domain_;
};

struct TORCH_API TensorDomain : public Val {
  ~TensorDomain() = default;

  TensorDomain(const TensorDomain& other) = delete;
  TensorDomain& operator=(const TensorDomain& other) = delete;

  TensorDomain(TensorDomain&& other) = delete;
  TensorDomain& operator=(TensorDomain&& other) = delete;

  TensorDomain(std::vector<const IterDomain*> domain_)
      : Val(ValType::TensorDomain), domain(domain_) {}

  std::vector<const IterDomain*>::size_type size() const {
    return domain.size();
  }
  const IterDomain* axis(std::vector<const IterDomain*>::size_type i) const {
    assert(i >= 0 && i < size());
    return domain[i];
  }

 private:
  const std::vector<const IterDomain*> domain;
};

struct TORCH_API Tensor : public Val {
  ~Tensor() = default;

  Tensor() = delete; // Don't ever want a default constructor, Vals are unique
                     // and immutable.

  Tensor(DataType dt, const TensorDomain* _td = nullptr)
      : Val(ValType::Tensor, dt), contiguity_(c10::nullopt), domain(_td) {}

  Tensor(const Tensor& other) = delete;
  Tensor& operator=(const Tensor& other) = delete;

  Tensor(Tensor&& other) = delete;
  Tensor& operator=(Tensor&& other) = delete;

  Tensor(const std::shared_ptr<c10::TensorType>& tensor_type);

  Tensor(const std::shared_ptr<Value>& jit_value);
  
  bool hasContiguityInfo() const;

  const c10::optional<TensorContiguity>& getContiguityInfo() const;

  static const Tensor* MakeDummyTensor(int ndims) {
    std::vector<const IterDomain*> sizes;
    for (int i = 0; i < ndims; i++) {
      sizes.push_back(new IterDomain(new Int()));
    }
    TensorDomain* td = new TensorDomain(sizes);

    return new Tensor(DataType::Float, td);
  }

  // protected:

  // Implementation details:
  const c10::optional<TensorContiguity> contiguity_;
  const TensorDomain* domain;
};

struct TORCH_API TensorView : public Val {
  ~TensorView() = default;

  TensorView(const TensorView& other) = delete;
  TensorView& operator=(const TensorView& other) = delete;

  TensorView(TensorView&& other) = delete;
  TensorView& operator=(TensorView&& other) = delete;

  TensorView(const Tensor* _tensor, const TensorDomain* _view)
      : Val(ValType::TensorView), tensor(_tensor), view(_view) {}

  const Tensor* tensor;
  const TensorDomain* view;
};

/*
 * Split an axis, by factor factor
 * TODO: Implement split by nparts
 */
struct TORCH_API Split : public Expr {
  ~Split() = default;
  Split(
      const TensorDomain* _out,
      const TensorDomain* _in,
      int _axis,
      const Int* _factor);

  const Val* out() const noexcept {
    return out_;
  }
  const Val* in() const noexcept {
    return in_;
  }
  int axis() const noexcept {
    return axis_;
  }
  const Val* factor() const noexcept {
    return factor_;
  }

  Split(const Split& other) = delete;
  Split& operator=(const Split& other) = delete;

  Split(Split&& other) = delete;
  Split& operator=(Split&& other) = delete;

 private:
  const TensorDomain* out_;
  const TensorDomain* in_;
  const int axis_;
  const Int* factor_;
};

/*
 * Merge axis _axis with the following axis. Both axis must be of the same
 * iter or reduction axis, as well as the same parallelization strategy if
 * there is one.
 * TODO: Should this be a unary op type?
 */
struct TORCH_API Merge : public Expr {
  ~Merge() = default;
  Merge(const TensorDomain* _out, const TensorDomain* _in, int _axis);

  Merge(const Merge& other) = delete;
  Merge& operator=(const Merge& other) = delete;

  Merge(Merge&& other) = delete;
  Merge& operator=(Merge&& other) = delete;

  const Val* out() const noexcept {
    return out_;
  }
  const Val* in() const noexcept {
    return in_;
  }
  int axis() const noexcept {
    return axis_;
  }

 private:
  const TensorDomain* out_;
  const TensorDomain* in_;
  const int axis_;
};

/*
 * Reorder axis of a tensor domain with the map
 * pos2axis[new_position] = old_position
 */
struct TORCH_API Reorder : public Expr {
  ~Reorder() = default;
  Reorder(
      const TensorDomain* _out,
      const TensorDomain* _in,
      std::vector<int> _pos2axis);

  Reorder(const Reorder& other) = delete;
  Reorder& operator=(const Reorder& other) = delete;

  Reorder(Reorder&& other) = delete;
  Reorder& operator=(Reorder&& other) = delete;

  const Val* out() const noexcept {
    return out_;
  }
  const Val* in() const noexcept {
    return in_;
  }
  const std::vector<int> pos2axis() const noexcept {
    return pos2axis_;
  }

 private:
  const TensorDomain* out_;
  const TensorDomain* in_;
  const std::vector<int> pos2axis_;
};

TORCH_API const TensorView* split(const Tensor*, int axis, int factor);
TORCH_API const TensorView* merge(const Tensor*, int axis);

} // namespace fuser
} // namespace jit
} // namespace torch
