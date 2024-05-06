#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/jit/runtime/static/impl.h>

namespace torch::jit {

// The following class facilitates code reuse between ProcessedNodeInputWrapper
// and ProcessedNodeOutputWrapper via CRTP
template <typename DerivedWrapper>
class ProcessedNodeWrapperBase {
 public:
  class ProcessedNodeWrapperBaseIter {
   public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = at::Tensor;
    using difference_type = size_t;
    using pointer = const at::Tensor*;
    using reference = const at::Tensor&;

    ProcessedNodeWrapperBaseIter() = default;

    ProcessedNodeWrapperBaseIter(
        const DerivedWrapper* container,
        size_t start_idx)
        : container_(container), idx_(start_idx) {}

    ProcessedNodeWrapperBaseIter& operator++() {
      TORCH_DCHECK_NE(idx_, container_->size());
      ++idx_;
      return *this;
    }

    ProcessedNodeWrapperBaseIter operator++(int) {
      ProcessedNodeWrapperBaseIter old = *this;
      ++(*this);
      return old;
    }

    reference operator*() const {
      TORCH_CHECK(container_ != nullptr);
      return (*container_)[idx_];
    }

    pointer operator->() const {
      TORCH_CHECK(container_ != nullptr);
      return &(*container_)[idx_];
    }

    friend bool operator==(
        ProcessedNodeWrapperBaseIter lhs,
        ProcessedNodeWrapperBaseIter rhs) {
      TORCH_DCHECK_EQ(lhs.container_, rhs.container_);
      return lhs.idx_ == rhs.idx_;
    }

    friend bool operator!=(
        ProcessedNodeWrapperBaseIter lhs,
        ProcessedNodeWrapperBaseIter rhs) {
      return !(lhs == rhs);
    }

   private:
    const DerivedWrapper* container_ = nullptr;
    size_t idx_ = 0;
  };

  // NB: to mimic the behavior of at::ArrayRef, both iterators are
  // the const version.
  using iterator = ProcessedNodeWrapperBaseIter;
  using const_iterator = ProcessedNodeWrapperBaseIter;
  using size_type = size_t;
  using value_type = at::Tensor;

  explicit ProcessedNodeWrapperBase(ProcessedNode& pnode) : pnode_(pnode) {}

  iterator begin() {
    return ProcessedNodeWrapperBaseIter(static_cast<DerivedWrapper*>(this), 0);
  }
  iterator end() {
    return ProcessedNodeWrapperBaseIter(
        static_cast<DerivedWrapper*>(this),
        static_cast<DerivedWrapper*>(this)->size());
  }

  const_iterator begin() const {
    return ProcessedNodeWrapperBaseIter(
        static_cast<const DerivedWrapper*>(this), 0);
  }
  const_iterator end() const {
    return ProcessedNodeWrapperBaseIter(
        static_cast<const DerivedWrapper*>(this),
        static_cast<const DerivedWrapper*>(this)->size());
  }

  const_iterator cbegin() const {
    return ProcessedNodeWrapperBaseIter(
        static_cast<const DerivedWrapper*>(this), 0);
  }
  const_iterator cend() const {
    return ProcessedNodeWrapperBaseIter(
        static_cast<const DerivedWrapper*>(this),
        static_cast<const DerivedWrapper*>(this)->size());
  }

  bool empty() const {
    return static_cast<const DerivedWrapper*>(this)->size() == 0;
  }

 protected:
  ProcessedNode& pnode_;
};

// A ProcessedNodeWrapperBase lets us use ProcessedNode directly in a context
// where a container of IValues is expected. This trick is handy for avoiding
// refcount bumps in perf-sensitive native ops. For example, suppose we have an
// op that takes a list of tensors as an argument and we've turned the op into a
// variadic variant in static runtime. To use the PyTorch library implementation
// of the op, we would have to pack the variadic arguments into a list:
//   std::vector<Tensor> tensor_list;
//   tensor_list.reserve(pnode->num_outputs());
//   for (const auto i : c10::irange(pnode->num_inputs())
//     tensor_list.push_back(pnode->Input(i).toTensor());
//   op_impl(tensor_list);
// Using ProcessedNodeWrapperBase, we can avoid this round of refcount bumps.
// All we need to do is turn `op_impl` into a template and pass it
// ProcessedNodeInputWrapper(*pnode)!
class ProcessedNodeInputWrapper
    : public ProcessedNodeWrapperBase<ProcessedNodeInputWrapper> {
 public:
  // The last `back_elements_ignored` elements are not considered.
  // Same for the first `front_elements_ignored` elements.
  // This is useful for ops where
  // only the first N elements are tensors (N < inputs.size()).
  // For instance, the last argument to VarStack is an integer dimension.
  explicit ProcessedNodeInputWrapper(
      ProcessedNode& pnode,
      size_t front_elements_ignored = 0,
      size_t back_elements_ignored = 1)
      : ProcessedNodeWrapperBase<ProcessedNodeInputWrapper>(pnode),
        front_elements_ignored_(front_elements_ignored),
        back_elements_ignored_(back_elements_ignored) {
    TORCH_CHECK(front_elements_ignored_ <= pnode_.num_inputs());
    TORCH_CHECK(
        back_elements_ignored_ <=
        pnode_.num_inputs() - front_elements_ignored_);
  }

  size_t size() const {
    return pnode_.num_inputs() - back_elements_ignored_ -
        front_elements_ignored_;
  }

  const at::Tensor& operator[](size_t idx) const {
    TORCH_CHECK(idx < size());
    return pnode_.Input(front_elements_ignored_ + idx).toTensor();
  }

  const at::Tensor& front() const {
    TORCH_CHECK(
        !empty(),
        "Attempted to access front() of empty ProcessedNodeInputWrapper");
    return pnode_.Input(front_elements_ignored_).toTensor();
  }

  const at::Tensor& back() const {
    TORCH_CHECK(
        !empty(),
        "Attempted to access back() of empty ProcessedNodeInputWrapper");
    return pnode_.Input(pnode_.num_inputs() - back_elements_ignored_ - 1)
        .toTensor();
  }

 private:
  size_t front_elements_ignored_;
  size_t back_elements_ignored_;
};

// Similar to ProcessedNodeInputWrapper, but wraps outputs and allows for
// writing.
class ProcessedNodeOutputWrapper
    : public ProcessedNodeWrapperBase<ProcessedNodeOutputWrapper> {
 public:
  using ProcessedNodeWrapperBase<
      ProcessedNodeOutputWrapper>::ProcessedNodeWrapperBase;

  size_t size() const {
    return pnode_.num_outputs();
  }

  at::Tensor& operator[](size_t idx) const {
    TORCH_CHECK(idx < size());
    return pnode_.Output(idx).toTensor();
  }

  at::Tensor& front() const {
    TORCH_CHECK(
        !empty(),
        "Attempted to access front() of empty ProcessedNodeOutputWrapper");
    return pnode_.Output(0).toTensor();
  }

  at::Tensor& back() const {
    TORCH_CHECK(
        !empty(),
        "Attempted to access back() of empty ProcessedNodeOutputWrapper");
    return pnode_.Output(size() - 1).toTensor();
  }
};

} // namespace torch::jit
