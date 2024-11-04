#pragma once

#include <cstddef>
#include <cstdint>

#include <memory>

#include <c10/macros/Macros.h>
#include <c10/util/Logging.h>

/**
 * Packed representation of input indices for ProcessedNode.
 */
class ProcessedNodeInputs {
 private:
  // This keeps the size usage for inputs + outputs down to 16 bytes;
  // we use 12 bytes, and then two 2-byte integers are used to store
  // the outputs.
  static constexpr size_t kMaxInlineInputs = 5;

 public:
  ProcessedNodeInputs() : ProcessedNodeInputs(0) {}

  explicit ProcessedNodeInputs(size_t size) {
    TORCH_DCHECK_LT(size, (1 << 16));
    if (size <= kMaxInlineInputs) {
      repr_.inline_repr_.size = size;
    } else {
      new (&repr_.outline_repr_) HeapArrayPtr(size);
    }
  }

  uint16_t operator[](uint16_t idx) const {
    return (*const_cast<ProcessedNodeInputs*>(this))[idx];
  }

  uint16_t& operator[](uint16_t idx) {
    if (C10_LIKELY(repr_.is_inline())) {
      TORCH_DCHECK_LT(idx, repr_.inline_repr_.size);
      return repr_.inline_repr_.inputs[idx];
    } else {
      return repr_.outline_repr_[idx];
    }
  }

  C10_NODISCARD uint16_t size() const {
    if (C10_LIKELY(repr_.is_inline())) {
      return repr_.inline_repr_.size;
    } else {
      return repr_.outline_repr_.size();
    }
  }

  C10_NODISCARD bool empty() const {
    return size() == 0;
  }

 private:
  class HeapArrayPtr {
   public:
    HeapArrayPtr() = default;
    ~HeapArrayPtr() = default;

    explicit HeapArrayPtr(uint16_t size) : array_(alloc(size)) {}

    HeapArrayPtr(const HeapArrayPtr& rhs) : array_(alloc(rhs.size())) {
      if (rhs.array_) {
        std::memcpy(
            array_.get(),
            rhs.array_.get(),
            (rhs.size() + 1) * sizeof(uint16_t));
      }
    }

    HeapArrayPtr& operator=(const HeapArrayPtr& rhs) {
      if (&rhs == this) {
        return *this;
      }

      if (size() != rhs.size()) {
        array_ = alloc(rhs.size());
      }

      if (rhs.array_) {
        std::memcpy(
            array_.get(),
            rhs.array_.get(),
            (rhs.size() + 1) * sizeof(uint16_t));
      }
      return *this;
    }

    HeapArrayPtr(HeapArrayPtr&&) noexcept = default;
    HeapArrayPtr& operator=(HeapArrayPtr&&) noexcept = default;

    C10_NODISCARD bool empty() const {
      return size() != 0;
    }

    C10_NODISCARD uint16_t size() const {
      return array_ ? array_[0] : 0;
    }

    uint16_t operator[](uint16_t idx) const {
      TORCH_DCHECK_LT(idx, size());
      return array_[idx + 1];
    }

    uint16_t& operator[](uint16_t idx) {
      TORCH_DCHECK_LT(idx, size());
      return array_[idx + 1];
    }

   private:
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays)
    std::unique_ptr<uint16_t[]> array_;

    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays)
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    static std::unique_ptr<uint16_t[]> alloc(uint16_t num_elts) {
      if (num_elts) {
        auto result = std::make_unique<uint16_t[]>(num_elts + 1);
        result[0] = num_elts;
        return result;
      } else {
        return nullptr;
      }
    }
  };

  // We want ProcessedNode to be able to pack two more `uint16_t`
  // fields after its ProcessedNodeInputs, and we'll end up being
  // aligned to an 8-byte boundary anyway. We could avoid this pragma
  // at the cost of having to move ProcessedNode::outputs_offset_ and
  // ProcessedNode::num_outputs_ into this class, which would be
  // awkward.
#pragma pack(push, 2)
  union Repr {
    C10_NODISCARD bool is_inline() const {
      uint8_t tag = 0;
      // Use of reinterpret_cast to pointer to char or unsigned char
      // is defined behavior; see
      // https://en.cppreference.com/w/cpp/language/reinterpret_cast .
      std::memcpy(&tag, reinterpret_cast<const uint8_t*>(this), 1);
      // HeapArrayPtr will be represented as a plain old pointer,
      // which will have alignment to at least a 2-byte boundary
      // (because it's uint16_t*) and more likely an 8- or 16-byte
      // boundary because malloc will tend to just align everything to
      // one of those. So, we just set tag to 1 when inline_repr_ is
      // active so as to be able to differentiate the two.
      return (tag & 1) != 0;
    }

    // NOLINTNEXTLINE(modernize-use-equals-default)
    Repr() {}

    ~Repr() {
      destroyIfOutline();
    }

    Repr(const Repr& rhs) {
      if (rhs.is_inline()) {
        std::memcpy(&inline_repr_, &rhs.inline_repr_, sizeof(inline_repr_));
      } else {
        new (&outline_repr_) OutlineRepr(rhs.outline_repr_);
      }
    }

    Repr& operator=(const Repr& rhs) {
      if (&rhs == this) {
        return *this;
      }
      if (rhs.is_inline()) {
        destroyIfOutline();
        new (&inline_repr_) InlineRepr();
        std::memcpy(&inline_repr_, &rhs.inline_repr_, sizeof(inline_repr_));
      } else {
        if (is_inline()) {
          new (&outline_repr_) OutlineRepr(rhs.outline_repr_);
        } else {
          outline_repr_ = rhs.outline_repr_;
        }
      }
      return *this;
    }

    Repr(Repr&& rhs) noexcept {
      if (rhs.is_inline()) {
        std::memcpy(&inline_repr_, &rhs.inline_repr_, sizeof(inline_repr_));
      } else {
        new (&outline_repr_) OutlineRepr(std::move(rhs.outline_repr_));
      }
    }

    Repr& operator=(Repr&& rhs) noexcept {
      if (&rhs == this) {
        return *this;
      }

      if (rhs.is_inline()) {
        destroyIfOutline();
        new (&inline_repr_) InlineRepr();
        std::memcpy(&inline_repr_, &rhs.inline_repr_, sizeof(inline_repr_));
      } else {
        if (is_inline()) {
          new (&outline_repr_) OutlineRepr(std::move(rhs.outline_repr_));
        } else {
          outline_repr_ = std::move(rhs.outline_repr_);
        }
      }

      return *this;
    }

    struct InlineRepr {
      uint8_t tag = 0x1;
      uint8_t size{};
      uint16_t inputs[kMaxInlineInputs]{};
    };

    using OutlineRepr = HeapArrayPtr;

    InlineRepr inline_repr_{};
    OutlineRepr outline_repr_;

   private:
    void destroyIfOutline() {
      if (!is_inline()) {
        outline_repr_.~OutlineRepr();
      }
    }
  } repr_;
#pragma pack(pop)
};

static_assert(
    sizeof(ProcessedNodeInputs) == 12,
    "ProcessedNodeInputs has the wrong size!");
