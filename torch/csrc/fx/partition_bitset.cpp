#include <torch/csrc/fx/partition_bitset.h>

#include <c10/util/sparse_bitset.h>
#include <torch/csrc/utils/pybind.h>

#include <limits>
#include <utility>
#include <vector>

namespace torch::fx {
namespace {

constexpr unsigned blockSize = 256;
using Element = c10::SparseBitVectorElement<blockSize>;
using BaseBits = c10::SparseBitVector<blockSize>;

// Storage is rounded up to whole blocks; only bits below size are set.
class DenseMask {
 public:
  explicit DenseMask(py::ssize_t size)
      : size_(checkedSize(size)),
        blocks_((size_ + blockSize - 1) / blockSize, fullBlock()) {
    if (size_ % blockSize != 0) {
      for (unsigned i = size_ % blockSize; i < blockSize; ++i) {
        blocks_.back().reset(i);
      }
    }
  }

  void reset(py::ssize_t index) {
    TORCH_CHECK_INDEX(
        index >= 0 && static_cast<size_t>(index) < size_,
        "Mask index out of range");
    blocks_[index / blockSize].reset(index % blockSize);
  }

  // Blocks outside the allocated storage are all zero.
  const Element& block(unsigned blockIndex) const {
    static const Element zero(0);
    return blockIndex < blocks_.size() ? blocks_[blockIndex] : zero;
  }

 private:
  static size_t checkedSize(py::ssize_t size) {
    TORCH_CHECK_VALUE(
        size >= 0 &&
            static_cast<size_t>(size) < std::numeric_limits<unsigned>::max(),
        "Mask size out of range");
    return static_cast<size_t>(size);
  }

  static Element fullBlock() {
    // Dense block positions come from the vector; Element's index is unused.
    Element mask(0);
    for (unsigned i = 0; i < blockSize; ++i) {
      mask.set(i);
    }
    return mask;
  }

  size_t size_;
  std::vector<Element> blocks_;
};

class SparseBits : public BaseBits {
 public:
  SparseBits() = default;
  explicit SparseBits(BaseBits bits) : BaseBits(std::move(bits)) {}

  bool intersects(const SparseBits& other, const DenseMask& mask) const {
    auto left = Elements.begin();
    auto right = other.Elements.begin();
    while (left != Elements.end() && right != other.Elements.end()) {
      if (left->index() < right->index()) {
        ++left;
      } else if (right->index() < left->index()) {
        ++right;
      } else {
        const auto& maskBlock = mask.block(left->index());
        for (unsigned i = 0; i < Element::BITWORDS_PER_ELEMENT; ++i) {
          if (left->word(i) & right->word(i) & maskBlock.word(i)) {
            return true;
          }
        }
        ++left;
        ++right;
      }
    }
    return false;
  }

  SparseBits masked(const DenseMask& mask) const {
    SparseBits result;
    for (const auto& element : Elements) {
      Element filtered = element;
      bool zero = false;
      filtered.intersectWith(mask.block(element.index()), zero);
      if (!zero) {
        // Appending preserves source order and keeps the fresh result's
        // cached end iterator valid.
        result.Elements.push_back(filtered);
      }
    }
    return result;
  }
};

} // namespace

void initPartitionBitsetBindings(PyObject* module) {
  auto m = py::reinterpret_borrow<py::module_>(module);
  py::class_<DenseMask>(m, "_PartitionMask")
      .def(py::init<py::ssize_t>(), py::arg("size"))
      .def("reset", &DenseMask::reset, py::arg("index"), py::pos_only());
  py::class_<SparseBits>(m, "_PartitionBitset")
      .def(py::init<>())
      .def(
          "__setitem__",
          [](SparseBits& bits, py::ssize_t index, bool value) {
            // The sparse iterator increments its last index.
            TORCH_CHECK_INDEX(
                index >= 0 &&
                    static_cast<size_t>(index) <
                        std::numeric_limits<unsigned>::max(),
                "Bitset index out of range");
            if (value) {
              bits.set(static_cast<unsigned>(index));
            } else {
              bits.reset(static_cast<unsigned>(index));
            }
          })
      .def(
          "__and__",
          [](const SparseBits& a, const SparseBits& b) {
            return SparseBits(a & b);
          },
          py::is_operator())
      .def(
          "__or__",
          [](const SparseBits& a, const SparseBits& b) {
            return SparseBits(a | b);
          },
          py::is_operator())
      .def(
          "__ior__",
          [](SparseBits& a, const SparseBits& b) -> SparseBits& {
            a |= b;
            return a;
          },
          py::return_value_policy::reference_internal,
          py::is_operator())
      .def("copy", [](const SparseBits& bits) { return SparseBits(bits); })
      .def(
          "difference",
          [](const SparseBits& a, const SparseBits& b) {
            return SparseBits(a - b);
          },
          py::arg("other"),
          py::pos_only())
      .def("empty", &SparseBits::empty)
      .def(
          "intersects",
          &SparseBits::intersects,
          py::arg("other"),
          py::arg("mask"),
          py::pos_only())
      .def("masked", &SparseBits::masked, py::arg("mask"), py::pos_only())
      .def("set_bits", [](const SparseBits& bits) {
        py::list result(bits.count());
        size_t position = 0;
        for (unsigned index : bits) {
          result[position++] = py::int_(index);
        }
        return result;
      });
}

} // namespace torch::fx
