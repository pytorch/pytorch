#include <c10/util/string_view.h>
#include <bits/stdint-intn.h>
#include <cstddef>
#include <string>
#include <torch/custom_class.h>

namespace torch {
namespace jit {

class basic_string_view final  : public CustomClassHolder {
    public:
    using value_type = char;
    using pointer = char*;
    using const_pointer = const char*;
    using reference = char&;
    using const_reference = const char&;
    using const_iterator = const char*;
    using iterator = const_iterator;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    basic_string_view(const char* str = "", int64_t count = 0)
        : begin_(str), size_(count) {}

    constexpr const_pointer data() const noexcept {
        return begin_;
    }

    constexpr int64_t size() const noexcept {
        return size_;
    }

    constexpr int64_t length() const noexcept {
        return size();
    }

    private:
        const_pointer begin_;
        int64_t size_;
    };

    TORCH_LIBRARY(cuda, m) {
    auto stringview_class = m.class_<torch::jit::basic_string_view>("string_view").def(torch::init<char, int64_t>());

    stringview_class.def("data", &basic_string_view::data)
        .def("size", &basic_string_view::size)
        .def("length", &basic_string_view::length);
    };
} // namespace jit
} // namespace torch
