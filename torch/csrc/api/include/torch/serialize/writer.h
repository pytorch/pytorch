#pragma once

#include <torch/detail/static.h>
#include <torch/tensor.h>

#include <string>
#include <vector>

namespace torch {
namespace serialize {

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                                 Writer
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Writer {
 public:
  virtual ~Writer() = default;

  virtual void write(
      const std::string& key,
      const Tensor& tensor,
      bool is_buffer = false) = 0;

  virtual void finish();

  void write(
      const std::string& key,
      const std::vector<Tensor>& tensors,
      bool is_buffer = false);

  template <
      typename Iterator,
      typename = detail::is_forward_tensor_iterator_t<Iterator>>
  void write(
      const std::string& key,
      Iterator begin,
      Iterator end,
      bool is_buffer = false);

  template <typename... Ts>
  void operator()(Ts&&... ts);
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                        Writer Implementation
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename Iterator, typename>
void Writer::write(
    const std::string& key,
    Iterator begin,
    Iterator end,
    bool is_buffer) {
  const int64_t size = std::distance(begin, end);
  write(key + ".size", torch::tensor(size));
  for (int64_t index = 0; begin != end; ++begin, ++index) {
    write(key + "." + std::to_string(index), *begin, is_buffer);
  }
}

template <typename... Ts>
void Writer::operator()(Ts&&... ts) {
  write(std::forward<Ts>(ts)...);
}

} // namespace serialize
} // namespace torch
