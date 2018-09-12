#pragma once

#include <torch/detail/static.h>
#include <torch/tensor.h>

#include <string>
#include <vector>

namespace torch {
namespace serialize {

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                                 Reader
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Reader {
 public:
  virtual ~Reader() = default;

  virtual void read(
      const std::string& key,
      Tensor& tensor,
      bool is_buffer = false) = 0;

  virtual void finish();

  void read(
      const std::string& key,
      std::vector<Tensor>& tensors,
      bool is_buffer = false);

  template <
      typename OutputIterator,
      typename = detail::is_output_iterator_t<OutputIterator>>
  void read(
      const std::string& key,
      OutputIterator output,
      bool is_buffer = false);

  template <typename... Ts>
  void operator()(Ts&&... ts);
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                        Reader Implementation
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename OutputIterator, typename>
void Reader::read(
    const std::string& key,
    OutputIterator output,
    bool is_buffer) {
  torch::Tensor size;
  read(key + ".size", size);
  for (int64_t index = 0; index < size.toCLong(); ++index) {
    torch::Tensor tensor;
    read(key + "." + std::to_string(index), tensor, is_buffer);
    *output = tensor;
  }
}

template <typename... Ts>
void Reader::operator()(Ts&&... ts) {
  read(std::forward<Ts>(ts)...);
}

} // namespace serialize
} // namespace torch
