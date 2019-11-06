#pragma once

#include "pickler.h"

namespace torch {
namespace jit {

// [unpickler refactor] there is some cruft around PickleOpCode::BUILD,
// PickleOpCode::NEWOBJ, and the last_opcode_ member below that should be deleted at
// some point, the Pickler doesn't produce it and it's only around to support
// models saved before 1.1
class Unpickler {
  TH_DISALLOW_COPY_AND_ASSIGN(Unpickler);

 public:
  // tensors inside the pickle are references to the tensor_table
  Unpickler(
      std::function<bool(char*, size_t)> reader,
      ClassResolver class_resolver,
      const std::vector<at::Tensor>* tensor_table)
      : reader_(reader),
        tensor_table_(tensor_table),
        class_resolver_(std::move(class_resolver)) {}

  // tensors inside the pickle contain meta-data, the raw tensor
  // dead is retrieved by calling `read_record`.
  Unpickler(
      std::function<bool(char*, size_t)> reader,
      ClassResolver class_resolver,
      std::function<at::DataPtr(const std::string&)> read_record,
      c10::optional<at::Device> device)
      : reader_(reader),
        tensor_table_(nullptr),
        class_resolver_(std::move(class_resolver)),
        read_record_(std::move(read_record)),
        device_(std::move(device)) {}

  IValue parse_ivalue();

 private:
  // No arguments ensures that a template arugment must be specified
  // so that the number of bytes read / type read is explicit
  template <typename T>
  T read() {
    T item;
    if (!reader_(reinterpret_cast<char*>(&item), sizeof(item))) {
      AT_ERROR("Unexpected end of pickler archive.");
    }
    return item;
  }

  std::string readBytes(size_t num_bytes);

  double readFloat();
  PickleOpCode readInstruction();
  PickleOpCode readOpCode();
  std::string readString();
  void readList(IValue list_ivalue);
  void setInput(size_t memo_id);
  void run();

  // Returns a pointer to the number of bytes requested. This should state-fully
  // remember how many bytes have been read
  std::function<bool(char*, size_t)> reader_;

  std::vector<IValue> stack_;

  // globals are represented on the stack as IValue integer indices
  // into this list
  std::vector<std::function<void(void)>> globals_;
  std::vector<IValue> memo_table_;
  std::vector<size_t> marks_;
  const std::vector<at::Tensor>* tensor_table_;

  // optionally nullptr, needs to be present for creating classes
  ClassResolver class_resolver_;
  IValue empty_tuple_;

  std::function<at::DataPtr(const std::string&)> read_record_;
  c10::optional<at::Device> device_;
};

} // namespace jit
} // namespace torch
