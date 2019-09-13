#pragma once

#include <string>
#include <vector>

#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <c10/util/ArrayRef.h>
#include <torch/csrc/utils/disallow_copy.h>

#include "pickler.h"

namespace torch {
namespace jit {

using ObjCallback =
    std::function<IValue(const c10::QualifiedName&, IValue)>;

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
      ObjCallback obj_callback,
      const std::vector<at::Tensor>* tensor_table)
      : reader_(reader),
        tensor_table_(tensor_table),
        obj_callback_(obj_callback) {}

  // tensors inside the pickle contain meta-data, the raw tensor
  // dead is retrieved by calling `read_record`.
  Unpickler(
      std::function<bool(char*, size_t)> reader,
      ObjCallback obj_callback,
      std::function<at::DataPtr(const std::string&)> read_record,
      c10::optional<at::Device> device,
      bool data_only = false)
      : reader_(reader),
        tensor_table_(nullptr),
        obj_callback_(obj_callback),
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
  void readList();
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
  ObjCallback obj_callback_;
  IValue empty_tuple_;

  std::function<at::DataPtr(const std::string&)> read_record_;
  c10::optional<at::Device> device_;
};

} // namespace jit
} // namespace torch
