#pragma once

#include <torch/csrc/python_headers.h>

// This file contains utilities used for handling PyObject preservation

namespace c10 {
class intrusive_ptr_target;
namespace impl {
struct PyObjectSlot;
} // namespace impl
} // namespace c10

namespace torch::utils {

class PyObjectPreservation {
 public:
  // Store a PyObject wrapper on a fresh c10 wrapper. The caller must hold
  // a unique reference to `target`.
  static void init_fresh_nonatomic(
      c10::intrusive_ptr_target* target,
      c10::impl::PyObjectSlot* slot,
      PyObject* pyobj);

  static PyObject* init_once(
      c10::intrusive_ptr_target* target,
      c10::impl::PyObjectSlot* slot,
      PyObject* pyobj);
};

} // namespace torch::utils
