#pragma once

#include <ATen/core/interned_strings.h>
#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/pickler.h>


namespace torch {
namespace distributed {
namespace rpc {

using torch::jit::IValue;
using torch::jit::Operator;
using torch::jit::Pickler;
using torch::jit::Unpickler;
using torch::jit::Symbol;
using torch::jit::Stack;


template <typename B, typename D>
std::unique_ptr<D> static_unique_ptr_cast(std::unique_ptr<B> base) {
  if (!base) {
      return std::unique_ptr<D>();
  } else {
    D* derived = static_cast<D*>(base.get());
    base.release();
    return std::move(std::unique_ptr<D>(derived));
  }
}

}
}
}
