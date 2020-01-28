#pragma once
#include <ATen/core/stack.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir.h>
#include <functional>
#include <iostream>

#include <stdexcept>

namespace torch {
namespace jit {
struct TORCH_API DebuggerHookException : public std::exception {};

template <typename Frame>
c10::optional<std::pair<Value*, IValue>> read(
    const std::string& debugName,
    std::vector<IValue>& registers,
    const Frame& frame) {
  for (auto entry : frame.function->value_to_reg_) {
    if (debugName == entry.first->debugName()) {
      size_t idx = registers.size() - entry.second;
      return std::make_pair(entry.first, registers.at(idx));
    }
  }
  return c10::nullopt;
}

template <typename Impl, typename ActiveFrame>
void TORCH_API run_debugger(Stack& stack, Impl* state, ActiveFrame* af) {
  std::string command;
  auto& frames = state->frames;
  auto& registers = state->registers;
  do {
    std::cout << "(tdb) ";
    std::cin >> command;
    if (command == "list") {
      size_t old_pc = frames.back().pc;
      frames.back().pc = af->pc;
      state->formatStackTrace(std::cout);
      frames.back().pc = old_pc;
    } else if (command == "dump") {
      state->dump(std::cout, stack);
    } else if (command == "err") {
      throw std::runtime_error(
          "Fake error created from the TorchScript debugger");
    } else if (command == "continue" || command == "c") {
      break;
    } else if (command == "info") {
      std::cout << "Have " << frames.size() << " frames\n";
    } else if (command == "graph") {
      const auto& frame = frames.back();
      std::cout << *frame.function->graph_ << "\n";
    } else if (command == "frame") {
      // Dump some info (registers and value) about this current frame
      const auto& frame = frames.back();
      std::cout << "Registers:\n";
      for (size_t i = 0; i < registers.size(); i++) {
        std::cout << "\t" << i << " " << registers.at(i) << "\n";
      }
      std::cout << "Values:\n";
      for (auto entry : frame.function->value_to_reg_) {
        auto v = entry.first;

        std::cout << entry.first->debugName() << "\n";
        std::cout << "\tsource node: " << *entry.first->node();
        std::cout << "\ttotal uses: " << v->uses().size() << "\n";

        auto use_entry = frame.function->use_count_.find(v);
        if (use_entry != frame.function->use_count_.end()) {
          std::cout << "\tuse count: " << use_entry->second << "\n";
        } else {
          std::cout << "\tuse count: <none>\n";
        }
        auto maybe_value = read(v->debugName(), state->registers, frames.back());
        if (maybe_value) {
          std::cout << "\tvalue: " << maybe_value.value().second << "\n";
        }
      }
    } else if (command == "read") {
      // Find a value that is in a register and get its value. Value must be
      // created before the breakpoint() and used after it, or the value
      // read (and the source node) will be incorrect.
      const auto& frame = frames.back();
      std::string name;
      std::cin >> name;
      auto maybe_value = read(name, state->registers, frames.back());
      if (maybe_value) {
        std::cout << "\tsource node: " << *maybe_value.value().first->node();
        std::cout << "\tvalue: " << maybe_value.value().second << "\n";
      } else {
        std::cout << "unknown value debugName: " << name << "\n";
      }
    } else if (command == "q" || command == "quit") {
      break;
    } else {
      std::cout << "unknown command: " << command << "\n";
    }
  } while (true);
}

} // namespace jit
} // namespace torch
