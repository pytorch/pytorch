#pragma once
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <functional>
#include <iostream>
#include <ATen/core/stack.h>

#include <stdexcept>

namespace torch {
namespace jit {
struct TORCH_API DebuggerHookException : public std::exception {};


template <typename Impl, typename ActiveFrame>
void TORCH_API run_debugger(Stack& stack, Impl* state, ActiveFrame* af) {
  std::string command;
  const auto& frames = state->frames;
  do {
    std::cout << "(tdb) ";
    std::cin >> command;
    if (command == "list") {
      //   size_t old_pc = frames.back().pc;
      //   frames.back().pc = af.pc;
      //   state->formatStackTrace(std::cout);
      //   frames.back().pc = old_pc;
    } else if (command == "dump") {
      state->dump(std::cout, stack);
    } else if (command == "err") {
      throw std::runtime_error("meow");
    } else if (command == "continue" || command == "c") {
      break;
    } else if (command == "info") {
      std::cout << "Have " << frames.size() << " frames\n";
    } else if (command == "graph") {
      const auto& frame = frames.back();
      std::cout << *frame.function->graph_ << "\n";
    } else if (command == "frame") {
      //   const auto& frame = frames.back();
      //   // std::cout << *frame.function->graph_ << "\n";
      //   std::cout << "Registers:\n";
      //   for (size_t i = 0; i < registers.size(); i++) {
      //     std::cout << "\t" << i << " " << registers.at(i) << "\n";
      //   }
      //   std::cout << "Values:\n";
      //   for (auto entry : frame.function->value_to_reg_) {
      //     auto v = entry.first;
      //     auto use_entry = frame.function->use_count_.find(v);
      //     if (use_entry != frame.function->use_count_.end()) {
      //       std::cout << use_entry->second << ", " << v->uses().size() << ":
      //       "; if (use_entry->second == v->uses().size()) {
      //         std::cout << "(dead) ";
      //       } else {
      //         std::cout << "(live) ";
      //       }
      //     } else {
      //     }
      //     std::cout << entry.second << " " << entry.first->debugName() << " "
      //               << *entry.first->node();
      //   }
    } else if (command == "read") {
      //   const auto& frame = frames.back();
      //   std::cout << "value name: ";
      //   std::string name;
      //   std::cin >> name;
      //   bool done = false;
      //   for (auto entry : frame.function->value_to_reg_) {
      //     if (name == entry.first->debugName()) {
      //       std::cout << name << ":\n";
      //       std::cout << "\t" << *entry.first->node();
      //       size_t idx = registers.size() - entry.second;
      //       std::cout << "\t" << registers.at(idx) << "\n";
      //       // std::cout << "\t" << reg(entry.second - 1) << "\n";
      //       done = true;
      //       break;
      //     }
      //   }
      //   if (!done) {
      //     std::cout << "unknown value debugName: " << name << "\n";
      //   }
    } else if (command == "q" || command == "quit") {
      break;
    } else {
      std::cout << "unknown command: " << command << "\n";
    }
    std::cout << state->frames.size() << "\n";
  } while (true);
}

} // namespace jit
} // namespace torch
