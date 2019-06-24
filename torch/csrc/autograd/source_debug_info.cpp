#include  <torch/csrc/autograd/source_debug_info.h>

namespace torch { namespace autograd { namespace profiler {

std::mutex module_source_debug_info_lock;

thread_local InstructionInfo inst_info;

void setInstructionInfo(int64_t code_id, size_t pc) {
  inst_info = InstructionInfo(code_id, pc);
}

InstructionInfo getInstructionInfo() {
  return inst_info;
}

// For nodes of JIT graph, InstructionDebufInfo stores instruction/node's
// debug information. Since instructions are serially
// executed from instruction buffer (each node is an instruction)
// we can use a vector and index into it using instruction's pc,
// which should be unique for each instruction of a graph.
using InstructionDebugInfo = std::vector<std::string>;
// Maps from thread_id to a scope of instructions for interpreter running
// on that thread.
std::vector<InstructionDebugInfo> module_source_debug_info;

int64_t createNextCodeId() {
  std::unique_lock<std::mutex> lock(module_source_debug_info_lock);
  module_source_debug_info.emplace_back(InstructionDebugInfo());
  return module_source_debug_info.size() - 1;
}

void setInstructionDebugInfo(const InstructionInfo& info , std::string&& inst_debug_info) {
  const int64_t code_id = info.code_id;
  if (code_id < 0 || code_id >= module_source_debug_info.size()) {
    TORCH_WARN("Found wrong code_id, found:", code_id);
    return;
  }
  std::unique_lock<std::mutex> lock(module_source_debug_info_lock);
  module_source_debug_info[code_id].emplace_back(inst_debug_info);
}

std::string unsafeGetInstructionDebugInfo(const InstructionInfo& info) {
  int64_t code_id = info.code_id;
  size_t inst_pc = info.instruction_pc;
  if (code_id < 0 || code_id >= module_source_debug_info.size()) {
    TORCH_WARN("Found wrong code_id, found:", code_id);
    return "";
  }
  if (inst_pc < 0 || inst_pc >= module_source_debug_info[code_id].size()) {
    TORCH_WARN("Instruction PC out of range, found:", inst_pc);
    return "";
  }
  return module_source_debug_info[code_id][inst_pc];
}

}}}
