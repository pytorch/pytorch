#pragma once

#include <c10/util/Exception.h>

#include <mutex>
#include <vector>
#include <unordered_map>
#include <thread>

namespace torch { namespace autograd { namespace profiler {

extern std::mutex module_source_debug_info_lock;

int64_t createNextCodeId();

struct InstructionInfo {
  InstructionInfo() : code_id(-1), instruction_pc(0) {}
  InstructionInfo(int64_t id, size_t pc) :
    code_id(id), instruction_pc(pc) {}
  int64_t code_id;
  size_t instruction_pc;
};

void setInstructionInfo(int64_t code_id, size_t pc);

InstructionInfo getInstructionInfo();

// Insert instructions' debug info in DebugInfo structure.
// There is one such instance for every instance of a Graph in a process.
// Thus graph_id uniquely identifies it.
// Furthermore it assumes that source_debug info is inserted in GraphebugInfo
// when the instructions are created. Therefore if we set the source_debug info
// at the time of instruction creation we do not need instruction_pc as
// it will always be, at the point of insertion, instruction_buffer.size() - 1.
// Thus instruction buffer and GraphSourceDebugInfo size shall be equal.
void setInstructionDebugInfo(const InstructionInfo& info, std::string&& inst_debug_info);

// Accessing module_source_debug_info requires locking the mutex.
// This function accesses without locking, assuming the caller
// acquires the lock. Hence the unsafe prefix.
std::string unsafeGetInstructionDebugInfo(const InstructionInfo& info);
}}}
