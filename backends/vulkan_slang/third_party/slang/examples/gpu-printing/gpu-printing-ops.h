// gpu-printing-op.h

// This file defines the various opcodes that
// will be used for GPU printing commands.
//
// Because the CPU will be doing printing on
// behalf of the GPU, the two processors need
// to agree on the values of these opcodes.
// Therefore we have set up this file to be
// included into both the C++ `gpu-printing.cpp`
// implementation and the Slang `printing.slang`
// file.
//
// Client code should defiine the `GPU_PRINTING_OP`
// macro appropriately, before including this file.
//
#ifndef GPU_PRINTING_OP
#error "Must define 'GPU_PRINTING_OP(NAME)' before including"
#endif

// The `Nop` opcode is used to represent a vacuous
// printing command that does nothing.
//
// It's main purpose is to allow GPU code to zero
// out parts of the printing buffer to disable
// or shorten a printing command that was started.
//
GPU_PRINTING_OP(Nop)

// The `NewLine` command is a compact way to
// print a newline character (`\n`)
GPU_PRINTING_OP(NewLine)

// Simple value types like `int`, `uint`, and `float`
// can have their own printing commands for when
// they will be printed directly.
//
GPU_PRINTING_OP(Int32)
GPU_PRINTING_OP(UInt32)
GPU_PRINTING_OP(Float32)

// String values are encoded in the print buffer as
// a 32-bit hash code, and are thus similar to
// the simple value cases in practice.
//
GPU_PRINTING_OP(String)

// The final opcode we define is a complex `printf()`
// style operation that combines a format string with
// a variable amount of argument data to be referenced
// by that format string.
//
GPU_PRINTING_OP(PrintF)

#undef GPU_PRINTING_OP
