#ifndef SLANG_VM_BYTE_CODE_H
#define SLANG_VM_BYTE_CODE_H

#include "core/slang-basic.h"
#include "core/slang-riff.h"
#include "slang-com-ptr.h"

namespace Slang
{

/*
Slang ByteCode Module File Format

# Header
    - (4 bytes) FourCC: 'S', 'V', 'M', 'C' (kSlangByteCodeFourCC)
    - (4 bytes uint) Version: 100

# Function Section
    - (4 bytes) FourCC: 'S', 'V', 'F', 'N' (kSlangByteCodeFunctionsFourCC)
    - (4 bytes uint) Function Count: number of functions in the module
    - (uint array) Function Offsets:
      array of "Function Count" 32-bit uints, storing byte offsets from
      start of file for each function.

## Function i:
    - (32 bytes `VMFuncHeader`) Function metadata, describing the name and other
      info needed for execution. VMFuncHeader::name is a `VMOperand` whose
      sectionId = kSlangByteCodeSectionConstants, pointing to the constant section for the
      function name.
    - (uint32 * parameterCount array): array of "parameterCount" 32-bit uints, storing byte offsets
      from start of the function's working set for each parameter.
    - (byte array) Code: array of `header.codeSize` bytes, containing the instruction stream for the
        function. Each instruction starts with a `VMInstHeader`, followed by
        `VMInstHeader::operandCount` `VMOperand` structs.

# Kernel Blob Section: binary data for the kernel blob
    - (4 bytes) FourCC: 'S', 'V', 'K', 'N' (kSlangByteCodeKernelBlobFourCC)
    - (4 bytes uint) Kernel Blob Size: size of the kernel blob in bytes.
    - (byte array) Kernel Blob: array of "Kernel Blob Size" bytes, containing the kernel blob data.

# Constants Section
    - (4 bytes) FourCC: 'S', 'V', 'C', 'S' (kSlangByteCodeConstantsFourCC)
    - (4 bytes uint) Constant Count: number of constants in the module.
    - (4 bytes uint) String Count: number of string literals in the constant section.
    - (uint array) String offsets: array of "String Count" 32-bit uints, storing byte offsets from
      start of the constant array blob (next item) for each string literal.
    - (uint array) Constants:
      array of "Constant Count" 32-bit uints, storing byte offsets from
      start of file for each constant.
*/

static const int kSlangByteCodeVersion = 100;

static const uint32_t kSlangByteCodeFourCC = SLANG_FOUR_CC('S', 'V', 'M', 'C');
static const uint32_t kSlangByteCodeFunctionsFourCC = SLANG_FOUR_CC('S', 'V', 'F', 'N');
static const uint32_t kSlangByteCodeKernelBlobFourCC = SLANG_FOUR_CC('S', 'V', 'K', 'N');
static const uint32_t kSlangByteCodeConstantsFourCC = SLANG_FOUR_CC('S', 'V', 'C', 'S');

static const int kSlangByteCodeSectionWorkingSet = 0;
static const int kSlangByteCodeSectionConstants = 1;
static const int kSlangByteCodeSectionInsts = 2;
static const int kSlangByteCodeSectionImmediate = 3;
static const int kSlangByteCodeSectionFuncs = 4;
static const int kSlangByteCodeSectionStrings = 5;


enum class VMOp : uint32_t
{
    Nop,
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Neg,
    And,
    Or,
    Not,
    BitAnd,
    BitOr,
    BitNot,
    BitXor,
    Shl,
    Shr,
    Ret,
    Less,
    Leq,
    Greater,
    Geq,
    Equal,
    Neq,
    Jump,
    JumpIf,
    Dispatch,
    Load,
    Store,
    Copy,
    GetWorkingSetPtr,
    GetElementPtr,
    OffsetPtr,
    GetElement,
    Swizzle,
    Cast,
    CallExt,
    Call,
    Print,
};

// Represents an operand in the VM bytecode.
// It consists of a section ID and a byte offset within that section.
struct VMOperand
{
    uint32_t sectionId;
    uint32_t padding;  // Padding to ensure section takes 8 bytes. sectionId will be replaced
                       // with actual pointers before execution.
    uint32_t type : 8; // type of the operand data.
    uint32_t size : 24;
    uint32_t offset;
    slang::OperandDataType getType() const { return slang::OperandDataType(type); }
    void setType(slang::OperandDataType newType) { type = uint32_t(newType); }
};

struct VMInstHeader
{
    VMOp opcode;
    uint32_t padding; // 32-bit padding, to ensure space are reserved to store function pointers for
                      // the opcode.
    uint32_t opcodeExtension;
    uint32_t operandCount;
    VMOperand& getOperand(Index index) const { return *((VMOperand*)(this + 1) + index); }
};

struct VMFuncHeader
{
    VMOperand name; // Name of the function as a VMOperand, pointing to the constant section.
    uint32_t workingSetSizeInBytes;  // Size of the working set in bytes.
    uint32_t codeSize;               // Size of the code in bytes.
    uint32_t parameterCount;         // Number of parameters for the function.
    uint32_t returnValueSizeInBytes; // Size of the return value in bytes.
    uint32_t parameterSizeInBytes;   // Size of the parameters in bytes.
    uint32_t getParameterOffset(Index index) const { return *((uint32_t*)(this + 1) + index); }
    uint8_t* getCode() const { return (uint8_t*)(this + 1) + parameterCount * sizeof(uint32_t); }
};

static const int kSlangByteCodeScalarTypeSignedInt = 0;
static const int kSlangByteCodeScalarTypeUnsignedInt = 1;
static const int kSlangByteCodeScalarTypeFloat = 2;

struct ArithmeticExtCode
{
    uint32_t scalarType : 2;     // 0: signed int, 1: unsigned int, 2: floating-point
    uint32_t scalarBitWidth : 2; // 0: 8, 1: 16, 2: 32, 3: 64
    uint32_t vectorSize : 12;    // number of elements in the vector.
    uint32_t unused : 16;
};

template<typename TOperand, typename TInstHeader>
struct VMInstIterator
{
    uint8_t* codePtr; // Pointer to the current instruction.

    void moveNext()
    {
        // Read the instruction header
        TInstHeader header;
        memcpy(&header, codePtr, sizeof(header));
        codePtr += sizeof(header);

        // Calculate the size of operand list.
        auto operandListSize = header.operandCount * sizeof(TOperand);

        // Advance the code pointer by the size of the header and operand list.
        codePtr += operandListSize;
    }

    VMInstIterator& operator++()
    {
        moveNext();
        return *this;
    }
    VMInstIterator operator++(int)
    {
        VMInstIterator rs = *this;
        rs.moveNext();
        return rs;
    }

    bool operator!=(const VMInstIterator& iter) const { return codePtr != iter.codePtr; }
    bool operator==(const VMInstIterator& iter) const { return codePtr == iter.codePtr; }
    TInstHeader* operator*() const { return reinterpret_cast<TInstHeader*>(codePtr); }
};

struct VMModuleView;

struct VMFunctionView
{
    const char* name = nullptr;
    VMFuncHeader* header; // Function header containing metadata.
    uint32_t* paramOffsets;
    uint8_t* functionCode;    // Pointer to the function code.
    uint8_t* functionCodeEnd; // Pointer to the end of the function code.
    VMModuleView* moduleView; // Pointer to start of the module.
    VMInstIterator<VMOperand, VMInstHeader> begin() const
    {
        VMInstIterator<VMOperand, VMInstHeader> iter;
        iter.codePtr = functionCode;
        return iter;
    }

    VMInstIterator<VMOperand, VMInstHeader> end() const
    {
        VMInstIterator<VMOperand, VMInstHeader> iter;
        iter.codePtr = functionCodeEnd;
        return iter;
    }
};

struct VMModuleView
{
    uint8_t* code;
    uint32_t functionCount;
    uint32_t* functionOffsets;
    uint8_t* constants;
    uint32_t constantBlobSize;
    uint32_t stringCount;
    uint32_t* stringOffsets; // Offsets to string literals in the constant section.
    uint8_t* kernelBlob;
    uint32_t kernelBlobSize;

    List<VMFunctionView> functionViews;

    VMFunctionView getFunction(Index index) const;

    template<typename T>
    SlangResult getConstant(VMOperand operand, T& outValue) const
    {
        if (operand.sectionId != kSlangByteCodeSectionConstants)
        {
            return SLANG_FAIL; // Invalid section
        }
        if (operand.offset + sizeof(T) > constantBlobSize)
        {
            return SLANG_FAIL; // Out of bounds
        }
        memcpy(&outValue, constants + operand.offset, sizeof(T));
        return SLANG_OK;
    }
};

SlangResult initVMModule(uint8_t* code, uint32_t codeSize, VMModuleView* moduleView);

StringBuilder& operator<<(StringBuilder& sb, VMOp op);
StringBuilder& operator<<(StringBuilder& sb, VMModuleView& module);
void printVMInst(StringBuilder& sb, VMModuleView* moduleView, VMInstHeader* inst);

} // namespace Slang


#endif
