#include "slang-vm.h"

#include "core/slang-blob.h"
#include "slang-vm-inst-impl.h"

namespace Slang
{

// Our VM insts need to be 8-byte aligned, so we can replace the opcode with function pointers and
// sectionId with data pointers.
static_assert(sizeof(VMOperand) % 8 == 0);
static_assert(sizeof(VMInstHeader) % 8 == 0);
static_assert(sizeof(VMOperand) == sizeof(VMExecOperand));
static_assert(sizeof(VMInstHeader) == sizeof(VMExecInstHeader));

ISlangUnknown* ByteCodeInterpreter::getInterface(const Guid& guid)
{
    if (guid == ISlangUnknown::getTypeGuid() || guid == IByteCodeRunner::getTypeGuid())
        return static_cast<IByteCodeRunner*>(this);

    return nullptr;
}

SlangResult ByteCodeInterpreter::prepareModuleForExecution()
{
    m_stringLits.clear();
    m_stringLits.setCount(m_moduleView.stringCount);
    for (uint32_t i = 0; i < m_moduleView.stringCount; i++)
    {
        auto strOffset = m_moduleView.stringOffsets[i];
        const char* str = (const char*)m_moduleView.constants + strOffset;
        m_stringLits[i] = str;
    }
    m_stringLitsPtr = m_stringLits.getBuffer();

    m_functions.setCount(m_moduleView.functionCount);
    for (uint32_t i = 0; i < m_moduleView.functionCount; i++)
    {
        auto func = m_moduleView.getFunction(i);
        auto& exeFunc = m_functions[i];
        exeFunc.m_codeBuffer.setCount(func.header->codeSize / sizeof(uint64_t));
        exeFunc.m_header = func.header;
        for (uint32_t j = 0; j < func.header->parameterCount; j++)
        {
            exeFunc.m_parameterOffsets.add(func.header->getParameterOffset(j));
        }
        exeFunc.m_parameterOffsets.add(func.header->parameterSizeInBytes);

        // Copy the code into the executable function buffer
        memcpy(exeFunc.m_codeBuffer.getBuffer(), func.functionCode, func.header->codeSize);

        // Replace the instruction headers with function pointers
        for (auto inst : exeFunc)
        {
            VMInstHeader* instHeader = reinterpret_cast<VMInstHeader*>(inst);
            auto handler = mapInstToFunction(instHeader, &m_moduleView, m_extInstHandlers);
            if (!handler)
            {
                StringBuilder instStr;
                printVMInst(instStr, &m_moduleView, instHeader);
                reportError(
                    "Cannot find execution handler for instruction %s\n",
                    instStr.toString().getBuffer());
                return SLANG_FAIL;
            }
            inst->functionPtr = handler;
            for (uint32_t operandIdx = 0; operandIdx < instHeader->operandCount; operandIdx++)
            {
                auto& operand = instHeader->getOperand(operandIdx);
                auto& execOpernad = inst->getOperand(operandIdx);
                switch (operand.sectionId)
                {
                case kSlangByteCodeSectionConstants:
                    execOpernad.section = &m_moduleView.constants;
                    break;
                case kSlangByteCodeSectionInsts:
                    execOpernad.section = (uint8_t**)&m_currentFuncCode;
                    break;
                case kSlangByteCodeSectionWorkingSet:
                    execOpernad.section = (uint8_t**)&m_currentWorkingSet;
                    break;
                case kSlangByteCodeSectionStrings:
                    execOpernad.section = (uint8_t**)&m_stringLitsPtr;
                    execOpernad.offset *= sizeof(const char*);
                    break;
                }
            }
        }
    }

    return SLANG_OK;
}

SLANG_NO_THROW SlangResult SLANG_MCALL ByteCodeInterpreter::loadModule(IBlob* moduleBlob)
{
    m_stack.reserve(128);
    m_workingSetBuffer.reserve(1024 * 1024); // Reserve 1MB for working set
    m_currentWorkingSet = m_workingSetBuffer.getBuffer();

    m_errorBuilder.clear();
    m_code.addRange((uint8_t*)(moduleBlob->getBufferPointer()), moduleBlob->getBufferSize());
    SLANG_RETURN_ON_FAIL(
        initVMModule(m_code.getBuffer(), (uint32_t)moduleBlob->getBufferSize(), &m_moduleView));
    SLANG_RETURN_ON_FAIL(prepareModuleForExecution());
    return SLANG_OK;
}

SLANG_NO_THROW void SLANG_MCALL ByteCodeInterpreter::getErrorString(slang::IBlob** outBlob)
{
    *outBlob = StringBlob::moveCreate(m_errorBuilder.produceString()).detach();
    m_errorBuilder.clear();
}

SLANG_NO_THROW int SLANG_MCALL ByteCodeInterpreter::findFunctionByName(const char* name)
{
    for (uint32_t i = 0; i < m_moduleView.functionCount; i++)
    {
        auto func = m_moduleView.getFunction(i);
        if (UnownedStringSlice(func.name) == name)
        {
            return (int)i;
        }
    }
    return -1; // Function not found
}

SLANG_NO_THROW SlangResult SLANG_MCALL
ByteCodeInterpreter::getFunctionInfo(uint32_t index, slang::ByteCodeFuncInfo* outInfo)
{
    if (index >= m_moduleView.functionCount)
    {
        return SLANG_FAIL;
    }
    auto func = m_moduleView.getFunction(index);
    outInfo->parameterCount = func.header->parameterCount;
    outInfo->returnValueSize = func.header->returnValueSizeInBytes;
    return SLANG_OK;
}

SLANG_NO_THROW SlangResult SLANG_MCALL
ByteCodeInterpreter::selectFunctionByIndex(uint32_t functionIndex)
{
    if (functionIndex >= m_moduleView.functionCount)
    {
        reportError(
            "Function index %u out of range [0, %u)",
            functionIndex,
            m_moduleView.functionCount);
        return SLANG_FAIL;
    }
    auto func = m_moduleView.getFunction(functionIndex);
    m_currentFuncCode = m_functions[functionIndex].m_codeBuffer.getBuffer();
    m_currentInst = reinterpret_cast<VMExecInstHeader*>(m_currentFuncCode);
    m_workingSetBuffer.setCount(func.header->workingSetSizeInBytes / sizeof(uint64_t));
    m_currentWorkingSet = m_workingSetBuffer.getBuffer();
    return SLANG_OK;
}

SLANG_NO_THROW SlangResult SLANG_MCALL
ByteCodeInterpreter::execute(void* argumentData, size_t argumentSize)
{
    if (!m_currentInst)
    {
        reportError("No function selected for execution");
        return SLANG_FAIL;
    }
    if (!m_currentWorkingSet)
    {
        reportError("No working set allocated for execution");
        return SLANG_FAIL;
    }
    if ((uint8_t*)m_currentWorkingSet + argumentSize >
        (uint8_t*)(m_workingSetBuffer.getBuffer() + m_workingSetBuffer.getCount()))
    {
        reportError("Argument size exceeds working set.");
        return SLANG_FAIL;
    }
    // Copy the arguments into the working set
    if (argumentData && argumentSize > 0)
    {
        memcpy(m_currentWorkingSet, argumentData, argumentSize);
    }
    m_returnValSize = 0;
    while (m_currentInst)
    {
        auto nextInst = m_currentInst->getNextInst();
        auto currentInst = m_currentInst;
        m_currentInst = nextInst;
        currentInst->functionPtr(this, currentInst, m_extInstHandlerUserData);
    }
    return SLANG_OK;
}

ByteCodeInterpreter::ByteCodeInterpreter()
{
    m_printCallback = defaultPrintCallback;
    m_printCallbackUserData = this;
}

SLANG_NO_THROW SlangResult SLANG_MCALL
ByteCodeInterpreter::setPrintCallback(slang::VMPrintFunc callback, void* userData)
{
    m_printCallback = callback;
    m_printCallbackUserData = userData;
    return SLANG_OK;
}

void ByteCodeInterpreter::defaultPrintCallback(const char* str, void* userData)
{
    SLANG_UNUSED(userData);
    printf("%s", str);
}

ExecutableFunction::InstIterator ExecutableFunction::begin()
{
    ExecutableFunction::InstIterator iter;
    iter.codePtr = (uint8_t*)m_codeBuffer.getBuffer();
    return iter;
}

ExecutableFunction::InstIterator ExecutableFunction::end()
{
    ExecutableFunction::InstIterator iter;
    iter.codePtr = (uint8_t*)(m_codeBuffer.getBuffer() + m_codeBuffer.getCount());
    return iter;
}


} // namespace Slang


SLANG_EXTERN_C SLANG_API SlangResult slang_createByteCodeRunner(
    const slang::ByteCodeRunnerDesc* desc,
    slang::IByteCodeRunner** outByteCodeRunner)
{
    SLANG_UNUSED(desc);
    Slang::RefPtr<Slang::ByteCodeInterpreter> runner = new Slang::ByteCodeInterpreter();
    *outByteCodeRunner = static_cast<slang::IByteCodeRunner*>(runner.detach());
    return SLANG_OK;
}

SLANG_EXTERN_C SLANG_API SlangResult
slang_disassembleByteCode(slang::IBlob* moduleBlob, slang::IBlob** outDisassemblyBlob)
{
    Slang::VMModuleView moduleView;
    SLANG_RETURN_ON_FAIL(Slang::initVMModule(
        (uint8_t*)moduleBlob->getBufferPointer(),
        (uint32_t)moduleBlob->getBufferSize(),
        &moduleView));
    Slang::StringBuilder sb;
    sb << moduleView;
    *outDisassemblyBlob = Slang::StringBlob::moveCreate(sb.produceString()).detach();
    return SLANG_OK;
}
