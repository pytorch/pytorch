// slang-serialize-ir.cpp
#include "slang-serialize-ir.h"

#include "../core/slang-byte-encode-util.h"
#include "../core/slang-math.h"
#include "../core/slang-text-io.h"
#include "slang-ir-insts.h"

namespace Slang
{

static bool _isConstant(IROp opIn)
{
    const int op = (kIROpMask_OpMask & opIn);
    return op >= kIROp_FirstConstant && op <= kIROp_LastConstant;
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! IRSerialWriter !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

void IRSerialWriter::_addInstruction(IRInst* inst)
{
    // It cannot already be in the map
    SLANG_ASSERT(!m_instMap.containsKey(inst));

    // Add to the map
    m_instMap.add(inst, Ser::InstIndex(m_insts.getCount()));
    m_insts.add(inst);
}

Result IRSerialWriter::_calcDebugInfo(SerialSourceLocWriter* sourceLocWriter)
{
    // We need to find the unique source Locs
    // We are not going to store SourceLocs directly, because there may be multiple views mapping
    // down to the same underlying source file

    // First find all the unique locs
    struct InstLoc
    {
        typedef InstLoc ThisType;

        SLANG_FORCE_INLINE bool operator<(const ThisType& rhs) const
        {
            return sourceLoc < rhs.sourceLoc ||
                   (sourceLoc == rhs.sourceLoc && instIndex < rhs.instIndex);
        }

        uint32_t instIndex;
        uint32_t sourceLoc;
    };

    // Find all of the source locations and their associated instructions
    List<InstLoc> instLocs;
    const Index numInsts = m_insts.getCount();
    for (Index i = 1; i < numInsts; i++)
    {
        IRInst* srcInst = m_insts[i];
        if (!srcInst->sourceLoc.isValid())
        {
            continue;
        }
        InstLoc instLoc;
        instLoc.instIndex = uint32_t(i);
        instLoc.sourceLoc = uint32_t(srcInst->sourceLoc.getRaw());
        instLocs.add(instLoc);
    }

    // Sort them
    instLocs.sort();

    // Look for runs
    const InstLoc* startInstLoc = instLocs.begin();
    const InstLoc* endInstLoc = instLocs.end();

    while (startInstLoc < endInstLoc)
    {
        const uint32_t startSourceLoc = startInstLoc->sourceLoc;

        // Find the run with the same source loc

        const InstLoc* curInstLoc = startInstLoc + 1;
        uint32_t curInstIndex = startInstLoc->instIndex + 1;

        // Find the run size with same source loc and run of instruction indices
        for (; curInstLoc < endInstLoc && curInstLoc->sourceLoc == startSourceLoc &&
               curInstLoc->instIndex == curInstIndex;
             ++curInstLoc, ++curInstIndex)
        {
        }

        // Add the run

        IRSerialData::SourceLocRun sourceLocRun;
        sourceLocRun.m_numInst = curInstIndex - startInstLoc->instIndex;
        ;
        sourceLocRun.m_startInstIndex = IRSerialData::InstIndex(startInstLoc->instIndex);
        sourceLocRun.m_sourceLoc =
            sourceLocWriter->addSourceLoc(SourceLoc::fromRaw(startSourceLoc));

        m_serialData->m_debugSourceLocRuns.add(sourceLocRun);

        // Next
        startInstLoc = curInstLoc;
    }

    return SLANG_OK;
}

Result IRSerialWriter::write(
    IRModule* module,
    SerialSourceLocWriter* sourceLocWriter,
    SerialOptionFlags options,
    IRSerialData* serialData)
{
    typedef Ser::Inst::PayloadType PayloadType;

    m_serialData = serialData;

    serialData->clear();

    // We reserve 0 for null
    m_insts.clear();
    m_insts.add(nullptr);

    // Reset
    m_instMap.clear();
    m_decorations.clear();

    // Stack for parentInst
    List<IRInst*> parentInstStack;

    IRModuleInst* moduleInst = module->getModuleInst();
    parentInstStack.add(moduleInst);

    // Add to the map
    _addInstruction(moduleInst);

    // Traverse all of the instructions
    while (parentInstStack.getCount())
    {
        // If it's in the stack it is assumed it is already in the inst map
        IRInst* parentInst = parentInstStack.getLast();
        parentInstStack.removeLast();
        SLANG_ASSERT(m_instMap.containsKey(parentInst));

        // Okay we go through each of the children in order. If they are IRInstParent derived, we
        // add to stack to process later cos we want breadth first so the order of children is the
        // same as their index order, meaning we don't need to store explicit indices
        const Ser::InstIndex startChildInstIndex = Ser::InstIndex(m_insts.getCount());

        IRInstListBase childrenList = parentInst->getDecorationsAndChildren();
        for (IRInst* child : childrenList)
        {
            // This instruction can't be in the map...
            SLANG_ASSERT(!m_instMap.containsKey(child));

            _addInstruction(child);

            parentInstStack.add(child);
        }

        // If it had any children, then store the information about it
        if (Ser::InstIndex(m_insts.getCount()) != startChildInstIndex)
        {
            Ser::InstRun run;
            run.m_parentIndex = m_instMap[parentInst];
            run.m_startInstIndex = startChildInstIndex;
            run.m_numChildren = Ser::SizeType(m_insts.getCount() - int(startChildInstIndex));

            m_serialData->m_childRuns.add(run);
        }
    }

#if 0
    {
        List<IRInst*> workInsts;
        calcInstructionList(module, workInsts);
        SLANG_ASSERT(workInsts.getCount() == m_insts.getCount());
        for (UInt i = 0; i < workInsts.getCount(); ++i)
        {
            SLANG_ASSERT(workInsts[i] == m_insts[i]);
        }
    }
#endif

    // Set to the right size
    m_serialData->m_insts.setCount(m_insts.getCount());
    // Clear all instructions
    memset(m_serialData->m_insts.begin(), 0, sizeof(Ser::Inst) * m_serialData->m_insts.getCount());

    // Need to set up the actual instructions
    {
        const Index numInsts = m_insts.getCount();

        for (Index i = 1; i < numInsts; ++i)
        {
            IRInst* srcInst = m_insts[i];
            Ser::Inst& dstInst = m_serialData->m_insts[i];

            dstInst.m_op = uint16_t(srcInst->getOp() & kIROpMask_OpMask);
            dstInst.m_payloadType = PayloadType::Empty;

            dstInst.m_resultTypeIndex = getInstIndex(srcInst->getFullType());

            IRConstant* irConst = as<IRConstant>(srcInst);
            if (irConst)
            {
                switch (srcInst->getOp())
                {
                // Special handling for the ir const derived types
                case kIROp_BlobLit:
                    {
                        // Blobs are serialized into string table like strings
                        auto stringLit = static_cast<IRBlobLit*>(srcInst);
                        dstInst.m_payloadType = PayloadType::String_1;
                        dstInst.m_payload.m_stringIndices[0] =
                            getStringIndex(stringLit->getStringSlice());
                        break;
                    }
                case kIROp_StringLit:
                    {
                        auto stringLit = static_cast<IRStringLit*>(srcInst);
                        dstInst.m_payloadType = PayloadType::String_1;
                        dstInst.m_payload.m_stringIndices[0] =
                            getStringIndex(stringLit->getStringSlice());
                        break;
                    }
                case kIROp_IntLit:
                    {
                        dstInst.m_payloadType = PayloadType::Int64;
                        dstInst.m_payload.m_int64 = irConst->value.intVal;
                        break;
                    }
                case kIROp_PtrLit:
                    {
                        dstInst.m_payloadType = PayloadType::Int64;
                        dstInst.m_payload.m_int64 = (intptr_t)irConst->value.ptrVal;
                        break;
                    }
                case kIROp_FloatLit:
                    {
                        dstInst.m_payloadType = PayloadType::Float64;
                        dstInst.m_payload.m_float64 = irConst->value.floatVal;
                        break;
                    }
                case kIROp_BoolLit:
                    {
                        dstInst.m_payloadType = PayloadType::UInt32;
                        dstInst.m_payload.m_uint32 = irConst->value.intVal ? 1 : 0;
                        break;
                    }
                case kIROp_VoidLit:
                    {
                        dstInst.m_payloadType = PayloadType::Empty;
                        break;
                    }
                default:
                    {
                        SLANG_RELEASE_ASSERT(!"Unhandled constant type");
                        return SLANG_FAIL;
                    }
                }
                continue;
            }

            // ModuleInst is different, in so far as it holds a pointer to IRModule, but we don't
            // need to save that off in a special way, so can just use regular path

            const int numOperands = int(srcInst->operandCount);
            Ser::InstIndex* dstOperands = nullptr;

            if (numOperands <= Ser::Inst::kMaxOperands)
            {
                // Checks the compile below is valid
                SLANG_COMPILE_TIME_ASSERT(
                    PayloadType(0) == PayloadType::Empty &&
                    PayloadType(1) == PayloadType::Operand_1 &&
                    PayloadType(2) == PayloadType::Operand_2);

                dstInst.m_payloadType = PayloadType(numOperands);
                dstOperands = dstInst.m_payload.m_operands;
            }
            else
            {
                dstInst.m_payloadType = PayloadType::OperandExternal;

                int operandArrayBaseIndex = int(m_serialData->m_externalOperands.getCount());
                m_serialData->m_externalOperands.setCount(operandArrayBaseIndex + numOperands);

                dstOperands = m_serialData->m_externalOperands.begin() + operandArrayBaseIndex;

                auto& externalOperands = dstInst.m_payload.m_externalOperand;
                externalOperands.m_arrayIndex = Ser::ArrayIndex(operandArrayBaseIndex);
                externalOperands.m_size = Ser::SizeType(numOperands);
            }

            for (int j = 0; j < numOperands; ++j)
            {
                const Ser::InstIndex dstInstIndex = getInstIndex(srcInst->getOperand(j));
                dstOperands[j] = dstInstIndex;
            }
        }
    }

    // Convert strings into a string table
    {
        SerialStringTableUtil::encodeStringTable(m_stringSlicePool, serialData->m_stringTable);
    }

    // If the option to use RawSourceLocations is enabled, serialize out as is
    if (options & SerialOptionFlag::RawSourceLocation)
    {
        const Index numInsts = m_insts.getCount();
        serialData->m_rawSourceLocs.setCount(numInsts);

        Ser::RawSourceLoc* dstLocs = serialData->m_rawSourceLocs.begin();
        // 0 is null, just mark as no location
        dstLocs[0] = Ser::RawSourceLoc(0);
        for (Index i = 1; i < numInsts; ++i)
        {
            IRInst* srcInst = m_insts[i];
            dstLocs[i] = Ser::RawSourceLoc(srcInst->sourceLoc.getRaw());
        }
    }

    if ((options & SerialOptionFlag::SourceLocation) && sourceLocWriter)
    {
        _calcDebugInfo(sourceLocWriter);
    }

    m_serialData = nullptr;
    return SLANG_OK;
}

Result _writeInstArrayChunk(
    FourCC chunkId,
    const List<IRSerialData::Inst>& array,
    RiffContainer* container)
{
    typedef RiffContainer::Chunk Chunk;
    typedef RiffContainer::ScopeChunk ScopeChunk;

    if (array.getCount() == 0)
    {
        return SLANG_OK;
    }

    return SerialRiffUtil::writeArrayChunk(chunkId, array, container);
}

/* static */ Result IRSerialWriter::writeContainer(
    const IRSerialData& data,
    RiffContainer* container)
{
    typedef RiffContainer::Chunk Chunk;
    typedef RiffContainer::ScopeChunk ScopeChunk;

    ScopeChunk scopeModule(container, Chunk::Kind::List, Bin::kIRModuleFourCc);

    SLANG_RETURN_ON_FAIL(_writeInstArrayChunk(Bin::kInstFourCc, data.m_insts, container));
    SLANG_RETURN_ON_FAIL(
        SerialRiffUtil::writeArrayChunk(Bin::kChildRunFourCc, data.m_childRuns, container));
    SLANG_RETURN_ON_FAIL(SerialRiffUtil::writeArrayChunk(
        Bin::kExternalOperandsFourCc,
        data.m_externalOperands,
        container));
    SLANG_RETURN_ON_FAIL(SerialRiffUtil::writeArrayChunk(
        SerialBinary::kStringTableFourCc,
        data.m_stringTable,
        container));

    SLANG_RETURN_ON_FAIL(SerialRiffUtil::writeArrayChunk(
        Bin::kUInt32RawSourceLocFourCc,
        data.m_rawSourceLocs,
        container));

    if (data.m_debugSourceLocRuns.getCount())
    {
        SerialRiffUtil::writeArrayChunk(
            Bin::kDebugSourceLocRunFourCc,
            data.m_debugSourceLocRuns,
            container);
    }

    return SLANG_OK;
}

/* static */ void IRSerialWriter::calcInstructionList(IRModule* module, List<IRInst*>& instsOut)
{
    // We reserve 0 for null
    instsOut.setCount(1);
    instsOut[0] = nullptr;

    // Stack for parentInst
    List<IRInst*> parentInstStack;

    IRModuleInst* moduleInst = module->getModuleInst();
    parentInstStack.add(moduleInst);

    // Add to list
    instsOut.add(moduleInst);

    // Traverse all of the instructions
    while (parentInstStack.getCount())
    {
        // If it's in the stack it is assumed it is already in the inst map
        IRInst* parentInst = parentInstStack.getLast();
        parentInstStack.removeLast();

        IRInstListBase childrenList = parentInst->getDecorationsAndChildren();
        for (IRInst* child : childrenList)
        {
            instsOut.add(child);
            parentInstStack.add(child);
        }
    }
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! IRSerialReader !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

static Result _readInstArrayChunk(
    RiffContainer::DataChunk* chunk,
    List<IRSerialData::Inst>& arrayOut)
{
    SerialRiffUtil::ListResizerForType<IRSerialData::Inst> resizer(arrayOut);
    return SerialRiffUtil::readArrayChunk(chunk, resizer);
}

/* static */ Result IRSerialReader::readContainer(
    RiffContainer::ListChunk* module,
    IRSerialData* outData)
{
    typedef IRSerialBinary Bin;

    outData->clear();

    for (RiffContainer::Chunk* chunk = module->m_containedChunks; chunk; chunk = chunk->m_next)
    {
        RiffContainer::DataChunk* dataChunk = as<RiffContainer::DataChunk>(chunk);
        if (!dataChunk)
        {
            continue;
        }

        switch (dataChunk->m_fourCC)
        {
        case Bin::kInstFourCc:
            {
                SLANG_RETURN_ON_FAIL(_readInstArrayChunk(dataChunk, outData->m_insts));
                break;
            }
        case Bin::kChildRunFourCc:
            {
                SLANG_RETURN_ON_FAIL(
                    SerialRiffUtil::readArrayChunk(dataChunk, outData->m_childRuns));
                break;
            }
        case Bin::kExternalOperandsFourCc:
            {
                SLANG_RETURN_ON_FAIL(
                    SerialRiffUtil::readArrayChunk(dataChunk, outData->m_externalOperands));
                break;
            }
        case SerialBinary::kStringTableFourCc:
            {
                SLANG_RETURN_ON_FAIL(
                    SerialRiffUtil::readArrayChunk(dataChunk, outData->m_stringTable));
                break;
            }
        case Bin::kUInt32RawSourceLocFourCc:
            {
                SLANG_RETURN_ON_FAIL(
                    SerialRiffUtil::readArrayChunk(dataChunk, outData->m_rawSourceLocs));
                break;
            }
        case Bin::kDebugSourceLocRunFourCc:
            {
                SLANG_RETURN_ON_FAIL(
                    SerialRiffUtil::readArrayChunk(dataChunk, outData->m_debugSourceLocRuns));
                break;
            }
        default:
            {
                break;
            }
        }
    }

    return SLANG_OK;
}

Result IRSerialReader::read(
    const IRSerialData& data,
    Session* session,
    SerialSourceLocReader* sourceLocReader,
    RefPtr<IRModule>& outModule)
{
    // Only used in debug builds
    [[maybe_unused]] typedef Ser::Inst::PayloadType PayloadType;

    m_serialData = &data;

    auto module = IRModule::create(session);
    outModule = module;
    m_module = module;

    // Convert m_stringTable into StringSlicePool.
    SerialStringTableUtil::decodeStringTable(
        data.m_stringTable.getBuffer(),
        data.m_stringTable.getCount(),
        m_stringTable);

    // Each IR instruction has:
    //
    // * An opcode
    // * Zero or more operands
    // * Zero or more children
    //
    // Most instructions are entirely defined by those properties.
    //
    // The instructions that represent simple constants (integers, strings, etc.) are
    // unique in that they have "payload" data that holds their value, instead of having
    // any operands.
    //
    // The deserialization logic here is set up to handle an arbitrary configuration
    // of IR instructions, which means it can handle cases where:
    //
    // * An instruction earlier in the serialized stream might refer to an instruction
    //   later in the stream, as one of its operands or (transitive) children.
    //
    // * An instruction in the stream transitively depends on itself via operand
    //   and/or child relationships.
    //
    // In order to handle these cases, deserialization proceeds in multiple passes.
    // In the first pass, `IRInst`s are allocated for each instruction in the stream,
    // based on their memory requirements (number of operands in the ordinary case
    // and payload size in the case of simple constants). Subsequent passes then
    // fill in the operands and/or children.
    //
    // Note that as a result of the strategy used here, it is not possible for the
    // deserialization logic to interact with any systems for deduplication or
    // simplification of instructions. An alternative version of the deserializer that
    // uses the `IRBuilder` interface instead might be possible, but would need a
    // plan for how to handle forward and/or circular references in the IR module.

    // Add all the instructions
    List<IRInst*> insts;

    const Index numInsts = data.m_insts.getCount();

    SLANG_ASSERT(numInsts > 0);

    insts.setCount(numInsts);
    insts[0] = nullptr;

    // 0 holds null
    // 1 holds the IRModuleInst
    {
        // Check that insts[1] is the module inst
        const Ser::Inst& srcInst = data.m_insts[1];
        SLANG_RELEASE_ASSERT(srcInst.m_op == kIROp_Module);
        SLANG_ASSERT(srcInst.m_payloadType == PayloadType::Empty);

        // The root IR instruction for the module will already have
        // been created as part of creating `module` above.
        //
        auto moduleInst = module->getModuleInst();

        // Set the IRModuleInst
        insts[1] = moduleInst;
    }

    for (Index i = 2; i < numInsts; ++i)
    {
        const Ser::Inst& srcInst = data.m_insts[i];

        const IROp op((IROp)srcInst.m_op);

        if (_isConstant(op))
        {
            // Handling of constants

            // Calculate the minimum object size (ie not including the payload of value)
            const size_t prefixSize = SLANG_OFFSET_OF(IRConstant, value);

            // All IR constants have zero operands.
            Int operandCount = 0;

            IRConstant* irConst = nullptr;
            switch (op)
            {
            case kIROp_BoolLit:
                {
                    // TODO: Most of these cases could use the templated `_allocateInst<T>`
                    // *if* we had distinct `IRConstant` subtypes to represent these
                    // cases and their subtype-specific payloads.

                    SLANG_ASSERT(srcInst.m_payloadType == PayloadType::UInt32);
                    irConst = static_cast<IRConstant*>(module->_allocateInst(
                        op,
                        operandCount,
                        prefixSize + sizeof(IRIntegerValue)));
                    irConst->value.intVal = srcInst.m_payload.m_uint32 != 0;
                    break;
                }
            case kIROp_IntLit:
                {
                    SLANG_ASSERT(srcInst.m_payloadType == PayloadType::Int64);
                    irConst = static_cast<IRConstant*>(module->_allocateInst(
                        op,
                        operandCount,
                        prefixSize + sizeof(IRIntegerValue)));
                    irConst->value.intVal = srcInst.m_payload.m_int64;
                    break;
                }
            case kIROp_PtrLit:
                {
                    SLANG_ASSERT(srcInst.m_payloadType == PayloadType::Int64);
                    irConst = static_cast<IRConstant*>(
                        module->_allocateInst(op, operandCount, prefixSize + sizeof(void*)));
                    irConst->value.ptrVal = (void*)(intptr_t)srcInst.m_payload.m_int64;
                    break;
                }
            case kIROp_FloatLit:
                {
                    SLANG_ASSERT(srcInst.m_payloadType == PayloadType::Float64);
                    irConst = static_cast<IRConstant*>(module->_allocateInst(
                        op,
                        operandCount,
                        prefixSize + sizeof(IRFloatingPointValue)));
                    irConst->value.floatVal = srcInst.m_payload.m_float64;
                    break;
                }
            case kIROp_VoidLit:
                {
                    SLANG_ASSERT(srcInst.m_payloadType == PayloadType::Empty);
                    irConst = static_cast<IRConstant*>(
                        module->_allocateInst(op, operandCount, prefixSize));
                    break;
                }
            case kIROp_BlobLit:
            case kIROp_StringLit:
                {
                    SLANG_ASSERT(srcInst.m_payloadType == PayloadType::String_1);

                    const UnownedStringSlice slice = m_stringTable.getSlice(
                        StringSlicePool::Handle(srcInst.m_payload.m_stringIndices[0]));

                    const size_t sliceSize = slice.getLength();
                    const size_t instSize =
                        prefixSize + SLANG_OFFSET_OF(IRConstant::StringValue, chars) + sliceSize;

                    irConst =
                        static_cast<IRConstant*>(module->_allocateInst(op, operandCount, instSize));

                    IRConstant::StringValue& dstString = irConst->value.stringVal;

                    dstString.numChars = uint32_t(sliceSize);
                    // Turn into pointer to avoid warning of array overrun
                    char* dstChars = dstString.chars;
                    // Copy the chars
                    memcpy(dstChars, slice.begin(), sliceSize);
                    break;
                }
            default:
                {
                    SLANG_ASSERT(!"Unknown constant type");
                    return SLANG_FAIL;
                }
            }

            insts[i] = irConst;
        }
        else
        {
            int numOperands = srcInst.getNumOperands();
            insts[i] = module->_allocateInst(op, numOperands);
        }
    }

    // Patch up the operands
    for (Index i = 1; i < numInsts; ++i)
    {
        const Ser::Inst& srcInst = data.m_insts[i];
        const IROp op((IROp)srcInst.m_op);
        SLANG_UNUSED(op);

        IRInst* dstInst = insts[i];

        // Set the result type
        if (srcInst.m_resultTypeIndex != Ser::InstIndex(0))
        {
            IRInst* resultInst = insts[int(srcInst.m_resultTypeIndex)];
            // NOTE! Counter intuitively the IRType* paramter may not be IRType* derived for example
            // IRGlobalGenericParam is valid, but isn't IRType* derived

            // SLANG_RELEASE_ASSERT(as<IRType>(resultInst));
            dstInst->setFullType(static_cast<IRType*>(resultInst));
        }

        // if (!isParentDerived(op))
        {
            const Ser::InstIndex* srcOperandIndices;
            const int numOperands = data.getOperands(srcInst, &srcOperandIndices);

            auto dstOperands = dstInst->getOperands();

            for (int j = 0; j < numOperands; j++)
            {
                dstOperands[j].init(dstInst, insts[int(srcOperandIndices[j])]);
            }
        }
    }

    // Patch up the children
    {
        const Index numChildRuns = data.m_childRuns.getCount();
        for (Index i = 0; i < numChildRuns; i++)
        {
            const auto& run = data.m_childRuns[i];

            IRInst* inst = insts[int(run.m_parentIndex)];

            for (int j = 0; j < int(run.m_numChildren); ++j)
            {
                IRInst* child = insts[j + int(run.m_startInstIndex)];
                SLANG_ASSERT(child->parent == nullptr);
                child->insertAtEnd(inst);
            }
        }
    }

    // Re-add source locations, if they are defined
    if (m_serialData->m_rawSourceLocs.getCount() == numInsts)
    {
        const Ser::RawSourceLoc* srcLocs = m_serialData->m_rawSourceLocs.begin();
        for (Index i = 1; i < numInsts; ++i)
        {
            IRInst* dstInst = insts[i];

            dstInst->sourceLoc.setRaw(Slang::SourceLoc::RawValue(srcLocs[i]));
        }
    }

    // We now need to apply the runs
    if (sourceLocReader && m_serialData->m_debugSourceLocRuns.getCount())
    {
        List<IRSerialData::SourceLocRun> sourceRuns(m_serialData->m_debugSourceLocRuns);
        // They are now in source location order
        sourceRuns.sort();

        // Just guess initially 0 for the source file that contains the initial run
        SerialSourceLocData::SourceRange range = SerialSourceLocData::SourceRange::getInvalid();
        int fix = 0;

        const Index numRuns = sourceRuns.getCount();
        for (Index i = 0; i < numRuns; ++i)
        {
            const auto& run = sourceRuns[i];

            // Work out the fixed source location
            SourceLoc sourceLoc;
            if (run.m_sourceLoc)
            {
                if (!range.contains(run.m_sourceLoc))
                {
                    fix = sourceLocReader->calcFixSourceLoc(run.m_sourceLoc, range);
                }
                sourceLoc = sourceLocReader->calcFixedLoc(run.m_sourceLoc, fix, range);
            }

            // Write to all the instructions
            SLANG_ASSERT(Index(uint32_t(run.m_startInstIndex) + run.m_numInst) <= insts.getCount());
            IRInst** dstInsts = insts.getBuffer() + int(run.m_startInstIndex);

            const int runSize = int(run.m_numInst);
            for (int j = 0; j < runSize; ++j)
            {
                dstInsts[j]->sourceLoc = sourceLoc;
            }
        }
    }

    outModule->buildMangledNameToGlobalInstMap();

    return SLANG_OK;
}

} // namespace Slang
