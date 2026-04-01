// slang-serialize-ir.h
#ifndef SLANG_SERIALIZE_IR_H_INCLUDED
#define SLANG_SERIALIZE_IR_H_INCLUDED

#include "../core/slang-riff.h"
#include "slang-ir.h"
#include "slang-serialize-ir-types.h"
#include "slang-serialize-source-loc.h"

// For TranslationUnitRequest
// and FrontEndCompileRequest::ExtraEntryPointInfo
#include "slang-compiler.h"

namespace Slang
{

struct IRSerialWriter
{
    typedef IRSerialData Ser;
    typedef IRSerialBinary Bin;

    Result write(
        IRModule* module,
        SerialSourceLocWriter* sourceLocWriter,
        SerialOptionFlags flags,
        IRSerialData* serialData);

    /// Write to a container
    static Result writeContainer(const IRSerialData& data, RiffContainer* container);

    /// Get an instruction index from an instruction
    Ser::InstIndex getInstIndex(IRInst* inst) const
    {
        return inst ? Ser::InstIndex(m_instMap.getValue(inst)) : Ser::InstIndex(0);
    }

    /// Get a slice from an index
    UnownedStringSlice getStringSlice(Ser::StringIndex index) const
    {
        return m_stringSlicePool.getSlice(StringSlicePool::Handle(index));
    }
    /// Get index from string representations
    Ser::StringIndex getStringIndex(StringRepresentation* string)
    {
        return Ser::StringIndex(m_stringSlicePool.add(string));
    }
    Ser::StringIndex getStringIndex(const UnownedStringSlice& slice)
    {
        return Ser::StringIndex(m_stringSlicePool.add(slice));
    }
    Ser::StringIndex getStringIndex(Name* name)
    {
        return name ? getStringIndex(name->text) : SerialStringData::kNullStringIndex;
    }
    Ser::StringIndex getStringIndex(const char* chars)
    {
        return Ser::StringIndex(m_stringSlicePool.add(chars));
    }
    Ser::StringIndex getStringIndex(const String& string)
    {
        return Ser::StringIndex(m_stringSlicePool.add(string.getUnownedSlice()));
    }

    StringSlicePool& getStringPool() { return m_stringSlicePool; }

    IRSerialWriter()
        : m_serialData(nullptr), m_stringSlicePool(StringSlicePool::Style::Default)
    {
    }

    /// Produces an instruction list which is in same order as written through IRSerialWriter
    static void calcInstructionList(IRModule* module, List<IRInst*>& instsOut);

protected:
    void _addInstruction(IRInst* inst);
    Result _calcDebugInfo(SerialSourceLocWriter* sourceLocWriter);

    List<IRInst*> m_insts; ///< Instructions in same order as stored in the

    List<IRDecoration*>
        m_decorations; ///< Holds all decorations in order of the instructions as found
    List<IRInst*> m_instWithFirstDecoration; ///< All decorations are held in this order after all
                                             ///< the regular instructions

    Dictionary<IRInst*, Ser::InstIndex> m_instMap; ///< Map an instruction to an instruction index

    StringSlicePool m_stringSlicePool;
    IRSerialData* m_serialData; ///< Where the data is stored
};

struct IRSerialReader
{
    typedef IRSerialData Ser;

    /// Read a stream to fill in dataOut IRSerialData
    static Result readContainer(RiffContainer::ListChunk* module, IRSerialData* outData);

    /// Read a module from serial data
    Result read(
        const IRSerialData& data,
        Session* session,
        SerialSourceLocReader* sourceLocReader,
        RefPtr<IRModule>& outModule);

    IRSerialReader()
        : m_serialData(nullptr), m_module(nullptr), m_stringTable(StringSlicePool::Style::Default)
    {
    }

protected:
    StringSlicePool m_stringTable;

    const IRSerialData* m_serialData;
    IRModule* m_module;
};

} // namespace Slang

#endif
