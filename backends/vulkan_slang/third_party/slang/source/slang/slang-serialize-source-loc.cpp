// slang-serialize-source-loc.cpp
#include "slang-serialize-source-loc.h"

#include "../core/slang-byte-encode-util.h"
#include "../core/slang-math.h"
#include "../core/slang-text-io.h"

namespace Slang
{

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DebugSerialData !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

size_t SerialSourceLocData::calcSizeInBytes() const
{
    return SerialListUtil::calcArraySize(m_stringTable) +
           SerialListUtil::calcArraySize(m_lineInfos) +
           SerialListUtil::calcArraySize(m_sourceInfos) +
           SerialListUtil::calcArraySize(m_adjustedLineInfos);
}

void SerialSourceLocData::clear()
{
    m_lineInfos.clear();
    m_adjustedLineInfos.clear();
    m_sourceInfos.clear();
    m_stringTable.clear();
}


bool SerialSourceLocData::operator==(const ThisType& rhs) const
{
    return (this == &rhs) ||
           (SerialListUtil::isEqual(m_stringTable, rhs.m_stringTable) &&
            SerialListUtil::isEqual(m_lineInfos, rhs.m_lineInfos) &&
            SerialListUtil::isEqual(m_adjustedLineInfos, rhs.m_adjustedLineInfos) &&
            SerialListUtil::isEqual(m_sourceInfos, rhs.m_sourceInfos));
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DebugSerialWriter !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

SerialSourceLocData::SourceLoc SerialSourceLocWriter::addSourceLoc(SourceLoc sourceLoc)
{
    // If it's not set we can ignore
    if (!sourceLoc.isValid())
    {
        return SerialSourceLocData::SourceLoc(0);
    }

    // Look up the view it's from
    SourceView* sourceView = m_sourceManager->findSourceView(sourceLoc);
    if (!sourceView)
    {
        // If not found we just ingore
        return SerialSourceLocData::SourceLoc(0);
    }

    SourceFile* sourceFile = sourceView->getSourceFile();
    Source* debugSourceFile;
    {
        RefPtr<Source>* ptrDebugSourceFile = m_sourceFileMap.tryGetValue(sourceFile);
        if (ptrDebugSourceFile == nullptr)
        {
            const SourceLoc::RawValue baseSourceLoc = m_freeSourceLoc;
            m_freeSourceLoc += SourceLoc::RawValue(sourceView->getRange().getSize() + 1);

            debugSourceFile = new Source(sourceFile, baseSourceLoc);
            m_sourceFileMap.add(sourceFile, debugSourceFile);
        }
        else
        {
            debugSourceFile = *ptrDebugSourceFile;
        }
    }

    // We need to work out the line index

    int offset = sourceView->getRange().getOffset(sourceLoc);
    int lineIndex = sourceFile->calcLineIndexFromOffset(offset);

    SerialSourceLocData::LineInfo lineInfo;
    lineInfo.m_lineStartOffset = sourceFile->getLineBreakOffsets()[lineIndex];
    lineInfo.m_lineIndex = lineIndex;

    if (!debugSourceFile->hasLineIndex(lineIndex))
    {
        // Add the information about the line
        int entryIndex = sourceView->findEntryIndex(sourceLoc);
        if (entryIndex < 0)
        {
            debugSourceFile->m_lineInfos.add(lineInfo);
        }
        else
        {
            const auto& entry = sourceView->getEntries()[entryIndex];

            SerialSourceLocData::AdjustedLineInfo adjustedLineInfo;
            adjustedLineInfo.m_lineInfo = lineInfo;
            adjustedLineInfo.m_pathStringIndex = SerialStringData::kNullStringIndex;

            const auto& pool = sourceView->getSourceManager()->getStringSlicePool();
            SLANG_ASSERT(pool.getStyle() == StringSlicePool::Style::Default);

            if (!pool.isDefaultHandle(entry.m_pathHandle))
            {
                UnownedStringSlice slice = pool.getSlice(entry.m_pathHandle);
                SLANG_ASSERT(slice.getLength() > 0);
                adjustedLineInfo.m_pathStringIndex =
                    SerialSourceLocData::StringIndex(m_stringSlicePool.add(slice));
            }

            adjustedLineInfo.m_adjustedLineIndex = lineIndex + entry.m_lineAdjust;

            debugSourceFile->m_adjustedLineInfos.add(adjustedLineInfo);
        }

        debugSourceFile->setHasLineIndex(lineIndex);
    }

    return SerialSourceLocData::SourceLoc(debugSourceFile->m_baseSourceLoc + offset);
}

void SerialSourceLocWriter::write(SerialSourceLocData* outSourceLocData)
{
    outSourceLocData->clear();

    // Okay we can now calculate the final source information

    for (const auto& [_, debugSourceFile] : m_sourceFileMap)
    {
        SourceFile* sourceFile = debugSourceFile->m_sourceFile;

        SerialSourceLocData::SourceInfo sourceInfo;

        sourceInfo.m_numLines =
            uint32_t(debugSourceFile->m_sourceFile->getLineBreakOffsets().getCount());

        sourceInfo.m_range.m_start = uint32_t(debugSourceFile->m_baseSourceLoc);
        sourceInfo.m_range.m_end =
            uint32_t(debugSourceFile->m_baseSourceLoc + sourceFile->getContentSize());

        sourceInfo.m_pathIndex = SerialSourceLocData::StringIndex(
            m_stringSlicePool.add(sourceFile->getPathInfo().foundPath));

        sourceInfo.m_lineInfosStartIndex = uint32_t(outSourceLocData->m_lineInfos.getCount());
        sourceInfo.m_adjustedLineInfosStartIndex =
            uint32_t(outSourceLocData->m_adjustedLineInfos.getCount());

        sourceInfo.m_numLineInfos = uint32_t(debugSourceFile->m_lineInfos.getCount());
        sourceInfo.m_numAdjustedLineInfos =
            uint32_t(debugSourceFile->m_adjustedLineInfos.getCount());

        // Add the line infos
        outSourceLocData->m_lineInfos.addRange(
            debugSourceFile->m_lineInfos.begin(),
            debugSourceFile->m_lineInfos.getCount());
        outSourceLocData->m_adjustedLineInfos.addRange(
            debugSourceFile->m_adjustedLineInfos.begin(),
            debugSourceFile->m_adjustedLineInfos.getCount());

        // Add the source info
        outSourceLocData->m_sourceInfos.add(sourceInfo);
    }

    // Convert the string pool
    SerialStringTableUtil::encodeStringTable(m_stringSlicePool, outSourceLocData->m_stringTable);
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! SerialSourceLocReader !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

Index SerialSourceLocReader::findViewIndex(SerialSourceLocData::SourceLoc loc)
{
    for (Index i = 0; i < m_views.getCount(); ++i)
    {
        if (m_views[i].m_range.contains(loc))
        {
            return i;
        }
    }
    return -1;
}


int SerialSourceLocReader::calcFixSourceLoc(
    SerialSourceLocData::SourceLoc loc,
    SerialSourceLocData::SourceRange& outRange)
{
    if (m_lastViewIndex < 0 || !m_views[m_lastViewIndex].m_range.contains(loc))
    {
        m_lastViewIndex = findViewIndex(loc);
    }

    if (m_lastViewIndex < 0)
    {
        // Set an invalid range, as couldn't find
        outRange = SerialSourceLocData::SourceRange::getInvalid();
        return 0;
    }

    const auto& view = m_views[m_lastViewIndex];

    SLANG_ASSERT(view.m_range.contains(loc));

    outRange = view.m_range;
    return view.m_sourceView->getRange().begin.getRaw() - view.m_range.m_start;
}

SourceLoc SerialSourceLocReader::getSourceLoc(SerialSourceLocData::SourceLoc loc)
{
    if (loc != 0)
    {
        if (m_lastViewIndex >= 0)
        {
            const auto& view = m_views[m_lastViewIndex];
            if (view.m_range.contains(loc))
            {
                return view.m_range.getSourceLoc(loc, view.m_sourceView);
            }
        }

        m_lastViewIndex = findViewIndex(loc);
        if (m_lastViewIndex >= 0)
        {
            const auto& view = m_views[m_lastViewIndex];
            return view.m_range.getSourceLoc(loc, view.m_sourceView);
        }
    }
    return SourceLoc();
}

SlangResult SerialSourceLocReader::read(
    const SerialSourceLocData* serialData,
    SourceManager* sourceManager)
{
    m_views.setCount(0);

    if (!sourceManager || serialData->m_sourceInfos.getCount() == 0)
    {
        return SLANG_OK;
    }

    List<UnownedStringSlice> debugStringSlices;
    SerialStringTableUtil::decodeStringTable(
        serialData->m_stringTable.getBuffer(),
        serialData->m_stringTable.getCount(),
        debugStringSlices);

    // All of the strings are placed in the manager (and its StringSlicePool) where the SourceView
    // and SourceFile are constructed from
    List<StringSlicePool::Handle> stringMap;
    SerialStringTableUtil::calcStringSlicePoolMap(
        debugStringSlices,
        sourceManager->getStringSlicePool(),
        stringMap);

    // Construct the source files
    const Index numSourceFiles = serialData->m_sourceInfos.getCount();

    // These hold the views (and SourceFile as there is only one SourceFile per view) in the same
    // order as the sourceInfos
    m_views.setCount(numSourceFiles);

    for (Index i = 0; i < numSourceFiles; ++i)
    {
        const auto& srcSourceInfo = serialData->m_sourceInfos[i];

        PathInfo pathInfo;
        pathInfo.type = PathInfo::Type::FoundPath;
        pathInfo.foundPath = debugStringSlices[UInt(srcSourceInfo.m_pathIndex)];

        SourceFile* sourceFile =
            sourceManager->createSourceFileWithSize(pathInfo, srcSourceInfo.m_range.getCount());

        // Here the initiatingSourecLoc is passed as 0, as that information is not currently saved
        // This simplifies the serialization - as currently for this data we save only a single view
        // per file.
        SourceView* sourceView =
            sourceManager->createSourceView(sourceFile, nullptr, SourceLoc::fromRaw(0));

        // We need to accumulate all line numbers, for this source file, both adjusted and
        // unadjusted
        List<SerialSourceLocData::LineInfo> lineInfos;
        // Add the adjusted lines
        {
            lineInfos.setCount(srcSourceInfo.m_numAdjustedLineInfos);
            const SerialSourceLocData::AdjustedLineInfo* srcAdjustedLineInfos =
                serialData->m_adjustedLineInfos.getBuffer() +
                srcSourceInfo.m_adjustedLineInfosStartIndex;
            const int numAdjustedLines = int(srcSourceInfo.m_numAdjustedLineInfos);
            for (int j = 0; j < numAdjustedLines; ++j)
            {
                lineInfos[j] = srcAdjustedLineInfos[j].m_lineInfo;
            }
        }
        // Add regular lines
        lineInfos.addRange(
            serialData->m_lineInfos.getBuffer() + srcSourceInfo.m_lineInfosStartIndex,
            srcSourceInfo.m_numLineInfos);

        // Put in sourceloc order
        lineInfos.sort();

        List<uint32_t> lineBreakOffsets;

        // We can now set up the line breaks array
        const int numLines = int(srcSourceInfo.m_numLines);
        lineBreakOffsets.setCount(numLines);

        {
            const Index numLineInfos = lineInfos.getCount();
            Index lineIndex = 0;

            // Every line up and including should hold the same offset
            for (Index lineInfoIndex = 0; lineInfoIndex < numLineInfos; ++lineInfoIndex)
            {
                const auto& lineInfo = lineInfos[lineInfoIndex];

                const uint32_t offset = lineInfo.m_lineStartOffset;
                const int finishIndex = int(lineInfo.m_lineIndex);

                SLANG_ASSERT(finishIndex < numLines);

                for (; lineIndex < finishIndex; ++lineIndex)
                {
                    SLANG_ASSERT(offset > 0);
                    lineBreakOffsets[lineIndex] = offset - 1;
                }
                lineBreakOffsets[lineIndex] = offset;
                lineIndex++;
            }

            // Do the remaining lines
            {
                const uint32_t endOffset = srcSourceInfo.m_range.getCount();
                for (; lineIndex < numLines; ++lineIndex)
                {
                    lineBreakOffsets[lineIndex] = endOffset;
                }
            }
        }

        sourceFile->setLineBreakOffsets(lineBreakOffsets.getBuffer(), lineBreakOffsets.getCount());

        if (srcSourceInfo.m_numAdjustedLineInfos)
        {
            List<SerialSourceLocData::AdjustedLineInfo> adjustedLineInfos;

            int numEntries = int(srcSourceInfo.m_numAdjustedLineInfos);

            adjustedLineInfos.addRange(
                serialData->m_adjustedLineInfos.getBuffer() +
                    srcSourceInfo.m_adjustedLineInfosStartIndex,
                numEntries);
            adjustedLineInfos.sort();

            // Work out the views adjustments, and place in dstEntries
            List<SourceView::Entry> dstEntries;
            dstEntries.setCount(numEntries);

            const uint32_t sourceLocOffset = uint32_t(sourceView->getRange().begin.getRaw());

            for (int j = 0; j < numEntries; ++j)
            {
                const auto& srcEntry = adjustedLineInfos[j];
                auto& dstEntry = dstEntries[j];

                dstEntry.m_pathHandle = stringMap[int(srcEntry.m_pathStringIndex)];
                dstEntry.m_startLoc =
                    SourceLoc::fromRaw(srcEntry.m_lineInfo.m_lineStartOffset + sourceLocOffset);
                dstEntry.m_lineAdjust = int32_t(srcEntry.m_adjustedLineIndex) -
                                        int32_t(srcEntry.m_lineInfo.m_lineIndex);
            }

            // Set the adjustments on the view
            sourceView->setEntries(dstEntries.getBuffer(), dstEntries.getCount());
        }

        // Set the view and the source range
        View& view = m_views[i];
        view.m_sourceView = sourceView;
        view.m_range = srcSourceInfo.m_range;
    }

    return SLANG_OK;
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DebugSerialData !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

/* static */ Result SerialSourceLocData::writeContainer(RiffContainer* container)
{
    RiffContainer::ScopeChunk debugChunkScope(
        container,
        RiffContainer::Chunk::Kind::List,
        SerialSourceLocData::kDebugFourCc);

    SLANG_RETURN_ON_FAIL(SerialRiffUtil::writeArrayChunk(
        SerialSourceLocData::kDebugStringFourCc,
        m_stringTable,
        container));
    SLANG_RETURN_ON_FAIL(SerialRiffUtil::writeArrayChunk(
        SerialSourceLocData::kDebugLineInfoFourCc,
        m_lineInfos,
        container));
    SLANG_RETURN_ON_FAIL(SerialRiffUtil::writeArrayChunk(
        SerialSourceLocData::kDebugAdjustedLineInfoFourCc,
        m_adjustedLineInfos,
        container));
    SLANG_RETURN_ON_FAIL(SerialRiffUtil::writeArrayChunk(
        SerialSourceLocData::kDebugSourceInfoFourCc,
        m_sourceInfos,
        container));

    return SLANG_OK;
}

/* static */ Result SerialSourceLocData::readContainer(RiffContainer::ListChunk* listChunk)
{
    SLANG_ASSERT(listChunk->getSubType() == SerialSourceLocData::kDebugFourCc);

    clear();
    for (RiffContainer::Chunk* chunk = listChunk->m_containedChunks; chunk; chunk = chunk->m_next)
    {
        RiffContainer::DataChunk* dataChunk = as<RiffContainer::DataChunk>(chunk);
        if (!dataChunk)
        {
            continue;
        }

        switch (dataChunk->m_fourCC)
        {
        case SerialSourceLocData::kDebugStringFourCc:
            {
                SLANG_RETURN_ON_FAIL(SerialRiffUtil::readArrayChunk(dataChunk, m_stringTable));
                break;
            }
        case SerialSourceLocData::kDebugLineInfoFourCc:
            {
                SLANG_RETURN_ON_FAIL(SerialRiffUtil::readArrayChunk(dataChunk, m_lineInfos));
                break;
            }
        case SerialSourceLocData::kDebugAdjustedLineInfoFourCc:
            {
                SLANG_RETURN_ON_FAIL(
                    SerialRiffUtil::readArrayChunk(dataChunk, m_adjustedLineInfos));
                break;
            }
        case SerialSourceLocData::kDebugSourceInfoFourCc:
            {
                SLANG_RETURN_ON_FAIL(SerialRiffUtil::readArrayChunk(dataChunk, m_sourceInfos));
                break;
            }
        }
    }

    return SLANG_OK;
}

} // namespace Slang
