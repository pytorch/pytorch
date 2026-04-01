// slang-source-loc.cpp
#include "slang-source-loc.h"

#include "../core/slang-char-encode.h"
#include "../core/slang-string-escape-util.h"
#include "../core/slang-string-util.h"
#include "slang-artifact-desc-util.h"
#include "slang-artifact-impl.h"
#include "slang-artifact-representation-impl.h"
#include "slang-artifact-util.h"

namespace Slang
{

/* !!!!!!!!!!!!!!!!!!!!!!!!! SourceView !!!!!!!!!!!!!!!!!!!!!!!!!!!! */

const String PathInfo::getMostUniqueIdentity() const
{
    switch (type)
    {
    case Type::Normal:
        return uniqueIdentity;
    case Type::FoundPath:
    case Type::FromString:
        {
            return foundPath;
        }
    default:
        return "";
    }
}

String PathInfo::getName() const
{
    switch (type)
    {
    case Type::Normal:
    case Type::FromString:
    case Type::FoundPath:
        {
            return foundPath;
        }
    }
    return String();
}

bool PathInfo::operator==(const ThisType& rhs) const
{
    // They must be the same type
    if (type != rhs.type)
    {
        return false;
    }

    switch (type)
    {
    case Type::TokenPaste:
    case Type::TypeParse:
    case Type::Unknown:
    case Type::CommandLine:
        {
            return true;
        }
    case Type::Normal:
        {
            return foundPath == rhs.foundPath && uniqueIdentity == rhs.uniqueIdentity;
        }
    case Type::FromString:
    case Type::FoundPath:
        {
            // Only have a found path
            return foundPath == rhs.foundPath;
        }
    default:
        break;
    }

    return false;
}

void PathInfo::appendDisplayName(StringBuilder& out) const
{
    switch (type)
    {
    case Type::TokenPaste:
        out << "[Token Paste]";
        break;
    case Type::TypeParse:
        out << "[Type Parse]";
        break;
    case Type::Unknown:
        out << "[Unknown]";
        break;
    case Type::CommandLine:
        out << "[Command Line]";
        break;
    case Type::Normal:
    case Type::FromString:
    case Type::FoundPath:
        {
            StringEscapeUtil::appendQuoted(
                StringEscapeUtil::getHandler(StringEscapeUtil::Style::Cpp),
                foundPath.getUnownedSlice(),
                out);
            break;
        }
    default:
        break;
    }
}

/* !!!!!!!!!!!!!!!!!!!!!!!!! SourceView !!!!!!!!!!!!!!!!!!!!!!!!!!!! */

int SourceView::findEntryIndex(SourceLoc sourceLoc) const
{
    if (!m_range.contains(sourceLoc))
    {
        return -1;
    }

    const auto rawValue = sourceLoc.getRaw();

    Index hi = m_entries.getCount();
    // If there are no entries, or it is in front of the first entry, then there is no associated
    // entry
    if (hi == 0 || m_entries[0].m_startLoc.getRaw() > sourceLoc.getRaw())
    {
        return -1;
    }

    Index lo = 0;
    while (lo + 1 < hi)
    {
        const Index mid = (hi + lo) >> 1;
        const Entry& midEntry = m_entries[mid];
        SourceLoc::RawValue midValue = midEntry.m_startLoc.getRaw();
        if (midValue <= rawValue)
        {
            // The location we seek is at or after this entry
            lo = mid;
        }
        else
        {
            // The location we seek is before this entry
            hi = mid;
        }
    }

    return int(lo);
}

void SourceView::addLineDirective(
    SourceLoc directiveLoc,
    StringSlicePool::Handle pathHandle,
    int line)
{
    SLANG_ASSERT(pathHandle != StringSlicePool::Handle(0));
    SLANG_ASSERT(m_range.contains(directiveLoc));

    // Check that the directiveLoc values are always increasing
    SLANG_ASSERT(
        m_entries.getCount() == 0 ||
        (m_entries.getLast().m_startLoc.getRaw() < directiveLoc.getRaw()));

    // Calculate the offset
    const int offset = m_range.getOffset(directiveLoc);

    // Get the line index in the original file
    const int lineIndex = m_sourceFile->calcLineIndexFromOffset(offset);

    Entry entry;
    entry.m_startLoc = directiveLoc;
    entry.m_pathHandle = pathHandle;

    // We also need to make sure that any lookups for line numbers will
    // get corrected based on this files location.
    // We assume the line number coming from the directive is a line number, NOT an index, so the
    // correction needs + 1 There is an additional + 1 because we want the NEXT line - ie the line
    // after the #line directive, to the specified value Taking both into account means +2 is
    // correct 'fix'
    entry.m_lineAdjust = line - (lineIndex + 2);

    m_entries.add(entry);
}

void SourceView::addLineDirective(SourceLoc directiveLoc, const String& path, int line)
{
    StringSlicePool::Handle pathHandle =
        getSourceManager()->getStringSlicePool().add(path.getUnownedSlice());
    return addLineDirective(directiveLoc, pathHandle, line);
}

void SourceView::addDefaultLineDirective(SourceLoc directiveLoc)
{
    SLANG_ASSERT(m_range.contains(directiveLoc));
    // Check that the directiveLoc values are always increasing
    SLANG_ASSERT(
        m_entries.getCount() == 0 ||
        (m_entries.getLast().m_startLoc.getRaw() < directiveLoc.getRaw()));

    // Well if there are no entries, or the last one puts it in default case, then we don't need to
    // add anything
    if (m_entries.getCount() == 0 || (m_entries.getCount() && m_entries.getLast().isDefault()))
    {
        return;
    }

    Entry entry;
    entry.m_startLoc = directiveLoc;
    entry.m_lineAdjust = 0; // No line adjustment... we are going back to default
    entry.m_pathHandle =
        StringSlicePool::Handle(0); // Mark that there is no path, and that this is a 'default'

    SLANG_ASSERT(entry.isDefault());

    m_entries.add(entry);
}

// Nominal-like types take into account line directives, and potentially source maps
static bool _isNominalLike(SourceLocType type)
{
    return type == SourceLocType::Nominal || type == SourceLocType::Emit;
}

static bool _canFollowSourceMap(SourceFile* sourceFile, SourceLocType type)
{
    // If we don't have a source map we have nothing to follow
    if (!sourceFile->getSourceMap())
    {
        return false;
    }

    // If it's obfuscated we can't follow if we are emitting
    if (sourceFile->getSourceMapKind() == SourceMapKind::Obfuscated && type == SourceLocType::Emit)
    {
        return false;
    }

    return _isNominalLike(type);
}

static SlangResult _findLocWithSourceMap(
    SourceManager* lookupSourceManager,
    SourceView* sourceView,
    SourceLoc loc,
    SourceLocType type,
    HandleSourceLoc& outLoc)
{
    auto sourceFile = sourceView->getSourceFile();

    if (!_canFollowSourceMap(sourceFile, type))
    {
        return SLANG_E_NOT_FOUND;
    }

    // Hold a list of sourceFiles visited so we can't end up in a loop of lookups
    ShortList<SourceFile*, 8> sourceFiles;
    sourceFiles.add(sourceFile);

    Index entryIndex = -1;

    // Do the initial lookup using the loc
    {
        const auto offset = sourceView->getRange().getOffset(loc);

        const auto lineIndex = sourceFile->calcLineIndexFromOffset(offset);
        const auto colIndex = sourceFile->calcColumnIndex(lineIndex, offset);

        // If we are in this function the sourceFile should have a map
        auto sourceMap = sourceFile->getSourceMap();
        SLANG_ASSERT(sourceMap);

        entryIndex = sourceMap->get().findEntry(lineIndex, colIndex);
    }

    if (entryIndex < 0)
    {
        return SLANG_FAIL;
    }

    // Keep searching through source maps
    do
    {
        auto sourceMap = sourceFile->getSourceMap()->getPtr();

        // Find the entry
        const auto& entry = sourceMap->getEntryByIndex(entryIndex);
        const auto sourceFileName = sourceMap->getSourceFileName(entry.sourceFileIndex);

        // If we have a source name, see if it already exists in source manager
        if (sourceFileName.getLength())
        {
            if (auto foundSourceFile =
                    lookupSourceManager->findSourceFileByPathRecursively(sourceFileName))
            {
                // We only follow if the source file hasn't already been visisted
                if (sourceFiles.indexOf(foundSourceFile) < 0)
                {
                    // Add so we don't reprocess
                    sourceFiles.add(foundSourceFile);

                    // If it has a source map, we try and look up the current location in it's
                    // source map
                    if (_canFollowSourceMap(foundSourceFile, type))
                    {
                        auto foundSourceMap = foundSourceFile->getSourceMap();

                        const auto foundEntryIndex =
                            foundSourceMap->get().findEntry(entry.sourceLine, entry.sourceColumn);

                        // If we found the entry repeat the lookup
                        if (foundEntryIndex >= 0)
                        {
                            sourceFile = foundSourceFile;
                            entryIndex = foundEntryIndex;
                            continue;
                        }
                    }
                }
            }
        }
    } while (false);

    // Generate the HandleSourceLoc
    auto sourceMap = sourceFile->getSourceMap()->getPtr();
    const auto& entry = sourceMap->getEntryByIndex(entryIndex);

    // We need to add the pool of the originating source view/file
    const auto originatingSourceManager = sourceView->getSourceManager();

    auto& managerPool = originatingSourceManager->getStringSlicePool();

    outLoc.line = entry.sourceLine + 1;
    outLoc.column = entry.sourceColumn + 1;
    outLoc.pathHandle = managerPool.add(sourceMap->getSourceFileName(entry.sourceFileIndex));

    return SLANG_OK;
}


SlangResult SourceView::_findSourceMapLoc(
    SourceLoc loc,
    SourceLocType type,
    HandleSourceLoc& outLoc)
{
    // We only do source map lookups with nominal
    if (!_isNominalLike(type))
    {
        return SLANG_E_NOT_FOUND;
    }

    // TODO(JS):
    // Ideally we'd do the lookup on the "current" source manager rather than the source manager on
    // this view, which may be a parent to the current one.
    auto lookupSourceManager = m_sourceFile->getSourceManager();

    SLANG_RETURN_ON_FAIL(_findLocWithSourceMap(lookupSourceManager, this, loc, type, outLoc));

    return SLANG_OK;
}

HandleSourceLoc SourceView::getHandleLoc(SourceLoc loc, SourceLocType type)
{
    {
        HandleSourceLoc handleLoc;
        if (SLANG_SUCCEEDED(_findSourceMapLoc(loc, type, handleLoc)))
        {
            return handleLoc;
        }
    }

    // Get the offset in bytes for this loc
    const int offset = m_range.getOffset(loc);

    // We need the line index from the original source file
    const int lineIndex = m_sourceFile->calcLineIndexFromOffset(offset);

    // TODO:
    // - Tab characters, which should really adjust how we report
    //   columns (although how are we supposed to know the setting
    //   that an IDE expects us to use when reporting locations?)
    //
    //   For now we just count tabs as single chars
    const int columnIndex = m_sourceFile->calcColumnIndex(lineIndex, offset);

    HandleSourceLoc handleLoc;
    handleLoc.column = columnIndex + 1;
    handleLoc.line = lineIndex + 1;

    // Only bother looking up the entry information if we want a 'Norminal'-like lookup
    if (_isNominalLike(type))
    {
        const int entryIndex = findEntryIndex(loc);
        if (entryIndex >= 0)
        {
            const Entry& entry = m_entries[entryIndex];
            // Adjust the line
            handleLoc.line += entry.m_lineAdjust;
            // Get the pathHandle..
            handleLoc.pathHandle = entry.m_pathHandle;
        }
    }

    return handleLoc;
}

HumaneSourceLoc SourceView::getHumaneLoc(SourceLoc loc, SourceLocType type)
{
    HandleSourceLoc handleLoc = getHandleLoc(loc, type);

    HumaneSourceLoc humaneLoc;
    humaneLoc.column = handleLoc.column;
    humaneLoc.line = handleLoc.line;
    humaneLoc.pathInfo = _getPathInfoFromHandle(handleLoc.pathHandle);
    return humaneLoc;
}

PathInfo SourceView::getViewPathInfo() const
{
    if (m_viewPath.getLength())
    {
        PathInfo pathInfo(m_sourceFile->getPathInfo());
        pathInfo.foundPath = m_viewPath;
        return pathInfo;
    }
    else
    {
        return m_sourceFile->getPathInfo();
    }
}

PathInfo SourceView::_getPathInfoFromHandle(StringSlicePool::Handle pathHandle) const
{
    // If there is no override path, then just the source files path
    if (pathHandle == StringSlicePool::Handle(0))
    {
        return getViewPathInfo();
    }
    else
    {
        return PathInfo::makePath(getSourceManager()->getStringSlicePool().getSlice(pathHandle));
    }
}

PathInfo SourceView::getPathInfo(SourceLoc loc, SourceLocType type)
{
    if (type == SourceLocType::Actual)
    {
        return getViewPathInfo();
    }

    {
        HandleSourceLoc handleLoc;
        if (SLANG_SUCCEEDED(_findSourceMapLoc(loc, type, handleLoc)))
        {
            return _getPathInfoFromHandle(handleLoc.pathHandle);
        }
    }

    const int entryIndex = findEntryIndex(loc);
    return _getPathInfoFromHandle(
        (entryIndex >= 0) ? m_entries[entryIndex].m_pathHandle : StringSlicePool::Handle(0));
}

/* !!!!!!!!!!!!!!!!!!!!!!! SourceFile !!!!!!!!!!!!!!!!!!!!!!!!!!!! */

void SourceFile::setLineBreakOffsets(const uint32_t* offsets, UInt numOffsets)
{
    m_lineBreakOffsets.clear();
    m_lineBreakOffsets.addRange(offsets, numOffsets);
}

const List<uint32_t>& SourceFile::getLineBreakOffsets()
{
    // We now have a raw input file that we can search for line breaks.
    // We obviously don't want to do a linear scan over and over, so we will
    // cache an array of line break locations in the file.
    if (m_lineBreakOffsets.getCount() == 0)
    {
        UnownedStringSlice content(getContent()), line;
        char const* contentBegin = content.begin();
        while (StringUtil::extractLine(content, line))
        {
            m_lineBreakOffsets.add(uint32_t(line.begin() - contentBegin));
        }
        // Note that we do *not* treat the end of the file as a line
        // break, because otherwise we would report errors like
        // "end of file inside string literal" with a line number
        // that points at a line that doesn't exist.
    }

    return m_lineBreakOffsets;
}

SourceFile::OffsetRange SourceFile::getOffsetRangeAtLineIndex(Index lineIndex)
{
    const List<uint32_t>& offsets = getLineBreakOffsets();
    const Index count = offsets.getCount();

    if (lineIndex >= count - 1)
    {
        // Work out the line start
        const uint32_t offsetEnd = uint32_t(getContentSize());
        const uint32_t offsetStart = (lineIndex >= count) ? offsetEnd : offsets[lineIndex];
        // The line is the span from start, to the end of the content
        return OffsetRange{offsetStart, offsetEnd};
    }
    else
    {
        const uint32_t offsetStart = offsets[lineIndex];
        const uint32_t offsetEnd = offsets[lineIndex + 1];
        return OffsetRange{offsetStart, offsetEnd};
    }
}

UnownedStringSlice SourceFile::getLineAtIndex(Index lineIndex)
{
    const OffsetRange range = getOffsetRangeAtLineIndex(lineIndex);

    if (range.isValid() && hasContent())
    {
        const UnownedStringSlice content = getContent();
        SLANG_ASSERT(range.end <= uint32_t(content.getLength()));

        const char* const text = content.begin();
        return UnownedStringSlice(text + range.start, text + range.end);
    }

    return UnownedStringSlice();
}

UnownedStringSlice SourceFile::getLineContainingOffset(uint32_t offset)
{
    const Index lineIndex = calcLineIndexFromOffset(offset);
    return getLineAtIndex(lineIndex);
}

bool SourceFile::isOffsetOnLine(uint32_t offset, Index lineIndex)
{
    const OffsetRange range = getOffsetRangeAtLineIndex(lineIndex);
    return range.isValid() && range.containsInclusive(offset);
}

int SourceFile::calcLineIndexFromOffset(int offset)
{
    SLANG_ASSERT(UInt(offset) <= getContentSize());

    // Make sure we have the line break offsets
    const auto& lineBreakOffsets = getLineBreakOffsets();

    // At this point we can assume the `lineBreakOffsets` array has been filled in.
    // We will use a binary search to find the line index that contains our
    // chosen offset.
    Index lo = 0;
    Index hi = lineBreakOffsets.getCount();

    while (lo + 1 < hi)
    {
        const Index mid = (hi + lo) >> 1;
        const uint32_t midOffset = lineBreakOffsets[mid];
        if (midOffset <= uint32_t(offset))
        {
            lo = mid;
        }
        else
        {
            hi = mid;
        }
    }

    return int(lo);
}

int SourceFile::calcColumnOffset(int lineIndex, int offset)
{
    const auto& lineBreakOffsets = getLineBreakOffsets();
    return offset - lineBreakOffsets[lineIndex];
}

int SourceFile::calcColumnIndex(int lineIndex, int offset, int tabSize)
{
    const int colOffset = calcColumnOffset(lineIndex, offset);

    // If we don't have the content of the file, the best we can do is to assume there is a char per
    // column
    if (!hasContent())
    {
        return colOffset;
    }

    const auto line = getLineAtIndex(lineIndex);

    const auto head = line.head(colOffset);

    auto colCount = UTF8Util::calcCodePointCount(head);

    if (tabSize >= 0)
    {
        Count tabCount = 0;
        for (auto c : head)
        {
            tabCount += Count(c == '\t');
        }

        // We substract one from tabSize, because colCount will already holds a +1 for each tab.
        colCount += tabCount * (tabSize - 1);
    }

    return int(colCount);
}

/* !!!!!!!!!!!!!!!!!!!!!!!!! SourceFile !!!!!!!!!!!!!!!!!!!!!!!!!!!! */

void SourceFile::setContents(ISlangBlob* blob)
{
    const UInt rawContentSize = blob->getBufferSize();

    SLANG_ASSERT(rawContentSize == m_contentSize);

    Byte* rawContentBegin = (Byte*)blob->getBufferPointer();

    // Query the encoding type and discard the Unicode Byte-Order-Marker before decoding
    size_t offset;
    auto type = CharEncoding::determineEncoding(rawContentBegin, rawContentSize, offset);
    SLANG_ASSERT(rawContentSize >= offset);

    List<char> decodedBuffer;
    CharEncoding::getEncoding(type)->decode(
        rawContentBegin + offset,
        int(rawContentSize - offset),
        decodedBuffer);

    m_contentBlob = RawBlob::create(decodedBuffer.getBuffer(), decodedBuffer.getCount());

    char const* decodedContentBegin = (char const*)m_contentBlob->getBufferPointer();
    const UInt decodedContentSize = m_contentBlob->getBufferSize();
    char const* decodedContentEnd = decodedContentBegin + decodedContentSize;

    m_content = UnownedStringSlice(decodedContentBegin, decodedContentEnd);
}

void SourceFile::setContents(const String& content)
{
    ComPtr<ISlangBlob> contentBlob = StringUtil::createStringBlob(content);
    setContents(contentBlob);
}

SourceFile::SourceFile(SourceManager* sourceManager, const PathInfo& pathInfo, size_t contentSize)
    : m_sourceManager(sourceManager), m_pathInfo(pathInfo), m_contentSize(contentSize)
{
}

SourceFile::~SourceFile() {}

SHA1::Digest SourceFile::getDigest()
{
    if (m_digest == SHA1::Digest())
    {
        DigestBuilder<SHA1> builder;
        builder.append(getContent());
        m_digest = builder.finalize();
    }
    return m_digest;
}

String SourceFile::calcVerbosePath() const
{
    ISlangFileSystemExt* fileSystemExt = getSourceManager()->getFileSystemExt();

    if (fileSystemExt)
    {
        String displayPath;
        ComPtr<ISlangBlob> displayPathBlob;
        if (SLANG_SUCCEEDED(fileSystemExt->getPath(
                PathKind::Display,
                m_pathInfo.foundPath.getBuffer(),
                displayPathBlob.writeRef())))
        {
            displayPath = StringUtil::getString(displayPathBlob);
        }
        if (displayPath.getLength() > 0)
        {
            return displayPath;
        }
    }

    return m_pathInfo.foundPath;
}

/* !!!!!!!!!!!!!!!!!!!!!!!!! SourceManager !!!!!!!!!!!!!!!!!!!!!!!!!!!! */

void SourceManager::initialize(SourceManager* p, ISlangFileSystemExt* fileSystemExt)
{
    m_fileSystemExt = fileSystemExt;

    m_parent = p;

    _resetLoc();
}

SourceManager::~SourceManager()
{
    _resetSource();
}

void SourceManager::_resetLoc()
{
    if (m_parent)
    {
        // If we have a parent source manager, then we assume that all code at that level
        // has already been loaded, and it is safe to start our own source locations
        // right after those from the parent.
        //
        // TODO: more clever allocation in cases where that might not be reasonable
        m_startLoc = m_parent->m_nextLoc;
    }
    else
    {
        // Location zero is reserved for an invalid location,
        // so we need to start reserving locations starting at 1.
        m_startLoc = SourceLoc::fromRaw(1);
    }

    m_nextLoc = m_startLoc;
}

void SourceManager::_resetSource()
{
    for (auto item : m_sourceViews)
    {
        delete item;
    }

    for (auto item : m_sourceFiles)
    {
        delete item;
    }

    m_sourceViews.clear();
    m_sourceFiles.clear();

    m_sourceFileMap.clear();
}


void SourceManager::reset()
{
    _resetSource();
    _resetLoc();
}

UnownedStringSlice SourceManager::allocateStringSlice(const UnownedStringSlice& slice)
{
    const UInt numChars = slice.getLength();

    char* dst = (char*)m_memoryArena.allocate(numChars);
    ::memcpy(dst, slice.begin(), numChars);

    return UnownedStringSlice(dst, numChars);
}

SourceRange SourceManager::allocateSourceRange(UInt size)
{
    // TODO: consider using atomics here


    SourceLoc beginLoc = m_nextLoc;
    SourceLoc endLoc = beginLoc + size;

    // We need to be able to represent the location that is *at* the end of
    // the input source, so the next available location for a new file
    // must be placed one after the end of this one.

    m_nextLoc = endLoc + 1;

    return SourceRange(beginLoc, endLoc);
}

SourceFile* SourceManager::createSourceFileWithSize(const PathInfo& pathInfo, size_t contentSize)
{
    SourceFile* sourceFile = new SourceFile(this, pathInfo, contentSize);
    m_sourceFiles.add(sourceFile);
    return sourceFile;
}

SourceFile* SourceManager::createSourceFileWithString(
    const PathInfo& pathInfo,
    const String& contents)
{
    SourceFile* sourceFile = new SourceFile(this, pathInfo, contents.getLength());
    m_sourceFiles.add(sourceFile);
    sourceFile->setContents(contents);
    return sourceFile;
}

SourceFile* SourceManager::createSourceFileWithBlob(const PathInfo& pathInfo, ISlangBlob* blob)
{
    SourceFile* sourceFile = new SourceFile(this, pathInfo, blob->getBufferSize());
    m_sourceFiles.add(sourceFile);
    sourceFile->setContents(blob);
    return sourceFile;
}

SourceView* SourceManager::createSourceView(
    SourceFile* sourceFile,
    const PathInfo* pathInfo,
    SourceLoc initiatingSourceLoc)
{
    SourceRange range = allocateSourceRange(sourceFile->getContentSize());

    SourceView* sourceView = nullptr;
    if (pathInfo && (pathInfo->foundPath.getLength() &&
                     sourceFile->getPathInfo().foundPath != pathInfo->foundPath))
    {
        sourceView = new SourceView(sourceFile, range, &pathInfo->foundPath, initiatingSourceLoc);
    }
    else
    {
        sourceView = new SourceView(sourceFile, range, nullptr, initiatingSourceLoc);
    }

    m_sourceViews.add(sourceView);

    return sourceView;
}

SourceView* SourceManager::findSourceView(SourceLoc loc) const
{
    Index hi = m_sourceViews.getCount();
    // It must be in the range of this manager and have associated views for it to possibly be a hit
    if (!getSourceRange().contains(loc) || hi == 0)
    {
        return nullptr;
    }

    // If we don't have very many, we may as well just linearly search
    if (hi <= 8)
    {
        for (int i = 0; i < hi; ++i)
        {
            SourceView* view = m_sourceViews[i];
            if (view->getRange().contains(loc))
            {
                return view;
            }
        }
        return nullptr;
    }

    const SourceLoc::RawValue rawLoc = loc.getRaw();

    // Binary chop to see if we can find the associated SourceUnit
    Index lo = 0;
    while (lo + 1 < hi)
    {
        Index mid = (hi + lo) >> 1;

        SourceView* midView = m_sourceViews[mid];
        if (midView->getRange().contains(loc))
        {
            return midView;
        }

        const SourceLoc::RawValue midValue = midView->getRange().begin.getRaw();
        if (midValue <= rawLoc)
        {
            // The location we seek is at or after this entry
            lo = mid;
        }
        else
        {
            // The location we seek is before this entry
            hi = mid;
        }
    }

    // Check if low is actually a hit
    SourceView* view = m_sourceViews[lo];
    return (view->getRange().contains(loc)) ? view : nullptr;
}

SourceView* SourceManager::findSourceViewRecursively(SourceLoc loc) const
{
    // Start with this manager
    const SourceManager* manager = this;
    do
    {
        SourceView* sourceView = manager->findSourceView(loc);
        // If we found a hit we are done
        if (sourceView)
        {
            return sourceView;
        }
        // Try the parent
        manager = manager->m_parent;
    } while (manager);
    // Didn't find it
    return nullptr;
}

SourceFile* SourceManager::findSourceFileByPathRecursively(const String& name) const
{
    // Start with this manager
    const SourceManager* manager = this;
    do
    {
        SourceFile* sourceFile = manager->findSourceFileByPath(name);
        // If we found a hit we are done
        if (sourceFile)
        {
            return sourceFile;
        }
        // Try the parent
        manager = manager->m_parent;
    } while (manager);
    // Didn't find it
    return nullptr;
}

SourceFile* SourceManager::findSourceFileByPath(const String& name) const
{
    for (auto sourceFile : m_sourceFiles)
    {
        if (sourceFile->getPathInfo().foundPath == name)
        {
            return sourceFile;
        }
    }
    return nullptr;
}

SourceFile* SourceManager::findSourceFile(const String& uniqueIdentity) const
{
    SourceFile* const* filePtr = m_sourceFileMap.tryGetValue(uniqueIdentity);
    return (filePtr) ? *filePtr : nullptr;
}

SourceFile* SourceManager::findSourceFileRecursively(const String& uniqueIdentity) const
{
    const SourceManager* manager = this;
    do
    {
        SourceFile* sourceFile = manager->findSourceFile(uniqueIdentity);
        if (sourceFile)
        {
            return sourceFile;
        }
        manager = manager->m_parent;
    } while (manager);
    return nullptr;
}

SourceFile* SourceManager::findSourceFileByContentRecursively(const char* text)
{
    const SourceManager* manager = this;
    do
    {
        SourceFile* sourceFile = manager->findSourceFileByContent(text);
        if (sourceFile)
        {
            return sourceFile;
        }
        manager = manager->m_parent;
    } while (manager);
    return nullptr;
}

SourceFile* SourceManager::findSourceFileByContent(const char* text) const
{
    for (SourceFile* sourceFile : getSourceFiles())
    {
        auto content = sourceFile->getContent();

        if (text >= content.begin() && text <= content.end())
        {
            return sourceFile;
        }
    }
    return nullptr;
}

void SourceManager::addSourceFile(const String& uniqueIdentity, SourceFile* sourceFile)
{
    SLANG_ASSERT(!findSourceFileRecursively(uniqueIdentity));
    m_sourceFileMap.add(uniqueIdentity, sourceFile);
}

void SourceManager::addSourceFileIfNotExist(const String& uniqueIdentity, SourceFile* sourceFile)
{
    if (findSourceFileRecursively(uniqueIdentity))
        return;
    m_sourceFileMap.addIfNotExists(uniqueIdentity, sourceFile);
}

HumaneSourceLoc SourceManager::getHumaneLoc(SourceLoc loc, SourceLocType type)
{
    SourceView* sourceView = findSourceViewRecursively(loc);
    if (sourceView)
    {
        return sourceView->getHumaneLoc(loc, type);
    }
    else
    {
        return HumaneSourceLoc();
    }
}

PathInfo SourceManager::getPathInfo(SourceLoc loc, SourceLocType type)
{
    SourceView* sourceView = findSourceViewRecursively(loc);
    if (sourceView)
    {
        return sourceView->getPathInfo(loc, type);
    }
    else
    {
        return PathInfo::makeUnknown();
    }
}

SourceLoc::RawValue SourceView::getAbsoluteLocation(SourceLoc location) const
{
    AbsoluteSegment segment;
    if (m_absSegments.getCount())
    {
        if (m_absSegments.getFirst().begin > location)
        {
            segment.begin = m_range.begin;
            segment.absoluteBegin = m_absoluteLocationBase;
        }
        else
        {
            auto it = std::upper_bound(
                          m_absSegments.begin(),
                          m_absSegments.end(),
                          location,
                          [](SourceLoc const& loc, AbsoluteSegment const& seg)
                          { return loc < seg.begin; }) -
                      1;
            segment = *it;
        }
    }
    else
    {
        segment = getLastSegment();
    }
    auto offset = SourceRange(segment.begin, location).getSize();
    return segment.absoluteBegin + offset;
}

SourceLoc::RawValue SourceManager::getAbsoluteLocation(SourceLoc location) const
{
    SourceLoc::RawValue res = 0;
    if (const SourceView* view = findSourceView(location))
    {
        res = view->getAbsoluteLocation(location);
    }
    return res;
}

} // namespace Slang
