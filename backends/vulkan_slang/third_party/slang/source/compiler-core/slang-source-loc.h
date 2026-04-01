// slang-source-loc.h
#ifndef SLANG_SOURCE_LOC_H_INCLUDED
#define SLANG_SOURCE_LOC_H_INCLUDED

#include "../core/slang-basic.h"
#include "../core/slang-castable.h"
#include "../core/slang-crypto.h"
#include "../core/slang-memory-arena.h"
#include "../core/slang-string-slice-pool.h"
#include "slang-com-ptr.h"
#include "slang-source-map.h"
#include "slang.h"

namespace Slang
{

/** Overview:

There needs to be a mechanism where we can easily and quickly track a specific locations in any
source file used during a compilation. This is important because that original location is
meaningful to the user as it relates to their original source. Thus SourceLoc are used so we can
display meaningful and accurate errors/warnings as well as being able to always map generated code
locations back to their origins.

A 'SourceLoc' along with associated structures (SourceView, SourceFile, SourceMangager) this can
pinpoint the location down to the byte across the compilation. This could be achieved by storing for
every token and instruction the file, line and column number came from. The SourceLoc is used in
lots of places - every AST node, every Token from the lexer, every IRInst - so we really want to
make it small. So for this reason we actually encode SourceLoc as a single integer and then use the
associated structures when needed to determine what the location actually refers to - the source
file, line and column number, or in effect the byte in the original file.

Unfortunately there is extra complications. When a source is parsed it's interpretation (in terms of
how a piece of source maps to an 'original' file etc) can be overridden - for example by using #line
directives. Moreover a single source file can be parsed multiple times. When it's parsed multiple
times the interpretation of the mapping (#line directives for example) can change. This is the
purpose of the SourceView - it holds the interpretation of a source file for a specific Lex/Parse.

Another complication is that not all 'source' comes from SourceFiles, a macro expansion, may
generate new 'source' we need to handle this, but also be able to have a SourceLoc map to the
expansion unambiguously. This is handled by creating a SourceFile and SourceView that holds only the
macro generated specific information.

SourceFile - Is the immutable text contents of a file (or perhaps some generated source - say from
doing a macro substitution) SourceView - Tracks a single parse of a SourceFile. Each SourceView
defines a range of source locations used. If a SourceFile is parsed twice, two SourceViews are
created, with unique SourceRanges. This is so that it is possible to tell which specific parse a
SourceLoc is from - and so know the right interpretation for that lex/parse.
*/

struct PathInfo
{
    typedef PathInfo ThisType;

    /// To be more rigorous about where a path comes from, the type identifies what a paths origin
    /// is
    enum class Type : uint8_t
    {
        Unknown,    ///< The path is not known
        Normal,     ///< Normal has both path and uniqueIdentity
        FoundPath,  ///< Just has a found path (uniqueIdentity is unknown, or even 'unknowable')
        FromString, ///< Created from a string (so found path might not be defined and should not be
                    ///< taken as to map to a loaded file)
        TokenPaste, ///< No paths, just created to do a macro expansion
        TypeParse,  ///< No path, just created to do a type parse
        CommandLine, ///< A macro constructed from the command line
    };

    /// True if has a canonical path
    SLANG_FORCE_INLINE bool hasUniqueIdentity() const
    {
        return type == Type::Normal && uniqueIdentity.getLength() > 0;
    }
    /// True if has a regular found path
    SLANG_FORCE_INLINE bool hasFoundPath() const
    {
        return (type == Type::Normal || type == Type::FoundPath || type == Type::FromString) &&
               foundPath.getLength() > 0;
    }
    /// True if has a found path that has originated from a file (as opposed to string or some other
    /// origin)
    SLANG_FORCE_INLINE bool hasFileFoundPath() const
    {
        return (type == Type::Normal || type == Type::FoundPath) && foundPath.getLength() > 0;
    }
    /// Get the 'name'/path of the item. Will return an empty string if not applicable or not set.
    String getName() const;

    bool operator==(const ThisType& rhs) const;
    bool operator!=(const ThisType& rhs) const { return !(*this == rhs); }

    /// Returns the 'most unique' identity for the path. If has a 'uniqueIdentity' returns that,
    /// else the foundPath, else "".
    const String getMostUniqueIdentity() const;

    /// Append to out, how to display the path
    void appendDisplayName(StringBuilder& out) const;

    // So simplify construction. In normal usage it's safer to use make methods over constructing
    // directly.
    static PathInfo makeUnknown() { return PathInfo{Type::Unknown, String(), String()}; }
    static PathInfo makeTokenPaste() { return PathInfo{Type::TokenPaste, "token paste", String()}; }
    static PathInfo makeNormal(const String& foundPathIn, const String& uniqueIdentity)
    {
        SLANG_ASSERT(uniqueIdentity.getLength() > 0 && foundPathIn.getLength() > 0);
        return PathInfo{Type::Normal, foundPathIn, uniqueIdentity};
    }
    static PathInfo makePath(const String& pathIn)
    {
        SLANG_ASSERT(pathIn.getLength() > 0);
        return PathInfo{Type::FoundPath, pathIn, String()};
    }
    static PathInfo makeTypeParse() { return PathInfo{Type::TypeParse, "type string", String()}; }
    static PathInfo makeCommandLine()
    {
        return PathInfo{Type::CommandLine, "command line", String()};
    }
    static PathInfo makeFromString(const String& userPath)
    {
        return PathInfo{Type::FromString, userPath, String()};
    }

    Type type;             ///< The type of path
    String foundPath;      ///< The path where the file was found (might contain relative elements)
    String uniqueIdentity; ///< The unique identity of the file on the path found
};

class SourceLoc
{
public:
    typedef SourceLoc ThisType;
    typedef uint32_t RawValue;

private:
    RawValue raw;

public:
    SourceLoc()
        : raw(0)
    {
    }

    SourceLoc(SourceLoc const& loc)
        : raw(loc.raw)
    {
    }

    SLANG_FORCE_INLINE bool operator==(const ThisType& rhs) const { return raw == rhs.raw; }
    SLANG_FORCE_INLINE bool operator!=(const ThisType& rhs) const { return !(raw == rhs.raw); }
    SLANG_FORCE_INLINE bool operator<(const ThisType& rhs) const { return raw < rhs.raw; }
    SLANG_FORCE_INLINE bool operator>(const ThisType& rhs) const { return raw > rhs.raw; }
    SLANG_FORCE_INLINE bool operator<=(const ThisType& rhs) const { return raw <= rhs.raw; }
    SLANG_FORCE_INLINE bool operator>=(const ThisType& rhs) const { return raw >= rhs.raw; }

    RawValue getRaw() const { return raw; }
    void setRaw(RawValue value) { raw = value; }

    static SourceLoc fromRaw(RawValue value)
    {
        SourceLoc result;
        result.setRaw(value);
        return result;
    }

    bool isValid() const { return raw != 0; }
    SourceLoc& operator=(const ThisType& rhs) = default;
};

inline SourceLoc operator+(SourceLoc loc, Int offset)
{
    return SourceLoc::fromRaw(SourceLoc::RawValue(Int(loc.getRaw()) + offset));
}

// A range of locations in the input source
struct SourceRange
{
    /// True if the loc is in the range. Range is inclusive on begin to end.
    bool contains(SourceLoc loc) const
    {
        const auto rawLoc = loc.getRaw();
        return rawLoc >= begin.getRaw() && rawLoc <= end.getRaw();
    }
    /// Get the total size
    SourceLoc::RawValue getSize() const { return end.getRaw() - begin.getRaw(); }

    /// Get the offset of a loc in this range
    int getOffset(SourceLoc loc) const
    {
        SLANG_ASSERT(contains(loc));
        return int(loc.getRaw() - begin.getRaw());
    }

    /// Convert an offset to a loc
    SourceLoc getSourceLocFromOffset(uint32_t offset) const
    {
        SLANG_ASSERT(offset <= getSize());
        return begin + offset;
    }

    SourceRange() {}

    SourceRange(SourceLoc loc)
        : begin(loc), end(loc)
    {
    }

    SourceRange(SourceLoc begin, SourceLoc end)
        : begin(begin), end(end)
    {
    }

    SourceLoc begin;
    SourceLoc end;
};

/// Source maps associated with files are could be of different uses. We use the SourceMapKind
/// to indicate the usage.
///
/// If the source map is obfuscated reasonable/desirable to ignore them on emit (if we didn't we
/// leak information, and we don't emit into the locations in the obfuscated intermediate "file").
enum class SourceMapKind
{
    Normal,     ///< A regular source map
    Obfuscated, ///< Obfuscated source map
};

// Pre-declare
struct SourceManager;

// A logical or physical storage object for a range of input code
// that has logically contiguous source locations.
class SourceFile
{
public:
    struct OffsetRange
    {
        /// We need a value to indicate an invalid range. We can't use 0 as that is valid for an
        /// offset range We can't use a negative number, and don't want to make signed so we get the
        /// full 32-bits. So we just use the max value as invalid
        static const uint32_t kInvalid = 0xffffffff;

        /// True if the range is valid
        SLANG_FORCE_INLINE bool isValid() const { return end >= start && start != kInvalid; }
        /// True if offset is within range (inclusively)
        SLANG_FORCE_INLINE bool containsInclusive(uint32_t offset) const
        {
            return offset >= start && offset <= end;
        }

        /// Get the count
        SLANG_FORCE_INLINE uint32_t getCount() const { return end - start; }

        /// Return an invalid range.
        static OffsetRange makeInvalid() { return OffsetRange{kInvalid, kInvalid}; }

        uint32_t start;
        uint32_t end;
    };

    /// Returns the line break offsets (in bytes from start of content)
    /// Note that this is lazily evaluated - the line breaks are only calculated on the first
    /// request
    const List<uint32_t>& getLineBreakOffsets();

    /// Returns true if the offset is on the specified line
    /// NOTE! If offsets are not fully setup (because we don't have source), will only be correct
    /// for lines that have offsets
    bool isOffsetOnLine(uint32_t offset, Index lineIndex);

    /// Get the line containing the offset. Requires that content is available, else will return an
    /// empty slice.
    UnownedStringSlice getLineContainingOffset(uint32_t offset);

    /// Get the line at the specified line index. Requires that content is available, else will
    /// return an empty slice.
    UnownedStringSlice getLineAtIndex(Index lineIndex);

    /// Get the offset range at the specified line index. Works without content.
    OffsetRange getOffsetRangeAtLineIndex(Index lineIndex);

    /// Set the line break offsets
    void setLineBreakOffsets(const uint32_t* offsets, UInt numOffsets);

    /// Calculate the line based on the offset
    int calcLineIndexFromOffset(int offset);

    /// Calculate the offset (in bytes) for a line
    int calcColumnOffset(int line, int offset);

    /// Given a line and offset (in bytes for the whole file), return the column index, taking into
    /// account tabs and utf8 encoding. Passing tabSize uses the default tab size (currently tab set
    /// to 1)
    int calcColumnIndex(int line, int offset, int tabSize = -1);

    /// Get the content holding blob
    ISlangBlob* getContentBlob() const { return m_contentBlob; }

    /// True if has full set content
    bool hasContent() const { return m_contentBlob != nullptr; }

    /// Get the content size
    size_t getContentSize() const { return m_contentSize; }

    /// Get the content
    const UnownedStringSlice& getContent() const { return m_content; }

    /// Get path info
    const PathInfo& getPathInfo() const { return m_pathInfo; }

    /// Set the content as a blob
    void setContents(ISlangBlob* blob);
    /// Set the content as a string
    void setContents(const String& content);

    /// Calculate a display path -> can canonicalize if necessary
    String calcVerbosePath() const;

    /// Get the source manager this was created on
    SourceManager* getSourceManager() const { return m_sourceManager; }

    /// Get the source map associated with this file. If it's set when doing
    /// lookup for source locations, the source map will be used
    IBoxValue<SourceMap>* getSourceMap() const { return m_sourceMap; }
    /// Get the source map kind
    SourceMapKind getSourceMapKind() const { return m_sourceMapKind; }

    /// Set a source map
    void setSourceMap(IBoxValue<SourceMap>* sourceMap, SourceMapKind sourceMapKind)
    {
        m_sourceMap = sourceMap;
        m_sourceMapKind = sourceMapKind;
    }

    /// Ctor
    SourceFile(SourceManager* sourceManager, const PathInfo& pathInfo, size_t contentSize);
    /// Dtor
    ~SourceFile();

    SHA1::Digest getDigest();

protected:
    SourceManager* m_sourceManager; ///< The source manager this belongs to
    PathInfo
        m_pathInfo; ///< The path The logical file path to report for locations inside this span.

    ComPtr<ISlangBlob> m_contentBlob; ///< A blob that owns the storage for the file contents. If
                                      ///< nullptr, there is no contents
    UnownedStringSlice m_content;     ///< The actual contents of the file.
    size_t m_contentSize;             ///< The size of the actual contents

    SHA1::Digest m_digest;

    // In order to speed up lookup of line number information,
    // we will cache the starting offset of each line break in
    // the input file:
    List<uint32_t> m_lineBreakOffsets;

    // If set then the locations in this file are really from locations from elsewhere,
    // where the SourceMap specifies that mapping
    ComPtr<IBoxValue<SourceMap>> m_sourceMap;
    // What kind of source map it is (if  there is one)
    SourceMapKind m_sourceMapKind = SourceMapKind::Normal;
};

enum class SourceLocType
{
    Nominal, ///< The normal interpretation which takes into account #line directives and source
             ///< maps
    Actual,  ///< Ignores #line directives/source maps - and is the location as seen in the actual
             ///< file
    Emit,    ///< Behaves the same as `Nominal` but ignores source maps. Used for Emit source
             ///< locations.
};

// A source location in a format a human might like to see
struct HumaneSourceLoc
{
    PathInfo pathInfo = PathInfo::makeUnknown();
    Int line = 0;
    Int column = 0;
};

// Same as HumaneSourceLoc but stores the path only as a handle.
struct HandleSourceLoc
{
    StringSlicePool::Handle pathHandle = StringSlicePool::Handle(0);
    Int line = 0;
    Int column = 0;
};

/* A SourceView maps to a single span of SourceLoc range and is equivalent to a single include or
more precisely use of a source file. It is distinct from a SourceFile - because a SourceFile may be
included multiple times, with different interpretations (depending on #defines for example).
*/
class SourceView
{
public:
    // Each entry represents some contiguous span of locations that
    // all map to the same logical file.
    struct Entry
    {
        /// True if this resets the line numbering. It is distinct from a m_lineAdjust being 0,
        /// because it also means the path returns to the default.
        bool isDefault() const { return m_pathHandle == StringSlicePool::Handle(0); }

        SourceLoc m_startLoc;                 ///< Where does this entry begin?
        StringSlicePool::Handle m_pathHandle; ///< What is the presumed path for this entry. If 0 it
                                              ///< means there is no path.
        int32_t m_lineAdjust; ///< Adjustment to apply to source line numbers when printing presumed
                              ///< locations. Relative to the line number in the underlying file.
    };

    // Represents a segment of a source.
    // All SourceLoc in segment are linearly mapped relative to absoluteBegin
    struct AbsoluteSegment
    {
        // SourceLoc in the file range.
        SourceLoc begin = {};
        // Location in the absolute mapping of locations ordered by includes.
        SourceLoc::RawValue absoluteBegin = {};
    };

    // Set the base of the absolute location mapping for this SourceView.
    void setAbsoluteLocationBase(SourceLoc::RawValue absLoc) { m_absoluteLocationBase = absLoc; }

    AbsoluteSegment getLastSegment() const
    {
        AbsoluteSegment res;
        if (m_absSegments.getCount())
        {
            res = m_absSegments.getLast();
        }
        else
        {
            res.begin = m_range.begin;
            res.absoluteBegin = m_absoluteLocationBase;
        }
        return res;
    }

    // Add a segment of absolute mapping after the previous ones.
    void addAbsoluteSegment(SourceLoc begin, SourceLoc::RawValue absoluteBegin)
    {
        SLANG_ASSERT(m_range.contains(begin));
        SLANG_ASSERT(getLastSegment().begin < begin);
        AbsoluteSegment seg;
        seg.begin = begin;
        seg.absoluteBegin = absoluteBegin;
        m_absSegments.add(seg);
    }

    // Maps a SourceLoc inside this SourceView to a unique absolute location ordered by includes.
    SourceLoc::RawValue getAbsoluteLocation(SourceLoc loc) const;

    /// Given a sourceLoc finds the entry associated with it. If returns -1 then no entry is
    /// associated with this location, and therefore the location should be interpreted as an offset
    /// into the underlying sourceFile.
    int findEntryIndex(SourceLoc sourceLoc) const;

    /// Add a line directive for this view. The directiveLoc must of course be in this SourceView
    /// The path handle, must have been constructed on the SourceManager associated with the view
    /// NOTE! Directives are assumed to be added IN ORDER during parsing such that every
    /// directiveLoc > previous
    void addLineDirective(SourceLoc directiveLoc, StringSlicePool::Handle pathHandle, int line);
    void addLineDirective(SourceLoc directiveLoc, const String& path, int line);

    /// Removes any corrections on line numbers and reverts to the source files path
    void addDefaultLineDirective(SourceLoc directiveLoc);

    /// Get the range that this view applies to
    const SourceRange& getRange() const { return m_range; }
    /// Get the entries
    const List<Entry>& getEntries() const { return m_entries; }
    /// Set the entries list
    void setEntries(const Entry* entries, UInt numEntries)
    {
        m_entries.clear();
        m_entries.addRange(entries, numEntries);
    }

    /// Get the source file holds the contents this view
    SourceFile* getSourceFile() const { return m_sourceFile; }
    /// Get the source manager
    SourceManager* getSourceManager() const { return m_sourceFile->getSourceManager(); }

    /// Get the associated 'content' (the source text)
    const UnownedStringSlice& getContent() const { return m_sourceFile->getContent(); }

    /// Get the size of the content
    size_t getContentSize() const { return m_sourceFile->getContentSize(); }

    /// Get the humane location
    /// Type determines if the location wanted is the original, or the 'normal' (which modifys
    /// behavior based on #line directives)
    HumaneSourceLoc getHumaneLoc(SourceLoc loc, SourceLocType type = SourceLocType::Nominal);

    /// Get the humane location, but store the path as a handle
    HandleSourceLoc getHandleLoc(SourceLoc loc, SourceLocType type = SourceLocType::Nominal);


    /// Get the path associated with a location
    PathInfo getPathInfo(SourceLoc loc, SourceLocType type = SourceLocType::Nominal);

    /// Get the initiating source location - that is the source location that caused the this
    /// SourceView to be created Can be SourceLoc(0) if there is no initiating location. For example
    /// for a #include - the view's initiating source loc for the view that is the contents of the
    /// view will be the location of the #include in the source. For the original source file (ie
    /// not an include) - the view will have an initiating source loc of SourceLoc(0)
    SourceLoc getInitiatingSourceLoc() const { return m_initiatingSourceLoc; }

    /// Gets the pathInfo for this view. It may be different from the m_sourceFile's if the path has
    /// been overridden by m_viewPath
    PathInfo getViewPathInfo() const;

    /// Ctor
    SourceView(
        SourceFile* sourceFile,
        SourceRange range,
        const String* viewPath,
        SourceLoc initiatingSourceLoc)
        : m_range(range), m_sourceFile(sourceFile), m_initiatingSourceLoc(initiatingSourceLoc)
    {
        if (viewPath)
        {
            m_viewPath = *viewPath;
        }
    }

protected:
    /// Get the pathInfo from a string handle. If it's 0, it will return the _getPathInfo
    PathInfo _getPathInfoFromHandle(StringSlicePool::Handle pathHandle) const;

    SlangResult _findSourceMapLoc(SourceLoc loc, SourceLocType type, HandleSourceLoc& outLoc);

    String m_viewPath; ///< Path to this view. If empty the path is the path to the SourceView

    SourceLoc m_initiatingSourceLoc; ///< An optional source loc that defines where this view was
                                     ///< initiated from. SourceLoc(0) if not defined.

    SourceRange m_range;      ///< The range that this SourceView applies to
    SourceFile* m_sourceFile; ///< The source file. Can hold the line breaks
    List<Entry> m_entries;    ///< An array entries describing how we should interpret a range,
                              ///< starting from the start location.
    SourceLoc::RawValue m_absoluteLocationBase = 0; ///< Base of the absolute location mapping.
    List<AbsoluteSegment> m_absSegments;            ///< Segments of absolute location mapping.
};

struct SourceManager
{
    // Initialize a source manager, with an optional parent
    void initialize(SourceManager* parent, ISlangFileSystemExt* fileSystemExt);

    /// Allocate a range of SourceLoc locations, these can be used to identify a specific location
    /// in the source
    SourceRange allocateSourceRange(UInt size);

    /// Returns the loc for start of next allocation
    SourceLoc getNextRangeStart() const { return m_nextLoc; }

    /// Create a SourceFile defined with the specified path, and content held within a blob
    SourceFile* createSourceFileWithSize(const PathInfo& pathInfo, size_t contentSize);
    SourceFile* createSourceFileWithString(const PathInfo& pathInfo, const String& contents);
    SourceFile* createSourceFileWithBlob(const PathInfo& pathInfo, ISlangBlob* blob);

    /// Get the humane source location
    HumaneSourceLoc getHumaneLoc(SourceLoc loc, SourceLocType type = SourceLocType::Nominal);

    /// Get the path associated with a location
    PathInfo getPathInfo(SourceLoc loc, SourceLocType type = SourceLocType::Nominal);

    /// Create a new source view from a file
    /// @param sourceFile is the source file that contains the source
    /// @param pathInfo is path used to read the file from
    /// @param initiatingSourceLoc the (optional) location in the source that led the the creation
    /// of this view. If there isn't an initiating source location pass SourceLoc(0)s
    SourceView* createSourceView(
        SourceFile* sourceFile,
        const PathInfo* pathInfo,
        SourceLoc initiatingSourceLoc);

    /// Find a view by a source file location.
    /// If not found in this manager will look in the parent SourceManager
    /// Returns nullptr if not found.
    SourceView* findSourceViewRecursively(SourceLoc loc) const;

    /// Find the SourceView associated with this manager for a specified location
    /// Returns nullptr if not found.
    SourceView* findSourceView(SourceLoc loc) const;

    /// Searches this manager, and then the parent to see if can find a match for path.
    /// If not found returns nullptr.
    SourceFile* findSourceFileRecursively(const String& uniqueIdentity) const;
    /// Find if the source file is defined on this manager.
    SourceFile* findSourceFile(const String& uniqueIdentity) const;

    /// Find a source file by path.
    SourceFile* findSourceFileByPath(const String& name) const;
    /// Find a source file by path recursively.
    SourceFile* findSourceFileByPathRecursively(const String& name) const;

    /// Searches this manager, and then the parent to see if can find a match
    SourceFile* findSourceFileByContentRecursively(const char* text);
    /// Find the source file that contains *the memory* text points to.
    SourceFile* findSourceFileByContent(const char* text) const;

    /// Get the file system associated with this source manager
    ISlangFileSystemExt* getFileSystemExt() const { return m_fileSystemExt; }
    /// Get the file system associated with this source manager
    void setFileSystemExt(ISlangFileSystemExt* fileSystemExt) { m_fileSystemExt = fileSystemExt; }

    /// Add a source file, uniqueIdentity must be unique for this manager AND any parents
    void addSourceFile(const String& uniqueIdentity, SourceFile* sourceFile);
    void addSourceFileIfNotExist(const String& uniqueIdentity, SourceFile* sourceFile);

    // Maps a SourceLoc to an absolute location
    SourceLoc::RawValue getAbsoluteLocation(SourceLoc location) const;

    /// Get the slice pool
    StringSlicePool& getStringSlicePool() { return m_slicePool; }

    /// Get the source range for just this manager
    /// Caution - the range will change if allocations are made to this manager.
    SourceRange getSourceRange() const { return SourceRange(m_startLoc, m_nextLoc); }

    /// Get the parent manager to this manager. Returns nullptr if there isn't any.
    SourceManager* getParent() const { return m_parent; }

    /// A memory arena to hold allocations that are in scope for the same time as SourceManager
    MemoryArena* getMemoryArena() { return &m_memoryArena; }

    /// Allocate a string slice
    UnownedStringSlice allocateStringSlice(const UnownedStringSlice& slice);

    /// Get all of the source files
    const List<SourceFile*>& getSourceFiles() const { return m_sourceFiles; }

    /// Get the source views
    const List<SourceView*>& getSourceViews() const { return m_sourceViews; }

    /// Resets state. Will release all views/source
    void reset();

    SourceManager()
        : m_memoryArena(2048), m_slicePool(StringSlicePool::Style::Default)
    {
    }
    ~SourceManager();

protected:
    void _resetLoc();
    void _resetSource();

    // The first location available to this source manager
    // (may not be the first location of all, because we might
    // have a parent source manager)
    SourceLoc m_startLoc;

    // The "parent" source manager that owns locations ahead of `startLoc`
    SourceManager* m_parent = nullptr;

    // The location to be used by the next source file to be loaded
    SourceLoc m_nextLoc;

    // All of the SourceViews constructed on this SourceManager. These are held in increasing order
    // of range, so can find by doing a binary chop.
    List<SourceView*> m_sourceViews;
    // All of the SourceFiles constructed on this SourceManager. This owns the SourceFile.
    List<SourceFile*> m_sourceFiles;

    StringSlicePool m_slicePool;

    // Memory arena that can be used for holding data to held in scope as long as the Source is
    // Can be used for storing the decoded contents of Token. Content for example.
    MemoryArena m_memoryArena;

    // Maps uniqueIdentities to source files
    Dictionary<String, SourceFile*> m_sourceFileMap;

    ComPtr<ISlangFileSystemExt> m_fileSystemExt;
};

} // namespace Slang

#endif
