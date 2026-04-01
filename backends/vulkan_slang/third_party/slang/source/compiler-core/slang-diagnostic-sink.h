#ifndef SLANG_DIAGNOSTIC_SINK_H
#define SLANG_DIAGNOSTIC_SINK_H

#include "../core/slang-basic.h"
#include "../core/slang-memory-arena.h"
#include "../core/slang-writer.h"
#include "slang-source-loc.h"
#include "slang-token.h"
#include "slang.h"

namespace Slang
{

enum class Severity
{
    Disable,
    Note,
    Warning,
    Error,
    Fatal,
    Internal
};

// Make sure that the slang.h severity constants match those defined here
static_assert(SLANG_SEVERITY_DISABLED == int(Severity::Disable), "mismatched Severity enum values");
static_assert(SLANG_SEVERITY_NOTE == int(Severity::Note), "mismatched Severity enum values");
static_assert(SLANG_SEVERITY_WARNING == int(Severity::Warning), "mismatched Severity enum values");
static_assert(SLANG_SEVERITY_ERROR == int(Severity::Error), "mismatched Severity enum values");
static_assert(SLANG_SEVERITY_FATAL == int(Severity::Fatal), "mismatched Severity enum values");
static_assert(
    SLANG_SEVERITY_INTERNAL == int(Severity::Internal),
    "mismatched Severity enum values");

// TODO(tfoley): move this into a source file...
inline const char* getSeverityName(Severity severity)
{
    switch (severity)
    {
    case Severity::Disable:
        return "ignored";
    case Severity::Note:
        return "note";
    case Severity::Warning:
        return "warning";
    case Severity::Error:
        return "error";
    case Severity::Fatal:
        return "fatal error";
    case Severity::Internal:
        return "internal error";
    default:
        return "unknown error";
    }
}

// A structure to be used in static data describing different
// diagnostic messages.
struct DiagnosticInfo
{
    int id;
    Severity severity;
    char const* name; ///< Unique name
    char const* messageFormat;
};

class Diagnostic
{
public:
    String Message;
    SourceLoc loc;
    int ErrorID;
    Severity severity;

    Diagnostic() { ErrorID = -1; }
    Diagnostic(const String& msg, int id, const SourceLoc& pos, Severity severity)
        : severity(severity)
    {
        Message = msg;
        ErrorID = id;
        loc = pos;
    }
};

struct SourceWarningStateTrackerBase : public RefObject
{
    virtual Severity consumeWarningSeverity(SourceLoc loc, int id, Severity severity) = 0;
};

class Name;

void printDiagnosticArg(StringBuilder& sb, char const* str);

void printDiagnosticArg(StringBuilder& sb, int32_t val);
void printDiagnosticArg(StringBuilder& sb, uint32_t val);

void printDiagnosticArg(StringBuilder& sb, int64_t val);
void printDiagnosticArg(StringBuilder& sb, uint64_t val);

void printDiagnosticArg(StringBuilder& sb, double val);

void printDiagnosticArg(StringBuilder& sb, Slang::String const& str);
void printDiagnosticArg(StringBuilder& sb, Slang::UnownedStringSlice const& str);
void printDiagnosticArg(StringBuilder& sb, Name* name);

void printDiagnosticArg(StringBuilder& sb, TokenType tokenType);
void printDiagnosticArg(StringBuilder& sb, Token const& token);

struct IRInst;
void printDiagnosticArg(StringBuilder& sb, IRInst* irObject);

class Modifier;
void printDiagnosticArg(StringBuilder& sb, Modifier* modifier);

template<typename T>
void printDiagnosticArg(StringBuilder& sb, RefPtr<T> ptr)
{
    printDiagnosticArg(sb, ptr.Ptr());
}

inline SourceLoc getDiagnosticPos(SourceLoc const& pos)
{
    return pos;
}

SourceLoc getDiagnosticPos(Token const& token);


template<typename T>
SourceLoc getDiagnosticPos(RefPtr<T> const& ptr)
{
    return getDiagnosticPos(ptr.Ptr());
}

struct DiagnosticArg
{
    void* data;
    void (*printFunc)(StringBuilder&, void*);

    template<typename T>
    struct Helper
    {
        static void printFunc(StringBuilder& sb, void* data) { printDiagnosticArg(sb, *(T*)data); }
    };

    template<typename T>
    DiagnosticArg(T const& arg)
        : data((void*)&arg), printFunc(&Helper<T>::printFunc)
    {
    }
};

class DiagnosticSink
{
public:
    /// Flags to control some aspects of Diagnostic sink behavior
    typedef uint32_t Flags;
    struct Flag
    {
        enum Enum : Flags
        {
            VerbosePath = 0x1, ///< Will display a more verbose path (if available) - such as a
                               ///< canonical or absolute path
            SourceLocationLine =
                0x2,         ///< If set will display the location line if source is available
            HumaneLoc = 0x4, ///< If set will display humane locs (filename/line number) information
            TreatWarningsAsErrors = 0x8, ///< If set will turn all Warning type messages (after
                                         ///< overrides) into Error type messages
            LanguageServer =
                0x10, ///< If set will format message in a way that is suitable for language server
        };
    };

    /// Used by diagnostic sink to be able to underline tokens. If not defined on the
    /// DiagnosticSink, will only display a caret at the SourceLoc
    typedef UnownedStringSlice (*SourceLocationLexer)(const UnownedStringSlice& text);

    /// Get the total amount of errors that have taken place on this DiagnosticSink
    SLANG_FORCE_INLINE int getErrorCount() { return m_errorCount; }

    template<typename P, typename... Args>
    bool diagnose(P const& pos, DiagnosticInfo const& info, Args const&... args)
    {
        DiagnosticArg as[] = {DiagnosticArg(args)...};
        return diagnoseImpl(getDiagnosticPos(pos), info, sizeof...(args), as);
    }

    template<typename P>
    bool diagnose(P const& pos, DiagnosticInfo const& info)
    {
        // MSVC gets upset with the zero sized array above, so overload that case here
        return diagnoseImpl(getDiagnosticPos(pos), info, 0, nullptr);
    }

    // Useful for notes on existing diagnostics, where it would be redundant to display the same
    // line again. (Ideally we would print the error/warning and notes in one call...)
    template<typename P, typename... Args>
    bool diagnoseWithoutSourceView(P const& pos, DiagnosticInfo const& info, Args const&... args)
    {
        const auto fs = this->getFlags();
        this->resetFlag(Flag::SourceLocationLine);

        auto result = diagnose(pos, info, args...);

        this->setFlags(fs);
        return result;
    }

    // Add a diagnostic with raw text
    // (used when we get errors from a downstream compiler)
    void diagnoseRaw(Severity severity, char const* message);
    void diagnoseRaw(Severity severity, const UnownedStringSlice& message);

    /// During propagation of an exception for an internal
    /// error, note that this source location was involved
    void noteInternalErrorLoc(SourceLoc const& loc);

    /// Create a blob containing diagnostics if there were any errors.
    /// *note* only works if writer is not set, the blob is created from outputBuffer
    SlangResult getBlobIfNeeded(ISlangBlob** outBlob);

    /// Get the source manager used
    SourceManager* getSourceManager() const { return m_sourceManager; }
    /// Set the source manager used for lookup of source locs
    void setSourceManager(SourceManager* inSourceManager) { m_sourceManager = inSourceManager; }

    /// Set the flags
    void setFlags(Flags flags) { m_flags = flags; }
    /// Get the flags
    Flags getFlags() const { return m_flags; }
    /// Set a flag
    void setFlag(Flag::Enum flag) { m_flags |= Flags(flag); }
    /// Reset a flag
    void resetFlag(Flag::Enum flag) { m_flags &= ~Flags(flag); }
    /// Test if flag is set
    bool isFlagSet(Flag::Enum flag) { return (m_flags & Flags(flag)) != 0; }

    /// Sets an override on the severity of a specific diagnostic message (by numeric identifier)
    /// info can be set to nullptr if only to override
    void overrideDiagnosticSeverity(
        int diagnosticId,
        Severity overrideSeverity,
        const DiagnosticInfo* info = nullptr);

    /// Get the (optional) diagnostic sink lexer. This is used to
    /// improve quality of highlighting a locations token. If not set, will just have a single
    /// character caret at location
    SourceLocationLexer getSourceLocationLexer() const { return m_sourceLocationLexer; }

    /// Set the maximum length (in chars) of a source line displayed. Set to 0 for no limit
    void setSourceLineMaxLength(Index length) { m_sourceLineMaxLength = length; }
    Index getSourceLineMaxLength() const { return m_sourceLineMaxLength; }

    /// The parent sink is another sink that will receive diagnostics from this sink.
    void setParentSink(DiagnosticSink* parentSink) { m_parentSink = parentSink; }
    DiagnosticSink* getParentSink() const { return m_parentSink; }

    void setSourceWarningStateTracker(SourceWarningStateTrackerBase* ptr)
    {
        m_sourceWarningStateTracker = ptr;
    }
    RefPtr<SourceWarningStateTrackerBase> getSourceWarningStateTracker() const
    {
        return m_sourceWarningStateTracker;
    }

    /// Reset state.
    /// Resets error counts. Resets the output buffer.
    void reset();

    /// Initialize state.
    void init(SourceManager* sourceManager, SourceLocationLexer sourceLocationLexer);

    /// Ctor
    DiagnosticSink(SourceManager* sourceManager, SourceLocationLexer sourceLocationLexer)
    {
        init(sourceManager, sourceLocationLexer);
    }
    /// Default Ctor
    DiagnosticSink()
        : m_sourceManager(nullptr), m_sourceLocationLexer(nullptr)
    {
    }

    // Public members

    /// The outputBuffer will contain any diagnostics *iff* the writer is *not* set
    StringBuilder outputBuffer;
    /// If a writer is set output will *not* be written to the outputBuffer
    ISlangWriter* writer = nullptr;

protected:
    // Returns true if a diagnostic is actually written.
    bool diagnoseImpl(
        SourceLoc const& pos,
        DiagnosticInfo info,
        int argCount,
        DiagnosticArg const* args);
    bool diagnoseImpl(DiagnosticInfo const& info, const UnownedStringSlice& formattedMessage);

    Severity getEffectiveMessageSeverity(DiagnosticInfo const& info, SourceLoc const& location);

    /// If set all diagnostics (as formatted by *this* sink, will be routed to the parent).
    DiagnosticSink* m_parentSink = nullptr;

    int m_errorCount = 0;
    int m_internalErrorLocsNoted = 0;

    /// If 0, then there is no limit, otherwise max amount of chars of the source line location
    /// We don't know the size of a terminal in general, but for now we'll guess 120.
    Index m_sourceLineMaxLength = 120;

    Flags m_flags = 0;

    // The source manager to use when mapping source locations to file+line info
    SourceManager* m_sourceManager = nullptr;

    SourceLocationLexer m_sourceLocationLexer;

    // Configuration that allows the user to control the severity of certain diagnostic messages
    Dictionary<int, Severity> m_severityOverrides;

    RefPtr<SourceWarningStateTrackerBase> m_sourceWarningStateTracker = nullptr;
};

/// An `ISlangWriter` that writes directly to a diagnostic sink.
class DiagnosticSinkWriter : public AppendBufferWriter
{
public:
    typedef AppendBufferWriter Super;

    DiagnosticSinkWriter(DiagnosticSink* sink)
        : Super(WriterFlag::IsStatic), m_sink(sink)
    {
    }

    // ISlangWriter
    SLANG_NO_THROW virtual SlangResult SLANG_MCALL write(const char* chars, size_t numChars)
        SLANG_OVERRIDE
    {
        m_sink->diagnoseRaw(Severity::Note, UnownedStringSlice(chars, chars + numChars));
        return SLANG_OK;
    }

private:
    DiagnosticSink* m_sink = nullptr;
};

class DiagnosticsLookup : public RefObject
{
public:
    static const Index kArenaInitialSize = 65536;

    /// Will take into account the slice name could be using different conventions
    const DiagnosticInfo* findDiagnosticByName(const UnownedStringSlice& slice) const;
    /// The name must be as defined in the diagnostics exactly, typically lower camel
    const DiagnosticInfo* findDiagnosticByExactName(const UnownedStringSlice& slice) const;

    /// Get a diagnostic by it's id.
    /// NOTE! That it is possible for multiple diagnostics to have the same id. This will return
    /// the first added
    const DiagnosticInfo* getDiagnosticById(Int id) const;

    /// info must stay in scope
    Index add(const DiagnosticInfo* info);
    /// Infos referenced must remain in scope
    void add(const DiagnosticInfo* const* infos, Index infosCount);

    /// NOTE! Name must stay in scope as long as the diagnostics lookup.
    /// If not possible add it to the arena to keep in scope.
    void addAlias(const char* name, const char* diagnosticName);

    /// Get the diagnostics held in this lookup
    const List<const DiagnosticInfo*>& getDiagnostics() const { return m_diagnostics; }

    /// Get the associated arena
    MemoryArena& getArena() { return m_arena; }

    /// NOTE! diagnostics must stay in scope for lifetime of lookup
    DiagnosticsLookup(const DiagnosticInfo* const* diagnostics, Index diagnosticsCount);
    DiagnosticsLookup();

protected:
    void _addName(const char* name, Index diagnosticIndex);

    Index _findDiagnosticIndexByExactName(const UnownedStringSlice& slice) const;

    List<const DiagnosticInfo*> m_diagnostics;

    StringBuilder m_work;
    Dictionary<UnownedStringSlice, Index> m_nameMap;
    Dictionary<Int, Index> m_idMap;

    MemoryArena m_arena;
};

} // namespace Slang

#endif
